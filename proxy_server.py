"""
Middleware proxy that sits between OpenCode and mlx-openai-server.
Injects tool-calling prompts, strips tools from payload, and parses
[TOOL_CALL] blocks from response into OpenAI tool_calls format.
"""
import re
import uuid
from contextlib import asynccontextmanager

import orjson
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, Response

# Backend mlx-openai-server URL
BACKEND_URL = "http://localhost:8000"

# Pre-compiled regex patterns (Change 3)
RE_TOOL_CALL = re.compile(r'\[TOOL_CALL\]\s*(.*?)\s*\[/TOOL_CALL\]', re.DOTALL)
RE_TOOL_CALL_STRIP = re.compile(r'\[TOOL_CALL\]\s*.*?\s*\[/TOOL_CALL\]', re.DOTALL)
RE_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)
RE_THINK_CLOSE = re.compile(r'</think>')
RE_BARE_JSON = re.compile(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:', re.DOTALL)


# Global connection-pooled httpx client (Change 1)
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=300)
    yield
    await app.state.client.aclose()


app = FastAPI(title="MLX Tool-Call Proxy", lifespan=lifespan)


def build_tool_system_prompt(tools: list) -> str:
    """Build a system prompt that instructs the model to use [TOOL_CALL] format."""
    tool_descriptions = []
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        required = params.get("required", [])
        props = params.get("properties", {})

        param_lines = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_lines.append(f'    - {pname} ({ptype}{req}): {pdesc}')

        tool_descriptions.append(
            f"Tool: {name}\n  Description: {desc}\n  Parameters:\n" + "\n".join(param_lines)
        )

    tools_block = "\n\n".join(tool_descriptions)

    return (
        "/no_think\nYou are a helpful AI coding assistant. Be concise. No reasoning. No thinking.\n\n"
        "TOOL FORMAT: wrap calls in [TOOL_CALL] and [/TOOL_CALL] brackets. Include ALL required params.\n"
        '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls -la", "description": "List files"}}\n[/TOOL_CALL]\n\n'
        f"TOOLS:\n{tools_block}"
    )


def parse_tool_calls(text: str) -> tuple[str | None, list | None]:
    """
    Parse [TOOL_CALL] blocks from model output.
    Returns (cleaned_text, tool_calls) where tool_calls is None if no calls found.
    """
    matches = RE_TOOL_CALL.findall(text)

    if not matches:
        return text, None

    tool_calls = []
    for idx, match in enumerate(matches):
        try:
            parsed = orjson.loads(match.strip())
            name = parsed.get("name", "unknown")
            arguments = parsed.get("arguments", {})
            tool_calls.append({
                "index": idx,
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": orjson.dumps(arguments).decode() if isinstance(arguments, dict) else str(arguments)
                }
            })
        except (orjson.JSONDecodeError, ValueError):
            continue

    if not tool_calls:
        return text, None

    # Remove tool_call blocks from the text
    remaining = RE_TOOL_CALL_STRIP.sub('', text).strip()
    # Strip thinking tags (Change 5: kept here since parse_tool_calls is also
    # called from rewrite_response where pre-stripping already happened, but
    # removing would change behavior for direct callers)
    remaining = RE_THINK_BLOCK.sub('', remaining).strip()
    remaining = RE_THINK_CLOSE.sub('', remaining).strip()

    return remaining if remaining else None, tool_calls


def rewrite_request(payload: dict) -> tuple[dict, list]:
    """Rewrite the request: inject tool prompt, strip tools/tool_choice.
    Returns (payload, original_tools)."""
    tools = payload.get("tools", []) or []
    if not tools:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
        return payload, [], {}

    messages = payload.get("messages", [])

    # Build tool-calling system prompt
    tool_prompt = build_tool_system_prompt(tools)

    # Inject into system message
    if messages and messages[0].get("role") == "system":
        orig = messages[0]["content"]
        if isinstance(orig, str):
            messages[0]["content"] = tool_prompt + "\n\n" + orig
        else:
            messages[0]["content"] = tool_prompt
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    # Save tool schemas for post-processing
    tool_names = []
    tool_schemas = {}
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "unknown")
        tool_names.append(name)
        tool_schemas[name] = fn.get("parameters", {})

    # Clean up message history: the backend doesn't understand tool_calls or tool role
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant" and msg.get("tool_calls"):
            # Convert assistant tool_call message into plain text
            tc_texts = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tc_texts.append(f'[TOOL_CALL]\n{{"name": "{fn.get("name")}", "arguments": {fn.get("arguments", "{}")}}}\n[/TOOL_CALL]')
            content = msg.get("content") or ""
            content = content + "\n" + "\n".join(tc_texts) if content else "\n".join(tc_texts)
            cleaned_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            # Convert tool result message into user message
            tool_name = msg.get("name", "tool")
            tool_content = msg.get("content", "")
            cleaned_messages.append({
                "role": "user",
                "content": f"[Tool Result from {tool_name}]:\n{tool_content}"
            })

        else:
            # Pass through as-is (but strip any tool_calls field)
            clean = {k: v for k, v in msg.items() if k != "tool_calls"}
            cleaned_messages.append(clean)

    # Strip tools and tool_choice so mlx server doesn't choke
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["messages"] = cleaned_messages

    return payload, tool_names, tool_schemas


def rewrite_response(data: dict, tool_names: list, tool_schemas: dict = None) -> dict:
    """Rewrite the response: parse [TOOL_CALL] blocks into OpenAI tool_calls format."""
    try:
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            content = msg.get("content")
            if not content:
                continue

            remaining_text, tool_calls = parse_tool_calls(content)

            # Fallback: if model didn't use XML tags but we detect a bare JSON tool call
            if not tool_calls:
                bare_json = RE_BARE_JSON.search(content)
                if bare_json:
                    # Try to extract the full JSON object
                    start = bare_json.start()
                    # Find matching closing brace
                    depth = 0
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                try:
                                    parsed = orjson.loads(content[start:i+1])
                                    name = parsed.get("name", "unknown")
                                    arguments = parsed.get("arguments", {})
                                    tool_calls = [{
                                        "index": 0,
                                        "id": f"call_{uuid.uuid4().hex[:24]}",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": orjson.dumps(arguments).decode() if isinstance(arguments, dict) else str(arguments)
                                        }
                                    }]
                                    remaining_text = None
                                    print(f"[proxy] Fallback: parsed bare JSON tool call for '{name}'")
                                except (orjson.JSONDecodeError, ValueError):
                                    pass
                                break

            if tool_calls:
                # Post-process: fill in missing required fields from schema
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        args = orjson.loads(fn.get("arguments", "{}"))
                    except (orjson.JSONDecodeError, ValueError, TypeError):
                        args = {}

                    schema = (tool_schemas or {}).get(fn_name, {})
                    required = schema.get("required", [])
                    props = schema.get("properties", {})

                    changed = False
                    for req_field in required:
                        if req_field not in args:
                            # Generate a sensible default
                            field_type = props.get(req_field, {}).get("type", "string")
                            if field_type == "string":
                                # Use a descriptive default based on other args
                                args[req_field] = f"{fn_name}: {', '.join(str(v) for v in args.values())}"
                            elif field_type == "boolean":
                                args[req_field] = False
                            elif field_type == "number" or field_type == "integer":
                                args[req_field] = 0
                            changed = True
                            print(f"[proxy] Auto-filled missing required field '{req_field}' for tool '{fn_name}'")

                    if changed:
                        fn["arguments"] = orjson.dumps(args).decode()
                    tc["function"] = fn

                msg["content"] = remaining_text
                msg["tool_calls"] = tool_calls
                choice["message"] = msg
                choice["finish_reason"] = "tool_calls"
            else:
                print(f"[proxy] WARNING: No tool_call found in model output")
    except Exception as e:
        print(f"[proxy] Response rewrite error: {e}")

    return data


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    has_tools = bool(payload.get("tools"))
    is_stream = payload.get("stream", False)

    print(f"[proxy] POST /v1/chat/completions stream={is_stream} tools={has_tools} model={payload.get('model')}")

    # Inject /no_think to disable reasoning for all requests
    messages = payload.get("messages", [])
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if isinstance(content, str) and "/no_think" not in content:
            messages[0]["content"] = "/no_think\n" + content
    elif messages:
        messages.insert(0, {"role": "system", "content": "/no_think\nBe concise."})
    payload["messages"] = messages

    # Rewrite request to inject tool prompts and strip tools
    payload, tool_names, tool_schemas = rewrite_request(payload)

    # For tool-call requests, force non-streaming and cap tokens
    if has_tools:
        payload["stream"] = False
        payload["max_tokens"] = min(payload.get("max_tokens", 4096), 4096)

    client = request.app.state.client  # Change 1: reuse pooled client

    if is_stream and not has_tools:
        # Stream: proxy SSE directly from backend
        async def stream_backend():
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            stream_backend(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming
        resp = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code != 200:
            print(f"[proxy] Backend error {resp.status_code}: {resp.text[:500]}")
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

        data = resp.json()

        # Strip thinking tags before any processing (uses pre-compiled regexes)
        try:
            raw = data["choices"][0]["message"]["content"]
            cleaned = RE_THINK_BLOCK.sub('', raw)
            cleaned = RE_THINK_CLOSE.sub('', cleaned).strip()
            data["choices"][0]["message"]["content"] = cleaned
            print(f"[proxy] RAW MODEL OUTPUT:\n{raw[:500]}")
        except Exception:
            pass

        if has_tools:
            data = rewrite_response(data, tool_names, tool_schemas)

        # If the original request wanted streaming, wrap our JSON response as SSE
        if is_stream and has_tools:
            def fake_stream():
                # Change 4: Pre-build base chunk dict once, use orjson
                base = {
                    "id": data.get("id", f"chatcmpl_{uuid.uuid4().hex}"),
                    "object": "chat.completion.chunk",
                    "created": data.get("created", 0),
                    "model": data.get("model", ""),
                }
                for choice in data.get("choices", []):
                    msg = choice.get("message", {})
                    tool_calls = msg.get("tool_calls")
                    content = msg.get("content")

                    # First chunk: role
                    role_delta = {"role": "assistant"}
                    if not tool_calls:
                        role_delta["content"] = ""
                    chunk = {**base, "choices": [{"index": 0, "delta": role_delta, "finish_reason": None}]}
                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"

                    # Content chunk if any
                    if content:
                        chunk["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                        yield f"data: {orjson.dumps(chunk).decode()}\n\n"

                    # Tool calls: send each tool call in its own delta chunk
                    if tool_calls:
                        for tc in tool_calls:
                            tc_delta = {
                                "index": tc.get("index", 0),
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                }
                            }
                            chunk["choices"] = [{"index": 0, "delta": {"tool_calls": [tc_delta]}, "finish_reason": None}]
                            yield f"data: {orjson.dumps(chunk).decode()}\n\n"

                    # Final chunk with finish_reason
                    finish = choice.get("finish_reason", "stop")
                    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
                    yield f"data: {orjson.dumps(chunk).decode()}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                fake_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        return JSONResponse(content=data)


@app.get("/v1/models")
async def list_models(request: Request):
    """Proxy the models endpoint."""
    print("[proxy] GET /v1/models")
    client = request.app.state.client
    resp = await client.get("/v1/models")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def fallback_proxy(request: Request, path: str):
    """Proxy any other endpoints directly."""
    print(f"[proxy] FALLBACK {request.method} /{path}")
    body = await request.body()
    client = request.app.state.client
    resp = await client.request(
        method=request.method,
        url=f"/{path}",
        headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
        content=body,
    )
    try:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception:
        return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


if __name__ == "__main__":
    print("[proxy] Starting MLX Tool-Call Proxy on port 5001")
    print(f"[proxy] Backend: {BACKEND_URL}")
    uvicorn.run(app, host="0.0.0.0", port=5001, loop="uvloop")  # Change 6: uvloop
