"""
Test suite for MLX Tool-Call Proxy.
Tests pure functions (unit) and FastAPI endpoints (integration) with mocked backend.
"""
import sys
import json
import pytest
import pytest_asyncio
import httpx
import respx

# Add the proxy's directory to sys.path so we can import it
sys.path.insert(0, "/Users/rogerrudin/mlx-env")

from proxy_server import (
    app,
    build_tool_system_prompt,
    parse_tool_calls,
    rewrite_request,
    rewrite_response,
    BACKEND_URL,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"},
                "description": {"type": "string", "description": "What it does"},
            },
            "required": ["command", "description"],
        },
    },
}

SAMPLE_TOOL_MINIMAL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
    },
}

SAMPLE_TOOL_DIRECT = {
    "name": "write",
    "description": "Write to a file",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    },
}


@pytest_asyncio.fixture
async def async_client():
    """httpx AsyncClient wired directly to the FastAPI app (no server needed).
    Sets up app.state.client to mimic the lifespan context manager."""
    # The lifespan creates app.state.client; we set it to a dummy that respx intercepts
    app.state.client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=300)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
    await app.state.client.aclose()


def _chat_response(content: str, model: str = "test-model") -> dict:
    """Build a canned backend chat completion response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _sse_chunks(content: str) -> str:
    """Build a minimal SSE stream the backend would return."""
    chunk = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    done_chunk = {
        **chunk,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk)}\n\n"
        f"data: {json.dumps(done_chunk)}\n\n"
        "data: [DONE]\n\n"
    )


# ===========================================================================
# Phase 1 — Unit tests for pure functions
# ===========================================================================


class TestBuildToolSystemPrompt:
    def test_single_tool_all_fields(self):
        prompt = build_tool_system_prompt([SAMPLE_TOOL])
        assert "bash" in prompt
        assert "command" in prompt
        assert "(required)" in prompt
        assert "[TOOL_CALL]" in prompt

    def test_tool_missing_optional_fields(self):
        prompt = build_tool_system_prompt([SAMPLE_TOOL_MINIMAL])
        assert "read_file" in prompt
        assert "Parameters:" in prompt  # section header still present

    def test_direct_object_vs_function_wrapper(self):
        wrapped = build_tool_system_prompt([SAMPLE_TOOL])
        direct = build_tool_system_prompt([SAMPLE_TOOL_DIRECT])
        assert "bash" in wrapped
        assert "write" in direct
        # Both should contain the TOOL FORMAT instruction
        assert "[TOOL_CALL]" in wrapped
        assert "[TOOL_CALL]" in direct


class TestParseToolCalls:
    def test_single_valid_block(self):
        text = '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls"}}\n[/TOOL_CALL]'
        remaining, tc = parse_tool_calls(text)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "bash"
        assert tc[0]["id"].startswith("call_")
        args = json.loads(tc[0]["function"]["arguments"])
        assert args["command"] == "ls"

    def test_multiple_blocks(self):
        text = (
            '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls"}}\n[/TOOL_CALL]\n'
            '[TOOL_CALL]\n{"name": "read", "arguments": {"path": "/tmp"}}\n[/TOOL_CALL]'
        )
        remaining, tc = parse_tool_calls(text)
        assert tc is not None
        assert len(tc) == 2
        assert tc[0]["index"] == 0
        assert tc[1]["index"] == 1
        assert tc[0]["function"]["name"] == "bash"
        assert tc[1]["function"]["name"] == "read"

    def test_no_blocks(self):
        text = "Hello, I can help you with that."
        remaining, tc = parse_tool_calls(text)
        assert remaining == text
        assert tc is None

    def test_malformed_json(self):
        text = "[TOOL_CALL]\n{this is not json}\n[/TOOL_CALL]"
        remaining, tc = parse_tool_calls(text)
        assert remaining == text
        assert tc is None

    def test_literal_newlines_in_json_strings(self):
        # Model emits literal newlines inside JSON string values (e.g. multiline oldString)
        old_code = "function foo() {\n    return 1;\n}"
        new_code = "function foo() {\n    return 2;\n}"
        # Build a [TOOL_CALL] block with real newlines inside the string values.
        # Use concatenation (not f-string) to avoid }} → } escaping confusion.
        raw_json = (
            '{"name": "edit", "arguments": {"filePath": "/a/b.js",'
            ' "oldString": "' + old_code + '", "newString": "' + new_code + '"}}'
        )
        # raw_json now has literal newlines inside the string values — invalid JSON
        text = "[TOOL_CALL]\n" + raw_json + "\n[/TOOL_CALL]"
        remaining, tc = parse_tool_calls(text)
        assert tc is not None, "Should recover from literal newlines in JSON strings"
        assert tc[0]["function"]["name"] == "edit"
        args = json.loads(tc[0]["function"]["arguments"])
        assert args["filePath"] == "/a/b.js"
        assert "return 2" in args["newString"]

    def test_mixed_valid_and_invalid(self):
        text = (
            '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls"}}\n[/TOOL_CALL]\n'
            "[TOOL_CALL]\n{bad json}\n[/TOOL_CALL]"
        )
        remaining, tc = parse_tool_calls(text)
        assert tc is not None
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "bash"

    def test_think_tags_stripped(self):
        text = '<think>reasoning here</think>[TOOL_CALL]\n{"name": "bash", "arguments": {}}\n[/TOOL_CALL]'
        remaining, tc = parse_tool_calls(text)
        assert tc is not None
        assert remaining is None or "<think>" not in (remaining or "")


class TestRewriteRequest:
    def test_no_tools(self):
        payload = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        result, names, schemas = rewrite_request(payload)
        assert "tools" not in result
        assert "tool_choice" not in result
        assert names == []

    def test_tools_with_existing_system(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ],
            "tools": [SAMPLE_TOOL],
            "tool_choice": "auto",
        }
        result, names, schemas = rewrite_request(payload)
        assert "tools" not in result
        assert "tool_choice" not in result
        assert names == ["bash"]
        sys_content = result["messages"][0]["content"]
        assert "[TOOL_CALL]" in sys_content
        assert "You are helpful." in sys_content

    def test_tools_no_system_message(self):
        payload = {
            "messages": [{"role": "user", "content": "run ls"}],
            "tools": [SAMPLE_TOOL],
        }
        result, names, schemas = rewrite_request(payload)
        assert result["messages"][0]["role"] == "system"
        assert "[TOOL_CALL]" in result["messages"][0]["content"]

    def test_assistant_tool_calls_in_history(self):
        payload = {
            "messages": [
                {"role": "user", "content": "run ls"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                        }
                    ],
                },
                {"role": "tool", "name": "bash", "content": "file1.txt\nfile2.txt"},
                {"role": "user", "content": "now cat file1"},
            ],
            "tools": [SAMPLE_TOOL],
        }
        result, names, schemas = rewrite_request(payload)
        # The assistant message should be converted to text with [TOOL_CALL]
        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        assert "[TOOL_CALL]" in assistant_msg["content"]
        assert "bash" in assistant_msg["content"]
        # The tool message should become a user message
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_msg = [m for m in user_msgs if "[Tool Result from bash]" in m["content"]]
        assert len(tool_result_msg) == 1
        assert "file1.txt" in tool_result_msg[0]["content"]

    def test_tool_role_converted(self):
        payload = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "name": "search", "content": "result here"},
            ],
            "tools": [SAMPLE_TOOL],
        }
        result, names, schemas = rewrite_request(payload)
        roles = [m["role"] for m in result["messages"]]
        assert "tool" not in roles
        converted = [m for m in result["messages"] if "Tool Result from search" in m.get("content", "")]
        assert len(converted) == 1
        assert converted[0]["role"] == "user"


class TestRewriteResponse:
    def test_valid_tool_call_extracted(self):
        data = _chat_response('[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls"}}\n[/TOOL_CALL]')
        result = rewrite_response(data, ["bash"])
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] is not None
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "bash"

    def test_no_tool_call_warning(self):
        data = _chat_response("I cannot help with that.")
        result = rewrite_response(data, ["bash"])
        choice = result["choices"][0]
        # No tool_calls should be added
        assert "tool_calls" not in choice["message"]
        assert choice["finish_reason"] == "stop"

    def test_bare_json_fallback(self):
        data = _chat_response('Sure, here: {"name": "bash", "arguments": {"command": "pwd"}}')
        result = rewrite_response(data, ["bash"])
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tc = choice["message"]["tool_calls"]
        assert tc[0]["function"]["name"] == "bash"

    def test_missing_required_field_autofilled(self):
        schema = {
            "bash": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "cmd"},
                    "description": {"type": "string", "description": "desc"},
                },
                "required": ["command", "description"],
            }
        }
        # Model only provided "command", missing "description"
        data = _chat_response('[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls"}}\n[/TOOL_CALL]')
        result = rewrite_response(data, ["bash"], tool_schemas=schema)
        tc = result["choices"][0]["message"]["tool_calls"][0]
        args = json.loads(tc["function"]["arguments"])
        assert "description" in args  # auto-filled

    def test_empty_content_skipped(self):
        data = _chat_response("")
        data["choices"][0]["message"]["content"] = None
        result = rewrite_response(data, ["bash"])
        # Should not crash
        assert result == data


# ===========================================================================
# Phase 1 — Integration tests (FastAPI + mocked backend)
# ===========================================================================


class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_no_tools_no_stream(self, async_client):
        backend_resp = _chat_response("Hello there!")
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=backend_resp)
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello there!"

    @pytest.mark.asyncio
    async def test_no_tools_stream(self, async_client):
        sse_body = _sse_chunks("Hello stream!")
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=sse_body.encode(),
                    headers={"content-type": "text/event-stream"},
                )
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        body = resp.text
        assert "Hello stream!" in body

    @pytest.mark.asyncio
    async def test_tools_present(self, async_client):
        sse_body = _sse_chunks(
            '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls", "description": "list"}}\n[/TOOL_CALL]'
        )
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=sse_body.encode(),
                    headers={"content-type": "text/event-stream"},
                )
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "list files"}],
                    "tools": [SAMPLE_TOOL],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        tc = data["choices"][0]["message"]["tool_calls"]
        assert tc[0]["function"]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_tools_stream_returns_sse(self, async_client):
        sse_body = _sse_chunks(
            '[TOOL_CALL]\n{"name": "bash", "arguments": {"command": "ls", "description": "list"}}\n[/TOOL_CALL]'
        )
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    200,
                    content=sse_body.encode(),
                    headers={"content-type": "text/event-stream"},
                )
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "list files"}],
                    "tools": [SAMPLE_TOOL],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        body = resp.text
        assert "data:" in body
        assert "[DONE]" in body
        # Parse one of the SSE chunks to verify tool_calls
        lines = [l for l in body.strip().split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
        found_tool_call = False
        for line in lines:
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                found_tool_call = True
                assert delta["tool_calls"][0]["function"]["name"] == "bash"
        assert found_tool_call

    @pytest.mark.asyncio
    async def test_backend_error(self, async_client):
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(500, json={"error": "internal"})
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 500
        assert resp.json()["error"] == "internal"

    @pytest.mark.asyncio
    async def test_think_tags_stripped(self, async_client):
        backend_resp = _chat_response(
            "<think>some reasoning</think>Here is the answer."
        )
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.post("/v1/chat/completions").mock(
                return_value=httpx.Response(200, json=backend_resp)
            )
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "<think>" not in content
        assert "Here is the answer." in content


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_proxy_models(self, async_client):
        models_resp = {"object": "list", "data": [{"id": "mlx-model", "object": "model"}]}
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.get("/v1/models").mock(
                return_value=httpx.Response(200, json=models_resp)
            )
            resp = await async_client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["data"][0]["id"] == "mlx-model"


class TestFallbackProxy:
    @pytest.mark.asyncio
    async def test_json_response(self, async_client):
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.get("/v1/some/path").mock(
                return_value=httpx.Response(200, json={"ok": True})
            )
            resp = await async_client.get("/v1/some/path")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_non_json_response(self, async_client):
        with respx.mock(base_url=BACKEND_URL) as mock:
            mock.get("/health").mock(
                return_value=httpx.Response(
                    200, content=b"OK", headers={"content-type": "text/plain"}
                )
            )
            resp = await async_client.get("/health")
        assert resp.status_code == 200
        assert resp.text == "OK"
