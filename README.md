# MLX Local Model + OpenCode with Tool Calling

Run **Qwen3.5-35B-A3B** (a 35B-parameter Mixture-of-Experts model, 4-bit quantized) locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx), and connect it to [OpenCode](https://opencode.ai) CLI with full **tool-calling support** through a lightweight middleware proxy.

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐      ┌────────────┐
│  OpenCode    │─────▶│  Proxy Server    │─────▶│  mlx-openai-server  │─────▶│  MLX/Metal  │
│  CLI         │◀─────│  :5001           │◀─────│  :8000              │◀─────│  GPU        │
└─────────────┘      └──────────────────┘      └─────────────────────┘      └────────────┘
                      Injects tool prompts      Runs Qwen3.5 inference       Apple Silicon
                      Parses [TOOL_CALL]        via mlx-vlm                  hardware accel
                      Returns OpenAI format
```

## Why a proxy?

MLX's OpenAI-compatible server doesn't support the `tools` / `tool_choice` fields in the chat completions API. OpenCode needs tool calling to work (file edits, shell commands, etc.). The proxy bridges the gap:

1. **Intercepts** requests from OpenCode that include tool definitions
2. **Injects** a system prompt teaching the model to emit `[TOOL_CALL]` blocks
3. **Strips** the `tools` array so the backend doesn't reject the request
4. **Parses** `[TOOL_CALL]` blocks from the model's response
5. **Rewrites** them into standard OpenAI `tool_calls` JSON format

### Why `[TOOL_CALL]` instead of `<tool_call>` XML?

The model needs to write HTML/XML file content inside tool call arguments. Using XML-style `<tool_call>` delimiters caused the model to confuse its own markup with HTML tags in the content — closing tags like `</div>` or `</html>` would derail the model mid-generation. Square bracket delimiters (`[TOOL_CALL]` / `[/TOOL_CALL]`) avoid this collision entirely.

## Prerequisites

- **macOS** on Apple Silicon (tested on M3 Pro, 32 GB RAM)
- **Python 3.12+** (ships with macOS or install via Homebrew)
- **~20 GB disk space** for the quantized model weights
- **Node.js 18+** (for OpenCode CLI)
- **OpenCode CLI** installed (`npm install -g opencode`)

## Step 1 — Create a Python venv and install packages

```bash
python3 -m venv ~/mlx-env
source ~/mlx-env/bin/activate
pip install mlx-vlm mlx-openai-server fastapi uvicorn httpx orjson uvloop
```

> `mlx-vlm` pulls in `mlx-lm`, `mlx`, `torchvision`, and other dependencies automatically.
> `orjson` and `uvloop` are optional but give ~3x faster JSON serialization and better async performance.

## Step 2 — Download the model

The model downloads automatically on first server start from HuggingFace. No manual step needed.

- **Model**: [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- **Architecture**: Mixture of Experts (MoE) — 35B total params, ~3B active per token
- **Quantization**: 4-bit (19 GB on disk)
- **Performance**: ~56 tokens/sec generation on M3 Pro, ~20.5 GB peak memory

## Step 3 — Patch mlx-openai-server

Several patches are needed to handle OpenCode's request format and disable unnecessary reasoning. Find the installed package files inside your venv:

```bash
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
```

### Patch A — Cap max_tokens (schema validation)

OpenCode sends `max_tokens: 32000` which exceeds what the local model can handle. Add a validator to cap it.

**File**: `$SITE_PACKAGES/app/schemas/openai.py`

Find the `ChatCompletionRequest` class and add this validator after the existing `check_temperature` validator:

```python
@validator("max_tokens")
def check_max_tokens(cls, v):
    if v is not None:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        if v > 8192:
            v = 8192  # Cap to reasonable limit for local models
    return v
```

Without this patch you'll get HTTP 422 errors:
```
ValidationError: max_tokens must be less than or equal to 4096
```

### Patch B — Handle array-style message content

OpenCode sends message content as an array (`[{"type": "text", "text": "..."}]`) instead of a plain string. The default handler doesn't handle this and crashes with "No user query found."

**File**: `$SITE_PACKAGES/app/handler/mlx_vlm.py`

Find the `_prepare_text_request` method (around line 375) and replace the message processing loop with:

```python
chat_messages = []
for message in request.messages:
    if isinstance(message.content, str):
        chat_messages.append({
            "role": message.role,
            "content": message.content
        })
    elif isinstance(message.content, list):
        # Handle array-style content e.g. [{"type": "text", "text": "hi"}]
        texts = []
        for item in message.content:
            if hasattr(item, 'type') and item.type == "text":
                text = getattr(item, "text", "").strip()
                if text:
                    texts.append(text)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)
        if texts:
            chat_messages.append({
                "role": message.role,
                "content": " ".join(texts)
            })
    else:
        continue
```

Without this patch you'll get HTTP 500 errors:
```
ValueError: No user query found in the messages
```

### Patch C — Performance tuning (optional but recommended)

These patches improve inference speed on Apple Silicon by tuning conservative defaults.

**File**: `$SITE_PACKAGES/app/models/mlx_vlm.py` — replace with the version in `mlx_vlm_model.py`

This file includes all performance optimizations:

- **`kv_bits=8`**: 8-bit KV cache quantization — halves KV memory, reduces attention bandwidth 15–25% per decoding step
- **System prompt KV cache**: Pre-fills the system prompt tokens once on first request, snapshots the KV state, and reuses it on every subsequent request. Only the new user message tokens need prefilling per request. This eliminates ~0.9s of prefill time per hot request
  - Cold request (first use / new system prompt): ~1.8s — builds the KV cache
  - Warm request (same system prompt): **~0.87s, 25 tok/s** — 2× speedup

Technical notes:
- Uses `generate_step(max_tokens=0)` for prefill-only (avoids a bug in `stream_generate` where the post-loop finalize yield crashes when 0 tokens are generated)
- Uses `cls.from_state(state, meta_state)` to restore cache — same protocol as `mlx_lm.save_prompt_cache`/`load_prompt_cache`, handles both `KVCache` and `ArraysCache` (Qwen3.5 uses 10 KVCache + 30 ArraysCache layers)
- System prefix tokenized directly as `<|im_start|>system\n...<|im_end|>\n` — bypasses Jinja template restriction (Qwen3.5's template requires at least one user turn)

**File**: `$SITE_PACKAGES/app/main.py` (line ~140) — Change GC interval from 50 to 200:

```python
if request.app.state.request_count % 200 == 0:
```

**File**: `$SITE_PACKAGES/app/core/queue.py` (line ~171) — Change GC interval from 10 to 50:

```python
if len(self.active_requests) % 50 == 0:
```

These GC changes reduce latency spikes from `gc.collect()` + `mx.clear_cache()` during active use. Safe on machines with 32 GB+ RAM.

### Patch D — Disable thinking mode (critical for performance)

Qwen3.5 has a "thinking" mode that generates hundreds of `<think>` reasoning tokens before each response. This wastes ~90% of inference time on tokens that get thrown away. The model's chat template controls this via an `enable_thinking` jinja variable — a text directive like `/no_think` does **not** work.

Two files need patching:

**File**: `$SITE_PACKAGES/app/models/mlx_vlm.py`

In the `__call__` method, replace the prompt preparation block (around line 64-72) with:

```python
# Prepare the prompt using the chat template
enable_thinking = kwargs.pop("enable_thinking", False)
# Use processor directly to ensure enable_thinking is passed to jinja template
formatted_prompt = self.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking
)
```

The original code used `mlx_vlm.prompt_utils.apply_chat_template` which silently drops the `enable_thinking` kwarg. Calling `self.processor.apply_chat_template()` directly passes it through to the jinja template.

**File**: `$SITE_PACKAGES/app/handler/mlx_vlm.py`

In the `_prepare_text_request` method, add `enable_thinking` to the `model_params` dict:

```python
model_params = {
    k: v for k, v in {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "frequency_penalty": request.frequency_penalty,
        "presence_penalty": request.presence_penalty,
        "stop": request.stop,
        "n": request.n,
        "seed": request.seed,
        "enable_thinking": getattr(request, "enable_thinking", False),
    }.items() if v is not None
}
```

Without this patch, tool calls take ~16s at ~3 tok/s. With it, they complete in ~1.8s at ~15 tok/s. With all patches applied (A–D plus Patch C kv_bits + KV cache reuse), warm tool calls reach **~0.87s at ~25 tok/s**.

## Step 4 — The proxy server

The proxy (`proxy_server.py`) sits between OpenCode and the MLX backend. See the architecture diagram above for how data flows.

Key behaviors:

- **Tool-call streaming with early stop**: Streams from the backend with `temperature=0` (argmax, fastest sampling) and breaks the stream the moment `[/TOOL_CALL]` is detected — no waiting for EOS. Hot tool calls complete in ~0.6–0.9s
- **Streaming passthrough**: Non-tool requests stream SSE directly from the backend at full speed
- **History rewriting**: Converts OpenAI `tool_calls` / `tool` role messages in chat history back into plain text the backend can understand
- **Flexible parsing**: Matches `[TOOL_CALL]`, `TOOL_CALL`, and `<tool_call>` variants — the model occasionally drops the brackets
- **Fallback parsing**: If the model emits a bare JSON tool call without any delimiters, the proxy still detects and parses it
- **Auto-fill missing fields**: If the model omits a required tool parameter, the proxy fills in a sensible default from the tool schema
- **Thinking suppression**: Thinking is disabled at the template level via Patch D (`enable_thinking=False`). The proxy also strips any residual `<think>` tags from responses as a safety net
- **SSE conversion**: Wraps buffered tool-call responses into proper OpenAI-format Server-Sent Events for streaming clients
- **Performance logging**: Logs tok/s and elapsed time for every tool-call request to the console

### Performance optimizations

The proxy uses several optimizations for minimal overhead:

- **Connection pooling**: Reuses a persistent `httpx.AsyncClient` with keep-alive connections to the backend (no TCP handshake per request)
- **Early stream stop**: Tool-call requests stream from the backend and stop reading as soon as `[/TOOL_CALL]` appears — saves 0.1–0.3s vs waiting for EOS
- **Compact system prompt**: Instruction overhead trimmed to ~60 tokens (down from ~170), reducing prefill on every tool-call request
- **orjson**: Uses `orjson` instead of `json` for ~3x faster JSON serialization/deserialization
- **uvloop**: Runs on `uvloop` for faster async event loop processing
- **Pre-compiled regex**: All regex patterns are compiled once at startup

## Step 5 — Configure OpenCode

Copy the example config into OpenCode's config directory:

```bash
mkdir -p ~/.config/opencode
cp opencode-config.json ~/.config/opencode/config.json
```

Or create `~/.config/opencode/config.json` manually:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local (Qwen3.5)",
      "options": {
        "baseURL": "http://localhost:5001/v1"
      },
      "models": {
        "mlx-community/Qwen3.5-35B-A3B-4bit": {
          "name": "Qwen3.5 35B-A3B MoE",
          "max_tokens": 4096
        }
      }
    }
  }
}
```

Note: `baseURL` points to the **proxy** on port 5001, not directly to the backend on 8000.

## Step 6 — Start the stack

### Option A — Use the start script

```bash
./start.sh
```

This starts both the backend and proxy, waits for the backend to be ready, and shuts everything down cleanly on Ctrl+C.

You can override the venv path:

```bash
MLX_VENV=~/my-other-venv ./start.sh
```

### Option B — Start manually in separate terminals

**Terminal 1** — MLX backend:

```bash
source ~/mlx-env/bin/activate
mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-35B-A3B-4bit \
  --model-type multimodal \
  --port 8000
```

**Terminal 2** — Proxy:

```bash
source ~/mlx-env/bin/activate
python proxy_server.py
```

**Terminal 3** — OpenCode:

```bash
opencode --provider mlx-local
```

## Troubleshooting

### "ValidationError: max_tokens must be less than or equal to 4096"
Apply Patch A from Step 3. OpenCode sends `max_tokens: 32000` which exceeds the hardcoded limit.

### "ValueError: No user query found in the messages"
Apply Patch B from Step 3. OpenCode sends array-style content that the default handler doesn't parse.

### "Model type qwen3_5_moe not supported"
Make sure you have `mlx-vlm >= 0.3.12` installed. Older versions don't know this model architecture:
```bash
pip install --upgrade mlx-vlm
```

### Model outputs bare `TOOL_CALL` without brackets
The proxy handles this automatically — it matches `[TOOL_CALL]`, `TOOL_CALL`, and `<tool_call>` variants via flexible regex.

### Tool calls fail — model omits required parameters
The proxy auto-fills missing required fields using the tool schema. If a string field like `description` is missing, it generates one from the other arguments.

### Tool calls fail for large file writes
Tool-call responses are capped at 256 tokens to keep inference fast. Very large files may exceed this. For best results, ask the model to create files incrementally or keep individual files small.

### Model outputs `</think>` tags in responses
Apply Patch D from Step 3. Qwen3.5's thinking mode is controlled by the `enable_thinking` jinja template variable, not by text directives like `/no_think`. Patch D sets `enable_thinking=False` at the template level, which makes the model emit an empty `<think>\n\n</think>` block instead of reasoning. The proxy strips any residual `<think>` tags as a safety net.

### Backend starts but model downloads slowly
The model is ~19 GB. First download from HuggingFace can take a while depending on your connection. Subsequent starts use the cached copy in `~/.cache/huggingface/`.

### High memory usage / swapping
The model uses ~20.5 GB peak. On a 32 GB machine this leaves headroom, but close other memory-heavy apps. On 16 GB machines this model may not fit — try a smaller quantized variant.

### Proxy crashes with connection error
Check that the backend is actually running and reachable on port 8000. The proxy assumes the backend is available and doesn't retry on connection failure. Restart the proxy after the backend is up.

### OpenCode shows "connection refused"
Make sure `baseURL` in your config points to `http://localhost:5001/v1` (the proxy), not port 8000.

### OpenCode hangs with no response
The proxy must support streaming — earlier versions blocked on non-streaming tool-call responses while OpenCode expected SSE. The current proxy wraps non-streaming responses into fake SSE chunks to prevent this.

## Project structure

```
mlx-opencode-toolcall/
├── README.md              ← This file
├── proxy_server.py        ← Tool-call middleware proxy
├── mlx_vlm_model.py       ← Patched model wrapper (Patches C+D + KV cache reuse)
├── opencode-config.json   ← Example OpenCode configuration
├── start.sh               ← Script to launch the full stack
└── test_proxy.py          ← Proxy unit tests
```

## License

This project is provided as-is for personal/educational use.
