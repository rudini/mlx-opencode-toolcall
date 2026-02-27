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
pip install mlx-vlm mlx-openai-server fastapi uvicorn httpx
```

> `mlx-vlm` pulls in `mlx-lm`, `mlx`, `torchvision`, and other dependencies automatically.

## Step 2 — Download the model

The model downloads automatically on first server start from HuggingFace. No manual step needed.

- **Model**: [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- **Architecture**: Mixture of Experts (MoE) — 35B total params, ~3B active per token
- **Quantization**: 4-bit (19 GB on disk)
- **Performance**: ~56 tokens/sec generation on M3 Pro, ~20.5 GB peak memory

## Step 3 — Patch mlx-openai-server

Two small patches are needed to handle OpenCode's request format. Find the installed package files inside your venv:

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
    """
    Validate max_tokens is positive and within reasonable limits.
    """
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

## Step 4 — The proxy server

The proxy (`proxy_server.py`) sits between OpenCode and the MLX backend. See the architecture diagram above for how data flows.

Key behaviors:
- **Tool requests**: Forces non-streaming mode, caps `max_tokens` at 4096 for tool calls
- **Streaming passthrough**: Non-tool requests stream SSE directly from the backend
- **History rewriting**: Converts OpenAI `tool_calls` / `tool` role messages in chat history back into plain text the backend can understand
- **Fallback parsing**: If the model emits a bare JSON tool call without `[TOOL_CALL]` brackets, the proxy still detects and parses it
- **Thinking suppression**: Injects `/no_think` to disable the model's reasoning mode, and strips any `<think>` tags from the response before processing
- **SSE conversion**: Wraps non-streaming backend responses into proper OpenAI-format Server-Sent Events for streaming clients

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
python -m mlx_vlm.server \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --host 0.0.0.0 \
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

### Tool calls fail for large file writes
The model has a 4096 token cap for tool-call responses. Very large files (full HTML pages, etc.) may exceed this. For best results, ask the model to create files incrementally or keep individual files small.

### Model outputs `</think>` tags in responses
The proxy injects `/no_think` to suppress the model's reasoning mode and strips any `<think>` / `</think>` tags from responses before processing. If you still see them, the model is ignoring the directive — this is cosmetic and doesn't affect tool-call parsing.

### Backend starts but model downloads slowly
The model is ~19 GB. First download from HuggingFace can take a while depending on your connection. Subsequent starts use the cached copy in `~/.cache/huggingface/`.

### High memory usage / swapping
The model uses ~20.5 GB peak. On a 32 GB machine this leaves headroom, but close other memory-heavy apps. On 16 GB machines this model may not fit — try a smaller quantized variant.

### Proxy crashes with NoneType error
Check that the backend is actually running and reachable on port 8000. The proxy assumes the backend is available and doesn't retry on connection failure.

### OpenCode shows "connection refused"
Make sure `baseURL` in your config points to `http://localhost:5001/v1` (the proxy), not port 8000.

### OpenCode SQLite / SQLITE_MISUSE errors
This is an OpenCode internal database issue, not a proxy problem. Restart OpenCode, or clear its session data if it persists:
```bash
rm -rf ~/.local/share/opencode/sessions
```

## Project structure

```
mlx-opencode-toolcall/
├── README.md              ← This file
├── proxy_server.py        ← Tool-call middleware proxy
├── opencode-config.json   ← Example OpenCode configuration
└── start.sh               ← Script to launch the full stack
```

## License

This project is provided as-is for personal/educational use.
