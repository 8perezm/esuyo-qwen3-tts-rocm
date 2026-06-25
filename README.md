# Qwen3-TTS — OpenAI-Compatible TTS Server (ROCm)

An OpenAI-compatible text-to-speech API server powered by [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign), running on AMD ROCm GPUs.

## Prerequisites

- An AMD GPU with ROCm support
- [Docker](https://docs.docker.com/engine/install/) with the `nvidia-container-toolkit` equivalent for ROCm — or a host with `/dev/kfd` and `/dev/dri` available
- Sufficient VRAM for the **Qwen3-TTS-12Hz-1.7B-VoiceDesign** model (~4 GB+)

## Quick Start

```bash
# Build and start the container
docker compose up -d

# Or rebuild without cache for a fresh install
docker compose build --no-cache
docker compose up -d
```

The API will be available at `http://localhost:8000`.

## API Usage

The server exposes an OpenAI-compatible `/v1/audio/speech` endpoint.

### cURL

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from Qwen3-TTS running on AMD ROCm.",
    "voice": "A warm, clear female narrator with a neutral accent."
  }' \
  --output speech.wav
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Hello from Qwen3-TTS running on AMD ROCm.",
        "voice": "A warm, clear female narrator with a neutral accent.",
    },
)

with open("speech.wav", "wb") as f:
    f.write(response.content)
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts",
    voice="A warm, clear female narrator with a neutral accent.",
    input="Hello from Qwen3-TTS running on AMD ROCm.",
) as response:
    response.stream_to_file("speech.wav")
```

## Request Format

| Field   | Type   | Default                                                    | Description                |
| ------- | ------ | ---------------------------------------------------------- | -------------------------- |
| `input` | string | *required*                                                  | Text to synthesize         |
| `voice` | string | `"A warm, clear female narrator with a neutral accent."`   | Voice description/prompt   |
| `model` | string | `"qwen3-tts"`                                              | Model identifier (ignored) |

## Voice Design

The `voice` field accepts natural language descriptions — get creative! Examples:

- `"A deep, resonant male voice with a British accent."`
- `"A cheerful, energetic young adult, speaking casually."`
- `"A calm, elderly woman with a soft Southern drawl."`

## Project Structure

```
├── app/
│   └── server.py          # FastAPI application
├── compose.yaml           # Docker Compose configuration (ROCm)
├── Dockerfile             # ROCm PyTorch container definition
└── README.md
```

## Development

The `compose.yaml` mounts the current directory into `/workspace` inside the container, so any changes to `app/server.py` take effect immediately on container restart.

```bash
# Attach to the running container
docker compose exec qwen-tts /bin/bash

# Restart after code changes
docker compose restart
```

## Notes

- The first startup downloads the ~1.7B parameter model from Hugging Face — this may take a few minutes depending on your connection.
- Model weights are cached in a named Docker volume (`hf_cache`) so subsequent restarts are faster.
- Default model dtype is `bfloat16`; adjust `dtype` in `app/server.py` if your GPU does not support it.

### `flash-attn` warning (informational, safe to ignore)

On startup you will see:

```
********
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
Please install flash-attn for faster inference.
********
```

This is **expected and not an error** — the server runs correctly without `flash-attn`. It is **not** installed in the Dockerfile by design, because:

- **No pre-built ROCm wheels** exist for our specific combo (ROCm 7.2.4 + PyTorch 2.10.0). CUDA has wheels for every combo, ROCm does not.
- **Building from source** is slow (10–30+ min), must match the PyTorch ABI exactly, and frequently fails — risking a broken image build.
- **PyTorch's native `scaled_dot_product_attention`** (SDPA), used automatically when `flash-attn` is absent, already routes to efficient backends on ROCm and typically achieves 80–95% of `flash-attn`'s speed with zero install pain.

#### Opting in anyway

If you want to try it on your specific GPU, install it at runtime first to confirm it compiles and gives a measurable speedup, then promote it to the `Dockerfile`:

```bash
# Inside the running container
docker compose exec qwen-tts pip install flash-attn --no-build-isolation

# Time a request before vs. after; only add to Dockerfile if the gain is meaningful
```

If you do promote it, add this line to the `pip install` block in the `Dockerfile`:

```dockerfile
pip install --no-cache-dir flash-attn
```
