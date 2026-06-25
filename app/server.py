import io
import os
from contextlib import asynccontextmanager
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

# Configurable via env vars so the same image can serve either model size.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
PORT = int(os.environ.get("PORT", "8000"))

# Detect which Qwen3-TTS variant is loaded so we can call the right API.
#   - "VoiceDesign" → generate_voice_design (description-driven, e.g. 1.7B-VoiceDesign)
#   - "CustomVoice" → generate_custom_voice (preset speakers + style, e.g. 0.6B-CustomVoice)
#   - "Base"        → generate_voice_clone (ref_audio/ref_text) — not implemented here
if "CustomVoice" in MODEL_ID:
    MODEL_VARIANT = "custom"
elif "Base" in MODEL_ID:
    MODEL_VARIANT = "base"
else:
    MODEL_VARIANT = "design"  # default: 1.7B-VoiceDesign

# Built-in speakers for the 0.6B-CustomVoice model. Source: HF model card.
CUSTOM_VOICE_SPEAKERS = {
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",  # Chinese
    "Ryan", "Aiden",                                  # English
    "Ono_Anna",                                       # Japanese
    "Sohee",                                          # Korean
}

# Default `voice` field per variant:
#   - VoiceDesign: a natural-language description of the voice to design.
#   - CustomVoice: a speaker name from CUSTOM_VOICE_SPEAKERS.
if MODEL_VARIANT == "custom":
    DEFAULT_VOICE = "Ryan"
else:
    DEFAULT_VOICE = "A warm, clear female narrator with a neutral accent."

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"Initializing Qwen3-TTS model on AMD GPU: {MODEL_ID}")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    print(f"Model ready (variant={MODEL_VARIANT}) on port {PORT}.")
    yield
    # Shutdown: release model resources
    model = None

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    input: str
    voice: str = DEFAULT_VOICE
    language: str = "English"
    instruct: str | None = None  # style instruction (CustomVoice only)
    model: str = "qwen3-tts"

@app.post("/v1/audio/speech")
async def text_to_speech(payload: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model initialization incomplete.")

    try:
        if MODEL_VARIANT == "custom":
            # CustomVoice: 'voice' must be a preset speaker name.
            speaker = next(
                (s for s in CUSTOM_VOICE_SPEAKERS if s.lower() == payload.voice.lower()),
                None,
            )
            if speaker is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown speaker '{payload.voice}'. "
                        f"Allowed speakers: {sorted(CUSTOM_VOICE_SPEAKERS)}"
                    ),
                )
            wavs, sample_rate = model.generate_custom_voice(
                text=[payload.input],
                language=payload.language,
                speaker=speaker,
                instruct=payload.instruct or "",
            )
        elif MODEL_VARIANT == "base":
            # Base model: voice cloning requires a reference audio sample.
            raise HTTPException(
                status_code=501,
                detail=(
                    "Base (voice-clone) mode requires ref_audio and ref_text; "
                    "not implemented in this server."
                ),
            )
        else:
            # VoiceDesign (default 1.7B): the 'voice' description drives generation.
            wavs, sample_rate = model.generate_voice_design(
                text=[payload.input],
                language=payload.language,
                instruct=payload.voice,
            )

        # --- FIX: Safe data conversion for soundfile layer ---
        # 1. If it's a PyTorch Tensor, drag it off the GPU to CPU memory
        if hasattr(wavs, "detach"):
            audio_data = wavs.detach().cpu().numpy()
        else:
            audio_data = np.array(wavs)

        # 2. Flatten out batch dimensions if nested [1, samples] or [[samples]]
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        # 3. Ensure float32 compliance to guarantee standard audio layouts
        audio_data = audio_data.astype(np.float32)
        # -----------------------------------------------------

        # Encode cleanly inside memory buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        audio_buffer.seek(0)

        return StreamingResponse(audio_buffer, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()  # echoes the exact breakdown block inside 'docker compose logs'
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
