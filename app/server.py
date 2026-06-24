import io
from contextlib import asynccontextmanager
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Initializing Qwen3-TTS model weights on AMD GPU...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    print("Model ready to serve requests.")
    yield
    # Shutdown: release model resources
    model = None

app = FastAPI(title="Qwen3-TTS OpenAI-Compatible API Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    input: str
    voice: str = "A warm, clear female narrator with a neutral accent."
    model: str = "qwen3-tts"

@app.post("/v1/audio/speech")
async def text_to_speech(payload: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model initialization incomplete.")

    try:
        # Generate the audio block
        wavs, sample_rate = model.generate_voice_design(
            text=[payload.input],
            language="English",
            instruct=payload.voice
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

    except Exception as e:
        import traceback
        traceback.print_exc() # This echoes the exact breakdown block inside 'docker compose logs'
        raise HTTPException(status_code=500, detail=f"Inference processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
