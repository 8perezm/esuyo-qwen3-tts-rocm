# Use your exact specified ROCm PyTorch base image
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Set non-interactive installation and avoid writing .pyc files
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies needed for audio processing and model compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Install the correct Python packages for text-to-speech processing
# Note: `--no-cache-dir` keeps the image slim. We pin packages to avoid version conflicts.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers>=4.48.0 \
    accelerate \
    soundfile \
    librosa \
    scipy \
    gradio \
    fastapi \
    uvicorn \
    huggingface_hub \
    qwen-tts

# Clone the official repo or copy your app code (adjust this as needed)
# RUN git clone https://github.com .

# Default command to run
CMD ["/bin/bash"]