#!/usr/bin/env bash
# Test the Qwen3-TTS API using cURL
# Usage: ./test_curl.sh [output.wav]

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
OUTPUT="${1:-speech.wav}"

curl -X POST "${BASE_URL}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from Qwen3-TTS running on AMD ROCm.",
    "voice": "A warm, clear female narrator with a neutral accent."
  }' \
  --output "$OUTPUT"

echo "Saved to $OUTPUT"
