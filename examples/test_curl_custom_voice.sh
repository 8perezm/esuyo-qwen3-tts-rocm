#!/usr/bin/env bash
# Test the 0.6B Qwen3-TTS CustomVoice service with cURL.
# Usage: ./test_curl_custom_voice.sh [output.wav]
#
# Targets the 0.6B service in compose.yaml (port 8001).
# For the 1.7B VoiceDesign service (port 8000), use test_curl.sh instead.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
OUTPUT="${1:-custom_voice.wav}"

curl -X POST "${BASE_URL}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from the 0.6B CustomVoice model.",
    "voice": "Ryan",
    "language": "English",
    "instruct": "Speak in a cheerful, energetic tone."
  }' \
  --output "$OUTPUT"

echo "Saved to $OUTPUT"

# Other speakers you can try by editing the 'voice' field above:
#   Vivian, Serena, Uncle_Fu, Dylan, Eric   (Chinese)
#   Ryan, Aiden                             (English)
#   Ono_Anna                                (Japanese)
#   Sohee                                   (Korean)
