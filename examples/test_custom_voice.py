"""Test the 0.6B Qwen3-TTS CustomVoice service.

This service uses preset speaker names (Vivian, Ryan, Aiden, etc.) and an
optional ``instruct`` field for style control. See the HF model card for the
full list of supported speakers and their native languages:
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

The 1.7B VoiceDesign service (port 8000) uses a different API: ``voice`` is
a natural-language description that drives voice generation. Use
``test_requests.py`` for that service.
"""

import argparse
import requests

# Default points at the 0.6B service defined in compose.yaml
DEFAULT_URL = "http://localhost:8001/v1/audio/speech"

# Preset speakers, sourced from the HF model card.
SPEAKERS = {
    "Vivian":   "Chinese - Bright young female voice",
    "Serena":   "Chinese - Warm, gentle young female voice",
    "Uncle_Fu": "Chinese - Seasoned male voice, mellow timbre",
    "Dylan":    "Chinese (Beijing) - Youthful Beijing male voice",
    "Eric":     "Chinese (Sichuan) - Lively Chengdu male voice",
    "Ryan":     "English - Dynamic male voice with rhythm",
    "Aiden":    "English - Sunny American male voice",
    "Ono_Anna": "Japanese - Playful Japanese female voice",
    "Sohee":    "Korean - Warm Korean female voice",
}

LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


def main():
    parser = argparse.ArgumentParser(
        description="Test the Qwen3-TTS 0.6B CustomVoice service",
    )
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="API endpoint URL (default: 0.6B service on :8001)")
    parser.add_argument("--text", default="Hello from the 0.6B CustomVoice model.",
                        help="Text to synthesize")
    parser.add_argument("--speaker", default="Ryan", choices=sorted(SPEAKERS.keys()),
                        help="Preset speaker (default: Ryan)")
    parser.add_argument("--language", default="English", choices=LANGUAGES,
                        help="Language of the text (default: English)")
    parser.add_argument("--instruct", default="",
                        help="Style instruction, e.g. 'Speak in a happy tone' (optional)")
    parser.add_argument("--output", default="custom_voice.wav",
                        help="Output WAV file (default: custom_voice.wav)")
    args = parser.parse_args()

    payload = {
        "input": args.text,
        "voice": args.speaker,
        "language": args.language,
    }
    if args.instruct:
        payload["instruct"] = args.instruct

    print(f"POST {args.url}")
    print(f"  speaker : {args.speaker} ({SPEAKERS[args.speaker]})")
    print(f"  language: {args.language}")
    if args.instruct:
        print(f"  instruct: {args.instruct}")

    response = requests.post(args.url, json=payload)
    response.raise_for_status()

    with open(args.output, "wb") as f:
        f.write(response.content)

    print(f"Saved to {args.output} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
