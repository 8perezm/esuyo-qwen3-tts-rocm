"""Test the Qwen3-TTS API using the requests library."""

import argparse
import requests

DEFAULT_URL = "http://localhost:8000/v1/audio/speech"


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS with requests")
    parser.add_argument("--url", default=DEFAULT_URL, help="API endpoint URL")
    parser.add_argument("--text", default="Hello from Qwen3-TTS running on AMD ROCm.",
                        help="Text to synthesize")
    parser.add_argument("--voice", default="A warm, clear female narrator with a neutral accent.",
                        help="Voice description")
    parser.add_argument("--output", default="speech.wav", help="Output WAV file")
    args = parser.parse_args()

    response = requests.post(
        args.url,
        json={
            "input": args.text,
            "voice": args.voice,
        },
    )
    response.raise_for_status()

    with open(args.output, "wb") as f:
        f.write(response.content)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
