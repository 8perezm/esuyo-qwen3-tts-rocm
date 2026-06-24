"""Test the Qwen3-TTS API using the OpenAI Python SDK."""

import argparse
from openai import OpenAI

BASE_URL = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS with OpenAI SDK")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    parser.add_argument("--text", default="Hello from Qwen3-TTS running on AMD ROCm.",
                        help="Text to synthesize")
    parser.add_argument("--voice", default="A warm, clear female narrator with a neutral accent.",
                        help="Voice description")
    parser.add_argument("--output", default="speech.wav", help="Output WAV file")
    args = parser.parse_args()

    client = OpenAI(base_url=f"{args.url}/v1", api_key="not-needed")

    with client.audio.speech.with_streaming_response.create(
        model="qwen3-tts",
        voice=args.voice,
        input=args.text,
    ) as response:
        response.stream_to_file(args.output)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
