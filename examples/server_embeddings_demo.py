#!/usr/bin/env python3
import argparse
import base64
import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests


def data_url(path: Path) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{payload}"


def headers() -> dict[str, str]:
    api_key = os.environ.get("API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def post_json(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        headers=headers(),
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def post_upload(base_url: str, payload: dict[str, Any], video_path: Path) -> dict[str, Any]:
    with video_path.open("rb") as video_file:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/embeddings/uploads",
            headers=headers(),
            data={"payload": json.dumps(payload)},
            files={"video_file": (video_path.name, video_file, mimetypes.guess_type(video_path.name)[0] or "video/mp4")},
            timeout=180,
        )
    response.raise_for_status()
    return response.json()


def embedding_at(response: dict[str, Any], index: int = 0) -> list[float]:
    return response["data"][index]["embedding"]


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create VLM2Vec embeddings through the FastAPI server.")
    parser.add_argument("--base-url", default=os.environ.get("VLM2VEC_SERVER_URL", "http://localhost:8000"))
    parser.add_argument("--image", type=Path, default=Path("assets/example.jpg"))
    parser.add_argument("--video", type=Path, default=Path("assets/example_video.mp4"))
    args = parser.parse_args()

    text_response = post_json(
        args.base_url,
        {
            "inputs": [
                {"type": "text", "text": "A cat and a dog"},
                {"type": "text", "text": "A person shoveling snow"},
            ]
        },
    )
    text_a = embedding_at(text_response, 0)
    text_b = embedding_at(text_response, 1)
    print(f"text dimensions: {text_response['dimensions']}")
    print(f"text cosine: {cosine(text_a, text_b):.4f}")

    image_payload = data_url(args.image)
    image_response = post_json(
        args.base_url,
        {
            "inputs": [
                {
                    "type": "image",
                    "instruction": "Represent the given image.",
                    "media": image_payload,
                }
            ]
        },
    )
    print(f"image dimensions: {image_response['dimensions']}")

    image_list_response = post_json(
        args.base_url,
        {
            "inputs": [
                {
                    "type": "images",
                    "instruction": "Represent the given images.",
                    "media": [image_payload, image_payload],
                }
            ]
        },
    )
    print(f"image-list dimensions: {image_list_response['dimensions']}")

    video_response = post_upload(
        args.base_url,
        {
            "inputs": [
                {
                    "type": "video",
                    "instruction": "Represent the given video.",
                    "media": "video_file",
                }
            ]
        },
        args.video,
    )
    print(f"video dimensions: {video_response['dimensions']}")


if __name__ == "__main__":
    main()

