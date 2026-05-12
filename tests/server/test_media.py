from __future__ import annotations

import asyncio
import base64
from io import BytesIO

import pytest
from fastapi import HTTPException
from PIL import Image

from src.server.config import Settings
from src.server.media import MediaResolver


def make_image_data_url() -> str:
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_data_url_image_decodes_to_rgb() -> None:
    resolver = MediaResolver(Settings(load_model_on_startup=False, max_media_mb=1))
    image = asyncio.run(resolver.image_from_ref(make_image_data_url()))
    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_invalid_base64_is_rejected() -> None:
    resolver = MediaResolver(Settings(load_model_on_startup=False, max_media_mb=1))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(resolver.bytes_from_ref("not valid base64"))
    assert exc_info.value.status_code == 400


def test_private_urls_are_blocked_by_default() -> None:
    resolver = MediaResolver(Settings(load_model_on_startup=False, allow_private_urls=False))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(resolver.bytes_from_ref("http://127.0.0.1/image.png"))
    assert exc_info.value.status_code == 400
    assert "private" in exc_info.value.detail


def test_oversized_base64_is_rejected_before_decode() -> None:
    resolver = MediaResolver(Settings(load_model_on_startup=False, max_media_mb=1))
    oversized = "a" * (2 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(resolver.bytes_from_ref(oversized))
    assert exc_info.value.status_code == 413

