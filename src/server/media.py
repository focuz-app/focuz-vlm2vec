import base64
import binascii
import ipaddress
import os
import socket
import tempfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from src.server.config import Settings
from src.server.schemas import EmbeddingInput, InputType
from src.server.types import ResolvedEmbeddingInput, UploadedBlob


class MediaResolver:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def read_upload(self, field_name: str, upload: UploadFile) -> UploadedBlob:
        data = await upload.read(self.settings.max_media_bytes + 1)
        if len(data) > self.settings.max_media_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"uploaded media exceeds {self.settings.max_media_mb} MB",
            )
        return UploadedBlob(
            field_name=field_name,
            filename=upload.filename,
            content_type=upload.content_type,
            data=data,
        )

    async def prepare_inputs(
        self,
        inputs: list[EmbeddingInput],
        uploads: Mapping[str, UploadedBlob] | None = None,
    ) -> list[ResolvedEmbeddingInput]:
        resolved: list[ResolvedEmbeddingInput] = []
        for index, item in enumerate(inputs):
            if item.type == InputType.text:
                resolved.append(
                    ResolvedEmbeddingInput(
                        index=index,
                        type=item.type,
                        text=item.text,
                        instruction=item.instruction,
                    )
                )
                continue

            if item.type == InputType.image:
                assert isinstance(item.media, str)
                resolved.append(
                    ResolvedEmbeddingInput(
                        index=index,
                        type=item.type,
                        text=item.text,
                        instruction=item.instruction,
                        media=await self.image_from_ref(item.media, uploads),
                    )
                )
                continue

            if item.type == InputType.images:
                assert isinstance(item.media, list)
                resolved.append(
                    ResolvedEmbeddingInput(
                        index=index,
                        type=item.type,
                        text=item.text,
                        instruction=item.instruction,
                        media=[await self.image_from_ref(ref, uploads) for ref in item.media],
                    )
                )
                continue

            if item.type == InputType.video:
                assert isinstance(item.media, str)
                video_path = await self.video_path_from_ref(item.media, uploads)
                resolved.append(
                    ResolvedEmbeddingInput(
                        index=index,
                        type=item.type,
                        text=item.text,
                        instruction=item.instruction,
                        media=video_path,
                        cleanup_paths=[video_path],
                    )
                )
        return resolved

    async def image_from_ref(
        self,
        ref: str,
        uploads: Mapping[str, UploadedBlob] | None = None,
    ) -> Image.Image:
        data = await self.bytes_from_ref(ref, uploads)
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                if width * height > self.settings.max_image_pixels:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"image exceeds {self.settings.max_image_pixels} pixels",
                    )
                return image.convert("RGB")
        except UnidentifiedImageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid image media") from exc
        except Image.DecompressionBombError as exc:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="image is too large") from exc

    async def video_path_from_ref(
        self,
        ref: str,
        uploads: Mapping[str, UploadedBlob] | None = None,
    ) -> Path:
        data = await self.bytes_from_ref(ref, uploads)
        suffix = self._guess_video_suffix(ref, uploads)
        fd, temp_path = tempfile.mkstemp(prefix="vlm2vec-video-", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        return Path(temp_path)

    async def bytes_from_ref(
        self,
        ref: str,
        uploads: Mapping[str, UploadedBlob] | None = None,
    ) -> bytes:
        if uploads and ref in uploads:
            return uploads[ref].data
        if ref.startswith("data:"):
            return self._decode_data_url(ref)
        if ref.startswith("http://") or ref.startswith("https://"):
            return await self._fetch_url(ref)
        return self._decode_base64(ref)

    def cleanup(self, resolved: list[ResolvedEmbeddingInput]) -> None:
        for item in resolved:
            for path in item.cleanup_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _decode_data_url(self, ref: str) -> bytes:
        header, separator, payload = ref.partition(",")
        if not separator or ";base64" not in header:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="data URLs must be base64 encoded")
        return self._decode_base64(payload)

    def _decode_base64(self, ref: str) -> bytes:
        if len(ref) > int(self.settings.max_media_bytes * 1.4) + 128:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"base64 media exceeds {self.settings.max_media_mb} MB",
            )
        try:
            data = base64.b64decode(ref, validate=True)
        except binascii.Error as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid base64 media") from exc
        if len(data) > self.settings.max_media_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"decoded media exceeds {self.settings.max_media_mb} MB",
            )
        return data

    async def _fetch_url(self, url: str) -> bytes:
        current_url = url
        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds, follow_redirects=False) as client:
            for _ in range(5):
                self._validate_url(current_url)
                try:
                    async with client.stream("GET", current_url) as response:
                        if response.is_redirect and "location" in response.headers:
                            current_url = urljoin(current_url, response.headers["location"])
                            continue

                        if response.status_code >= 400:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"media URL returned HTTP {response.status_code}",
                            )

                        content_length = response.headers.get("content-length")
                        if content_length and content_length.isdigit() and int(content_length) > self.settings.max_media_bytes:
                            raise HTTPException(
                                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"URL media exceeds {self.settings.max_media_mb} MB",
                            )

                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > self.settings.max_media_bytes:
                                raise HTTPException(
                                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                    detail=f"URL media exceeds {self.settings.max_media_mb} MB",
                                )
                        return bytes(data)
                except httpx.HTTPError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"failed to fetch media URL: {exc}",
                    ) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="too many media URL redirects")

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="media URL must be HTTP(S)")
        if self.settings.allow_private_urls:
            return
        try:
            addr_infos = socket.getaddrinfo(parsed.hostname, None)
        except socket.gaierror as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="media URL host cannot be resolved") from exc
        for addr_info in addr_infos:
            ip = ipaddress.ip_address(addr_info[4][0])
            if not ip.is_global:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="private media URLs are not allowed")

    def _guess_video_suffix(self, ref: str, uploads: Mapping[str, UploadedBlob] | None = None) -> str:
        filename = uploads[ref].filename if uploads and ref in uploads else urlparse(ref).path
        suffix = Path(filename or "").suffix.lower()
        return suffix if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"} else ".mp4"
