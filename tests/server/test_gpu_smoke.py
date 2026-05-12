from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 to load the real model")
def test_real_model_text_and_image_smoke() -> None:
    from src.server.config import Settings
    from src.server.media import MediaResolver
    from src.server.schemas import EmbeddingInput, InputType
    from src.server.service import EmbeddingService

    settings = Settings(load_model_on_startup=False, max_batch_size=2)
    service = EmbeddingService(settings)
    resolver = MediaResolver(settings)

    async def run_smoke() -> None:
        await service.load()
        image_b64 = Path("assets/example.jpg").read_bytes()
        import base64

        request_inputs = [
            EmbeddingInput(type=InputType.text, text="A cat and a dog"),
            EmbeddingInput(type=InputType.image, media=base64.b64encode(image_b64).decode("ascii")),
        ]
        resolved = await resolver.prepare_inputs(request_inputs)
        embeddings = await service.embed(resolved, batch_size=2)
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert len(embeddings[0]) == len(embeddings[1])

    asyncio.run(run_smoke())

