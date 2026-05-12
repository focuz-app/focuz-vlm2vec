import json
import logging
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from starlette.datastructures import UploadFile as StarletteUploadFile

from src.server.config import Settings, get_settings
from src.server.media import MediaResolver
from src.server.schemas import EmbeddingRequest, EmbeddingResponse, ModelInfo, ModelsResponse
from src.server.types import UploadedBlob

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingServiceProtocol(Protocol):
    device: str

    @property
    def is_ready(self) -> bool: ...

    async def load(self) -> None: ...

    async def embed(self, resolved, batch_size=None) -> list[list[float]]: ...


class LazyEmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.device
        self._service: EmbeddingServiceProtocol | None = None

    @property
    def is_ready(self) -> bool:
        return self._service is not None and self._service.is_ready

    async def load(self) -> None:
        if self._service is None:
            from src.server.service import EmbeddingService

            self._service = EmbeddingService(self.settings)
        await self._service.load()
        self.device = self._service.device

    async def embed(self, resolved, batch_size=None) -> list[list[float]]:
        if self._service is None:
            raise RuntimeError("model is not loaded")
        return await self._service.embed(resolved, batch_size=batch_size)


def create_app(settings: Settings | None = None, service: EmbeddingServiceProtocol | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    embedding_service = service or LazyEmbeddingService(resolved_settings)
    media_resolver = MediaResolver(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.embedding_service = embedding_service
        app.state.media_resolver = media_resolver
        if resolved_settings.load_model_on_startup:
            await embedding_service.load()
        yield

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.embedding_service = embedding_service
    app.state.media_resolver = media_resolver

    async def require_api_key(
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = resolved_settings.api_key
        if not expected:
            return
        bearer = None
        if authorization and authorization.lower().startswith("bearer "):
            bearer = authorization[7:]
        if x_api_key == expected or bearer == expected:
            return
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")

    def ensure_ready() -> None:
        if not embedding_service.is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model is not ready")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        ensure_ready()
        return {"status": "ready"}

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    async def models() -> ModelsResponse:
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=resolved_settings.model_name,
                    backbone=resolved_settings.model_backbone,
                    device=embedding_service.device,
                    ready=embedding_service.is_ready,
                )
            ]
        )

    @app.post("/v1/embeddings", dependencies=[Depends(require_api_key)])
    async def embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        ensure_ready()
        if request.batch_size and request.batch_size > resolved_settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"batch_size must be <= {resolved_settings.max_batch_size}",
            )
        resolved = await media_resolver.prepare_inputs(request.inputs)
        try:
            vectors = await embedding_service.embed(resolved, batch_size=request.batch_size)
        finally:
            media_resolver.cleanup(resolved)
        return _embedding_response(resolved_settings.model_name, vectors)

    @app.post("/v1/embeddings/uploads", dependencies=[Depends(require_api_key)])
    async def upload_embeddings(request: Request) -> EmbeddingResponse:
        ensure_ready()
        form = await request.form()
        payload_value = form.get("payload")
        if not isinstance(payload_value, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="multipart form requires payload field")
        try:
            embedding_request = EmbeddingRequest.model_validate(json.loads(payload_value))
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"invalid payload: {exc}") from exc
        if embedding_request.batch_size and embedding_request.batch_size > resolved_settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"batch_size must be <= {resolved_settings.max_batch_size}",
            )
        uploads = await _collect_uploads(form.multi_items(), media_resolver)
        resolved = await media_resolver.prepare_inputs(embedding_request.inputs, uploads=uploads)
        try:
            vectors = await embedding_service.embed(resolved, batch_size=embedding_request.batch_size)
        finally:
            media_resolver.cleanup(resolved)
        return _embedding_response(resolved_settings.model_name, vectors)

    return app


async def _collect_uploads(items, media_resolver: MediaResolver) -> Mapping[str, UploadedBlob]:
    uploads: dict[str, UploadedBlob] = {}
    for key, value in items:
        if key == "payload":
            continue
        if not isinstance(value, StarletteUploadFile):
            continue
        if key in uploads:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"duplicate upload field: {key}")
        uploads[key] = await media_resolver.read_upload(key, value)
    return uploads


def _embedding_response(model_name: str, vectors: list[list[float]]) -> EmbeddingResponse:
    dimensions = len(vectors[0]) if vectors else 0
    return EmbeddingResponse(
        model=model_name,
        dimensions=dimensions,
        data=[{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
    )


app = create_app()
