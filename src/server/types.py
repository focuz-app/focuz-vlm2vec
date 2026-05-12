from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.server.schemas import InputType


@dataclass(slots=True)
class UploadedBlob:
    field_name: str
    filename: str | None
    content_type: str | None
    data: bytes


@dataclass(slots=True)
class ResolvedEmbeddingInput:
    index: int
    type: InputType
    text: str | None
    instruction: str | None
    media: Any = None
    cleanup_paths: list[Path] = field(default_factory=list)

