from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputType(str, Enum):
    text = "text"
    image = "image"
    images = "images"
    video = "video"


class EmbeddingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: InputType = Field(description="Input modality.")
    text: str | None = Field(default=None, description="Text content or extra text paired with visual input.")
    instruction: str | None = Field(default=None, description="Optional embedding instruction.")
    media: str | list[str] | None = Field(
        default=None,
        description="Base64/data URL/HTTP(S) URL for JSON requests, or uploaded field name for multipart requests.",
    )
    metadata: dict[str, Any] | None = Field(default=None, description="Optional client metadata echoed nowhere.")

    @model_validator(mode="after")
    def validate_input(self) -> "EmbeddingInput":
        if self.type == InputType.text:
            if not self.text or not self.text.strip():
                raise ValueError("text inputs require non-empty text")
            if self.media is not None:
                raise ValueError("text inputs must not include media")
        elif self.type in {InputType.image, InputType.video}:
            if not isinstance(self.media, str) or not self.media:
                raise ValueError(f"{self.type.value} inputs require media as a string")
        elif self.type == InputType.images:
            if not isinstance(self.media, list) or not self.media:
                raise ValueError("images inputs require media as a non-empty list")
            if not all(isinstance(item, str) and item for item in self.media):
                raise ValueError("images media entries must be non-empty strings")
        return self


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[EmbeddingInput] = Field(min_length=1, description="Inputs to embed.")
    batch_size: int | None = Field(default=None, ge=1, description="Optional per-request micro-batch size.")


class EmbeddingData(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    data: list[EmbeddingData]


class ModelInfo(BaseModel):
    id: str
    backbone: str
    device: str
    ready: bool


class ModelsResponse(BaseModel):
    data: list[ModelInfo]

