from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "VLM2Vec Embedding Server"
    app_version: str = "0.1.0"
    api_key: str | None = Field(default=None, description="Optional API key for private deployments.")

    model_name: str = "VLM2Vec/VLM2Vec-V2.0"
    model_backbone: str = "qwen2_vl"
    model_type: str | None = None
    processor_name: str | None = None
    checkpoint_path: str | None = None
    model_lora: bool = True
    model_pooling: str = "last"
    model_normalize: bool = True
    model_temperature: float = 0.02
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    load_model_on_startup: bool = True

    max_batch_size: int = 16
    max_length: int | None = None
    resize_min_pixels: int = 28 * 28 * 4
    resize_max_pixels: int = 28 * 28 * 1280

    max_media_mb: int = 128
    max_image_pixels: int = 25_000_000
    request_timeout_seconds: float = 20.0
    allow_private_urls: bool = False

    default_image_instruction: str = "Represent the given image."
    default_images_instruction: str = "Represent the given images."
    default_video_instruction: str = "Represent the given video."
    video_fps: float = 1.0
    video_max_pixels: int = 360 * 420

    @property
    def max_media_bytes(self) -> int:
        return self.max_media_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

