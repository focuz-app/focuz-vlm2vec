import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

import torch

from src.arguments import DataArguments, ModelArguments
from src.model.model import MMEBModel
from src.model.processor import QWEN2_VL, VLM_IMAGE_TOKENS, load_processor, process_input_text, process_vlm_inputs_fns
from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info
from src.server.config import Settings
from src.server.schemas import InputType
from src.server.types import ResolvedEmbeddingInput
from src.utils.basic_utils import batch_to_device

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: MMEBModel | None = None
        self.processor = None
        self.model_args: ModelArguments | None = None
        self.data_args: DataArguments | None = None
        self.device = settings.device
        self.dtype = self._resolve_dtype(settings.torch_dtype)
        self._gpu_lock = asyncio.Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None and self.model_args is not None

    @property
    def model_name(self) -> str:
        return self.settings.model_name

    @property
    def model_backbone(self) -> str:
        return self.settings.model_backbone

    async def load(self) -> None:
        if self.is_ready:
            return
        if self.settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")

        self.model_args = ModelArguments(
            model_name=self.settings.model_name,
            model_type=self.settings.model_type,
            processor_name=self.settings.processor_name,
            model_backbone=self.settings.model_backbone,
            checkpoint_path=self.settings.checkpoint_path,
            pooling=self.settings.model_pooling,
            normalize=self.settings.model_normalize,
            temperature=self.settings.model_temperature,
            lora=self.settings.model_lora,
        )
        self.data_args = DataArguments(
            max_len=self.settings.max_length,
            resize_min_pixels=self.settings.resize_min_pixels,
            resize_max_pixels=self.settings.resize_max_pixels,
        )
        logger.info("Loading processor for %s", self.settings.model_name)
        self.processor = load_processor(self.model_args, self.data_args)
        logger.info("Loading model %s on %s", self.settings.model_name, self.device)
        self.model = MMEBModel.load(self.model_args, is_trainable=False, processor=self.processor)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        logger.info("Model loaded")

    async def embed(self, inputs: list[ResolvedEmbeddingInput], batch_size: int | None = None) -> list[list[float]]:
        if not self.is_ready:
            raise RuntimeError("model is not loaded")
        resolved_batch_size = min(batch_size or self.settings.max_batch_size, self.settings.max_batch_size)
        results: dict[int, list[float]] = {}

        async with self._gpu_lock:
            non_video = [item for item in inputs if item.type != InputType.video]
            for chunk in _chunks(non_video, resolved_batch_size):
                embeddings = self._embed_non_video_batch(chunk)
                for item, embedding in zip(chunk, embeddings, strict=True):
                    results[item.index] = embedding

            for item in [item for item in inputs if item.type == InputType.video]:
                results[item.index] = self._embed_video(item)

        return [results[index] for index in range(len(inputs))]

    def _embed_non_video_batch(self, items: list[ResolvedEmbeddingInput]) -> list[list[float]]:
        assert self.model is not None
        assert self.processor is not None
        assert self.model_args is not None
        assert self.data_args is not None

        process_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        texts: list[str] = []
        visuals = []
        for item in items:
            text, visual = self._format_non_video_item(item)
            texts.append(text)
            visuals.append(visual)

        processed = process_fn(
            {"text": texts, "images": visuals},
            processor=self.processor,
            max_length=self.data_args.max_len,
        )
        processed = batch_to_device(processed, self.device)
        reps = self._forward(processed)
        return reps.float().cpu().tolist()

    def _embed_video(self, item: ResolvedEmbeddingInput) -> list[float]:
        assert self.model is not None
        assert self.processor is not None
        assert self.model_args is not None
        assert self.data_args is not None
        if self.model_args.model_backbone != QWEN2_VL:
            raise RuntimeError("video uploads are currently implemented for qwen2_vl-backed VLM2Vec models")
        if not isinstance(item.media, Path):
            raise RuntimeError("video input was not resolved to a temporary file")

        prompt = self._format_video_prompt(item)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(item.media),
                        "max_pixels": self.settings.video_max_pixels,
                        "fps": self.settings.video_fps,
                    },
                    {"type": "text", "text": item.text or ""},
                ],
            }
        ]
        _, video_inputs = process_vision_info(messages)
        processed = self.processor(
            text=prompt,
            videos=video_inputs,
            return_tensors="pt",
            max_length=self.data_args.max_len,
            truncation=self.data_args.max_len is not None,
        )
        for key in ("pixel_values_videos", "video_grid_thw"):
            if key in processed and isinstance(processed[key], torch.Tensor):
                processed[key] = processed[key].unsqueeze(0)
        processed = batch_to_device(processed, self.device)
        reps = self._forward(processed)
        return reps[0].float().cpu().tolist()

    def _forward(self, processed: dict) -> torch.Tensor:
        assert self.model is not None
        with torch.inference_mode():
            if self.device.startswith("cuda") and self.dtype in {torch.bfloat16, torch.float16}:
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    output = self.model(qry=processed)
            else:
                output = self.model(qry=processed)
        reps = output["qry_reps"].detach()
        if reps.dim() != 2:
            raise RuntimeError(f"expected dense 2D embeddings, got tensor shape {tuple(reps.shape)}")
        return reps

    def _format_non_video_item(self, item: ResolvedEmbeddingInput) -> tuple[str, object | None]:
        assert self.model_args is not None
        if item.type == InputType.text:
            assert item.text is not None
            if item.instruction:
                return process_input_text(item.instruction, self.model_args.model_backbone, text=item.text), None
            return item.text, None
        if item.type == InputType.image:
            instruction = item.instruction or self.settings.default_image_instruction
            return (
                process_input_text(
                    instruction,
                    self.model_args.model_backbone,
                    text=item.text,
                    add_image_token=True,
                ),
                item.media,
            )
        if item.type == InputType.images:
            instruction = item.instruction or self.settings.default_images_instruction
            prompt = process_input_text(
                instruction,
                self.model_args.model_backbone,
                text=item.text,
                add_image_token=True,
            )
            if isinstance(item.media, list) and len(item.media) > 1:
                prompt = self._repeat_image_token(prompt, len(item.media))
            return prompt, item.media
        raise RuntimeError(f"unexpected non-video input type: {item.type}")

    def _format_video_prompt(self, item: ResolvedEmbeddingInput) -> str:
        assert self.model_args is not None
        instruction = item.instruction or self.settings.default_video_instruction
        return process_input_text(
            instruction,
            self.model_args.model_backbone,
            text=item.text,
            add_video_token=True,
        )

    def _repeat_image_token(self, prompt: str, count: int) -> str:
        assert self.model_args is not None
        token = VLM_IMAGE_TOKENS.get(self.model_args.model_backbone)
        if not token or not prompt.startswith(token):
            return prompt
        return f"{' '.join([token] * count)}{prompt[len(token):]}"

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        if dtype_name == "bfloat16":
            return torch.bfloat16
        if dtype_name == "float16":
            return torch.float16
        if dtype_name == "float32":
            return torch.float32
        raise ValueError(f"unsupported TORCH_DTYPE: {dtype_name}")


def _chunks(items: list[ResolvedEmbeddingInput], size: int) -> Iterable[list[ResolvedEmbeddingInput]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
