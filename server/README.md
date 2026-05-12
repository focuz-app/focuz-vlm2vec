# VLM2Vec FastAPI Embedding Server

This server exposes VLM2Vec embeddings for text, images, image lists, and videos through a FastAPI API. It defaults to `VLM2Vec/VLM2Vec-V2.0` with the `qwen2_vl` backbone, LoRA enabled, normalized embeddings, and CUDA bf16 inference.

## Host Requirements

For Docker deployment on Ubuntu:

- NVIDIA driver installed and visible through `nvidia-smi`.
- Docker Engine and Docker Compose plugin.
- NVIDIA Container Toolkit configured for Docker.

Quick host check:

```bash
nvidia-smi
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## Run With Docker Compose

```bash
docker compose up --build
```

The first startup downloads the model into the `hf-cache` Docker volume. The API listens on `http://localhost:8000`.

With an API key:

```bash
API_KEY=change-me docker compose up --build
```

## Local Development

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-server.txt
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Use one worker. Each worker loads its own model copy, which wastes GPU memory.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_NAME` | `VLM2Vec/VLM2Vec-V2.0` | Hugging Face model name or mounted model path. |
| `MODEL_BACKBONE` | `qwen2_vl` | Repo model backbone name. |
| `CHECKPOINT_PATH` | unset | Optional LoRA/checkpoint path. |
| `MODEL_LORA` | `true` | Load LoRA adapter weights. |
| `MODEL_NORMALIZE` | `true` | Return normalized embeddings. |
| `MODEL_POOLING` | `last` | Pooling mode passed to `MMEBModel`. |
| `DEVICE` | `cuda` | `cuda`, `cuda:0`, `cpu`, or `auto`. |
| `TORCH_DTYPE` | `bfloat16` | `bfloat16`, `float16`, or `float32`. |
| `MAX_BATCH_SIZE` | `16` | Upper bound for request micro-batches. |
| `MAX_MEDIA_MB` | `128` | Max decoded media bytes per item. |
| `MAX_IMAGE_PIXELS` | `25000000` | Max decoded image pixels. |
| `REQUEST_TIMEOUT_SECONDS` | `20` | HTTP(S) media fetch timeout. |
| `ALLOW_PRIVATE_URLS` | `false` | Allow URL fetches to private or loopback networks. |
| `API_KEY` | unset | Optional bearer or `X-API-Key` auth token. |
| `VIDEO_FPS` | `1.0` | Video frame sampling rate. |
| `VIDEO_MAX_PIXELS` | `151200` | Max pixels passed to Qwen video preprocessing. |

The Compose GPU reservation also accepts `NVIDIA_DEVICE_ID` on the host side. For MIG deployments, set it to the MIG UUID shown by `nvidia-smi -L`, for example:

```bash
NVIDIA_DEVICE_ID=MIG-2d8def8e-1c22-5493-9185-71da1202e4da docker compose up --build
```

If running Compose through `sudo`, either use a `.env` file or preserve the variable:

```bash
sudo -E docker compose up --build
```

## API

### Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/v1/models
```

When `API_KEY` is set, pass either:

```bash
-H "Authorization: Bearer $API_KEY"
```

or:

```bash
-H "X-API-Key: $API_KEY"
```

### JSON Embeddings

`POST /v1/embeddings`

```json
{
  "inputs": [
    {
      "type": "text",
      "text": "A cat and a dog"
    },
    {
      "type": "image",
      "instruction": "Represent the given image.",
      "media": "data:image/jpeg;base64,..."
    },
    {
      "type": "images",
      "instruction": "Represent the given images.",
      "media": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."]
    },
    {
      "type": "video",
      "instruction": "Represent the given video.",
      "media": "https://example.com/video.mp4"
    }
  ],
  "batch_size": 4
}
```

Response:

```json
{
  "model": "VLM2Vec/VLM2Vec-V2.0",
  "dimensions": 1536,
  "data": [
    {
      "index": 0,
      "embedding": [0.01, 0.02]
    }
  ]
}
```

`media` accepts base64 strings, data URLs, or HTTP(S) URLs. Private-network URLs are blocked by default.

### Multipart Upload Embeddings

`POST /v1/embeddings/uploads`

The `payload` form field uses the same schema as JSON requests, but `media` values reference uploaded form field names.

```bash
curl -X POST http://localhost:8000/v1/embeddings/uploads \
  -F 'payload={
    "inputs": [
      {"type": "image", "media": "image_file"},
      {"type": "video", "media": "video_file"}
    ]
  }' \
  -F image_file=@assets/example.jpg \
  -F video_file=@assets/example_video.mp4
```

## Demo Client

Run:

```bash
python examples/server_embeddings_demo.py \
  --base-url http://localhost:8000 \
  --image assets/example.jpg \
  --video assets/example_video.mp4
```

With auth:

```bash
API_KEY=change-me python examples/server_embeddings_demo.py --base-url http://localhost:8000
```

The demo prints embedding dimensions and a cosine similarity between two returned vectors.

## Tests

API and media tests use a fake embedding service, so they do not load model weights:

```bash
python -m pip install -r requirements-dev.txt
pytest tests/server
```

Run the optional real GPU smoke test only on a CUDA host:

```bash
RUN_GPU_TESTS=1 pytest tests/server/test_gpu_smoke.py
```

## Troubleshooting

- `503 model is not ready`: startup is still loading or model loading failed. Check container logs.
- CUDA is unavailable: verify `docker run --rm --gpus all ... nvidia-smi` on the host.
- Out of memory: lower `MAX_BATCH_SIZE`, lower `VIDEO_MAX_PIXELS`, or run one container per GPU.
- Video decode fails: make sure `ffmpeg` is available in the image and the file is a supported video container.
- URL media fails: private URLs are blocked unless `ALLOW_PRIVATE_URLS=true`.
