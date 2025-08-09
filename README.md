# Pipeline Loader and Caching Service

A lightweight, production-ready pipeline builder that downloads model assets from S3-compatible storage, builds pipelines with optional LoRA overlays, and caches them both in-memory and on-disk for fast reuse. Designed to minimize cold start times and support large weights efficiently.

## Key features

- Disk and in-memory pipeline caching
- Optional LoRA application with per-LoRA scaling
- Pluggable S3-compatible storage (e.g., Backblaze B2, MinIO, R2)
- Safe local caching with reproducible manifests
- Optional flash-attention acceleration with graceful fallback
- Headless test suite that runs without GPU, CUDA, or real models

## Requirements

- Python 3.11+
- pip
- For Docker builds: recent Docker engine
- For acceleration (optional): NVIDIA CUDA matching your Torch version
    - If building flash-attn from source, ensure a capable build VM with sufficient CPU/RAM

## Quick start

### 1) Clone and prepare environment

```shell script
# bash
git clone <your-repo-url>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install diffusers transformers safetensors
```


Set required environment variables (see “Environment variables” below). For local testing, you can set placeholders.

```shell script
# bash
export B2_BUCKET=<your-bucket>
export B2_REGION=<your-region>              # e.g., us-east-005
export B2_KEY_ID=<your-key-id>
export B2_SECRET_KEY=<your-secret>
export PIPELINE_STORE=/absolute/path/to/cache
export HF_TOKEN=<your-hf-token-optional>
```


Run your entrypoint:

```shell script
# bash
python handler.py
```


### 2) Docker build and run

Build the image:

```shell script
# bash
docker build -t pipeline-builder:latest .
```


Run:

```shell script
# bash
docker run --rm \
  -e B2_BUCKET=<your-bucket> \
  -e B2_REGION=<your-region> \
  -e B2_KEY_ID=<your-key-id> \
  -e B2_SECRET_KEY=<your-secret> \
  -e PIPELINE_STORE=/workspace/pretrained \
  -e HF_TOKEN=<your-hf-token-optional> \
  -v /local/models:/workspace/models \
  pipeline-builder:latest
```


Notes:
- Mounting /workspace/models is recommended to persist downloaded weights and LoRAs between runs.
- The container creates and uses a non-root user by default.

## Environment variables

- B2_BUCKET: S3-compatible bucket name for your model and LoRA objects.
- B2_REGION: Region string for the S3-compatible endpoint (example: us-east-005).
- B2_KEY_ID: Access key ID for your S3-compatible storage.
- B2_SECRET_KEY: Secret key for your S3-compatible storage.
- PIPELINE_STORE: Absolute path for on-disk pipeline cache (e.g., /workspace/pretrained).
- HF_TOKEN: Optional token for fetching base pipelines from Hugging Face when needed.

Advanced/optional:
- Custom S3 endpoint: set endpoint_url via your client wiring if you front the origin with a CDN (see “CDN and custom endpoints”).
- Addressing style (path vs virtual-host): ensure your client is configured appropriately when using a CNAME.

## Testing

Fast, safe tests that mock GPU and network dependencies.

```shell script
# bash
pip install pytest
pytest -q
```


- No real GPU, CUDA, or model downloads are triggered.
- Works on CPU-only and AMD GPU workstations.

Optional pytest.ini (quiet defaults):

```textmate
# ini
[pytest]
addopts = -q
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
```


## Caching behavior

- In-memory cache: reuses pipelines created in the current process based on a cache key derived from model type and LoRA set.
- On-disk cache: serialized pipelines stored under PIPELINE_STORE with a deterministic directory derived from model configuration and weight fingerprints.
- A build manifest is written alongside cached pipelines to aid reproducibility and troubleshooting.

Tips:
- Use versioned, immutable object names for large artifacts (include content hash) to simplify cache invalidation.
- Mount a persistent volume for PIPELINE_STORE and /workspace/models to avoid cold rebuilds on restart.

## LoRA workflow

- Provide LoRA entries with path and optional scale.
- LoRAs are applied on top of a clean base pipeline during load.
- Use consistent naming and versioning; cache keys include LoRA identity and scale.

Example LoRA list (conceptual):

```python
# python
loras = [
    {"path": "/workspace/models/loras/my-style.safetensors", "scale": 0.7},
    {"path": "/workspace/models/loras/extra-detail.safetensors"},
]
```


## Acceleration: flash-attention (optional)

- If present and compatible, attention kernels may switch to accelerated implementations.
- If absent, execution falls back to standard attention.
- Building flash-attn from source can be slow; prefer prebuilt wheels when available or constrain the CUDA architectures:
    - Set TORCH_CUDA_ARCH_LIST to only the architectures you deploy (e.g., “8.9” for Ada, “9.0” for Hopper).
    - Consider using a shared “wheelhouse” and install the built wheel across environments.

## CDN and custom endpoints (optional)

For faster cross-region cold starts:
- Place your S3-compatible storage behind a CDN and use long Cache-Control TTLs on immutable, versioned artifacts.
- Prefer unsigned GETs for cacheability at the edge.
- Ensure Accept-Ranges is supported for partial content requests.
- You can set a custom endpoint URL in your S3 client to point at your CDN CNAME.

## Troubleshooting

- Slow flash-attn build:
    - Constrain TORCH_CUDA_ARCH_LIST (e.g., export TORCH_CUDA_ARCH_LIST="8.9").
    - Use more CPU/RAM for the build host.
    - Try prebuilt wheels first; otherwise cache a built wheel for reuse.
- Long cold starts due to large models:
    - Use CDN with tiered cache, persistent local cache volumes, and versioned artifacts.
    - Consider pre-baked images that include your most used models.
- Permission or 403 errors fetching objects:
    - Verify keys, bucket policy, and endpoint/region values.
    - For CDN caching, prefer public objects and unsigned requests for downloads.

## Security notes

- Do not commit secrets. Use environment variables or a secrets manager.
- Avoid logging sensitive values. Redact keys/tokens from logs.
- Use least-privilege IAM for object storage access.

## Contributing

- Open issues and PRs with a clear description and small, focused changes.
- Please add or update tests for new functionality.

## License

- Add your chosen license here.

## Support

- File issues for bugs or questions.
- For optimization help (caching, CDN, flash-attn builds), open a discussion with:
    - Your target GPU architecture(s)
    - Torch/CUDA/Python versions
    - Storage/CDN provider and region(s)