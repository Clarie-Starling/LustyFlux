import base64
from PIL import Image
from io import BytesIO
import os
import logging
import random
import threading
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)

# Idempotent environment + client init
_env_ready = False
_env_lock = threading.Lock()
_s3_client = None

def get_bucket_name() -> str:
    return os.getenv("B2_BUCKET")

def _build_endpoint(region: str) -> str:
    region = region or "us-east-005"
    return f"https://s3.{region}.backblazeb2.com"

def ensure_env_ready():
    global _env_ready, _s3_client
    if _env_ready:
        return
    with _env_lock:
        if _env_ready:
            return
        # Read env (can be set before calling this function)
        region = os.getenv("B2_REGION", "us-east-005")
        endpoint = _build_endpoint(region)
        key = os.getenv("B2_KEY_ID")
        secret = os.getenv("B2_SECRET_KEY")
        # Lazily create S3 client
        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region,
        )
        _env_ready = True
        logging.info("Environment initialized (utils)")

def get_s3_client():
    ensure_env_ready()
    return _s3_client

# Backblaze S3-compatible config from env vars
B2_BUCKET = os.getenv("B2_BUCKET")
B2_REGION = os.getenv("B2_REGION", "us-east-005")
B2_ENDPOINT = f"https://s3.{B2_REGION}.backblazeb2.com"
B2_ACCESS_KEY = os.getenv("B2_KEY_ID")
B2_SECRET_KEY = os.getenv("B2_SECRET_KEY")
IMG_RES_BASE = os.getenv("IMG_RES_BASE", 1024)

s3_client = boto3.client(
    "s3",
    endpoint_url=B2_ENDPOINT,
    aws_access_key_id=B2_ACCESS_KEY,
    aws_secret_access_key=B2_SECRET_KEY,
    region_name=B2_REGION,
)

def load_image_input(inputs):
    b64 = inputs.get("image_base64")
    if not b64:
        return None
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[-1]
    image_data = base64.b64decode(b64)
    return Image.open(BytesIO(image_data)).convert("RGB")

def upscale_image(image: Image.Image, factor: float) -> Image.Image:
    w, h = image.size
    return image.resize((int(w * factor), int(h * factor)), resample=Image.BILINEAR)

def get_presigned_url(s3_key: str, expires_in: int = 604800) -> str:
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": B2_BUCKET, "Key": s3_key},
            ExpiresIn=expires_in,
        )
    except (BotoCoreError, ClientError) as e:
        logging.error(f"Failed to generate presigned URL for {s3_key}: {e}")
        raise

def upload_to_s3(image: Image.Image, job_id: str) -> Dict[str, str]:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    key = f"{job_id}.png"

    try:
        s3_client.upload_fileobj(buffer, B2_BUCKET, key, ExtraArgs={"ContentType": "image/png"})
        logging.info(f"Uploaded {key} to Backblaze bucket {B2_BUCKET}")
        return {"url": get_presigned_url(key), "s3_key": key}

    except (BotoCoreError, ClientError) as e:
        logging.error(f"S3 upload failed: {e}")
        raise

def aspect_to_resolution(aspect_ratio: str, max_dim: int = IMG_RES_BASE) -> tuple[int, int]:
    try:
        w, h = map(int, aspect_ratio.split(":"))
    except ValueError:
        raise ValueError(f"Invalid aspect ratio: {aspect_ratio}")

    scale = max_dim / max(w, h)
    scaled_w, scaled_h = round(w * scale), round(h * scale)

    # Enforce divisibility by 16
    adjusted_w = (scaled_w // 16) * 16
    adjusted_h = (scaled_h // 16) * 16

    return adjusted_w, adjusted_h

def generate_seed() -> int:
    return random.randint(0, 2**32 - 1)

def initialize_worker_environment():
    """Initialize all required model files from S3 storage"""
    ensure_env_ready()
    s3_client = get_s3_client()
    bucket = get_bucket_name()

    model_files = {
        "main_model": {
            "env_var": "MODEL_FILENAME",
            "default": "getphatFLUXReality_v8.safetensors",
            "s3_key": "models/{filename}",
            "local_path": "/workspace/models/{filename}"
        },
        "t5_encoder": {
            "s3_key": "models/t5xxl_fp16.safetensors",
            "local_path": "/workspace/models/t5xxl_fp16.safetensors"
        },
        "clip_encoder": {
            "s3_key": "models/clip_l.safetensors",
            "local_path": "/workspace/models/clip_l.safetensors"
        },
        "vae": {
            "s3_key": "models/ae.safetensors",
            "local_path": "/workspace/models/ae.safetensors"
        }
    }

    # Download missing files
    for name, config in model_files.items():
        if name == "main_model":
            filename = os.getenv(config["env_var"], config["default"])
            local_path = config["local_path"].format(filename=filename)
            s3_key = config["s3_key"].format(filename=filename)
        else:
            local_path = config["local_path"]
            s3_key = config["s3_key"]

        if os.path.exists(local_path):
            logging.info(f'{name.replace("_", " ").title()} exists at {local_path}')
            continue

        logging.info(f'Downloading {name.replace("_", " ")}...')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            with open(local_path, "wb") as f:
                s3_client.download_fileobj(bucket, s3_key, f)
            logging.info(f'Successfully downloaded {name} to {local_path}')
        except (BotoCoreError, ClientError, OSError) as e:
            logging.error(f'Failed to download {name}: {e}')
            if name in ["t5_encoder", "clip_encoder"]:
                raise RuntimeError(f"Critical text encoder missing: {name}")
            elif name == "main_model":
                raise

    # Ensure loras directory exists
    os.makedirs("/workspace/models/loras", exist_ok=True)
    logging.info("LORA directory ready for on-demand downloads")
