import os
import json
import time
import hashlib
import logging
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel, T5Config, CLIPTextConfig
from safetensors.torch import load_file
import utils as u


from botocore.exceptions import BotoCoreError, ClientError
# add near other imports
from transformers.utils import is_flash_attn_2_available

logging.basicConfig(level=logging.INFO)

# Enhanced cache system (in-memory for current process)
pipe_cache = {}
lora_cache = {}


# Where to store serialized pipelines
PIPELINE_STORE = os.getenv("PIPELINE_STORE", "/runpod-volume/pretrained")

# Precision configuration (use bfloat16 for best throughput)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dtype = torch.bfloat16


t5_config = T5Config(
    vocab_size=32128,
    d_model=4096,
    d_kv=64,
    d_ff=10240,
    num_layers=24,
    num_heads=64,
    relative_attention_num_buckets=32,
    dropout_rate=0.0,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="gated-gelu",
    is_encoder_decoder=False,
    use_cache=False,
    pad_token_id=0,
    eos_token_id=1,
    torch_dtype=torch.bfloat16,
)

clip_config = CLIPTextConfig(
    vocab_size=49408,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=77,
    projection_dim=768,
)

vae_config = {
    "sample_size": 256,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 32,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
}
vae = AutoencoderKL(**vae_config).to("cuda", dtype=dtype)


def _file_fingerprint(path: str) -> dict:
    """
    Cheap fingerprint for large weights: size + mtime.
    Avoid hashing GBs of data on every run.
    """
    try:
        st = os.stat(path)
        return {"path": os.path.basename(path), "size": st.st_size, "mtime": int(st.st_mtime)}
    except FileNotFoundError:
        return {"path": os.path.basename(path), "missing": True}


def _stable_hash(payload: dict) -> str:
    """
    Create a stable short hash from a dict.
    """
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _pipeline_dir(key: dict) -> str:
    """
    Map a logical key to a directory under PIPELINE_STORE.
    """
    base = _stable_hash(key)
    sub = key.get("model_type", "model")
    return os.path.join(PIPELINE_STORE, f"{sub}-{base}")


def get_lora_path(lora_config):
    filename = os.path.basename(lora_config["path"])
    local_path = f"/workspace/models/loras/{filename}"
    s3_key = f"models/loras/{filename}"

    if not os.path.exists(local_path):
        logging.info(f"Downloading LORA: {filename}")
        try:
            u.initialize_worker_environment()
            s3_client = u.get_s3_client()
            bucket = u.get_bucket_name()
            with open(local_path, "wb") as f:
                s3_client.download_fileobj(bucket, s3_key, f)

            logging.info(f"Downloaded LORA to {local_path}")
        except (BotoCoreError, ClientError) as e:
            logging.error(f"Failed to download LORA {filename}: {e}")
            return None

    return local_path


def get_lora_key(lora_list):
    if not lora_list:
        return "no_lora"
    # Include per-LORA scale if provided (weight, alpha, etc.)
    parts = []
    for l in sorted(lora_list, key=lambda x: os.path.basename(x["path"])):  # deterministic
        name = os.path.basename(l["path"])
        scale = l.get("scale") or l.get("weight") or l.get("alpha")
        parts.append(f"{name}:{scale}" if scale is not None else name)
    return "-".join(parts)


def validate_and_load_loras(lora_list):
    valid_loras = []
    for lora in lora_list or []:
        filename = os.path.basename(lora["path"])
        path = f"/workspace/models/loras/{filename}"
        if os.path.exists(path):
            valid_loras.append({"path": path, "scale": lora.get("scale")})
        else:
            logging.warning(f"LoRA file {path} not found, skipping")
    return valid_loras


def apply_loras(pipe, lora_items):
    """
    Apply LoRAs to pipeline. If a scale is provided, pass it in.
    """
    if hasattr(pipe, "disable_lora"):
        pipe.disable_lora()

    for item in lora_items or []:
        path = item["path"]
        scale = item.get("scale", None)
        if path not in lora_cache:
            # If API supports passing weight/scale, do it; otherwise just load.
            try:
                if scale is not None:
                    pipe.load_lora_weights(path, weight_name=None, weight=scale)
                else:
                    pipe.load_lora_weights(path, weight_name=None)
                lora_cache[path] = True
            except TypeError:
                # Fallback if weight arg not supported
                pipe.load_lora_weights(path, weight_name=None)
                lora_cache[path] = True
        else:
            # Re-enable already loaded LORA (load again ensures active)
            if scale is not None:
                try:
                    pipe.load_lora_weights(path, weight_name=None, weight=scale)
                except TypeError:
                    pipe.load_lora_weights(path, weight_name=None)
            else:
                pipe.load_lora_weights(path, weight_name=None)
    return pipe


def _load_local_pipeline_if_exists(save_dir: str):
    if os.path.isdir(save_dir) and os.path.isfile(os.path.join(save_dir, "model_index.json")):
        logging.info(f"Loading pipeline from local cache: {save_dir}")
        pipe = FluxPipeline.from_pretrained(
            save_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
        return pipe.to("cuda", dtype=dtype)
    logging.info(f"No local pipeline found at {save_dir}")
    return None



def _save_pipeline(pipe, save_dir: str):
    _ensure_dir(save_dir)
    # Write a small manifest for debugging and reproducibility
    manifest_path = os.path.join(save_dir, "build_manifest.json")
    try:
        pipe.save_pretrained(save_dir, safe_serialization=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": int(time.time()),
                    "dtype": str(dtype),
                    "torch_version": torch.__version__,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        logging.info(f"Saved pipeline to {save_dir}")
    except Exception as e:
        logging.error(f"Failed to save pipeline to {save_dir}: {e}")
        # Best effort; do not crash here.


def get_pipeline(model_type, lora_list):
    """
    Build-or-load pipeline with disk-level caching + in-memory cache.
    The cache key includes:
      - model_type
      - base model id or custom weight fingerprints
      - dtype
      - scheduler config (if modified)
      - LoRA set (name + scale)
    """
    global pipe_cache, lora_cache

    # Normalize and validate LoRAs first (and ensure they’re downloaded, if needed)
    resolved_loras = []
    for lora in lora_list or []:
        path = get_lora_path(lora)
        if path:
            resolved_loras.append({"path": path, "scale": lora.get("scale")})

    lora_key = get_lora_key(lora_list or [])
    cache_key = f"{model_type}_{lora_key}"
    if cache_key in pipe_cache:
        return pipe_cache[cache_key]

    # Disk cache key setup
    if model_type == "flux_kontext":
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev"
        disk_key = {
            "model_type": model_type,
            "base": base_repo,
            "dtype": "bfloat16",
            # If you tweak scheduler defaults, include those here as well
        }
        save_dir = _pipeline_dir(disk_key)

        # Try loading a locally saved pipeline first
        pipe = _load_local_pipeline_if_exists(save_dir)
        if pipe is None:
            logging.info("Building pipeline (Kontext) and saving locally for reuse...")
            pipe = FluxPipeline.from_pretrained(
                base_repo,
                torch_dtype=dtype,
                use_safetensors=True,
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True,
            ).to("cuda", dtype=dtype)

            # Optional: set VAE if you need to override
            # pipe.vae = vae

            _save_pipeline(pipe, save_dir)

        # Apply LoRAs (not baked in; we keep base clean)
        pipe = apply_loras(pipe, resolved_loras)
        pipe_cache[cache_key] = pipe
        return pipe

    elif model_type == "flux_dev":
        # Custom-weight build
        ckpt_path = "/workspace/models/getphatFLUXReality_v8.safetensors"
        txt1_path = "/workspace/models/t5xxl_fp16.safetensors"
        txt2_path = "/workspace/models/clip_l.safetensors"
        base_repo = "black-forest-labs/FLUX.1-dev"
        hftoken = os.getenv("HF_TOKEN")

        # Validate existence early

        if not os.path.exists(ckpt_path):
            logging.info(f"Copy from BB...")
            u.initialize_worker_environment()



        # Disk key includes fingerprints of custom weights
        disk_key = {
            "model_type": model_type,
            "base": base_repo,
            "dtype": "bfloat16",
            "transformer": _file_fingerprint(ckpt_path),
            "t5": _file_fingerprint(txt1_path),
            "clip": _file_fingerprint(txt2_path),
            # If you change scheduler params, include them
            "scheduler": {"type": "FlowMatchEulerDiscreteScheduler", "beta_sigmas": True, "dyn_shift": True},
        }
        save_dir = _pipeline_dir(disk_key)

        # Try loading a locally saved pipeline first
        pipe = _load_local_pipeline_if_exists(save_dir)
        if pipe is None:
            logging.info("Building pipeline (custom flux_dev) and saving locally for reuse...")

            # Load text encoders
            text_encoder = T5EncoderModel(t5_config)
            try:
                text_encoder.load_state_dict(load_file(txt1_path), strict=True)
            except RuntimeError as e:
                logging.error(f"Failed to load T5 state dict: {e}")
                raise
            text_encoder = text_encoder.to(dtype=dtype).to("cuda")
            enable_flash_attn_for_transformers(text_encoder)
            text_encoder_2 = CLIPTextModel(clip_config)
            try:
                text_encoder_2.load_state_dict(load_file(txt2_path), strict=True)
            except RuntimeError as e:
                logging.error(f"Failed to load CLIP state dict: {e}")
                raise
            text_encoder_2 = text_encoder_2.to(dtype=dtype).to("cuda")
            enable_flash_attn_for_transformers(text_encoder_2)
            transformer = FluxTransformer2DModel.from_single_file(
                ckpt_path,
                subfolder="transformer",
                torch_dtype=dtype,
                token=hftoken,
            ).to("cuda", dtype=dtype)
            enable_flash_attn_for_transformers(transformer)
            pipe = FluxPipeline.from_pretrained(
                base_repo,
                transformer=transformer,
                text_encoder=text_encoder_2,
                text_encoder_2=text_encoder,
                use_safetensors=True,
                torch_dtype=dtype,
                token=hftoken,
                low_cpu_mem_usage=True,
            ).to("cuda", dtype=dtype)

            # Optional: override VAE if desired
            # pipe.vae = vae

            # Set scheduler consistently
            from diffusers import FlowMatchEulerDiscreteScheduler

            pipe.scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                use_beta_sigmas=True,
                num_train_timesteps=1000,
            )

            _save_pipeline(pipe, save_dir)

        # Apply LoRAs to the loaded base
        pipe = apply_loras(pipe, resolved_loras)
        pipe_cache[cache_key] = pipe
        return pipe

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def enable_flash_attn_for_transformers(model):
    """
    Enable flash-attn v2 for a Transformers model if available.
    Falls back silently to default attention if not available/supported.
    """
    try:
        if is_flash_attn_2_available():
            # Newer Transformers support setting this at runtime
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("flash_attention_2")
            # Also set on config to persist across save/load if supported
            if hasattr(model, "config"):
                try:
                    model.config.attn_implementation = "flash_attention_2"
                except Exception:
                    pass
    except Exception:
        # If anything goes wrong, just keep default attention
        pass

# Example usage when you construct/load your text encoders and transformer:
# After text_encoder/text_encoder_2/transformer are created and moved to CUDA:
# enable_flash_attn_for_transformers(text_encoder)
# enable_flash_attn_for_transformers(text_encoder_2)
# For the diffusion transformer if it inherits HF attention modules:
# enable_flash_attn_for_transformers(transformer)

# Optional: ensure PyTorch uses its fastest SDPA kernels (good fallback on Hopper)
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass
