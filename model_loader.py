import os
import json
import hashlib
import logging
import shutil
import torch
from diffusers import FluxPipeline, AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel, T5Config, CLIPTextConfig
from safetensors.torch import load_file
import utils as u
from lib.PipelineLRU import PipelineLRU



from botocore.exceptions import BotoCoreError, ClientError
# add near other imports
from transformers.utils import is_flash_attn_2_available

logging.basicConfig(level=logging.INFO)

# Enhanced cache system (in-memory for the current process)
pipe_lru = PipelineLRU(capacity=int(os.getenv("PIPELINE_LRU_CAPACITY", "2")))
lora_cache = {}



# Where to store serialized pipelines
PIPELINE_STORE = os.getenv("PIPELINE_STORE", "/runpod-volume/pretrained")

# Precision configuration (use bfloat16 for the best throughput)
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
    dropout_rate=0.1,
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
    projection_dim=768,
)


vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev",
                                    token=os.getenv("HF_TOKEN"),
                                    subfolder="vae",
                                    safetensors=True).to("cuda", dtype=dtype)

def _read_saved_at(dir_path: str) -> int:
    """
    Return a sortable timestamp for a cached pipeline directory.
    Prefer build_manifest.json 'saved_at', else directory mtime.
    """
    manifest = os.path.join(dir_path, "build_manifest.json")
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
            ts = int(data.get("saved_at", 0))
            if ts > 0:
                return ts
    except Exception:
        pass
    try:
        return int(os.path.getmtime(dir_path))
    except Exception:
        return 0


def _prune_pipeline_store(keep_latest_per_prefix: int = 1):
    """
    Delete old cached pipelines under PIPELINE_STORE, keeping the most recent
    per model prefix: 'flux_kontext-' and 'flux_dev-'.

    Safe, best-effort cleanup with logging.
    """
    if not os.path.isdir(PIPELINE_STORE):
        return

    prefixes = ("flux_kontext-", "flux_dev-")
    try:
        entries = [e for e in os.listdir(PIPELINE_STORE)]
    except Exception as e:
        logging.warning(f"Cache prune skipped: cannot list {PIPELINE_STORE}: {e}")
        return

    for prefix in prefixes:
        dirs = []
        for name in entries:
            if not name.startswith(prefix):
                continue
            full = os.path.join(PIPELINE_STORE, name)
            if os.path.isdir(full):
                dirs.append((full, _read_saved_at(full)))

        if len(dirs) <= keep_latest_per_prefix:
            continue

        # Sort by timestamp desc, keep newest N
        dirs.sort(key=lambda x: x[1], reverse=True)
        to_keep = set(p for p, _ in dirs[:keep_latest_per_prefix])
        to_delete = [p for p, _ in dirs[keep_latest_per_prefix:]]

        for path in to_delete:
            try:
                logging.info(f"Pruning old cache dir: {path}")
                shutil.rmtree(path, ignore_errors=False)
            except Exception as e:
                logging.warning(f"Failed to remove cache dir {path}: {e}")


def _stable_sha256(path: str) -> str | None:
    """
    Stream the file and compute sha256. Persist a sidecar '<file>.sha256'
    so subsequent runs are O(1). Returns None if file is missing.
    """
    if not os.path.exists(path):
        return None
    sidecar = f"{path}.sha256"
    try:
        if os.path.isfile(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                saved = f.read().strip()
                if saved:
                    return saved
    except Exception:
        pass

    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        try:
            with open(sidecar, "w", encoding="utf-8") as f:
                f.write(digest)
        except Exception:
            pass  # best effort; fingerprint still returned
        return digest
    except FileNotFoundError:
        return None


def _file_fingerprint(path: str) -> dict:
    """
    Stable fingerprint across machines: filename + size + sha256.
    Avoid mtime since downloads on different hosts yield different mtimes.
    """
    base = os.path.basename(path)
    if not os.path.exists(path):
        return {"path": base, "missing": True}
    st = os.stat(path)
    sha = _stable_sha256(path)
    return {"path": base, "size": st.st_size, "sha256": sha}



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
    local_path = f"/runpod-volume/loras/{filename}"
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
    def norm_scale(v):
        if v is None:
            return None
        try:
            # Normalize numeric to a compact canonical form
            return f"{float(v):.6g}"
        except Exception:
            return str(v)
    parts = []
    for l in sorted(lora_list, key=lambda x: os.path.basename(x["path"])):
        name = os.path.basename(l["path"])
        scale = l.get("scale") or l.get("weight") or l.get("alpha")
        s = norm_scale(scale)
        parts.append(f"{name}:{s}" if s is not None else name)
    return "-".join(parts)



def validate_and_load_loras(lora_list):
    valid_loras = []
    for lora in lora_list or []:
        filename = os.path.basename(lora["path"])
        path = f"/runpod-volume/loras/{filename}"
        if os.path.exists(path):
            valid_loras.append({"path": path, "scale": lora.get("scale")})
        else:
            logging.warning(f"LoRA file {path} not found, skipping")
    return valid_loras



def apply_loras(pipe, lora_items):
    """
    Apply LoRAs to the pipeline with support for per-component scaling:
      - strength_model -> 'transformer'
      - strength_clip  -> 'text_encoder'
    Falls back to 'scale' (single scalar) if provided or when per-component control
    is not supported by the current pipeline/version.
    """
    if hasattr(pipe, "disable_lora"):
        pipe.disable_lora()

    active_adapter_names = []

    for item in lora_items or []:
        path = item["path"]
        # New fields:
        s_model = item.get("strength_model", None)
        s_clip = item.get("strength_clip", None)
        # Backward-compatible fallback:
        scale = item.get("scale", None) or item.get("weight") or item.get("alpha")

        # Give each LoRA a stable adapter name so they can co-exist
        adapter_name = os.path.splitext(os.path.basename(path))[0]
        active_adapter_names.append(adapter_name)

        # Attempt the most granular route first: per-component weights dict
        # For FluxPipeline the component names are typically:
        #   'transformer' (main model), 'text_encoder' (CLIP), and optionally 'text_encoder_2' (T5)
        per_component_weight = None
        if s_model is not None or s_clip is not None:
            per_component_weight = {}
            if s_model is not None:
                per_component_weight["transformer"] = float(s_model)
            if s_clip is not None:
                per_component_weight["text_encoder"] = float(s_clip)
            # If you want to also scale T5 with clip strength, uncomment the next line:
            # per_component_weight["text_encoder_2"] = float(s_clip)

        try:
            # Prefer loading by adapter so multiple LoRAs can be used together
            if per_component_weight is not None:
                # Some newer diffusers versions accept a dict in 'weight'
                pipe.load_lora_weights(
                    path,
                    adapter_name=adapter_name,
                    weight=per_component_weight,
                )
            elif scale is not None:
                # Fallback: single scalar for all affected modules
                pipe.load_lora_weights(
                    path,
                    adapter_name=adapter_name,
                    weight=float(scale),
                )
            else:
                pipe.load_lora_weights(path, adapter_name=adapter_name)
            lora_cache[path] = True
        except TypeError:
            # Older API: no adapter_name or weight dict support
            if per_component_weight is not None:
                # Last-resort: collapse to a single scalar if provided; else default to 1.0
                merged = float(s_model if s_model is not None else (s_clip if s_clip is not None else 1.0))
                pipe.load_lora_weights(path, weight=merged)
            elif scale is not None:
                pipe.load_lora_weights(path, weight=float(scale))
            else:
                pipe.load_lora_weights(path)

    # Activate all loaded adapters together when supported
    try:
        if active_adapter_names:
            # If you want to keep per-adapter global weights, you could pass adapter_weights here.
            # We rely on the weights applied at load-time above.
            pipe.set_adapters(active_adapter_names)
    except Exception:
        # If the pipeline/version doesn’t support set_adapters, it’s fine: the LoRAs are already active.
        pass

    return pipe


def _load_local_pipeline_if_exists(save_dir: str):
    if os.path.isdir(save_dir) and os.path.isfile(os.path.join(save_dir, "model_index.json")):
        logging.info(f"Loading pipeline from local cache: {save_dir}")
        pipe = FluxPipeline.from_pretrained(
            save_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe.to("cuda", dtype=dtype)
    logging.info(f"No local pipeline found at {save_dir}")
    return None



def _save_pipeline(pipe, save_dir: str):
    _ensure_dir(save_dir)
    manifest_path = os.path.join(save_dir, "build_manifest.json")
    def _do_save():
        pipe.save_pretrained(save_dir, safe_serialization=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": int(__import__("time").time()),
                    "dtype": str(dtype),
                    "torch_version": torch.__version__,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        logging.info(f"Saved pipeline to {save_dir}")

    try:
        _do_save()
    except Exception as e:
        logging.error(f"Failed to save pipeline to {save_dir}: {e}. Triggering cache prune and retry...")
        # Burnout: prune old caches, then retry once
        _prune_pipeline_store(keep_latest_per_prefix=1)
        try:
            _do_save()
        except Exception as e2:
            logging.error(f"Retry save failed for {save_dir}: {e2}")
            # Best effort: surface the error or continue depending on your policy
            # raise # Uncomment to make this fatal



def get_pipeline(model_type, lora_list):
    """
    Build-or-load pipeline with disk-level caching and LRU-based in-memory cache.
    The cache key includes:
      - model_type
      - base model id or custom weight fingerprints
      - dtype
      - scheduler config (if modified)
      - LoRA set (name and scale)
    """
    global pipe_lru, lora_cache

    # Normalize and validate LoRAs first (and ensure they’re downloaded, if needed)
    resolved_loras = []
    for lora in lora_list or []:
        path = get_lora_path(lora)
        if path:
            # carry through any per-component strengths if present
            resolved = {
                "path": path,
                "scale": lora.get("scale"),
                "strength_model": lora.get("strength_model"),
                "strength_clip": lora.get("strength_clip"),
            }
            resolved_loras.append(resolved)

    lora_key = get_lora_key(lora_list or [])
    cache_key = f"{model_type}_{lora_key}"

    # Fast path: in-memory LRU
    cached = pipe_lru.get(cache_key)
    if cached is not None:
        return cached

    # Disk cache key setup
    if model_type == "flux_kontext":
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev"
        disk_key = {
            "model_type": model_type,
            "base": base_repo,
            "dtype": "bfloat16",
        }
        save_dir = _pipeline_dir(disk_key)

        # Use a churn guard around heavy transitions: load/build/apply LoRAs
        with u.churn_guard():
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
                _save_pipeline(pipe, save_dir)

            # Apply LoRAs (not baked in; we keep base clean)
            pipe = apply_loras(pipe, resolved_loras)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

        # Put into LRU (eviction will unload old pipelines)
        pipe_lru.put(cache_key, pipe)
        return pipe

    elif model_type == "flux_dev":
        # Custom-weight build
        ckpt_path = "/runpod-volume/models/Flux_Model_001.safetensors"
        txt1_path = "/runpod-volume/models/t5xxl_fp16.safetensors"
        txt2_path = "/runpod-volume/models/clip_l.safetensors"
        hftoken = os.getenv("HF_TOKEN")

        if not os.path.exists(ckpt_path):
            logging.info("Copy from BB...")
            u.initialize_worker_environment()

        disk_key = {
            "model_type": model_type,
            "base": ckpt_path,
            "dtype": "bfloat16",
            "transformer": _file_fingerprint(ckpt_path),
            "t5": _file_fingerprint(txt1_path),
            "clip": _file_fingerprint(txt2_path),
            "scheduler": {"type": "FlowMatchEulerDiscreteScheduler", "beta_sigmas": True, "dyn_shift": True},
        }
        save_dir = _pipeline_dir(disk_key)

        with u.churn_guard():
            pipe = _load_local_pipeline_if_exists(save_dir)
            if pipe is None:
                logging.info("Building pipeline (custom flux_dev) and saving locally for reuse...")

                # Load text encoders
                text_encoder = T5EncoderModel(t5_config)
                text_encoder.load_state_dict(load_file(txt1_path), strict=True)
                text_encoder = text_encoder.to(dtype=dtype).to("cuda")
                enable_flash_attn_for_transformers(text_encoder)

                text_encoder_2 = CLIPTextModel(clip_config)
                text_encoder_2.load_state_dict(load_file(txt2_path), strict=True)
                text_encoder_2 = text_encoder_2.to(dtype=dtype).to("cuda")
                enable_flash_attn_for_transformers(text_encoder_2)

                pipe = FluxPipeline.from_single_file(
                    ckpt_path,
                    vae=vae,
                    text_encoder=text_encoder_2,
                    text_encoder_2=text_encoder,
                    use_safetensors=True,
                    torch_dtype=dtype,
                    token=hftoken,
                    low_cpu_mem_usage=True,
                ).to("cuda", dtype=dtype)

                from diffusers import FlowMatchEulerDiscreteScheduler
                pipe.scheduler = FlowMatchEulerDiscreteScheduler(
                    use_dynamic_shifting=True,
                    use_beta_sigmas=True,
                    num_train_timesteps=1000,
                )

                _save_pipeline(pipe, save_dir)

            # Apply LoRAs to the loaded base
            pipe = apply_loras(pipe, resolved_loras)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass

        pipe_lru.put(cache_key, pipe)
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


try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass
