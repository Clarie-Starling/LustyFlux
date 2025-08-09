# python
import sys
import types
import importlib
import json
import os
import time
import tempfile
from pathlib import Path


def make_dummy_modules(monkeypatch):
    # ----- Dummy torch -----
    class DummyBackends:
        class cuda:
            class matmul:
                allow_tf32 = True
            class cudnn:
                allow_tf32 = True

            @staticmethod
            def enable_flash_sdp(flag):  # noqa: ARG002
                return True

            @staticmethod
            def enable_mem_efficient_sdp(flag):  # noqa: ARG002
                return True

            @staticmethod
            def enable_math_sdp(flag):  # noqa: ARG002
                return True

        class cudnn:
            allow_tf32 = True

    torch = types.SimpleNamespace(
        backends=DummyBackends(),
        bfloat16="bfloat16",
        __version__="0.0.0-test",
    )
    sys.modules["torch"] = torch

    # ----- Dummy diffusers -----
    class DummyAutoencoderKL:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def to(self, *args, **kwargs):  # noqa: ARG002
            return self

    class DummyFluxPipeline:
        def __init__(self, **kwargs):  # noqa: ARG002
            self.scheduler = None
            self.vae = None
            self._saved = {}
            self._loras = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # noqa: ARG002
            return cls()

        def to(self, *args, **kwargs):  # noqa: ARG002
            return self

        def save_pretrained(self, save_dir, safe_serialization=True):  # noqa: ARG002
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            # create a dummy model_index.json file to simulate proper cache
            with open(os.path.join(save_dir, "model_index.json"), "w", encoding="utf-8") as f:
                json.dump({"ok": True}, f)

        # LORA handling expected by apply_loras
        def disable_lora(self):
            self._loras.clear()

        def load_lora_weights(self, path, weight_name=None, weight=None):  # noqa: ARG002
            self._loras.append((path, weight))

    class DummyFluxTransformer2DModel:
        @classmethod
        def from_single_file(cls, *args, **kwargs):  # noqa: ARG002
            obj = cls()
            obj._is_on_cuda = False
            return obj

        def to(self, *args, **kwargs):  # noqa: ARG002
            self._is_on_cuda = True
            return self

    class DummyFlowMatchEulerDiscreteScheduler:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    diffusers = types.ModuleType("diffusers")
    diffusers.FluxPipeline = DummyFluxPipeline
    diffusers.FluxTransformer2DModel = DummyFluxTransformer2DModel
    diffusers.AutoencoderKL = DummyAutoencoderKL
    diffusers.FlowMatchEulerDiscreteScheduler = DummyFlowMatchEulerDiscreteScheduler
    sys.modules["diffusers"] = diffusers

    # ----- Dummy transformers -----
    class DummyCfg:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DummyModel:
        def __init__(self, config=None):  # noqa: ARG002
            self.config = types.SimpleNamespace()

        def to(self, *args, **kwargs):  # noqa: ARG002
            return self

        # set_attn_implementation may be called
        def set_attn_implementation(self, impl):  # noqa: ARG002
            return None

        # load_state_dict may be called in flux_dev path (we won't hit it in tests)
        def load_state_dict(self, *args, **kwargs):  # noqa: ARG002
            return None

    transformers = types.ModuleType("transformers")
    transformers.T5EncoderModel = DummyModel
    transformers.CLIPTextModel = DummyModel
    transformers.T5Config = DummyCfg
    transformers.CLIPTextConfig = DummyCfg

    transformers_utils = types.ModuleType("transformers.utils")

    def is_flash_attn_2_available():
        return False

    transformers_utils.is_flash_attn_2_available = is_flash_attn_2_available
    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = transformers_utils

    # ----- Dummy safetensors.torch -----
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")

    def load_file(path):  # noqa: ARG001
        return {}

    safetensors_torch.load_file = load_file
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch

    # ----- Dummy boto3 and botocore.exceptions -----
    class DummyS3Client:
        def download_fileobj(self, bucket, key, f):  # noqa: ARG002
            f.write(b"dummy")

    def boto3_client(*args, **kwargs):  # noqa: ARG002
        return DummyS3Client()

    boto3 = types.ModuleType("boto3")
    boto3.client = boto3_client
    sys.modules["boto3"] = boto3

    botocore_ex = types.ModuleType("botocore.exceptions")

    class BotoCoreError(Exception):
        pass

    class ClientError(Exception):
        pass

    botocore_ex.BotoCoreError = BotoCoreError
    botocore_ex.ClientError = ClientError
    sys.modules["botocore.exceptions"] = botocore_ex

    # ----- Dummy utils.initialize_worker_environment -----
    utils = types.ModuleType("utils")
    def initialize_worker_environment():  # noqa: D401
        return None
    utils.initialize_worker_environment = initialize_worker_environment
    sys.modules["utils"] = utils


def import_model_loader(monkeypatch):
    # Ensure envs are set but harmless
    monkeypatch.setenv("PIPELINE_STORE", os.path.join(tempfile.gettempdir(), "pipestore"))
    monkeypatch.setenv("B2_BUCKET", "dummy-bucket")
    monkeypatch.setenv("B2_KEY_ID", "dummy-key")
    monkeypatch.setenv("B2_SECRET_KEY", "dummy-secret")
    monkeypatch.setenv("B2_REGION", "us-east-005")
    # Build dummy modules and import the module under test fresh
    make_dummy_modules(monkeypatch)
    if "model_loader" in sys.modules:
        del sys.modules["model_loader"]
    import model_loader  # noqa: WPS433
    importlib.reload(model_loader)
    return model_loader


def test_stable_hash_deterministic(monkeypatch):
    ml = import_model_loader(monkeypatch)
    a = {"x": 1, "y": [3, 2], "z": {"a": True}}
    b = {"z": {"a": True}, "y": [3, 2], "x": 1}
    h1 = ml._stable_hash(a)
    h2 = ml._stable_hash(b)
    assert isinstance(h1, str) and len(h1) == 12
    assert h1 == h2


def test_file_fingerprint_present_and_missing(monkeypatch, tmp_path):
    ml = import_model_loader(monkeypatch)
    f = tmp_path / "weights.bin"
    f.write_bytes(b"abc")
    info = ml._file_fingerprint(str(f))
    assert info["path"] == "weights.bin"
    assert info["size"] == 3
    assert isinstance(info["mtime"], int)

    missing = ml._file_fingerprint(str(tmp_path / "nope.bin"))
    assert missing["path"] == "nope.bin"
    assert missing.get("missing") is True


def test_pipeline_dir_uses_store_and_model_type(monkeypatch):
    ml = import_model_loader(monkeypatch)
    key = {"model_type": "flux_kontext", "base": "abc", "dtype": "bfloat16"}
    p = ml._pipeline_dir(key)
    assert p.startswith(os.getenv("PIPELINE_STORE"))
    assert "flux_kontext-" in p


def test_get_lora_key_sorting_and_scale(monkeypatch):
    ml = import_model_loader(monkeypatch)
    loras_a = [
        {"path": "/some/loras/B.safetensors", "scale": 0.5},
        {"path": "/some/loras/A.safetensors", "scale": 0.8},
    ]
    loras_b = [
        {"path": "/some/loras/A.safetensors", "scale": 0.8},
        {"path": "/some/loras/B.safetensors", "scale": 0.5},
    ]
    k1 = ml.get_lora_key(loras_a)
    k2 = ml.get_lora_key(loras_b)
    assert k1 == k2
    assert "A.safetensors:0.8" in k1
    assert "B.safetensors:0.5" in k1


def test_validate_and_load_loras(monkeypatch, tmp_path):
    ml = import_model_loader(monkeypatch)

    # Simulate /workspace/models/loras directory logic without touching that path
    existing = {"present.safetensors"}

    def fake_exists(path):
        # Only claim existence for our pretend workspace lora paths
        if path.startswith("/workspace/models/loras/"):
            return os.path.basename(path) in existing
        return os.path.exists(path)

    monkeypatch.setattr(ml.os.path, "exists", fake_exists, raising=True)

    inp = [
        {"path": "/wherever/present.safetensors", "scale": 0.7},
        {"path": "/wherever/missing.safetensors", "scale": 0.3},
    ]
    out = ml.validate_and_load_loras(inp)
    assert len(out) == 1
    assert out[0]["path"].endswith("/workspace/models/loras/present.safetensors")
    assert out[0]["scale"] == 0.7


def test_apply_loras_calls_disable_and_load(monkeypatch):
    ml = import_model_loader(monkeypatch)

    class DummyPipe:
        def __init__(self):
            self.calls = []
            self._loras_disabled = False

        def disable_lora(self):
            self._loras_disabled = True
            self.calls.append(("disable_lora",))

        def load_lora_weights(self, path, weight_name=None, weight=None):  # noqa: ARG002
            self.calls.append(("load_lora_weights", path, weight))

    # reset lora_cache for test isolation
    ml.lora_cache.clear()

    pipe = DummyPipe()
    loras = [
        {"path": "/workspace/models/loras/A.safetensors", "scale": 0.8},
        {"path": "/workspace/models/loras/B.safetensors"},
    ]
    res = ml.apply_loras(pipe, loras)
    assert res is pipe
    assert pipe._loras_disabled is True
    # Expect two loads, with scale for the first and None for the second
    assert ("load_lora_weights", "/workspace/models/loras/A.safetensors", 0.8) in pipe.calls
    assert ("load_lora_weights", "/workspace/models/loras/B.safetensors", None) in pipe.calls


def test_get_pipeline_kontext_uses_cache_and_local_load(monkeypatch, tmp_path):
    ml = import_model_loader(monkeypatch)

    # Make a fake cached pipeline directory
    disk_key = {"model_type": "flux_kontext", "base": "black-forest-labs/FLUX.1-Kontext-dev", "dtype": "bfloat16"}
    save_dir = ml._pipeline_dir(disk_key)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"ok": True}, f)

    # Spy on FluxPipeline.from_pretrained to ensure it isn't called when cache exists
    calls = {"from_pretrained": 0}

    class SpyFluxPipeline(ml.FluxPipeline.__class__):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # noqa: ARG002
            calls["from_pretrained"] += 1
            return ml.diffusers.FluxPipeline.from_pretrained(*args, **kwargs)

    # Patch only inside model_loader namespace
    monkeypatch.setattr(ml, "FluxPipeline", ml.diffusers.FluxPipeline, raising=True)

    # Ensure apply_loras is called but returns pipe as-is
    monkeypatch.setattr(ml, "apply_loras", lambda pipe, l: pipe, raising=True)

    # First call: should load from local cache and store in memory cache
    p1 = ml.get_pipeline("flux_kontext", lora_list=[])
    # Second call: should return in-memory cached object
    p2 = ml.get_pipeline("flux_kontext", lora_list=[])
    assert p1 is p2
    # from_pretrained should not be called because local cache was used
    assert calls["from_pretrained"] == 0