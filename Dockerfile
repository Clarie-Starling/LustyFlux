# ---- builder: only for compiling flash-attn ----
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1

# Build-time knobs
ARG ENABLE_FLASH_ATTN=0
# Set to the architectures you actually deploy on to avoid compiling everything
# Examples: "8.0" (A100), "8.9" (Ada), "9.0" (H100). Separate multiple with ';'
ARG TORCH_CUDA_ARCH_LIST="8.9;9.0"
# Pin to a known-good version if you have one; otherwise leave empty to pick latest
ARG FLASH_ATTN_VERSION=""
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}


# Tooling required just for build
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Ensure torch matches runtime (provided by base images already)
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc ccache g++  && \
    rm -rf /var/lib/apt/lists/*
ENV CC="ccache gcc" CXX="ccache g++" CUDAHOSTCXX="ccache g++"

# Build the wheel
RUN if [ "$ENABLE_FLASH_ATTN" = "1" ]; then \
      set -eux; \
      PKG="flash-attn"; \
      VER_SPEC=""; \
      if [ -n "$FLASH_ATTN_VERSION" ]; then VER_SPEC=="==$FLASH_ATTN_VERSION"; fi; \
      mkdir -p /build/wheels; \
      (pip download --no-deps --only-binary=:all: -d /build/wheels "${PKG}${VER_SPEC}" \
        || pip wheel --no-build-isolation --no-deps -w /build/wheels "${PKG}${VER_SPEC}"); \
    fi


# ---- runtime: small final image ----
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
WORKDIR /workspace

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Torch comes from base; install only the rest
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir diffusers transformers safetensors && \
    rm -rf /root/.cache

## Install the prebuilt flash-attn wheel
#COPY --from=builder /build/wheels /tmp/wheels
#RUN pip install --no-cache-dir /tmp/wheels/flash_attn-*.whl && rm -rf /tmp/wheels

COPY . .

RUN useradd -m -u 1000 user && \
    chown -R user:user /workspace
USER user

ENV HF_HOME=/workspace/models/

CMD ["python", "/workspace/handler.py"]
