#!/bin/bash
set -euo pipefail
echo "=== [Provision] start ==="

# ---------- NVIDIA / system sanity ----------
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
else
  echo "[WARN] nvidia-smi not found (container still may have GPU via runtime)"
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  wget git ca-certificates build-essential \
  libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 libxcursor1 \
  libosmesa6 libegl1 patchelf \
  && rm -rf /var/lib/apt/lists/*

# Headless GPU rendering for MuJoCo
export MUJOCO_GL=egl

# Avoid JAX preallocating full VRAM
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------- Miniconda ----------
cd /root
if [ ! -d /root/miniconda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda
fi
export PATH="/root/miniconda/bin:$PATH"
source /root/miniconda/etc/profile.d/conda.sh
conda --version

# ---------- Create lean env ----------
ENV_NAME="jax_env"
if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "${ENV_NAME}" python=3.10 pip
fi
conda activate "${ENV_NAME}"

# Make your workspace importable for "from Models import ..." etc.
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# ---------- Minimal deps (match your imports) ----------
pip install -U \
  numpy pyyaml matplotlib tensorboard glfw pynput typing_extensions \
  flax optax mujoco

# CPU-only torch just for SummaryWriter
pip install -U torch --index-url https://download.pytorch.org/whl/cpu

# ---------- JAX: try GPU (CUDA 12) first, fallback to CPU ----------
set +e
echo "Attempting GPU JAX install (cuda12 wheels)..."
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
JAX_DEVICES=$(python - <<'PY'
import jax; print([d.platform for d in jax.devices()])
PY
)
if echo "$JAX_DEVICES" | grep -qi "gpu"; then
  echo "GPU JAX OK: $JAX_DEVICES"
else
  echo "GPU JAX not detected ($JAX_DEVICES). Installing CPU JAX..."
  pip uninstall -y jax jaxlib >/dev/null 2>&1 || true
  pip install -U "jax[cpu]"
fi
set -e

# ---------- Optional merge from your env spec if mounted ----------
if [ -f /workspace/environment.yml ]; then
  echo "Found /workspace/environment.yml — attempting a non-fatal merge update."
  conda env update -n "${ENV_NAME}" -f /workspace/environment.yml --prune || \
    echo "[WARN] env update from environment.yml failed; continuing with minimal env."
fi

# ---------- Smoke tests ----------
python - <<'PY'
import importlib
mods = ["jax","jax.numpy","flax","flax.linen","optax","mujoco","yaml","matplotlib","numpy","tensorboard","pynput","glfw"]
fail=False
for m in mods:
    try: importlib.import_module(m)
    except Exception as e:
        print(f"[FAIL] import {m}: {e}"); fail=True
import jax, mujoco
print("JAX devices:", jax.devices())
print("MuJoCo version:", mujoco.__version__)
raise SystemExit(1) if fail else None
PY

echo "=== [Provision] done ==="
