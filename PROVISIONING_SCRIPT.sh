#!/bin/bash
set -euo pipefail
cd /workspace

# If the image has conda already:
if [ -f /root/miniconda/etc/profile.d/conda.sh ]; then
  source /root/miniconda/etc/profile.d/conda.sh
  conda activate jax_env || (conda create -y -n jax_env python=3.10 pip && conda activate jax_env)
else
  # Vast base images also ship a venv at /venv/main; use it if conda isn't present
  . /venv/main/bin/activate
fi

# Headless-friendly settings
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Only the deps your code imports
pip install -U numpy pyyaml matplotlib tensorboard glfw pynput typing_extensions \
  flax optax mujoco torch --index-url https://download.pytorch.org/whl/cpu

# Try GPU JAX (CUDA 12 wheels), fall back to CPU if no GPU wheel
pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
 || pip install -U "jax[cpu]"

python - <<'PY'
import jax, mujoco
print("JAX devices:", jax.devices())
print("MuJoCo:", mujoco.__version__)
PY
