#!/usr/bin/env bash
# Phase 1 — DHG-Bench environment for A100 (CUDA 12.1), venv version.
# Run on YOUR machine from the DHG-Bench repo root. Mirrors arXiv:2508.12244v2 Appendix B.3.
# REQUIREMENT: a system Python 3.9.x must exist (the paper used 3.9.21).
#   check with:  python3.9 --version
set -euo pipefail

PYBIN="${PYBIN:-python3.9}"          # override: PYBIN=python3.12 bash phase1_setup_env_venv.sh
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYBIN' not found."
  echo "  Your system python: $(python3 --version 2>&1)"
  echo "  Option A (reproduce B.3 pins): install Python 3.9, then rerun as-is."
  echo "  Option B (use what you have):  PYBIN=python3 bash phase1_setup_env_venv.sh"
  echo "    ...but then ALSO bump the torch/PyG pins below (see header notes)."
  exit 1
fi
"$PYBIN" --version

# --- create & activate venv ---
"$PYBIN" -m venv .venv-dhgbench
# shellcheck disable=SC1091
source .venv-dhgbench/bin/activate
python -m pip install --upgrade pip wheel setuptools

# --- torch 2.2.2 + CUDA 12.1 ---
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# --- PyG core + compiled companions (wheels must match torch-2.2.2+cu121) ---
pip install torch_geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
            torch-spline-conv==1.2.2 \
            -f https://data.pyg.org/whl/torch-2.2.2+cu121.html

# --- remaining pins from B.3 ---
pip install deeprobust==0.2.11 ipdb==0.13.13 numpy==1.24.3 dask==2024.8.0

# --- analysis stack (per-seed significance layer) ---
pip install scipy scikit-posthocs statsmodels pandas matplotlib huggingface_hub

echo "=== sanity ==="
python - <<'PY'
import torch, torch_geometric, torch_scatter, torch_sparse
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "dev", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("PyG", torch_geometric.__version__)
PY
echo "Env ready. In every NEW shell, reactivate with:  source .venv-dhgbench/bin/activate"
