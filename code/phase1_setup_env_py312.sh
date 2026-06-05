#!/usr/bin/env bash
# Phase 1 (Option B) — Python 3.12 + torch 2.4.0+cu121 (NO system Python 3.9 needed).
# Pins differ from paper B.3 (torch 2.2.2/py3.9) -> frame repro-check as 'different stack'.
# Run on YOUR machine from the DHG-Bench repo root. Based on arXiv:2508.12244v2 (adapted stack).
# REQUIREMENT: system python3 >= 3.12 (uses your existing interpreter).
#   check with:  python3.9 --version
set -euo pipefail

PYBIN="${PYBIN:-python3}"          # override: PYBIN=python3.12 bash phase1_setup_env_venv.sh
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYBIN' not found."
  echo "  Your system python: $(python3 --version 2>&1)"
  exit 1
fi
"$PYBIN" --version

# --- create & activate venv ---
"$PYBIN" -m venv .venv-dhgbench
# shellcheck disable=SC1091
source .venv-dhgbench/bin/activate
python -m pip install --upgrade pip wheel setuptools

# --- torch 2.4.0 + CUDA 12.1 runtime (works under driver 595 / CUDA 13.2) ---
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# --- PyG core + compiled companions (wheels must match torch-2.4.0+cu121) ---
pip install torch_geometric==2.6.1
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
            torch-spline-conv==1.2.2 \
            -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# --- remaining deps (versions adapted to py3.12 / torch 2.4) ---
pip install deeprobust==0.2.11 ipdb==0.13.13 numpy==1.26.4 dask==2024.8.0

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
