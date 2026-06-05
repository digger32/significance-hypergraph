#!/usr/bin/env bash
# One-time: create correct configs for CEGCN/CEGAT (they ship without yaml).
# CEGNN needs: All_num_layers, MLP_hidden, dropout, heads, concat, use_bn.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YDIR="$SCRIPT_DIR/dhgbench/lib_yamls/node_yamls"
[ -d "$YDIR" ] || { echo "FATAL: $YDIR not found (run from repo root)"; exit 1; }

for b in cegcn cegat; do
cat > "$YDIR/config_$b.yaml" <<YAML
default:
  All_num_layers: 2
  embedding_hidden: 128
  MLP_hidden: 128
  epochs: 100
  dropout: 0.5
  lr: 1.0e-3
  wd: 0.0
  heads: 4
  concat: false
  use_bn: true
  normalization: "ln"
  InputNorm: false
YAML
echo "wrote $YDIR/config_$b.yaml"
done
echo "Done. CEGCN/CEGAT now have all fields parse_model/CEGNN require (baseline HPs)."
