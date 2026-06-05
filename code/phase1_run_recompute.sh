#!/usr/bin/env bash
# Phase 1 — node-classification recompute on A100. Run from ANYWHERE (cd's into dhgbench itself).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"; DHGB="$REPO/dhgbench"; DATA="$REPO/data"

[ -f "$DHGB/main.py" ] || { echo "FATAL: $DHGB/main.py not found (put script in repo root)"; exit 1; }
[ -d "$DATA" ]        || { echo "FATAL: $DATA missing (unzip data.zip in repo root)"; exit 1; }

# --- bake-in: ensure label remap is present in data_base.py (idempotent, no git needed) ---
DB="$DHGB/lib_dataset/data_base.py"
if ! grep -q "DHG_REMAP_LABELS" "$DB"; then
  python3 - "$DB" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
old='    def _initialization_(self):\n        \n        self.num_classes= len(self.y.unique())'
new='''    def _initialization_(self):
        
        # Auto-injected by phase1_run_recompute.sh: dense-remap labels so
        # max(label) == num_classes-1. Fixes trivago's sparse-label crash
        # (CUDA assert t < n_classes). Active when DHG_REMAP_LABELS=1.
        import os as _os
        if _os.environ.get("DHG_REMAP_LABELS") == "1":
            _u, _inv = self.y.unique(return_inverse=True)
            self.y = _inv.view(self.y.shape).to(self.y.device)
            self.data.y = self.y
        self.num_classes= len(self.y.unique())'''
assert old in s, "anchor not found — manual patch needed"
open(p,"w").write(s.replace(old,new,1))
print("injected label-remap into data_base.py")
PY
else
  echo "label-remap already present in data_base.py — skipping injection"
fi

export DHG_DUMP_DIR="${DHG_DUMP_DIR:-$REPO/results_phase1}"
export DHG_REMAP_LABELS=1   # dense-remap labels (fixes trivago 'assert t<n_classes'); needs phase1_label_remap.patch applied
SEEDS="${SEEDS:-20}"
LOGDIR="$REPO/logs_phase1"; mkdir -p "$LOGDIR" "$DHG_DUMP_DIR"
: > "$DHG_DUMP_DIR/_status.txt"

# Method names MUST match parse_model() strings AND config_<name>.yaml (lowercased).
# Paper-name -> harness-name fixes: AllSetTransformer->AllSetformer, ED-HNN->EDHNN,
#   TF-HNN->TFHNN, T-HyperGNN->TMPHN, UniGNN->UniGCNII.
METHODS=(MLP HGNN HyperGCN HCHA LEGCN HyperND PhenomNN SheafHyperGNN \
         HJRL DPHGNN TFHNN HNHN UniGCNII AllSetformer EDHNN HyperGT EHNN TMPHN)
# CEGCN/CEGAT have NO yaml and need a couple defaults supplied on the CLI:
BASELINES_NOYAML=(CEGCN CEGAT)

DATASETS=(cora pubmed coauthor_cora coauthor_dblp walmart-trips-100 trivago \
          actor amazon twitch pokec yelp NTU2012 ModelNet40)

cd "$DHGB"

# CEGCN/CEGAT need correct configs — create them first (one-time):
#   bash phase1_fix_baseline_configs.sh
YDIR="$DHGB/lib_yamls/node_yamls"
for b in cegcn cegat; do
  [ -f "$YDIR/config_$b.yaml" ] || echo "WARN: config_$b.yaml missing — run phase1_fix_baseline_configs.sh first"
done

run() {  # $1=dataset $2=method
  local ds="$1" m="$2"
  local tag="${ds}__${m}"
  local log="$LOGDIR/${tag}.log"
  echo "[$(date +%H:%M:%S)] >>> $tag (seeds=$SEEDS)"
  timeout 3600 python main.py --dname="$ds" --task_type=node_cls --method="$m" \
      --is_default=True --num_seeds="$SEEDS" --device=cuda:0 > "$log" 2>&1
  local rc=$?
  if   [ $rc -eq 124 ]; then echo "    TIMEOUT $tag"; echo "TIMEOUT $tag" >> "$DHG_DUMP_DIR/_status.txt"
  elif [ $rc -ne 0 ];  then
    if grep -qiE "out of memory" "$log"; then echo "    OOM     $tag"; echo "OOM $tag" >> "$DHG_DUMP_DIR/_status.txt"
    else echo "    FAIL($rc) $tag"; echo "FAIL $tag rc=$rc" >> "$DHG_DUMP_DIR/_status.txt"; fi
  else echo "    ok      $tag"; fi
}

START=$(date +%s)
for ds in "${DATASETS[@]}"; do
  for m in "${METHODS[@]}";          do run "$ds" "$m"; done
  for m in "${BASELINES_NOYAML[@]}"; do run "$ds" "$m"; done
done
echo "Done in $(( ($(date +%s)-START)/60 )) min. JSON -> $DHG_DUMP_DIR"
echo "Failures/OOM:"; cat "$DHG_DUMP_DIR/_status.txt"
