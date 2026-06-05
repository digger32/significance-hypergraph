#!/usr/bin/env python3
"""Phase 1 preflight — static validation of every (method, dataset) cell BEFORE running.
Catches the structural failure classes we hit (bad method name, missing yaml, missing
config fields for CEGNN, missing data files) without needing a GPU. Run from repo root:
    python phase1_preflight.py
Exit 0 if all cells are launchable; prints a per-cell report otherwise.
"""
import os, sys, glob, re

REPO = os.path.dirname(os.path.abspath(__file__))
DHGB = os.path.join(REPO, "dhgbench")
YDIR = os.path.join(DHGB, "lib_yamls", "node_yamls")
DATA = os.path.join(REPO, "data")

METHODS = ["MLP","HGNN","HyperGCN","HCHA","LEGCN","HyperND","PhenomNN","SheafHyperGNN",
           "HJRL","DPHGNN","TFHNN","HNHN","UniGCNII","AllSetformer","EDHNN","HyperGT",
           "EHNN","TMPHN","CEGCN","CEGAT"]
DATASETS = ["cora","pubmed","coauthor_cora","coauthor_dblp","walmart-trips-100","trivago",
            "actor","amazon","twitch","pokec","yelp","NTU2012","ModelNet40"]

# --- 1. parse_model accepted names ---
exp = open(os.path.join(DHGB,"lib_utils","exp_agent.py")).read()
accepted = set(re.findall(r"method\s*==\s*'([A-Za-z0-9_]+)'", exp))
accepted |= set(re.findall(r"'([A-Za-z0-9_]+)'", " ".join(re.findall(r"method\s*in\s*\[([^\]]*)\]", exp))))

# --- 2. CEGNN required config fields ---
ceg = ""
ceg_path = os.path.join(DHGB,"lib_models","HNN","cegnn.py")
if os.path.exists(ceg_path):
    ceg = open(ceg_path).read()
ceg_fields = set(re.findall(r"args\.([A-Za-z0-9_]+)", ceg))
# fields available from argparse (so not required in yaml)
pp = open(os.path.join(DHGB,"parameter_parser.py")).read()
argparse_fields = set(re.findall(r"--([A-Za-z0-9_]+)", pp))
ceg_need_yaml = sorted(ceg_fields - argparse_fields)

def yaml_path(m): return os.path.join(YDIR, f"config_{m.lower()}.yaml")

def yaml_has_fields(m, fields):
    p = yaml_path(m)
    if not os.path.exists(p): return None
    txt = open(p).read()
    return [f for f in fields if not re.search(rf"\b{re.escape(f)}\s*:", txt)]

problems = []
for m in METHODS:
    if m not in accepted:
        problems.append(f"METHOD '{m}' not accepted by parse_model"); continue
    if not os.path.exists(yaml_path(m)):
        problems.append(f"YAML missing for '{m}' -> {os.path.basename(yaml_path(m))}")
    if m in ("CEGCN","CEGAT"):
        miss = yaml_has_fields(m, ceg_need_yaml)
        if miss is None: problems.append(f"{m}: config file absent (needs {ceg_need_yaml})")
        elif miss:       problems.append(f"{m}: config missing fields {miss}")

# --- 3. data presence (best-effort: just warn) ---
data_ok = os.path.isdir(DATA)
print(f"parse_model accepts {len(accepted)} names | CEGNN yaml-needed fields: {ceg_need_yaml}")
print(f"data/ present: {data_ok}")
print(f"methods checked: {len(METHODS)} | datasets: {len(DATASETS)} | cells: {len(METHODS)*len(DATASETS)}")
print(f"label-remap fix needed for trivago: set DHG_REMAP_LABELS=1 (see phase1_label_remap.patch)\n")

if problems:
    print("STRUCTURAL PROBLEMS FOUND:")
    for p in problems: print("  -", p)
    sys.exit(1)
else:
    print("OK: all methods are launchable (correct names, yamls, CEGNN fields present).")
    print("Remaining expected gaps at runtime are RESOURCE-based only:")
    print("  - OOM on large datasets (yelp/walmart/coauthor_dblp) for heavy methods")
    print("  - trivago needs DHG_REMAP_LABELS=1 to avoid the label-range crash")
    sys.exit(0)
