#!/usr/bin/env python3
"""Layer-1 source check: verify every `paper` value in docs/repro_check.csv against
DHG-Bench Tables 1 and A5 (arXiv:2508.12244v2, 29 Sep 2025), transcribed below
directly from the paper text. Also verifies the OOM pattern and the trivago
reported range quoted in the manuscript. Non-zero exit on any mismatch.

Run: python check_layer1_sources.py docs/repro_check.csv
"""
import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "docs/repro_check.csv"

# ---- DHG-Bench Table 1 (mean accuracy, %; OOM = None) ----------------------
T1_COLS = ["Cora", "Pubmed", "Cora-CA", "DBLP-CA", "Walmart", "Trivago",
           "Actor", "Gamers", "Pokec", "Yelp"]
T1 = {
 "MLP":               [75.33, 86.62, 75.57, 85.54, 63.21, 36.76, 86.06, 52.57, 59.64, 31.84],
 "CEGCN":             [76.90, 86.03, 78.40, 89.75, 70.40, 47.24, 67.41, 51.02, 57.37, None],
 "CEGAT":             [77.22, 86.09, 78.02, 89.61, 65.83, None,  73.87, 51.05, 57.34, None],
 "HGNN":              [77.90, 86.17, 82.84, 91.00, 77.12, 57.67, 77.83, 52.38, 57.87, 33.71],
 "HyperGCN":          [78.38, 87.42, 81.65, 89.51, 68.75, 42.39, 81.82, 51.32, 57.51, 29.29],
 "HCHA":              [77.84, 86.33, 83.01, 91.18, 77.66, 52.50, 78.30, 52.35, 58.19, 33.13],
 "LEGCN":             [74.36, 87.52, 74.59, 85.16, 62.98, 33.45, 85.34, 51.31, 59.66, None],
 "HyperND":           [79.23, 86.73, 83.19, 91.34, 75.10, 87.19, 83.19, 52.39, 57.65, None],
 "PhenomNN":          [78.97, 87.81, 84.05, 91.83, None,  None,  83.14, 51.80, 58.43, None],
 "SheafHyperGNN":     [79.03, 87.10, 84.08, 91.09, None,  None,  85.00, 52.07, 59.06, None],
 "HJRL":              [78.67, 87.98, 83.72, None,  None,  None,  71.54, 51.62, 57.57, None],
 "DPHGNN":            [76.40, 86.72, 82.13, None,  None,  None,  83.65, 52.36, 58.20, None],
 "TF-HNN":            [79.47, 87.90, 84.19, 91.38, 77.04, 90.79, 85.96, 52.34, 59.17, 35.16],
 "HNHN":              [75.24, 85.66, 76.51, 85.84, 65.21, 53.75, 81.20, 51.12, 58.55, 25.86],
 "UniGNN":            [79.41, 87.57, 83.49, 91.71, 76.26, 36.15, 84.61, 52.50, 58.56, 31.09],
 "AllSetTransformer": [78.02, 87.79, 82.95, 91.51, 78.61, 59.92, 85.66, 51.74, 58.55, 33.18],
 "ED-HNN":            [78.58, 87.65, 82.98, 91.55, 77.90, 75.99, 85.77, 50.54, 58.68, 34.84],
 "HyperGT":           [75.57, 86.06, 75.42, 84.53, None,  None,  84.43, 51.19, 57.73, None],
 "EHNN":              [76.51, 87.12, 81.68, 90.47, 77.95, None,  86.21, 52.14, 58.23, 34.09],
 "T-HyperGNN":        [74.20, 86.28, 75.01, 85.44, 73.48, None,  85.32, 51.82, 58.82, None],
}

# ---- DHG-Bench Table A5 -----------------------------------------------------
A5_COLS = ["NTU2012", "ModelNet40", "Ratings"]
A5 = {
 "MLP":               [88.59, 96.88, 28.47],
 "CEGCN":             [84.93, 92.34, 26.65],
 "CEGAT":             [84.14, 92.02, 28.23],
 "HGNN":              [90.13, 97.43, 28.05],
 "HyperGCN":          [75.78, 91.15, 27.34],
 "HCHA":              [90.53, 97.68, 28.33],
 "LEGCN":             [89.82, 96.82, 28.21],
 "HyperND":           [88.98, 97.18, 28.32],
 "PhenomNN":          [88.78, 98.28, 28.49],
 "SheafHyperGNN":     [90.81, 98.30, 28.35],
 "HJRL":              [88.15, 96.33, 26.90],
 "DPHGNN":            [84.77, 97.19, 28.57],
 "TF-HNN":            [91.69, 98.38, 28.56],
 "HNHN":              [87.27, 97.30, 27.29],
 "UniGNN":            [89.86, 98.42, 28.39],
 "AllSetTransformer": [90.17, 98.07, 27.32],
 "ED-HNN":            [91.45, 98.51, 28.38],
 "HyperGT":           [86.00, 96.83, 26.58],
 "EHNN":              [87.99, 97.97, 28.95],
 "T-HyperGNN":        [89.15, 97.76, 24.63],
}

PAPER = {}
for m, row in T1.items():
    for d, v in zip(T1_COLS, row):
        PAPER[(m, d)] = v
for m, row in A5.items():
    for d, v in zip(A5_COLS, row):
        PAPER[(m, d)] = v

rep = pd.read_csv(CSV)
fails, checked = [], 0

# A. every csv `paper` value must equal the table value exactly
for _, r in rep.iterrows():
    key = (r.method, r.dataset)
    if key not in PAPER:
        fails.append(f"{key}: not present in Tables 1/A5"); continue
    want = PAPER[key]
    if want is None:
        fails.append(f"{key}: csv has paper={r.paper} but the paper reports OOM"); continue
    checked += 1
    if abs(float(r.paper) - want) > 1e-9:
        fails.append(f"{key}: csv paper={r.paper} vs table {want}")

# B. every non-OOM (method, dataset) in the tables must appear in the csv,
#    except trivago cells, where our side has no comparable recompute
present = {(r.method, r.dataset) for _, r in rep.iterrows()}
# Cells reported upstream but lost on OUR side (from the run status), hence not comparable:
OUR_MISSING = {("CEGAT", "Walmart"),        # CUDA OOM in our recompute
               ("T-HyperGNN", "DBLP-CA"),   # one-hour timeout, no seed completed
               ("T-HyperGNN", "Walmart")}   # one-hour timeout, no seed completed
for key, v in PAPER.items():
    if v is None or key in present:
        continue
    m, d = key
    if d == "Trivago":
        continue  # trivago excluded from the analysis; csv keeps only HNHN/LEGCN
    if key in OUR_MISSING:
        continue  # reported upstream, lost on our side; correctly absent from the csv
    fails.append(f"{key}: reported {v} in the paper but absent from the csv")

# C. the csv trivago rows must be exactly HNHN and LEGCN with the right values
tri = rep[rep.dataset == "Trivago"]
got = sorted((r.method, float(r.paper)) for _, r in tri.iterrows())
if got != [("HNHN", 53.75), ("LEGCN", 33.45)]:
    fails.append(f"trivago rows: {got}, want HNHN=53.75, LEGCN=33.45")

# D. the trivago reported range across methods (for the manuscript sentence)
tvals = sorted(v for (m, d), v in PAPER.items() if d == "Trivago" and v is not None)
print(f"trivago reported values across methods: {tvals}")
print(f"  -> range {tvals[0]:.2f} .. {tvals[-1]:.2f} over {len(tvals)} methods")

# E. cell count
print(f"\ncells checked against the tables: {checked} (csv rows: {len(rep)})")
tab_nonoom = sum(1 for v in PAPER.values() if v is not None)
print(f"non-OOM cells in Tables 1/A5: {tab_nonoom}; "
      f"absent from csv: {tab_nonoom - checked} "
      f"(trivago excluded: {sum(1 for (m,d),v in PAPER.items() if d=='Trivago' and v is not None and (m,d) not in present)}, "
      f"our-side missing others)")

print()
if fails:
    for f in fails:
        print("  FAIL:", f)
    print(f"\nLAYER-1 SOURCE CHECK FAIL: {len(fails)} mismatches")
    sys.exit(1)
print("LAYER-1 SOURCE CHECK PASS: every reported value in repro_check.csv matches "
      "DHG-Bench Tables 1/A5 exactly, and the OOM pattern is consistent.")
