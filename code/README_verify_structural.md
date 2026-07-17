# >>> UPDATE (re-run needed) <<<
# The first run reported `avg_e` FAIL on most datasets and one H_edge mismatch (coauthor_cora).
# Cause: DHG-Bench stores the released incidence as [V|E ; E|V] (forward V->E plus reverse E->V),
# so a naive pass double-counts memberships. The script now replicates DHG-Bench's own `ExtractV2E`
# and keeps only the V->E half, which fixes avg_e (and the coauthor_cora homophily). Just RE-RUN the
# updated `verify_structural_features.py` exactly as before. nodes / feat / classes already PASSed and
# are unchanged.

# Verifying the structural features on the server (optional, upgrades §3.4)

**Goal.** Re-derive the hypergraph-structural features (nodes, hyperedges, mean hyperedge size,
features, classes, hyperedge homophily H_edge) directly from the raw DHG-Bench data and confirm
they match the values cited from DHG-Bench Table A1. If they all match, §3.4 can change from
*"taken from the benchmark's own dataset documentation"* to *"verified against the raw data"* — this
closes a likely reviewer question ("did you recompute these or copy them?"). It is **optional**: the
current wording is already honest, so the paper is submittable without this.

## The script is now wired to DHG-Bench — no code editing needed

The earlier version had a `load_hypergraph()` stub you had to adapt. This version loads through the
benchmark's own `lib_dataset.data_base.HyperDataset` (the same path the recompute used), so you just
run it. It reads `.x`, `.y` and `.hyperedge_index` and computes the structural features; no model is
run and no GPU is needed.

## Steps

1. Put two files in the **DHG-Bench repo root** (the folder that contains `dhgbench/` and `data/`):
   - `verify_structural_features.py`
   - `structure_table.csv`  (the cited values; its `Hedge` column is Table A1's H_edge)
   You already placed them correctly (next to `dhgbench/` and `data/`), per your screenshot.

2. Make sure `data.zip` is extracted so `data/trad_data`, `data/hete_data`, `data/pyg_data/...` exist
   (the recompute already did this).

3. Run from the repo root with the project venv active (the loader resolves `../data/...` relative to
   the `dhgbench/` package, which the script handles by chdir-ing into it automatically):
   ```bash
   source .venv-dhgbench/bin/activate
   tmux new -s structcheck
   python verify_structural_features.py --cited structure_table.csv --out structural_recomputed.csv
   # Ctrl-b d to detach;  tmux a -t structcheck to reattach
   ```
   Sanity-check a couple of small datasets first:
   ```bash
   python verify_structural_features.py --only cora NTU2012 --cited structure_table.csv
   ```
   If the package is not auto-found, pass it: `--dhgbench-dir /home/ii/Documents/MAKE-bench/DHG-Bench/dhgbench`.

4. **Read the diff.** It prints a PASS/FAIL row per dataset for nodes / feat / mean |e| / classes, and
   reports which **H_edge convention** matches the cited value:
   - `dom`  = dominant-label fraction  (mean over hyperedges of `max_c |{v in e: y=c}| / |e|`)
   - `pair` = intra-class pair fraction (mean over hyperedges of same-label pairs / all pairs)
   Note the matching convention in a one-line footnote on Table 5 / §3.4.

5. **If ALL PASS:** change the §3.4 sentence to e.g. *"the structural statistics were re-derived from
   the raw data and verified against DHG-Bench Table A1"*, add the H_edge convention footnote, and add
   `verify_structural_features.py` + `structural_recomputed.csv` to the repo `code/`.
   **If something FAILs:** keep the current "taken from" wording (still correct) and inspect. The two
   likely causes are (a) a different H_edge convention from Table A1 (the script already reports both),
   or (b) a different hypergraph construction for the visual-object datasets, which is itself worth one
   sentence.

## Notes
- **walmart-trips-100** is featureless; DHG-Bench synthesises its features from `feature_noise`, so its
  **feature dimensionality** depends on `--feature_noise` (default 1.0) and is the one configuration-
  dependent structural column. The diff therefore does not fail walmart on feat dim; its counts, |e|,
  classes and H_edge are unaffected.
- This script touches only the **structural** columns of Table 5. The ρ_d values and the leading-family
  analysis come from the released per-seed recompute and are already locked; nothing here can alter
  W=0.447, 0/136 Holm, or twitch ρ_d=0.8.
