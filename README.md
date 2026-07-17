# significance-hypergraph

Code, patches, and per-seed results for the paper:

**Is There a Best Hypergraph Neural Network? A Significance-Aware Recomputation and Statistical Audit of DHG-Bench.**

This repository releases everything needed to regenerate the statistical
results and every figure in the paper from the recomputed per-seed accuracies
of the node-classification track of DHG-Bench.

## Contents

```
code/                analysis and recompute scripts
  phase1_analyze_perseed.py    four-layer statistical analysis (Layers 1-4 + leaderboard)
  make_paper_figs.py           regenerates ALL manuscript figures (vector PDF) from the
                               released per-seed JSON, with built-in assertions
  gate_reviewer_round.py       review-proofing gate: recomputes and asserts every number
                               reported in the paper (non-zero exit on mismatch)
  verify_structural_features.py  re-derives the structural features from raw DHG-Bench data
  structure_merge.py / make_figs.py / gate_structure.py   structural-analysis pipeline
  phase1_final_figs.py         builds docs/repro_check.csv (reproducibility table)
  phase1_*.py / *.sh           recompute harness, preflight, env setup, launch scripts
patches/
  phase1_perseed_dump.patch    env-gated hook that dumps per-seed train/val/test accuracy
  phase1_label_remap.patch     env-gated dense-label remap, added as a candidate fix while
                               diagnosing the trivago crash; the released trivago labels are
                               already dense, so there it is a no-op (see the paper, Appendix B)
docs/
  repro_check.csv              full per-cell reproducibility table (222 comparable cells)
  structure_table.csv          structural features + rankability + leading family per dataset
  structural_recomputed.csv    structural features re-derived from the raw DHG-Bench data
results_phase1/node_cls/       225 per-seed JSON result files (20 methods x 13 datasets)
requirements-lock.txt          exact package pins of the recompute environment (py3.12/torch 2.4)
requirements-analysis.txt      the six packages needed for the analysis step (no GPU)
```

**Note on truncated cells.** Seven of the 225 released cells were truncated by
the one-hour wall-clock budget after two seeds (TMPHN on ModelNet40, actor,
amazon, pokec, pubmed and twitch; HyperGCN on yelp); the incremental per-seed
dump preserved the completed seeds. The per-seed analysis layers use only the
cells with all twenty seeds, and `gate_reviewer_round.py` asserts this
inventory together with every number reported in the paper.

## Reproducing the analysis and the figures

The statistical results, tables, and figures are regenerated from the
released per-seed JSON; no GPU is required for this step.

```bash
pip install -r requirements-analysis.txt
python code/phase1_analyze_perseed.py results_phase1/node_cls          # Layers 1-4 + leaderboard
python code/make_paper_figs.py results_phase1/node_cls \
       docs/repro_check.csv docs/structure_table.csv                   # all figures (vector PDF)
python code/gate_reviewer_round.py results_phase1/node_cls             # assert every paper number
```

## Reproducing the recompute (optional, requires a GPU and the DHG-Bench data)

The recompute itself runs the public DHG-Bench harness with the two
environment-gated patches applied. It requires the DHG-Bench code and
datasets (not redistributed here) and a CUDA GPU. The exact package pins of
the environment used are in `requirements-lock.txt`.

- Every cell is run in the harness default configuration (`is_default`),
  because the public release ships dataset-specific configuration sections
  only for `cora`; the absence of the other per-dataset configurations is
  one of the reproducibility findings of the paper.
- On trivago, training fails for 18 of the 20 methods under the public
  release (14 label-range assertions at a dense label tensor, 3 host-memory
  exhaustions, 1 GPU out-of-memory); the verified failure taxonomy is in
  Appendix B of the paper.
- Long-running jobs should be wrapped in `tmux` or `nohup` so they survive
  SSH disconnection.

## Data availability

The benchmark under audit (DHG-Bench) and its datasets are third-party
resources available from their original authors; this repository
redistributes neither. No new experimental data are generated beyond the
recomputed per-seed accuracies released here.

## License

Code is released for research use. See `LICENSE` (add an SPDX license, e.g.
MIT or Apache-2.0, before publishing the repository).

## Citation

If you use this code or the released results, please cite the paper (see the
manuscript for the full reference).
