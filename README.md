# significance-hypergraph

Code, patches, and per-seed results for the paper:

**Is There a Best Hypergraph Neural Network? A Significance-Aware Recomputation and Statistical Audit of DHG-Bench.**

This repository releases everything needed to regenerate the statistical
results in the paper from the recomputed per-seed accuracies of the
node-classification track of DHG-Bench.

## Contents

```
code/                analysis and recompute scripts
  phase1_analyze_perseed.py   four-layer statistical analysis (Layers 1-4 + leaderboard)
  phase1_final_figs.py        critical-difference diagram + reproducibility table
  phase1_*.py / *.sh          recompute harness, preflight, env setup, launch scripts
patches/
  phase1_perseed_dump.patch   env-gated hook that dumps per-seed train/val/test accuracy
  phase1_label_remap.patch    env-gated dense-label remap for the trivago label-range fault
docs/
  repro_check.csv             full per-cell reproducibility table (222 comparable cells)
results_phase1/node_cls/      225 per-seed JSON result files (20 methods x 13 datasets)
```

## Reproducing the analysis

The statistical results, tables, and figures are regenerated from the
released per-seed JSON; no GPU is required for this step.

```bash
pip install numpy scipy pandas statsmodels scikit-posthocs matplotlib
python code/phase1_analyze_perseed.py results_phase1/node_cls   # Layers 1-4 + leaderboard
python code/phase1_final_figs.py                                # CD diagram + repro table
```

## Reproducing the recompute (optional, requires a GPU and the DHG-Bench data)

The recompute itself runs the public DHG-Bench harness with the two
environment-gated patches applied. It requires the DHG-Bench code and
datasets (not redistributed here) and a CUDA GPU.

- Every cell is run in the harness default configuration (`is_default`),
  because the public release ships dataset-specific configuration sections
  only for `cora`; the absence of the other per-dataset configurations is
  one of the reproducibility findings of the paper.
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
