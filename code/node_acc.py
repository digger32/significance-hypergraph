"""
DHG-Bench node-classification accuracy (mean over 5 runs), transcribed from
arXiv:2508.12244v2 Table 1 (10 datasets) + Table A5 (NTU2012, ModelNet40, Ratings).
OOM cells -> None (treated as missing, NOT as worst rank).
Rows = methods (17 HNNs + 3 baselines: MLP, CEGCN, CEGAT).
"""

# column order
DATASETS_T1 = ["Cora","Pubmed","Cora-CA","DBLP-CA","Walmart","Trivago","Actor","Gamers","Pokec","Yelp"]
DATASETS_A5 = ["NTU2012","ModelNet40","Ratings"]
DATASETS = DATASETS_T1 + DATASETS_A5

# homophily labels (from paper Sec 3.1): 8 homophilic + 5 heterophilic node datasets
HOMOPHILIC = {"Cora","Pubmed","Cora-CA","DBLP-CA","NTU2012","ModelNet40","Walmart","Trivago"}
HETEROPHILIC = {"Actor","Yelp","Ratings","Gamers","Pokec"}

BASELINES = {"MLP","CEGCN","CEGAT"}

# method -> {dataset: acc or None}
O = None  # OOM
T1 = {
 "MLP":              [75.33,86.62,75.57,85.54,63.21,36.76,86.06,52.57,59.64,31.84],
 "CEGCN":            [76.90,86.03,78.40,89.75,70.40,47.24,67.41,51.02,57.37,O],
 "CEGAT":            [77.22,86.09,78.02,89.61,65.83,O,    73.87,51.05,57.34,O],
 "HGNN":             [77.90,86.17,82.84,91.00,77.12,57.67,77.83,52.38,57.87,33.71],
 "HyperGCN":         [78.38,87.42,81.65,89.51,68.75,42.39,81.82,51.32,57.51,29.29],
 "HCHA":             [77.84,86.33,83.01,91.18,77.66,52.50,78.30,52.35,58.19,33.13],
 "LEGCN":            [74.36,87.52,74.59,85.16,62.98,33.45,85.34,51.31,59.66,O],
 "HyperND":          [79.23,86.73,83.19,91.34,75.10,87.19,83.19,52.39,57.65,O],
 "PhenomNN":         [78.97,87.81,84.05,91.83,O,    O,    83.14,51.80,58.43,O],
 "SheafHyperGNN":    [79.03,87.10,84.08,91.09,O,    O,    85.00,52.07,59.06,O],
 "HJRL":             [78.67,87.98,83.72,O,    O,    O,    71.54,51.62,57.57,O],
 "DPHGNN":           [76.40,86.72,82.13,O,    O,    O,    83.65,52.36,58.20,O],
 "TF-HNN":           [79.47,87.90,84.19,91.38,77.04,90.79,85.96,52.34,59.17,35.16],
 "HNHN":             [75.24,85.66,76.51,85.84,65.21,53.75,81.20,51.12,58.55,25.86],
 "UniGNN":           [79.41,87.57,83.49,91.71,76.26,36.15,84.61,52.50,58.56,31.09],
 "AllSetTransformer":[78.02,87.79,82.95,91.51,78.61,59.92,85.66,51.74,58.55,33.18],
 "ED-HNN":           [78.58,87.65,82.98,91.55,77.90,75.99,85.77,50.54,58.68,34.84],
 "HyperGT":          [75.57,86.06,75.42,84.53,O,    O,    84.43,51.19,57.73,O],
 "EHNN":             [76.51,87.12,81.68,90.47,77.95,O,    86.21,52.14,58.23,34.09],
 "T-HyperGNN":       [74.20,86.28,75.01,85.44,73.48,O,    85.32,51.82,58.82,O],
}
A5 = {
 "MLP":              [88.59,96.88,28.47],
 "CEGCN":            [84.93,92.34,26.65],
 "CEGAT":            [84.14,92.02,28.23],
 "HGNN":             [90.13,97.43,28.05],
 "HyperGCN":         [75.78,91.15,27.34],
 "HCHA":             [90.53,97.68,28.33],
 "LEGCN":            [89.82,96.82,28.21],
 "HyperND":          [88.98,97.18,28.32],
 "PhenomNN":         [88.78,98.28,28.49],
 "SheafHyperGNN":    [90.81,98.30,28.35],
 "HJRL":             [88.15,96.33,26.90],
 "DPHGNN":           [84.77,97.19,28.57],
 "TF-HNN":           [91.69,98.38,28.56],
 "HNHN":             [87.27,97.30,27.29],
 "UniGNN":           [89.86,98.42,28.39],
 "AllSetTransformer":[90.17,98.07,27.32],
 "ED-HNN":           [91.45,98.51,28.38],
 "HyperGT":          [86.00,96.83,26.58],
 "EHNN":             [87.99,97.97,28.95],
 "T-HyperGNN":       [89.15,97.76,24.63],
}

def build_matrix():
    import pandas as pd
    rows = {}
    for m in T1:
        rows[m] = T1[m] + A5[m]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=DATASETS)
    return df

if __name__ == "__main__":
    df = build_matrix()
    print(df.shape, "methods x datasets")
    print("OOM cells:", int(df.isna().sum().sum()))
    print("Complete datasets (no OOM for any method):",
          [c for c in df.columns if df[c].notna().all()])
