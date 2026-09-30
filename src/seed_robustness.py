"""Repeat the leakage-free binary experiment over several random seeds.

Each seed changes the 70:30 split, SMOTETomek sampling and model randomness,
so the spread shows whether differences between the hybrid and its components
are larger than run-to-run noise.

Usage:  python src/seed_robustness.py [--seeds 0 1 2 3 4]
"""
import argparse

import pandas as pd

import config
from data_loader import load_dapt2020
from metrics import binary_metrics, scores
from models import HybridRFXGB, random_forest, xgboost
from preprocessing import clean, make_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    X, labels = clean(load_dapt2020())
    y = labels["Label"].to_numpy()
    rows = []
    for seed in args.seeds:
        split = make_splits(X, y, ["Benign", "APT"], "leakage_free", seed=seed)
        models = {"Random forest": random_forest(seed), "XGBoost": xgboost(seed),
                  "Proposed hybrid (RF+XGB)": HybridRFXGB("combo", seed),
                  "Hybrid RF+XGB (soft vote)": HybridRFXGB("soft", seed)}
        for name, model in models.items():
            model.fit(split.X_train, split.y_train)
            m = binary_metrics(split.y_test, model.predict(split.X_test), scores(model, split.X_test))
            rows.append({"seed": seed, "Model": name, **m})
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "binary_leakage_free"
    df.to_csv(out / "seed_robustness_raw.csv", index=False)
    cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "AUC"]
    agg = df.groupby("Model")[cols].agg(["mean", "std"]) * 100
    table = pd.DataFrame({c: agg[(c, "mean")].map("{:.2f}".format) + " ± " + agg[(c, "std")].map("{:.2f}".format)
                          for c in cols})
    text = (f"# Leakage-free binary detection over {len(args.seeds)} seeds (mean ± std, %)\n\n"
            + table.to_markdown() + "\n")
    (out / "seed_robustness.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
