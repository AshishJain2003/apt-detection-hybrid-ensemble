"""Run the Saini et al. (2023) pipeline on the APT-specific DAPT2020 dataset.

Experiments
  binary_paper_protocol  Faithful replication: SMOTETomek on the full data, THEN 70:30 split.
  binary_leakage_free    Same pipeline, but split first and SMOTETomek only on training data;
                         the test set is untouched real traffic.
  stage_leakage_free     Extension: classify the APT *stage* of each flow (5 classes),
                         which DAPT2020 labels but the paper's datasets cannot.

Usage:  python src/run_experiments.py [--experiments ...] [--no-deep]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from data_loader import STAGE_ORDER, load_dapt2020
from metrics import binary_metrics, multiclass_metrics, scores
from models import classical_models
from preprocessing import clean, make_splits

EXPERIMENTS = {
    "binary_paper_protocol": {"target": "Label", "protocol": "paper"},
    "binary_leakage_free": {"target": "Label", "protocol": "leakage_free"},
    "stage_leakage_free": {"target": "Stage", "protocol": "leakage_free"},
}


def build_models(binary, use_deep):
    models = classical_models(binary)
    if use_deep:
        try:
            from deep_models import deep_models
            models.update(deep_models())
        except ImportError as e:
            print(f"  ! Skipping CNN/ANN/MLP: PyTorch unavailable ({e})")
    return models


def run(name, target, protocol, X, labels, use_deep):
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = target == "Label"
    class_names = ["Benign", "APT"] if binary else STAGE_ORDER
    y = labels["Label"].to_numpy() if binary else labels["Stage"].map(STAGE_ORDER.index).to_numpy()

    print(f"\n=== {name}  (target={target}, protocol={protocol})")
    t0 = time.time()
    split = make_splits(X, y, class_names, protocol)
    print(f"  preprocessing {time.time() - t0:.0f}s | train {split.X_train.shape} test {split.X_test.shape}"
          f" | {split.n_after_correlation} features after Pearson filter -> top {len(split.features)} by MI")

    info = {"experiment": name, "target": target, "protocol": protocol, "class_names": class_names,
            "features_selected": split.features, "n_after_correlation": split.n_after_correlation,
            "mutual_information": split.mi_scores.round(6).to_dict(),
            "class_counts": {k: {class_names[int(c)]: int(n) for c, n in v.items()}
                             for k, v in split.counts.items()}}
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2))

    rows, preds = [], {"y_test": split.y_test}
    for model_name, model in build_models(binary, use_deep).items():
        t = time.time()
        model.fit(split.X_train, split.y_train)
        fit_s = time.time() - t
        t = time.time()
        y_pred = model.predict(split.X_test)
        y_score = scores(model, split.X_test)
        pred_s = time.time() - t

        m = (binary_metrics(split.y_test, y_pred, y_score) if binary
             else multiclass_metrics(split.y_test, y_pred, y_score, class_names))
        m.update({"Model": model_name, "Train time (s)": fit_s, "Predict time (s)": pred_s})
        if hasattr(model, "epochs_trained_"):
            m["Epochs"] = model.epochs_trained_
        rows.append(m)
        preds[f"{model_name}|pred"] = y_pred
        preds[f"{model_name}|score"] = y_score
        key = "F1" if binary else "Macro F1"
        print(f"  {model_name:<28} acc={m['Accuracy']:.4f}  {key}={m[key]:.4f}  "
              f"FPR={m['FPR']:.4f}  ({fit_s:.1f}s)")

    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(out_dir / "metrics.csv")
    np.savez_compressed(out_dir / "predictions.npz", **preds)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--no-deep", action="store_true", help="skip CNN/ANN/MLP")
    args = ap.parse_args()

    df = load_dapt2020()
    X, labels = clean(df)
    print(f"DAPT2020: {len(df):,} flows loaded, {len(X):,} after cleaning, {X.shape[1]} features")
    for name in args.experiments:
        run(name, **EXPERIMENTS[name], X=X, labels=labels, use_deep=not args.no_deep)


if __name__ == "__main__":
    main()
