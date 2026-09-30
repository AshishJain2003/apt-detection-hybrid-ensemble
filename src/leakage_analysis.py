"""How much of the paper protocol's advantage comes from leakage, and how much from class balance?

Balancing before splitting changes two things at once:
  1. the test set becomes ~50% APT instead of the real ~23%, which flatters precision and F1;
  2. synthetic SMOTE flows (interpolated from flows that may sit in the training set) enter
     the test set, which flatters recall.
This script separates the two effects for the proposed hybrid:
  - it re-scores the paper-protocol predictions at the real class ratio (TPR and FPR do not
    depend on the class ratio, so precision/F1 can be recomputed exactly), and
  - it rebuilds the paper-protocol split, marks which test flows are synthetic (not present
    in the real data), and compares the miss rate on synthetic vs real APT flows.

Usage:  python src/leakage_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_dapt2020
from preprocessing import _smote_tomek, clean

MODEL = "Proposed hybrid (RF+XGB)"


def main():
    paper = pd.read_csv(config.RESULTS_DIR / "binary_paper_protocol/metrics.csv", index_col="Model").loc[MODEL]
    free = pd.read_csv(config.RESULTS_DIR / "binary_leakage_free/metrics.csv", index_col="Model").loc[MODEL]

    # (1) class-ratio effect: paper-protocol TPR/FPR at the leakage-free test set's real class counts
    P, N = free["TP"] + free["FN"], free["TN"] + free["FP"]
    tp, fp = paper["Recall"] * P, paper["FPR"] * N
    prec = tp / (tp + fp)
    f1 = 2 * prec * paper["Recall"] / (prec + paper["Recall"])

    # (2) synthetic flows in the paper-protocol test set
    X, labels = clean(load_dapt2020())
    y = labels["Label"].to_numpy()
    X_bal, y_bal = _smote_tomek(X, y, config.SEED)
    real_rows = set(map(tuple, X.to_numpy()))
    synthetic = np.array([tuple(r) not in real_rows for r in X_bal.to_numpy()])
    _, test_idx = train_test_split(np.arange(len(y_bal)), test_size=config.TEST_SIZE,
                                   stratify=y_bal, random_state=config.SEED)
    preds = np.load(config.RESULTS_DIR / "binary_paper_protocol/predictions.npz")
    y_test, y_pred = preds["y_test"], preds[f"{MODEL}|pred"]
    assert np.array_equal(y_bal[test_idx], y_test), "split mismatch"
    syn = synthetic[test_idx]
    apt = y_test == 1
    miss_syn = np.mean(y_pred[apt & syn] == 0)
    miss_real = np.mean(y_pred[apt & ~syn] == 0)

    rows = [
        ["Paper protocol, as reported (test set ~50% APT, includes synthetic flows)", paper["Precision"], paper["Recall"], paper["F1"]],
        ["Paper protocol, re-scored at the real class ratio", prec, paper["Recall"], f1],
        ["Leakage-free protocol (real flows, real class ratio)", free["Precision"], free["Recall"], free["F1"]],
    ]
    table = pd.DataFrame(rows, columns=["Setting", "Precision", "Recall", "F1"]).set_index("Setting")
    lines = [f"# Leakage analysis: {MODEL}", "",
             "Balancing before splitting inflates results in two ways. This separates them.", "",
             (table * 100).round(2).to_markdown(), "",
             f"- **Class-ratio effect:** re-scoring at the real ratio ({int(P):,} APT vs {int(N):,} benign) moves F1 from "
             f"{paper['F1']:.2%} to {f1:.2%} and precision from {paper['Precision']:.2%} to {prec:.2%}: "
             f"{(paper['F1'] - f1) * 100:.1f} of the {(paper['F1'] - free['F1']) * 100:.1f} F1 points of difference.",
             f"- **Leakage effect:** the remaining {(f1 - free['F1']) * 100:.1f} F1 points. {syn[apt].mean():.0%} of the APT flows in the "
             f"paper-protocol test set are synthetic; the model misses {miss_syn:.2%} of synthetic APT flows but {miss_real:.2%} of real ones "
             f"(leakage-free: {free['FNR']:.2%}).", ""]
    (config.RESULTS_DIR / "binary_paper_protocol" / "leakage_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
