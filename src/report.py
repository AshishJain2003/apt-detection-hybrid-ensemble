"""Build figures and a Markdown summary from results/<experiment>/ outputs.

Usage:  python src/report.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_curve

import config
from data_loader import STAGE_ORDER, load_dapt2020
from preprocessing import clean

FIG_DIR = config.RESULTS_DIR / "figures"

# Reference palette (dataviz skill): light surface, text tokens, validated categorical slots.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, DEEMPH = "#e1e0d9", "#c3c2b7", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

HYBRID = "Proposed hybrid (RF+XGB)"
SOFT = "Hybrid RF+XGB (soft vote)"

# Table 7 of Saini et al. (2023), CSE-CIC-IDS2018, in %.
PAPER_TABLE7 = pd.DataFrame(
    [["K-nearest neighbor", 98.40, 98.56, 98.22, 98.39, 98.58, 1.42, 1.78],
     ["Decision tree", 98.31, 98.56, 98.04, 98.30, 98.58, 1.42, 1.96],
     ["SVM", 88.70, 83.96, 95.55, 89.38, 81.91, 18.09, 4.45],
     ["Random forest", 98.66, 99.58, 97.72, 98.64, 99.60, 0.40, 2.28],
     ["XGBoost", 98.29, 99.42, 97.14, 98.27, 99.44, 0.56, 2.86],
     ["Gradient boosting", 97.38, 98.56, 96.15, 97.34, 98.61, 1.39, 3.85],
     ["Kernel SVM", 97.16, 98.66, 95.59, 97.10, 98.71, 1.29, 4.41],
     ["AdaBoost", 96.98, 97.93, 95.95, 96.93, 98.00, 2.00, 4.05],
     ["Naive Bayes", 86.77, 81.45, 95.08, 87.74, 78.53, 21.47, 4.92],
     ["Logistic regression", 85.04, 82.48, 88.83, 85.54, 81.29, 18.71, 11.17],
     [HYBRID, 98.92, 99.47, 98.35, 98.90, 99.48, 0.52, 1.65],
     ["CNN", 98.22, 98.25, 98.21, 98.22, 99.51, 0.49, 1.79],
     ["ANN", 97.73, 97.79, 97.72, 97.73, 99.49, 0.51, 2.28],
     ["MLP", 97.98, 98.03, 97.98, 97.98, 99.42, 0.58, 2.02]],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "TNR", "FPR", "FNR"]).set_index("Model")


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": INK_2, "ytick.major.size": 0, "axes.edgecolor": BASELINE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
        "savefig.bbox": "tight"})


def _load(exp):
    d = config.RESULTS_DIR / exp
    if not (d / "metrics.csv").exists():
        return None
    return {"metrics": pd.read_csv(d / "metrics.csv", index_col="Model"),
            "info": json.loads((d / "split_info.json").read_text()),
            "preds": np.load(d / "predictions.npz", allow_pickle=False)}


def _save(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print("  wrote", FIG_DIR / name)


def fig_stage_distribution():
    X, labels = clean(load_dapt2020())
    counts = labels["Stage"].value_counts().reindex(STAGE_ORDER)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(counts.index[::-1], counts.values[::-1], height=0.55, color=SERIES[0])
    ax.set_xscale("log")
    ax.set_xlim(5, 4e5)
    for y, v in enumerate(counts.values[::-1]):
        ax.text(v * 1.15, y, f"{v:,}", va="center", color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Flows after cleaning (log scale)")
    ax.set_title("DAPT2020 flows by APT stage")
    _save(fig, "01_stage_distribution.png")


def fig_mutual_information(res):
    mi = pd.Series(res["info"]["mutual_information"])
    top = mi.iloc[:config.MI_TOP_K][::-1]
    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.barh(top.index, top.values, height=0.6, color=SERIES[0])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("Mutual information with the Benign/APT label")
    ax.set_title(f"Top {config.MI_TOP_K} features by mutual information")
    _save(fig, "02_mutual_information.png")


def fig_protocol_comparison(paper, free):
    models = [m for m in free["metrics"].index if m in paper["metrics"].index]
    order = free["metrics"].loc[models, "F1"].sort_values().index
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True)
    for ax, metric, title in [(axes[0], "F1", "F1-score (higher is better)"),
                              (axes[1], "FPR", "False positive rate (lower is better)")]:
        ax.barh(y + 0.19, paper["metrics"].loc[order, metric], height=0.36, color=SERIES[0],
                label="Paper protocol: balance, then split")
        ax.barh(y - 0.19, free["metrics"].loc[order, metric], height=0.36, color=SERIES[1],
                label="Leakage-free: split, then balance train only")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, order)
    axes[0].set_xlim(0, 1)
    axes[1].set_xlim(0, max(0.05, float(free["metrics"].loc[order, "FPR"].max()) * 1.1))
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.07), ncol=2)
    fig.suptitle("Binary APT detection on DAPT2020: effect of SMOTETomek before the split",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    _save(fig, "03_protocol_comparison.png")


def _cm_panel(ax, cm, names, title):
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=30 if len(names) > 2 else 0, ha="right" if len(names) > 2 else "center")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    ax.set_title(title, fontsize=10.5)


def fig_confusion_binary(paper, free):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, res, title in [(axes[0], paper, "Paper protocol"), (axes[1], free, "Leakage-free protocol")]:
        yt, yp = res["preds"]["y_test"], res["preds"][f"{HYBRID}|pred"]
        _cm_panel(ax, confusion_matrix(yt, yp, labels=[0, 1]), ["Benign", "APT"], title)
    fig.suptitle("Proposed hybrid (RF+XGB) confusion matrices, DAPT2020 test set",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "04_confusion_hybrid_binary.png")


def fig_roc(free):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    yt = free["preds"]["y_test"]
    for color, name in zip(SERIES, [HYBRID, "Random forest", "XGBoost"]):
        s = free["preds"][f"{name}|score"]
        fpr, tpr, _ = roc_curve(yt, s[:, 1] if s.ndim == 2 else s)
        auc = free["metrics"].loc[name, "AUC"]
        for ax in axes:
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC {auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color=BASELINE, lw=1)
    axes[0].set_title("ROC curve", fontsize=11)
    axes[1].set_xlim(0, 0.05)
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_title("Zoom: FPR below 5%", fontsize=11)
    for ax in axes:
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Leakage-free binary detection: hybrid vs its components",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "05_roc_binary.png")


def fig_stage(stage):
    names = stage["info"]["class_names"]
    recall = stage["metrics"][[f"Recall: {n}" for n in names]]
    recall.columns = names
    recall = recall.loc[stage["metrics"]["Macro F1"].sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(recall) + 1.5))
    ax.imshow(recall.values, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for i in range(recall.shape[0]):
        for j in range(recall.shape[1]):
            v = recall.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.55 else INK)
    n_test = stage["info"]["class_counts"]["test"]
    ax.set_xticks(range(len(names)), [f"{n}\n(n={n_test[n]:,} test flows)" for n in names], fontsize=8.5)
    ax.tick_params(axis="x", colors=INK_2)
    ax.set_yticks(range(len(recall)), recall.index)
    ax.grid(False)
    ax.set_title("Per-stage recall on DAPT2020 (models sorted by macro F1)")
    _save(fig, "06_stage_recall_heatmap.png")

    yt, yp = stage["preds"]["y_test"], stage["preds"][f"{SOFT}|pred"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    _cm_panel(ax, confusion_matrix(yt, yp, labels=range(len(names))), names,
              "Hybrid RF+XGB (soft vote): APT-stage confusion matrix")
    _save(fig, "07_confusion_stage.png")


def _fmt_pct(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = (out[c] * 100).map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    return out


def summary_markdown(paper, free, stage):
    lines = ["# Results summary (auto-generated by src/report.py)", ""]
    cols = ["Accuracy", "Precision", "Recall", "F1", "TNR", "FPR", "FNR", "AUC"]
    for title, res in [("Binary, paper protocol (SMOTETomek on all data, then 70:30 split)", paper),
                       ("Binary, leakage-free protocol (70:30 split, SMOTETomek on train only)", free)]:
        if res is None:
            continue
        m = res["metrics"]
        t = _fmt_pct(m, cols)
        t["Train (s)"] = m["Train time (s)"].map(lambda v: f"{v:.1f}")
        lines += [f"## {title}", "", "All values in %.", "", t.to_markdown(), ""]
    if stage is not None:
        m = stage["metrics"]
        cols = ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1",
                "FPR", "APT detection rate", "Macro AUC (OvR)"]
        lines += ["## APT-stage classification (5 classes, leakage-free)", "",
                  "FPR = benign flows raised as any APT stage. All values in %.", "",
                  _fmt_pct(m, cols).to_markdown(), "",
                  "### Per-stage recall (%)", "",
                  _fmt_pct(m, [c for c in m.columns if c.startswith("Recall: ")]).to_markdown(), ""]
    lines += ["## Reference: Saini et al. (2023) Table 7, CSE-CIC-IDS2018 (%)", "",
              PAPER_TABLE7.to_markdown(floatfmt=".2f"), ""]
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    print("  wrote", config.RESULTS_DIR / "summary.md")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    paper, free, stage = (_load("binary_paper_protocol"), _load("binary_leakage_free"),
                          _load("stage_leakage_free"))
    fig_stage_distribution()
    if free is not None:
        fig_mutual_information(free)
        fig_roc(free)
    if paper is not None and free is not None:
        fig_protocol_comparison(paper, free)
        fig_confusion_binary(paper, free)
    if stage is not None:
        fig_stage(stage)
    summary_markdown(paper, free, stage)


if __name__ == "__main__":
    main()
