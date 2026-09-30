"""Evaluation metrics from Section 4 of the paper, plus multi-class extensions."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def scores(model, X):
    """Continuous scores for ROC/AUC: probabilities if available, else decision values."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:
            pass
    return model.decision_function(X)


def binary_metrics(y_true, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pos = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),          # = TPR / detection rate
        "F1": f1_score(y_true, y_pred),
        "TNR": tn / (tn + fp),
        "FPR": fp / (fp + tn),                          # benign flows raised as APT
        "FNR": fn / (fn + tp),                          # APT flows missed
        "AUC": roc_auc_score(y_true, pos),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def multiclass_metrics(y_true, y_pred, y_score, class_names) -> dict:
    """Stage-level metrics. Class 0 is Benign; every other class is an APT stage."""
    labels = list(range(len(class_names)))
    benign = y_true == 0
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        # Binary view of a multi-class detector: did it raise *an* alarm at all?
        "FPR": float(np.mean(y_pred[benign] != 0)),
        "APT detection rate": float(np.mean(y_pred[~benign] != 0)),
    }
    # One-vs-rest AUC needs probabilities; SVMs only give uncalibrated margins.
    is_proba = np.ndim(y_score) == 2 and np.allclose(y_score.sum(axis=1), 1, atol=1e-3)
    out["Macro AUC (OvR)"] = (roc_auc_score(y_true, y_score, multi_class="ovr", average="macro",
                                            labels=labels) if is_proba else np.nan)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, r in zip(class_names, per_class_recall):
        out[f"Recall: {name}"] = r
    return out
