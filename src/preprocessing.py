"""Preprocessing from Section 3.2 of Saini et al. (2023), applied to DAPT2020.

Paper order:  clean -> SMOTETomek -> 70:30 split -> Pearson filter -> mutual-information top-k
The paper balances the WHOLE dataset before splitting, so synthetic SMOTE rows
(interpolated from real test rows) end up in the test set. `make_splits` can run
that exact order ("paper") or a leakage-free order ("leakage_free") in which the
split happens first and SMOTETomek only touches the training data. Everything
else is identical, so the gap between the two isolates the effect of the leak.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature engineering / data cleaning. Returns (X, labels).

    `labels` also carries Src/Dst IP as metadata for error analysis; they are never features.
    """
    labels = df[config.LABEL_COLUMNS + ["Src IP", "Dst IP"]].copy()
    X = df.drop(columns=config.ID_COLUMNS + config.LABEL_COLUMNS)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    keep = X.notna().all(axis=1)                      # remove NaN / infinity rows
    X, labels = X[keep], labels[keep]

    X = pd.get_dummies(X, columns=config.CATEGORICAL_COLUMNS, dtype=float)  # dummy variables
    X = X.loc[:, X.nunique() > 1]                    # delete constant features

    dup = pd.concat([X, labels["Stage"]], axis=1).duplicated()  # remove duplicate data
    X, labels = X[~dup], labels[~dup]
    return X.reset_index(drop=True).astype(np.float64), labels.reset_index(drop=True)


def correlation_filter(X: pd.DataFrame, threshold: float = config.CORRELATION_THRESHOLD) -> list[str]:
    """Pearson filter: drop a feature if it is >threshold correlated with an earlier one."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    return [c for c in X.columns if c not in dropped]


def mutual_information_ranking(X: pd.DataFrame, y, seed: int = config.SEED) -> pd.Series:
    scores = mutual_info_classif(X, y, random_state=seed)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def _smote_tomek(X, y, seed):
    # SMOTE needs k_neighbors < size of the smallest class (Data Exfiltration is tiny).
    k = int(min(5, pd.Series(y).value_counts().min() - 1))
    # sampling_strategy="all" is SMOTETomek's own default: Tomek links are removed from every
    # class. (TomekLinks' standalone default would clean only one class after balancing.)
    st = SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed),
                    tomek=TomekLinks(sampling_strategy="all", n_jobs=-1), random_state=seed)
    X_res, y_res = st.fit_resample(X, y)
    return X_res.reset_index(drop=True), np.asarray(y_res)


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    features: list
    class_names: list
    mi_scores: pd.Series
    n_after_correlation: int
    counts: dict = field(default_factory=dict)


def make_splits(X: pd.DataFrame, y: np.ndarray, class_names, protocol: str,
                seed: int = config.SEED, top_k: int = config.MI_TOP_K) -> Split:
    counts = {"original": pd.Series(y).value_counts().sort_index().to_dict()}

    if protocol == "paper":
        X_bal, y_bal = _smote_tomek(X, y, seed)
        counts["after_smote_tomek"] = pd.Series(y_bal).value_counts().sort_index().to_dict()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_bal, y_bal, test_size=config.TEST_SIZE, stratify=y_bal, random_state=seed)
    elif protocol == "leakage_free":
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
        X_tr, y_tr = _smote_tomek(X_tr, y_tr, seed)
        counts["after_smote_tomek"] = pd.Series(y_tr).value_counts().sort_index().to_dict()
    else:
        raise ValueError(protocol)

    # Feature selection is fitted on the training data only.
    X_tr = X_tr.loc[:, X_tr.nunique() > 1]
    kept = correlation_filter(X_tr)
    mi = mutual_information_ranking(X_tr[kept], y_tr, seed)
    features = list(mi.index[:top_k])

    scaler = StandardScaler().fit(X_tr[features])
    counts["train"] = pd.Series(y_tr).value_counts().sort_index().to_dict()
    counts["test"] = pd.Series(y_te).value_counts().sort_index().to_dict()
    return Split(
        X_train=scaler.transform(X_tr[features]).astype(np.float32),
        X_test=scaler.transform(X_te[features]).astype(np.float32),
        y_train=np.asarray(y_tr), y_test=np.asarray(y_te),
        features=features, class_names=list(class_names), mi_scores=mi,
        n_after_correlation=len(kept), counts=counts)
