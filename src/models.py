"""The classifiers compared in Table 7 of Saini et al. (2023), plus the proposed hybrid.

Baselines use library defaults (the paper gives no settings for them), except
Random Forest and XGBoost, which use the same settings as inside the hybrid so
that the hybrid is compared against its own components.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config

KERNEL_SVM_MAX_TRAIN = 20_000  # RBF SVC is O(n^2); train it on a stratified subsample.


def random_forest(seed=config.SEED):
    return RandomForestClassifier(n_estimators=config.N_ESTIMATORS, n_jobs=-1, random_state=seed)


def xgboost(seed=config.SEED):
    return XGBClassifier(n_estimators=config.N_ESTIMATORS, learning_rate=config.XGB_LEARNING_RATE,
                         max_depth=config.XGB_MAX_DEPTH, n_jobs=-1, random_state=seed,
                         tree_method="hist")


class HybridRFXGB(ClassifierMixin, BaseEstimator):
    """Hybrid ensemble of Random Forest and XGBoost (Algorithm 1).

    vote="combo" reproduces combo.SimpleClassifierAggregator(method="average")
    exactly, as used in the paper: predict() averages the two models' predicted
    *labels* and thresholds at >= 0.5, so for binary data a flow is flagged as
    APT when either model flags it. predict_proba() averages probabilities.
    This label-averaging only makes sense for 0/1 labels.

    vote="soft" averages the two models' class probabilities and takes the
    argmax (equivalent to sklearn's soft VotingClassifier); it also works for
    multi-class targets such as APT stages.
    """

    def __init__(self, vote="combo", seed=config.SEED):
        self.vote = vote
        self.seed = seed

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.vote == "combo" and len(self.classes_) != 2:
            raise ValueError("combo label-averaging is only defined for binary targets")
        self.estimators_ = [random_forest(self.seed).fit(X, y), xgboost(self.seed).fit(X, y)]
        return self

    def predict_proba(self, X):
        return np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)

    def predict(self, X):
        if self.vote == "combo":
            labels = np.column_stack([est.predict(X) for est in self.estimators_])
            return (labels.mean(axis=1) >= 0.5).astype(int)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SubsampledFit(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on a stratified random subsample of at most `max_train` rows."""

    def __init__(self, estimator, max_train=KERNEL_SVM_MAX_TRAIN, seed=config.SEED):
        self.estimator = estimator
        self.max_train = max_train
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_train:
            idx = []
            for c in np.unique(y):
                members = np.flatnonzero(y == c)
                n = max(1, round(self.max_train * len(members) / len(y)))
                idx.append(rng.choice(members, size=min(n, len(members)), replace=False))
            idx = np.concatenate(idx)
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def classical_models(binary: bool, seed=config.SEED) -> dict:
    models = {
        "K-nearest neighbor": KNeighborsClassifier(n_jobs=-1),
        "Decision tree": DecisionTreeClassifier(random_state=seed),
        "SVM": LinearSVC(dual=False, random_state=seed),
        "Random forest": random_forest(seed),
        "XGBoost": xgboost(seed),
        "Gradient boosting": GradientBoostingClassifier(random_state=seed),
        "Kernel SVM": SubsampledFit(SVC(kernel="rbf", random_state=seed), seed=seed),
        "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=seed),
        "Naive Bayes": GaussianNB(),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=seed),
    }
    if binary:
        models["Proposed hybrid (RF+XGB)"] = HybridRFXGB(vote="combo", seed=seed)
    models["Hybrid RF+XGB (soft vote)"] = HybridRFXGB(vote="soft", seed=seed)
    return models
