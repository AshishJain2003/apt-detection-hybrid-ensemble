"""CNN, ANN and MLP baselines (Table 7), with an sklearn-style fit/predict API.

The paper trains them for 200 epochs but gives no architectures, so these are
standard choices for tabular flow features. Training stops early when the loss
on a 10% validation split of the training data has not improved for
`config.DL_PATIENCE` epochs, and the best weights are restored.

Training uses a GPU when one is available: CUDA, or Apple Silicon (MPS).
"""
import copy
import os

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn

import config

torch.set_num_threads(os.cpu_count() or 1)
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")


def _ann(n_features, n_classes):
    return nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, n_classes))


def _mlp(n_features, n_classes):
    return nn.Sequential(
        nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_classes))


def _cnn(n_features, n_classes):
    return nn.Sequential(
        nn.Unflatten(1, (1, n_features)),                 # treat the feature vector as a 1-D signal
        nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(64 * (n_features // 2), 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, n_classes))


ARCHITECTURES = {"CNN": _cnn, "ANN": _ann, "MLP": _mlp}


class TorchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, arch="MLP", max_epochs=config.DL_MAX_EPOCHS, patience=config.DL_PATIENCE,
                 batch_size=config.DL_BATCH_SIZE, lr=1e-3, seed=config.SEED):
        self.arch = arch
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_idx, test_size=0.1, stratify=y_idx, random_state=self.seed)
        X_tr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
        y_tr = torch.tensor(y_tr, device=DEVICE)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
        y_val = torch.tensor(y_val, device=DEVICE)

        self.model_ = ARCHITECTURES[self.arch](X.shape[1], len(self.classes_)).to(DEVICE)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, stale = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            self.model_.train()
            for batch in torch.randperm(len(y_tr), generator=gen).to(DEVICE).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(X_tr[batch]), y_tr[batch]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(X_val), y_val).item()
            self.history_.append(val_loss)
            if epoch % 20 == 0:
                print(f"    {self.arch} epoch {epoch:3d}  val_loss={val_loss:.4f}  ({DEVICE})", flush=True)
            if val_loss < best - 1e-5:
                best, best_state, stale = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.epochs_trained_ = len(self.history_)
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def deep_models(seed=config.SEED) -> dict:
    return {name: TorchClassifier(arch=name, seed=seed) for name in ARCHITECTURES}
