"""Нейросеть для табличных данных на PyTorch (вопросы 21–24, 30).

Обёрнута в интерфейс scikit-learn (fit / predict / predict_proba), поэтому встаёт
в тот же Pipeline, что и baseline, и сохраняется в один joblib-файл.
Установка PyTorch (CPU): uv sync --group torch. Для GPU в Colab torch уже стоит.
"""

from __future__ import annotations

import copy
import importlib.util
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _dense(X) -> np.ndarray:
    return (X.toarray() if sparse.issparse(X) else np.asarray(X)).astype(np.float32)


class _TorchMLPBase(BaseEstimator):
    def __init__(self, hidden=(128, 64), dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=60,
                 batch_size=128, patience=8, seed=42, device="auto"):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed
        self.device = device

    # --- то, чем отличаются классификатор и регрессор
    def _n_out(self, y) -> int: ...
    def _loss(self): ...
    def _target(self, y) -> np.ndarray: ...

    def _build(self, n_in: int, n_out: int):
        import torch.nn as nn

        layers, width = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            width = h
        layers.append(nn.Linear(width, n_out))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = "cuda" if self.device == "auto" and torch.cuda.is_available() else (
            "cpu" if self.device == "auto" else self.device)
        X = _dense(X)
        t = self._target(np.asarray(y))
        # 10 % train отдаём под раннюю остановку, test и val не трогаем
        idx = rng.permutation(len(X))
        n_hold = max(1, int(0.1 * len(X)))
        hold, tr = idx[:n_hold], idx[n_hold:]
        Xt = torch.tensor(X[tr], device=dev)
        yt = torch.tensor(t[tr], device=dev)
        Xh = torch.tensor(X[hold], device=dev)
        yh = torch.tensor(t[hold], device=dev)

        self.model_ = self._build(X.shape[1], self._n_out(y)).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        loss_fn = self._loss()
        best, best_state, bad = np.inf, None, 0
        self.history_ = []
        for _epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt), device=dev)
            for start in range(0, len(Xt), self.batch_size):
                b = perm[start : start + self.batch_size]
                if len(b) < 2:  # BatchNorm не работает на батче из одного объекта
                    continue
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[b]).squeeze(-1), yt[b])
                loss.backward()
                opt.step()
            sched.step()
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(Xh).squeeze(-1), yh))
            self.history_.append(val_loss)
            if val_loss < best - 1e-4:
                best, best_state, bad = val_loss, copy.deepcopy(self.model_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:  # early stopping
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu").eval()
        self.n_epochs_ = len(self.history_)
        return self

    def _forward(self, X) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.model_(torch.tensor(_dense(X))).numpy()


class TorchMLPClassifier(ClassifierMixin, _TorchMLPBase):
    def _n_out(self, y) -> int:
        self.classes_ = np.unique(y)
        return 1 if len(self.classes_) == 2 else len(self.classes_)

    def _loss(self):
        import torch.nn as nn

        return nn.BCEWithLogitsLoss() if len(self.classes_) == 2 else nn.CrossEntropyLoss()

    def _target(self, y) -> np.ndarray:
        classes = np.unique(y)
        codes = np.searchsorted(classes, y)
        return codes.astype(np.float32) if len(classes) == 2 else codes.astype(np.int64)

    def predict_proba(self, X) -> np.ndarray:
        logits = self._forward(X)
        if len(self.classes_) == 2:
            p = 1 / (1 + np.exp(-logits.reshape(-1)))
            return np.column_stack([1 - p, p])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class TorchMLPRegressor(RegressorMixin, _TorchMLPBase):
    def _n_out(self, y) -> int:
        return 1

    def _loss(self):
        import torch.nn as nn

        return nn.MSELoss()

    def _target(self, y) -> np.ndarray:
        # обучаемся на стандартизованном target, иначе MSE на больших числах взрывается
        self.y_mean_, self.y_std_ = float(np.mean(y)), float(np.std(y) or 1.0)
        return ((y - self.y_mean_) / self.y_std_).astype(np.float32)

    def predict(self, X) -> np.ndarray:
        return self._forward(X).reshape(-1) * self.y_std_ + self.y_mean_


def make_nn(task: str, params: dict[str, Any], seed: int) -> tuple[Any, str]:
    """Возвращает (модель, framework). Без PyTorch — MLP из scikit-learn с предупреждением."""
    framework = params.get("framework", "auto")
    use_torch = framework == "torch" or (framework == "auto" and TORCH_AVAILABLE)
    kw = {k: params[k] for k in ("hidden", "dropout", "lr", "weight_decay", "epochs",
                                 "batch_size", "patience") if k in params}
    if use_torch:
        if not TORCH_AVAILABLE:
            raise RuntimeError("framework: torch, но PyTorch не установлен: uv sync --group torch")
        kw["hidden"] = tuple(kw.get("hidden", (128, 64)))
        cls = TorchMLPRegressor if task == "regression" else TorchMLPClassifier
        return cls(seed=seed, **kw), "torch"

    from sklearn.neural_network import MLPClassifier, MLPRegressor

    cls = MLPRegressor if task == "regression" else MLPClassifier
    model = cls(
        hidden_layer_sizes=tuple(params.get("hidden", (128, 64))),
        alpha=params.get("weight_decay", 1e-4),
        learning_rate_init=params.get("lr", 1e-3),
        max_iter=params.get("epochs", 60),
        batch_size=min(params.get("batch_size", 128), 200),
        early_stopping=True,
        n_iter_no_change=params.get("patience", 8),
        random_state=seed,
    )
    return model, "sklearn"
