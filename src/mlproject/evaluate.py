"""Этап 4: оценка с доверительными интервалами, калибровка, интерпретация, анализ ошибок.

    uv run python -m mlproject.evaluate           # после mlproject.train
    uv run python -m mlproject.evaluate --smoke --output-dir .cache/smoke

Что получится (в reports/):
    evaluation.json         — метрики с 95 % ДИ, разница best − baseline, калибровка
    feature_importance.csv  — SHAP (для бустинга) и permutation importance
    errors.csv, slices.csv  — самые уверенные ошибки и метрики по срезам
    figures/*.png           — матрица ошибок, ROC/PR, калибровка, важность признаков
и обновится автоматический блок метрик в docs/model_card.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    brier_score_loss,
)

from mlproject import data as data_mod  # noqa: E402
from mlproject.config import load_config, paths  # noqa: E402
from mlproject.features import feature_names  # noqa: E402
from mlproject.metrics import (  # noqa: E402
    bootstrap_ci,
    compute_all,
    expected_calibration_error,
    metric_fn,
    paired_delta_ci,
)

log = logging.getLogger("mlproject.evaluate")

SCORING = {
    "roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1", "f1_macro": "f1_macro",
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "recall": "recall",
    "precision": "precision", "log_loss": "neg_log_loss", "brier": "neg_brier_score",
    "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2",
    "mape": "neg_mean_absolute_percentage_error",
}
CARD_START, CARD_END = "<!-- auto:metrics:start -->", "<!-- auto:metrics:end -->"


def model_outputs(preds: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    pred = preds[f"{name}__pred"].to_numpy()
    cols = [c for c in preds.columns if c.startswith(f"{name}__proba__")]
    return pred, (preds[cols].to_numpy() if cols else None)


def intervals(cfg, metrics, preds) -> dict[str, Any]:
    task, metric = metrics["task"], metrics["primary_metric"]
    ev = cfg["evaluation"]
    y = preds["y_true"].to_numpy()
    out: dict[str, Any] = {"primary_metric": metric, "level": ev["ci_level"],
                           "n_bootstrap": ev["n_bootstrap"], "models": {}}
    for name in metrics["models"]:
        pred, proba = model_outputs(preds, name)
        out["models"][name] = bootstrap_ci(metric, task, y, pred, proba, n=ev["n_bootstrap"],
                                           level=ev["ci_level"], seed=cfg["project"]["seed"])
    best, base = metrics["best"], metrics["baseline"]
    out["delta_best_vs_baseline"] = {
        "best": best, "baseline": base,
        **paired_delta_ci(metric, task, y, model_outputs(preds, best),
                          model_outputs(preds, base), n=ev["n_bootstrap"],
                          level=ev["ci_level"], seed=cfg["project"]["seed"]),
    }
    return out


def calibration(task: str, y: np.ndarray, proba: np.ndarray | None, pred: np.ndarray,
                classes: list, fig_dir: Path) -> dict[str, Any] | None:
    if task == "regression":
        resid = y - pred
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(pred, resid, s=8, alpha=0.5)
        axes[0].axhline(0, color="black", lw=1)
        axes[0].set(xlabel="прогноз", ylabel="остаток", title="Остатки")
        axes[1].hist(resid, bins=40)
        axes[1].set(title="Распределение остатков")
        fig.tight_layout()
        fig.savefig(fig_dir / "residuals.png", dpi=120)
        plt.close(fig)
        return {"residual_mean": float(resid.mean()), "residual_std": float(resid.std())}
    if proba is None:
        return None
    if task == "binary":
        conf, hit = proba[:, 1], (y == classes[1]).astype(int)
        brier = float(brier_score_loss(hit, conf))
    else:  # мультикласс: калибровка уверенности в предсказанном классе (top-label)
        conf = proba.max(axis=1)
        hit = (np.asarray(classes)[proba.argmax(axis=1)] == y).astype(int)
        brier = None
    frac, mean_pred = calibration_curve(hit, conf, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="идеальная калибровка")
    ax.plot(mean_pred, frac, "o-", label="модель")
    ax.set(xlabel="предсказанная вероятность", ylabel="доля положительных",
           title="Калибровка (reliability diagram)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=120)
    plt.close(fig)
    return {"brier": None if brier is None else round(brier, 6),
            "ece": expected_calibration_error(hit, conf), "n_bins": 10}


def diagnostic_plots(task, y, pred, proba, classes, fig_dir: Path) -> None:
    if task == "regression":
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
    ax.set_title("Матрица ошибок (test)")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    if task == "binary" and proba is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y, proba[:, 1], pos_label=classes[1], ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y, proba[:, 1], pos_label=classes[1], ax=axes[1])
        axes[0].set_title("ROC")
        axes[1].set_title("Precision–Recall")
        fig.tight_layout()
        fig.savefig(fig_dir / "roc_pr.png", dpi=120)
        plt.close(fig)


def importance(cfg, pipe, X: pd.DataFrame, y: pd.Series, metric: str, task: str,
               fig_dir: Path) -> pd.DataFrame:
    rows = []
    # 1) SHAP-значения: у LightGBM они встроены (pred_contrib=True), пакет shap не нужен
    model = pipe.named_steps["model"]
    if type(model).__module__.startswith("lightgbm"):
        Xt = pipe[:-1].transform(X)
        contrib = model.predict(Xt, pred_contrib=True)
        contrib = contrib.toarray() if hasattr(contrib, "toarray") else np.asarray(contrib)
        names = feature_names(pipe)
        k = len(names) + 1
        if contrib.shape[1] != k:  # мультикласс: блоки по классам
            contrib = contrib.reshape(len(Xt), -1, k).mean(axis=1)
        values = np.abs(contrib[:, :-1]).mean(axis=0)
        rows += [{"feature": n, "method": "shap_mean_abs", "importance": float(v), "std": None}
                 for n, v in zip(names, values, strict=True)]
        try:  # beeswarm, если установлен shap (uv sync --group explain)
            import shap

            shap.summary_plot(contrib[:, :-1], Xt, feature_names=names, show=False,
                              max_display=15)
            plt.tight_layout()
            plt.savefig(fig_dir / "shap_beeswarm.png", dpi=120)
            plt.close("all")
        except Exception:  # shap не установлен или не справился — это необязательный график
            plt.close("all")
    # 2) permutation importance по исходным колонкам — работает для любой модели
    n = min(len(X), 2000)
    Xs, ys = X.iloc[:n], y.iloc[:n]
    scoring = SCORING.get(metric, None)
    if task == "multiclass" and metric == "roc_auc":
        scoring = "roc_auc_ovr"
    perm = permutation_importance(pipe, Xs, ys, scoring=scoring, n_repeats=5,
                                  random_state=cfg["project"]["seed"], n_jobs=1)
    skip = {cfg["data"].get("id_column"), cfg["data"].get("time_column")}
    rows += [{"feature": c, "method": "permutation", "importance": float(m), "std": float(s)}
             for c, m, s in zip(X.columns, perm.importances_mean, perm.importances_std,
                                strict=True) if c not in skip]
    imp = pd.DataFrame(rows).sort_values(["method", "importance"], ascending=[True, False])

    method = "shap_mean_abs" if (imp["method"] == "shap_mean_abs").any() else "permutation"
    top = imp[imp["method"] == method].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set(title=f"Важность признаков ({method})", xlabel="важность")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_importance.png", dpi=120)
    plt.close(fig)
    return imp


def errors_table(cfg, X: pd.DataFrame, y: pd.Series, pred, proba, classes, task) -> pd.DataFrame:
    out = X.copy()
    out["y_true"] = y.to_numpy()
    out["y_pred"] = pred
    if task == "regression":
        out["abs_error"] = np.abs(out["y_true"] - out["y_pred"])
        return out.sort_values("abs_error", ascending=False).head(cfg["evaluation"]["n_errors"])
    if proba is not None:
        out["confidence"] = proba.max(axis=1)
    wrong = out[out["y_true"] != out["y_pred"]]
    if "confidence" in wrong:
        wrong = wrong.sort_values("confidence", ascending=False)
    return wrong.head(cfg["evaluation"]["n_errors"])


def slices(cfg, X, y, pred, proba, task, metric) -> pd.DataFrame:
    col = cfg["evaluation"].get("slice_column")
    if not col or col not in X:
        return pd.DataFrame()
    fn = metric_fn(metric, task)
    rows = []
    for value, idx in X.groupby(col, observed=True).indices.items():
        yi, pi = y.to_numpy()[idx], pred[idx]
        pri = None if proba is None else proba[idx]
        try:
            score = float(fn(yi, pi, pri))
        except ValueError:
            score = float("nan")
        rows.append({"slice_column": col, "value": value, "n": len(idx), metric: round(score, 4)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def update_model_card(card: Path, metrics: dict[str, Any], ev: dict[str, Any]) -> None:
    if not card.exists():
        return
    m = metrics["primary_metric"]
    lines = [CARD_START, "",
             f"Основная метрика: **{m}** на test, {int(ev['level'] * 100)} % ДИ "
             f"(бутстреп, {ev['n_bootstrap']} выборок). Лучшая модель выбрана по val.", "",
             f"| Модель | Семейство | {m} | ДИ |", "|---|---|---|---|"]
    for name, rec in metrics["models"].items():
        ci = ev["models"][name]
        mark = " (лучшая)" if name == metrics["best"] else ""
        lines.append(f"| {name}{mark} | {rec['family']}/{rec['framework']} | {ci['value']:.4f} | "
                     f"[{ci['ci_low']:.4f}; {ci['ci_high']:.4f}] |")
    d = ev["delta_best_vs_baseline"]
    lines += ["", f"Разница {d['best']} − {d['baseline']}: {d['delta']:+.4f} "
                  f"[{d['ci_low']:+.4f}; {d['ci_high']:+.4f}].", ""]
    if ev.get("calibration") and ev["calibration"].get("ece") is not None:
        c = ev["calibration"]
        lines += [f"Калибровка: Brier = {c['brier']}, ECE = {c['ece']}.", ""]
    lines.append(CARD_END)
    text = card.read_text("utf-8")
    block = "\n".join(lines)
    pattern = re.compile(re.escape(CARD_START) + ".*?" + re.escape(CARD_END), re.S)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text + "\n" + block
    card.write_text(text, "utf-8")


def run(cfg: dict[str, Any], p: dict[str, Any], update_card: bool = True) -> dict[str, Any]:
    reports, figs = Path(p["reports"]), Path(p["figures"])
    metrics = json.loads((reports / "metrics.json").read_text("utf-8"))
    preds = pd.read_parquet(reports / "predictions_test.parquet")
    pipe = joblib.load(Path(p["models"]) / "model.joblib")
    test = data_mod.load_splits(cfg, p)["test"]
    X, y = data_mod.xy(test, cfg)
    task, metric = metrics["task"], metrics["primary_metric"]
    best = metrics["best"]
    pred, proba = model_outputs(preds, best)
    classes = list(getattr(pipe, "classes_", []))

    ev = intervals(cfg, metrics, preds)
    ev["best"] = best
    ev["test_metrics_best"] = compute_all(task, y, pred, proba)
    ev["calibration"] = calibration(task, y.to_numpy(), proba, pred, classes, figs)
    diagnostic_plots(task, y, pred, proba, classes, figs)
    imp = importance(cfg, pipe, X, y, metric, task, figs)
    imp.to_csv(reports / "feature_importance.csv", index=False)
    errors_table(cfg, X, y, pred, proba, classes, task).to_csv(reports / "errors.csv", index=False)
    slices(cfg, X, y, pred, proba, task, metric).to_csv(reports / "slices.csv", index=False)
    (reports / "evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2,
                                                        default=str), "utf-8")
    if update_card:
        update_model_card(Path(p["docs"]) / "model_card.md", metrics, ev)
    return ev


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Этап 4: оценка модели")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(smoke=args.smoke)
    p = paths(cfg, args.output_dir)
    ev = run(cfg, p, update_card=args.output_dir is None)
    best = ev["models"][ev["best"]]
    print(f"{ev['best']}: {ev['primary_metric']}={best['value']:.4f} "
          f"[{best['ci_low']:.4f}; {best['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
