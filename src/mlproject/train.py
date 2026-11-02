"""Этап 3: обучение и сравнение моделей с трекингом в MLflow.

    uv run python -m mlproject.train            # полный прогон
    uv run python -m mlproject.train --smoke    # быстрый прогон для CI (десятки секунд)

Что получится:
    models/model.joblib, models/model_meta.json   — лучшая модель (по val) и её паспорт
    reports/metrics.json                          — метрики всех моделей на val и test
    reports/optuna_trials.csv, reports/mlflow_runs.csv
    reports/predictions_test.parquet              — предсказания на test для этапа 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from mlproject import data as data_mod  # noqa: E402
from mlproject.config import load_config, paths  # noqa: E402
from mlproject.features import make_pipeline  # noqa: E402
from mlproject.metrics import HIGHER_IS_BETTER, compute_all  # noqa: E402
from mlproject.models import (  # noqa: E402
    boosting_space,
    make_baseline,
    make_boosting,
    make_dummy,
    make_nn_model,
)

log = logging.getLogger("mlproject.train")


def predict(pipe, X: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """Предсказания и вероятности. Для бинарной задачи класс выбирается по decision_threshold."""
    task = cfg["task"]["type"]
    if task == "regression":
        return pipe.predict(X), None
    proba = pipe.predict_proba(X)
    if task == "binary":
        classes = pipe.classes_
        thr = cfg["task"].get("decision_threshold", 0.5)
        pred = np.where(proba[:, 1] >= thr, classes[1], classes[0])
        return pred, proba
    return pipe.predict(X), proba


def better(a: float, b: float, metric: str) -> bool:
    return a > b if HIGHER_IS_BETTER.get(metric, True) else a < b


class Trainer:
    def __init__(self, cfg: dict[str, Any], p: dict[str, Any]):
        self.cfg, self.p = cfg, p
        self.task = cfg["task"]["type"]
        self.metric = cfg["metric"]["primary"]
        self.seed = cfg["project"]["seed"]
        self.splits = data_mod.prepare(cfg, p)
        self.X = {}
        self.y = {}
        for name, part in self.splits.items():
            self.X[name], self.y[name] = data_mod.xy(part, cfg)
        self.records: dict[str, dict[str, Any]] = {}
        self.pipes: dict[str, Any] = {}
        self.test_preds: dict[str, tuple] = {}

    def fit_eval(self, name: str, estimator, family: str, framework: str,
                 params: dict[str, Any], log_run: bool = True) -> dict[str, Any]:
        import mlflow

        pipe = make_pipeline(self.cfg, estimator)
        start = time.perf_counter()
        pipe.fit(self.X["train"], self.y["train"])
        fit_seconds = round(time.perf_counter() - start, 3)
        rec: dict[str, Any] = {"family": family, "framework": framework,
                               "params": params, "fit_seconds": fit_seconds}
        for part in ("val", "test"):
            pred, proba = predict(pipe, self.X[part], self.cfg)
            rec[part] = compute_all(self.task, self.y[part], pred, proba)
            if part == "test":
                self.test_preds[name] = (pred, proba)
        if log_run:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"model_name": name, "family": family, "framework": framework})
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                mlflow.log_metric("fit_seconds", fit_seconds)
                for part in ("val", "test"):
                    mlflow.log_metrics({f"{part}_{k}": v for k, v in rec[part].items()})
        self.records[name] = rec
        self.pipes[name] = pipe
        log.info("%-9s val %s=%.4f  test %s=%.4f  (%.1f c)", name, self.metric,
                 rec["val"].get(self.metric, float("nan")), self.metric,
                 rec["test"].get(self.metric, float("nan")), fit_seconds)
        return rec

    def tune_boosting(self) -> tuple[dict[str, Any], pd.DataFrame]:
        import mlflow
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        base = dict(self.cfg["models"]["boosting"])
        direction = "maximize" if HIGHER_IS_BETTER.get(self.metric, True) else "minimize"

        def objective(trial: optuna.Trial) -> float:
            params = {**base, **boosting_space(trial)}
            pipe = make_pipeline(self.cfg, make_boosting(self.task, params, self.seed))
            pipe.fit(self.X["train"], self.y["train"])
            pred, proba = predict(pipe, self.X["val"], self.cfg)
            value = compute_all(self.task, self.y["val"], pred, proba)[self.metric]
            with mlflow.start_run(run_name=f"optuna-{trial.number:03d}", nested=True):
                mlflow.set_tags({"family": "boosting", "optuna_trial": trial.number})
                mlflow.log_params(trial.params)
                mlflow.log_metric(f"val_{self.metric}", value)
            return value

        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=self.seed),
            study_name=f"{self.cfg['project']['name']}-boosting",
        )
        o = self.cfg["optuna"]
        study.optimize(objective, n_trials=o["n_trials"], timeout=o.get("timeout_seconds"))
        log.info("Optuna: %d проб, лучшее val %s=%.4f", len(study.trials), self.metric,
                 study.best_value)
        return {**base, **study.best_params, "subsample_freq": 1}, study.trials_dataframe()

    def run(self) -> dict[str, Any]:
        import mlflow

        cfg, p = self.cfg, self.p
        mlflow.set_tracking_uri(p["mlflow_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        with mlflow.start_run(run_name=f"train-{stamp}") as parent:
            mlflow.set_tags({"variant": cfg["project"]["variant"], "smoke": cfg["_smoke"],
                             "task": self.task})
            mlflow.log_params({"split": cfg["split"]["strategy"], "seed": self.seed,
                               "primary_metric": self.metric,
                               **{f"n_{k}": len(v) for k, v in self.splits.items()}})

            self.fit_eval("dummy", make_dummy(self.task), "dummy", "sklearn", {})
            bl = cfg["models"]["baseline"]
            self.fit_eval("baseline", make_baseline(self.task, bl, self.seed), "linear",
                          "sklearn", bl)
            best_params, trials = self.tune_boosting()
            self.fit_eval("boosting", make_boosting(self.task, best_params, self.seed),
                          "boosting", cfg["models"]["boosting"].get("library", "lightgbm"),
                          best_params)
            nn_cfg = cfg["models"]["nn"]
            nn_model, framework = make_nn_model(self.task, nn_cfg, self.seed)
            if framework != "torch":
                log.warning("PyTorch не найден — нейросеть обучена через sklearn MLP. "
                            "Для зачёта этапа 3 нужен PyTorch: uv sync --group torch")
            self.fit_eval("nn", nn_model, "nn", framework, nn_cfg)

            candidates = [n for n in self.records if n != "dummy"]
            best = candidates[0]
            for name in candidates[1:]:
                if better(self.records[name]["val"][self.metric],
                          self.records[best]["val"][self.metric], self.metric):
                    best = name
            mlflow.set_tag("best_model", best)
            mlflow.log_metric(f"best_test_{self.metric}", self.records[best]["test"][self.metric])
            result = self.save(best, trials, parent.info.run_id)
            if cfg["mlflow"].get("log_model"):
                self.log_model(best)
        self.export_runs()
        return result

    def save(self, best: str, trials: pd.DataFrame, run_id: str) -> dict[str, Any]:
        cfg, p = self.cfg, self.p
        reports, models = Path(p["reports"]), Path(p["models"])
        pipe = self.pipes[best]
        joblib.dump(pipe, models / "model.joblib")
        now = datetime.now(UTC)
        meta = {
            "model_version": f"{cfg['project']['name']}-{best}-{now:%Y%m%d%H%M}",
            "model_name": best,
            "family": self.records[best]["family"],
            "framework": self.records[best]["framework"],
            "task": self.task,
            "target": cfg["data"]["target"],
            "input_columns": [c for c in self.X["train"].columns
                              if c not in (cfg["data"].get("id_column"),
                                           cfg["data"].get("time_column"))],
            "classes": [v.item() if hasattr(v, "item") else v
                        for v in getattr(pipe, "classes_", [])],
            "decision_threshold": cfg["task"].get("decision_threshold"),
            "test_metrics": self.records[best]["test"],
            "trained_at": now.isoformat(timespec="seconds"),
            "mlflow_run_id": run_id,
            "sklearn_version": sklearn.__version__,
            "smoke": cfg["_smoke"],
        }
        (models / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                "utf-8")
        metrics = {
            "schema_version": 1,
            "dataset": cfg["project"]["variant"],
            "track": cfg["project"].get("track", "classic"),
            "task": self.task,
            "primary_metric": self.metric,
            "higher_is_better": HIGHER_IS_BETTER.get(self.metric, True),
            "success_threshold": cfg["metric"].get("threshold"),
            "split": {"strategy": cfg["split"]["strategy"],
                      **{k: len(v) for k, v in self.splits.items()}},
            "smoke": cfg["_smoke"],
            "baseline": "baseline",
            "best": best,
            "selected_by": f"val_{self.metric}",
            "models": self.records,
            "mlflow_run_id": run_id,
            "generated_at": now.isoformat(timespec="seconds"),
        }
        (reports / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                                         default=str), "utf-8")
        trials.to_csv(reports / "optuna_trials.csv", index=False)
        self.save_predictions(reports / "predictions_test.parquet")
        log.info("лучшая модель: %s; артефакты в %s", best, p["base"])
        return metrics

    def save_predictions(self, path: Path) -> None:
        id_col = self.cfg["data"].get("id_column")
        out = pd.DataFrame({"y_true": self.y["test"].to_numpy()})
        if id_col and id_col in self.X["test"]:
            out.insert(0, id_col, self.X["test"][id_col].to_numpy())
        for name, (pred, proba) in self.test_preds.items():
            out[f"{name}__pred"] = pred
            if proba is not None:
                for j, cls in enumerate(self.pipes[name].classes_):
                    out[f"{name}__proba__{cls}"] = proba[:, j]
        out.to_parquet(path, index=False)

    def log_model(self, best: str) -> None:
        import mlflow

        try:
            mlflow.log_artifact(str(Path(self.p["models"]) / "model_meta.json"))
            # MLflow 3 сохраняет sklearn-модели в формате skops и требует явно перечислить
            # «доверенные» типы: pickle из чужих рук может выполнить произвольный код.
            # Здесь доверяем только тому, что обучили сами.
            import skops.io as sio

            pipe = self.pipes[best]
            trusted = sio.get_untrusted_types(data=sio.dumps(pipe))
            mlflow.sklearn.log_model(
                sk_model=pipe, name="model", skops_trusted_types=trusted,
                registered_model_name=self.cfg["mlflow"]["experiment"],
            )
        except Exception as exc:  # регистрация модели — приятный бонус, а не повод падать
            log.warning("не удалось записать модель в MLflow: %s", exc)

    def export_runs(self) -> None:
        import mlflow

        runs = mlflow.search_runs(experiment_names=[self.cfg["mlflow"]["experiment"]])
        keep = [c for c in runs.columns if c in ("run_id", "status", "start_time")
                or c.startswith(("params.", "metrics.", "tags.mlflow.runName",
                                 "tags.mlflow.parentRunId", "tags.family", "tags.framework"))]
        runs[keep].to_csv(Path(self.p["reports"]) / "mlflow_runs.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Этап 3: обучение моделей")
    ap.add_argument("--smoke", action="store_true", help="быстрый прогон для CI")
    ap.add_argument("--output-dir", default=None, help="куда писать артефакты")
    args = ap.parse_args(argv)
    cfg = load_config(smoke=args.smoke)
    p = paths(cfg, args.output_dir)
    result = Trainer(cfg, p).run()
    best = result["best"]
    print(f"best={best} test {result['primary_metric']}="
          f"{result['models'][best]['test'][result['primary_metric']]:.4f}")


if __name__ == "__main__":
    main()
