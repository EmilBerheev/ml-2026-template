"""Этап 6: отчёт о дрейфе данных и предсказаний (Evidently 0.7, вопрос 35).

    uv run python -m mlproject.monitor                     # сдвиг из config.yaml → monitoring.shift
    uv run python -m mlproject.monitor --current logs/predictions.jsonl   # журнал сервиса
    uv run python -m mlproject.monitor --smoke --output-dir .cache/smoke

Что получится: reports/drift_report.html (откройте в браузере) и reports/drift_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mlproject import data as data_mod
from mlproject.config import load_config, paths
from mlproject.metrics import HIGHER_IS_BETTER, compute_all
from mlproject.train import predict

log = logging.getLogger("mlproject.monitor")


def simulate_shift(df: pd.DataFrame, rules: list[dict[str, Any]], seed: int) -> pd.DataFrame:
    """Портит копию данных так, как это могло бы случиться в жизни."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for r in rules or []:
        col, kind = r["column"], r["kind"]
        if col not in out:
            log.warning("колонки %s нет — правило сдвига пропущено", col)
            continue
        if kind == "scale":  # инфляция, смена валюты, другая единица измерения
            out[col] = out[col] * r["factor"]
        elif kind == "add":
            out[col] = out[col] + r["value"]
        elif kind == "noise":  # деградация датчика
            out[col] = out[col] + rng.normal(0, out[col].std() * r.get("sigma", 0.5), len(out))
        elif kind == "missing":  # сломалась интеграция с источником
            out.loc[rng.uniform(size=len(out)) < r["share"], col] = np.nan
        elif kind == "set_share":  # поменялся состав клиентов
            mask = rng.uniform(size=len(out)) < r["share"]
            out.loc[mask, col] = r["value"]
        else:
            raise ValueError(f"неизвестный вид сдвига: {kind}")
    return out


def _drift_columns(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    columns, overall = [], {}
    for m in snapshot.get("metrics", []):
        conf = m.get("config", {})
        kind = conf.get("type", "")
        if kind.endswith("DriftedColumnsCount"):
            overall = {"count": m["value"]["count"], "share": m["value"]["share"],
                       "drift_share_threshold": conf.get("drift_share")}
        elif kind.endswith("ValueDrift"):
            method, thr, value = conf.get("method", ""), conf.get("threshold"), m.get("value")
            drifted = value < thr if "p_value" in method else value >= thr
            columns.append({"column": conf.get("column"), "method": method,
                            "threshold": thr, "value": value, "drifted": bool(drifted)})
    return columns, overall


def run(cfg: dict[str, Any], p: dict[str, Any], current_path: str | None = None) -> dict[str, Any]:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    mon = cfg["monitoring"]
    task, metric = cfg["task"]["type"], cfg["metric"]["primary"]
    model = joblib.load(Path(p["models"]) / "model.joblib")
    splits = data_mod.load_splits(cfg, p)
    reference = splits[mon.get("reference", "train")]
    clean_test = splits["test"]
    if current_path:
        current = (pd.read_csv(current_path) if current_path.endswith(".csv")
                   else pd.read_json(current_path, lines=True))
        shift = {"source": current_path}
    else:
        current = simulate_shift(clean_test, mon.get("shift"), cfg["project"]["seed"])
        shift = {"source": "simulated", "rules": mon.get("shift")}

    num = [c for c in cfg["features"]["numeric"] if c in reference and c in current]
    cat = [c for c in cfg["features"]["categorical"] if c in reference and c in current]
    cols = num + cat

    def with_prediction(df: pd.DataFrame) -> pd.DataFrame:
        out = df[cols].copy()
        pred, proba = predict(model, df, cfg)
        out["prediction"] = proba[:, 1] if task == "binary" else pred
        return out

    ref_df, cur_df = with_prediction(reference), with_prediction(current)
    pred_is_num = task in ("binary", "regression")
    definition = DataDefinition(
        numerical_columns=num + (["prediction"] if pred_is_num else []),
        categorical_columns=cat + ([] if pred_is_num else ["prediction"]),
    )
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        current_data=Dataset.from_pandas(cur_df, data_definition=definition),
        reference_data=Dataset.from_pandas(ref_df, data_definition=definition),
    )
    reports = Path(p["reports"])
    snapshot.save_html(str(reports / "drift_report.html"))
    columns, overall = _drift_columns(snapshot.dict())

    quality = None
    target = cfg["data"]["target"]
    if target in current:
        def score(df):
            pr, pb = predict(model, df, cfg)
            return compute_all(task, df[target], pr, pb).get(metric)

        before, after = score(clean_test), score(current)
        drop = (before - after) if HIGHER_IS_BETTER.get(metric, True) else (after - before)
        quality = {"metric": metric, "reference": round(before, 4), "current": round(after, 4),
                   "drop": round(drop, 4)}

    feature_cols = [c for c in columns if c["column"] != "prediction"]
    drifted = [c["column"] for c in feature_cols if c["drifted"]]
    share = len(drifted) / max(1, len(feature_cols))
    alerts = []
    if share >= mon["alerts"]["drift_share"]:
        alerts.append(f"дрейф в {len(drifted)} из {len(feature_cols)} признаков ({share:.0%})")
    if quality and quality["drop"] >= mon["alerts"]["metric_drop"]:
        alerts.append(f"{metric} упала на {quality['drop']:.3f}")
    import evidently

    summary = {
        "evidently_version": evidently.__version__,
        "reference": mon.get("reference", "train"),
        "reference_rows": len(ref_df),
        "current_rows": len(cur_df),
        "shift": shift,
        "columns_checked": len(feature_cols),
        "drifted_columns": drifted,
        "share_drifted": round(share, 4),
        "dataset_drift": share >= mon["alerts"]["drift_share"],
        "prediction_drift": any(c["drifted"] for c in columns if c["column"] == "prediction"),
        "evidently_overall": overall,
        "columns": columns,
        "quality": quality,
        "alerts": alerts,
        "action": "переобучить модель" if alerts else "наблюдать",
        "smoke": cfg.get("_smoke", False),
    }
    (reports / "drift_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), "utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Этап 6: отчёт о дрейфе")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--current", default=None, help="CSV/JSONL с текущими данными")
    args = ap.parse_args(argv)
    cfg = load_config(smoke=args.smoke)
    p = paths(cfg, args.output_dir)
    s = run(cfg, p, args.current)
    print(f"дрейф: {len(s['drifted_columns'])}/{s['columns_checked']} признаков "
          f"{s['drifted_columns']}; тревоги: {s['alerts'] or 'нет'}")


if __name__ == "__main__":
    main()
