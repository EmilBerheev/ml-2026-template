"""Этап 2: загрузка, очистка и разбиение данных.

    uv run python -m mlproject.data            # полный датасет (или sample, если полного нет)
    uv run python -m mlproject.data --smoke    # быстрый прогон на части sample

Результат: data/processed/{train,val,test}.parquet и reports/split_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from mlproject.config import load_config, paths

log = logging.getLogger("mlproject.data")
SPLITS = ("train", "val", "test")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_dataset(cfg: dict[str, Any], smoke: bool | None = None) -> pd.DataFrame:
    """Полный датасет, если он скачан; иначе data/sample.csv. В smoke-режиме — часть sample."""
    smoke = cfg.get("_smoke", False) if smoke is None else smoke
    root = Path(cfg["_root"])
    full = root / cfg["data"]["full_path"]
    sample = root / cfg["data"]["sample_path"]
    if not smoke and full.exists():
        df = read_table(full)
        source = full
    else:
        if not smoke:
            log.warning("полный датасет %s не найден — работаю на %s", full, sample)
        df = read_table(sample)
        source = sample
        rows = cfg.get("smoke", {}).get("rows")
        if smoke and rows and len(df) > rows:
            df = df.sample(n=rows, random_state=cfg["project"]["seed"])
    df.attrs["source"] = str(source.relative_to(root))
    return df


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Убирает дубликаты и строки без target. Всё, что удалено, попадает в отчёт."""
    d = cfg["data"]
    report = {"rows_in": len(df)}
    out = df.drop_duplicates()
    report["exact_duplicates"] = report["rows_in"] - len(out)
    if d.get("id_column") and d["id_column"] in out:
        before = len(out)
        out = out.drop_duplicates(subset=d["id_column"], keep="first")
        report["duplicate_ids"] = before - len(out)
    before = len(out)
    out = out[out[d["target"]].notna()]
    report["missing_target"] = before - len(out)
    if d.get("time_column") and d["time_column"] in out:
        out = out.assign(**{d["time_column"]: pd.to_datetime(out[d["time_column"]])})
    if d.get("drop_columns"):
        out = out.drop(columns=[c for c in d["drop_columns"] if c in out])
    report["rows_out"] = len(out)
    return out.reset_index(drop=True), report


def split(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """train / val / test по стратегии из config.yaml: random, stratified, group или time."""
    s, d = cfg["split"], cfg["data"]
    seed = cfg["project"]["seed"]
    val_size, test_size = s["val_size"], s["test_size"]
    strategy = s["strategy"]
    if strategy == "time":
        ordered = df.sort_values(d["time_column"], kind="stable")
        n = len(ordered)
        n_test, n_val = int(n * test_size), int(n * val_size)
        train = ordered.iloc[: n - n_val - n_test]
        val = ordered.iloc[n - n_val - n_test : n - n_test]
        test = ordered.iloc[n - n_test :]
    elif strategy == "group":
        groups = df[d["group_column"]]
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        rest_idx, test_idx = next(gss.split(df, groups=groups))
        rest, test = df.iloc[rest_idx], df.iloc[test_idx]
        gss = GroupShuffleSplit(
            n_splits=1, test_size=val_size / (1 - test_size), random_state=seed
        )
        train_idx, val_idx = next(gss.split(rest, groups=rest[d["group_column"]]))
        train, val = rest.iloc[train_idx], rest.iloc[val_idx]
    elif strategy in ("random", "stratified"):
        strat = df[d["target"]] if strategy == "stratified" else None
        rest, test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=strat
        )
        strat = rest[d["target"]] if strategy == "stratified" else None
        train, val = train_test_split(
            rest, test_size=val_size / (1 - test_size), random_state=seed, stratify=strat
        )
    else:
        raise ValueError(f"неизвестная стратегия разбиения: {strategy}")
    return {"train": train.copy(), "val": val.copy(), "test": test.copy()}


def leakage_report(splits: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> dict[str, Any]:
    """Пересечения ID и групп между частями, порядок во времени. Всё должно быть нулевым."""
    d = cfg["data"]
    rep: dict[str, Any] = {}
    for key in ("id_column", "group_column"):
        col = d.get(key)
        if not col or col not in splits["train"]:
            continue
        sets = {name: set(part[col]) for name, part in splits.items()}
        rep[f"{key}_overlap"] = {
            "train_val": len(sets["train"] & sets["val"]),
            "train_test": len(sets["train"] & sets["test"]),
            "val_test": len(sets["val"] & sets["test"]),
        }
    tcol = d.get("time_column")
    if cfg["split"]["strategy"] == "time" and tcol:
        rep["time_order_ok"] = bool(
            splits["train"][tcol].max() <= splits["val"][tcol].min()
            and splits["val"][tcol].max() <= splits["test"][tcol].min()
        )
    rep["ok"] = all(
        v == 0 for k, ov in rep.items() if k.endswith("_overlap") for v in ov.values()
    ) and rep.get("time_order_ok", True)
    return rep


def summarize(
    splits: dict[str, pd.DataFrame], cfg: dict[str, Any], cleaning: dict[str, int], source: str
) -> dict[str, Any]:
    target = cfg["data"]["target"]
    summary: dict[str, Any] = {
        "dataset": cfg["project"]["variant"],
        "source": source,
        "strategy": cfg["split"]["strategy"],
        "seed": cfg["project"]["seed"],
        "smoke": cfg.get("_smoke", False),
        "cleaning": cleaning,
        "sizes": {name: len(part) for name, part in splits.items()},
        "leakage": leakage_report(splits, cfg),
    }
    if cfg["task"]["type"] != "regression":
        summary["target_share"] = {
            name: {str(k): round(float(v), 4) for k, v in
                   part[target].value_counts(normalize=True).sort_index().items()}
            for name, part in splits.items()
        }
    else:
        summary["target_mean"] = {n: float(p[target].mean()) for n, p in splits.items()}
    return summary


def prepare(cfg: dict[str, Any], p: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Полный цикл этапа 2: загрузить → очистить → разбить → сохранить → отчёт."""
    raw = load_dataset(cfg)
    df, cleaning = clean(raw, cfg)
    splits = split(df, cfg)
    for name, part in splits.items():
        part.to_parquet(Path(p["processed"]) / f"{name}.parquet", index=False)
    summary = summarize(splits, cfg, cleaning, raw.attrs.get("source", "?"))
    out = Path(p["reports"]) / "split_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), "utf-8")
    if not summary["leakage"]["ok"]:
        log.error("разбиение с утечкой: %s", summary["leakage"])
    return splits


def load_splits(cfg: dict[str, Any], p: dict[str, Any]) -> dict[str, pd.DataFrame]:
    folder = Path(p["processed"])
    if all((folder / f"{n}.parquet").exists() for n in SPLITS) and not cfg.get("_smoke"):
        return {n: pd.read_parquet(folder / f"{n}.parquet") for n in SPLITS}
    return prepare(cfg, p)


def xy(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg["data"]["target"]
    return df.drop(columns=[target]), df[target]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Этап 2: подготовка разбиений")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(smoke=args.smoke)
    p = paths(cfg, args.output_dir)
    splits = prepare(cfg, p)
    sizes = {n: len(s) for n, s in splits.items()}
    print(f"готово: {sizes}; отчёт {Path(p['reports']) / 'split_summary.json'}")


if __name__ == "__main__":
    main()
