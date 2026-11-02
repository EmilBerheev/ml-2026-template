"""Загрузка конфигурации и путей проекта."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = Path("configs") / "config.yaml"
SCHEMA_NAME = Path("configs") / "schema.yaml"


def find_root(start: Path | None = None) -> Path:
    """Корень проекта: переменная MLPROJECT_ROOT или ближайшая папка с configs/config.yaml."""
    env = os.environ.get("MLPROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).exists():
            return folder
    raise FileNotFoundError("Не нашёл configs/config.yaml — запускайте команды из корня проекта")


def load_config(path: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    root = find_root()
    cfg_path = Path(path) if path else root / CONFIG_NAME
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(root)
    cfg["_smoke"] = smoke
    if smoke:
        cfg = apply_smoke(cfg)
    return cfg


def apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Урезает объём работы для CI: меньше строк, проб Optuna, деревьев и эпох."""
    cfg = copy.deepcopy(cfg)
    smoke = cfg.get("smoke", {})
    cfg["optuna"]["n_trials"] = smoke.get("n_trials", 3)
    cfg["optuna"]["timeout_seconds"] = 60
    cfg["models"]["boosting"]["n_estimators"] = smoke.get("n_estimators", 60)
    cfg["models"]["nn"]["epochs"] = smoke.get("epochs", 5)
    cfg["evaluation"]["n_bootstrap"] = smoke.get("n_bootstrap", 100)
    cfg["mlflow"]["log_model"] = smoke.get("log_model", False)
    return cfg


def load_schema() -> dict[str, Any]:
    with open(find_root() / SCHEMA_NAME, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def paths(cfg: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Path | str]:
    """Куда писать артефакты. С output_dir всё уходит туда (так грейдер не трогает ваши отчёты)."""
    root = Path(cfg["_root"])
    base = Path(output_dir).resolve() if output_dir else root
    p: dict[str, Path | str] = {
        "root": root,
        "base": base,
        "reports": base / "reports",
        "figures": base / "reports" / "figures",
        "models": base / "models",
        "processed": base / "data" / "processed",
        "docs": root / "docs",
    }
    if output_dir:
        p["mlflow_uri"] = "sqlite:///" + (base / "mlflow.db").as_posix()
    else:
        uri = cfg["mlflow"]["tracking_uri"]
        if uri.startswith("sqlite:///") and not Path(uri.removeprefix("sqlite:///")).is_absolute():
            uri = "sqlite:///" + (root / uri.removeprefix("sqlite:///")).as_posix()
        p["mlflow_uri"] = uri
    for key in ("reports", "figures", "models", "processed"):
        Path(p[key]).mkdir(parents=True, exist_ok=True)
    return p
