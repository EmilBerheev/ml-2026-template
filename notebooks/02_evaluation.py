import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Этап 4 — разбор результатов после uv run python -m mlproject.evaluate.
    # Открыть: uv run marimo edit notebooks/02_evaluation.py
    import json

    import marimo as mo
    import pandas as pd

    from mlproject.config import find_root

    reports = find_root() / "reports"
    return json, mo, pd, reports


@app.cell
def _(json, mo, pd, reports):
    _metrics = json.loads((reports / "metrics.json").read_text("utf-8"))
    _evaluation = json.loads((reports / "evaluation.json").read_text("utf-8"))
    _m = _metrics["primary_metric"]
    _table = pd.DataFrame([
        {"модель": name, "семейство": rec["family"], "framework": rec["framework"],
         f"val {_m}": rec["val"].get(_m), f"test {_m}": rec["test"].get(_m),
         "ДИ": f"[{_evaluation['models'][name]['ci_low']:.3f}; "
               f"{_evaluation['models'][name]['ci_high']:.3f}]"}
        for name, rec in _metrics["models"].items()
    ])
    _d = _evaluation["delta_best_vs_baseline"]
    mo.vstack([
        mo.md(f"## Сравнение моделей (лучшая — **{_metrics['best']}**)"),
        _table,
        mo.md(f"Разница {_d['best']} − {_d['baseline']}: {_d['delta']:+.4f} "
              f"[{_d['ci_low']:+.4f}; {_d['ci_high']:+.4f}]. Если интервал задевает ноль, "
              "преимущество не доказано."),
    ])
    return


@app.cell
def _(mo, reports):
    _figs = [p for p in sorted((reports / "figures").glob("*.png")) if not p.name.startswith("eda_")]
    mo.hstack([mo.image(src=str(p), width=360) for p in _figs], wrap=True)
    return


@app.cell
def _(mo, pd, reports):
    _errors = pd.read_csv(reports / "errors.csv")
    _slices = pd.read_csv(reports / "slices.csv")
    mo.vstack([
        mo.md("## Самые уверенные ошибки — что у них общего?"),
        _errors.head(20),
        mo.md("## Метрика по срезам"),
        _slices,
    ])
    return


if __name__ == "__main__":
    app.run()
