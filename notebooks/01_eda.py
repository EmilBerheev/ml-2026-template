import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Этап 2 — разведочный анализ (вопрос 10).
    # Открыть:      uv run marimo edit notebooks/01_eda.py
    # Как скрипт:   uv run python notebooks/01_eda.py   (сохранит графики в reports/figures)
    # Для Colab:    uv run marimo export ipynb notebooks/01_eda.py -o 01_eda.ipynb
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from mlproject.config import load_config
    from mlproject.data import load_dataset

    return Path, load_config, load_dataset, mo, np, pd, plt


@app.cell
def _(Path, load_config, load_dataset, mo):
    cfg = load_config()
    df = load_dataset(cfg)
    target = cfg["data"]["target"]
    num_cols = [c for c in cfg["features"]["numeric"] if c in df]
    cat_cols = [c for c in cfg["features"]["categorical"] if c in df]
    fig_dir = Path(cfg["_root"]) / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mo.md(
        f"# EDA: {cfg['project']['variant']}\n\n"
        f"Источник: `{df.attrs.get('source')}`, {len(df)} строк, {df.shape[1]} колонок."
    )
    return cat_cols, cfg, df, fig_dir, num_cols, target


@app.cell
def _(df, pd):
    # Паспорт колонок: тип, пропуски, уникальные значения, пример
    _overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_share": df.isna().mean().round(4),
        "n_unique": df.nunique(),
        "example": df.iloc[0],
    })
    _overview
    return


@app.cell
def _(cfg, df, fig_dir, mo, plt, target):
    # Баланс классов (или распределение target для регрессии)
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    if cfg["task"]["type"] == "regression":
        df[target].plot.hist(bins=40, ax=_ax)
    else:
        df[target].value_counts(normalize=True).sort_index().plot.bar(ax=_ax)
        _ax.set_ylabel("доля")
    _ax.set_title(f"Распределение {target}")
    _fig.tight_layout()
    _fig.savefig(fig_dir / "eda_target.png", dpi=120)
    mo.vstack([mo.md(f"Дубликатов строк: **{int(df.duplicated().sum())}**"), _fig])
    return


@app.cell
def _(df, fig_dir, np, num_cols, plt, target):
    # Распределения числовых признаков в разрезе target
    _n = len(num_cols)
    _cols = 3
    _rows = max(1, int(np.ceil(_n / _cols)))
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(12, 3 * _rows), squeeze=False)
    for _ax, _col in zip(_axes.ravel(), num_cols):
        for _value, _part in df.groupby(target):
            _ax.hist(_part[_col].dropna(), bins=30, alpha=0.5, density=True, label=str(_value))
        _ax.set_title(_col)
    for _ax in _axes.ravel()[_n:]:
        _ax.axis("off")
    _axes[0, 0].legend(title=target)
    _fig.tight_layout()
    _fig.savefig(fig_dir / "eda_numeric.png", dpi=120)
    _fig
    return


@app.cell
def _(cat_cols, df, fig_dir, np, plt, target):
    # Доля положительного класса по категориям: где модель найдёт сигнал
    _cols = 2
    _rows = max(1, int(np.ceil(len(cat_cols) / _cols)))
    _fig, _axes = plt.subplots(_rows, _cols, figsize=(11, 3.2 * _rows), squeeze=False)
    for _ax, _col in zip(_axes.ravel(), cat_cols):
        df.groupby(_col, observed=True)[target].mean().sort_values().plot.barh(ax=_ax)
        _ax.set_title(f"среднее {target} по {_col}")
    for _ax in _axes.ravel()[len(cat_cols):]:
        _ax.axis("off")
    _fig.tight_layout()
    _fig.savefig(fig_dir / "eda_categorical.png", dpi=120)
    _fig
    return


@app.cell
def _(df, fig_dir, num_cols, plt, target):
    # Корреляции (Спирмен — устойчивее к выбросам). Подозрительно высокая связь с target —
    # повод проверить признак на утечку.
    corr = df[[*num_cols, target]].corr(method="spearman")
    _fig, _ax = plt.subplots(figsize=(6.5, 5.5))
    _im = _ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    _ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    _ax.set_yticks(range(len(corr)), corr.columns)
    for _i in range(len(corr)):
        for _j in range(len(corr)):
            _ax.text(_j, _i, f"{corr.iloc[_i, _j]:.2f}", ha="center", va="center", fontsize=7)
    _fig.colorbar(_im)
    _ax.set_title("Корреляции Спирмена")
    _fig.tight_layout()
    _fig.savefig(fig_dir / "eda_correlation.png", dpi=120)
    _fig
    return (corr,)


@app.cell
def _(cfg, df, fig_dir, pd, plt, target):
    # Target во времени: есть ли тренд или сезонность (тогда нужен временной split)
    _tcol = cfg["data"].get("time_column")
    if _tcol and _tcol in df:
        _monthly = (df.assign(_month=pd.to_datetime(df[_tcol]).dt.to_period("M"))
                    .groupby("_month")[target].mean())
        _fig, _ax = plt.subplots(figsize=(9, 3.2))
        _monthly.plot(ax=_ax)
        _ax.set_title(f"среднее {target} по месяцам ({_tcol})")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "eda_target_time.png", dpi=120)
        _out = _fig
    else:
        _out = "Колонки времени нет — график пропущен."
    _out
    return


@app.cell
def _(corr, mo, target):
    mo.md(
        "## Что записать в README (этап 2)\n\n"
        "1. Баланс классов и что он значит для метрики.\n"
        "2. Признаки с пропусками и как вы их заполняете.\n"
        "3. Самые сильные связи с target "
        f"(здесь: {', '.join(corr[target].drop(target).abs().sort_values()[::-1].index[:3])}).\n"
        "4. Признаки-подозреваемые в утечке.\n"
        "5. Есть ли тренд во времени и какое разбиение выбрать.\n"
    )
    return


if __name__ == "__main__":
    app.run()
