# data/

- `sample.csv` — маленькая выборка (до 5 МБ), на ней идут тесты и CI. Коммитится.
- `raw/` — полный датасет. Не коммитится: как его получить, написано в `docs/data_card.md`.
- `processed/` — разбиения train/val/test, их создаёт `uv run python -m mlproject.data`.

В шаблоне `sample.csv` — синтетика «отток клиентов связи» (CC0), её генерирует
`uv run python -m mlproject.make_sample`. Замените её выборкой из своего варианта:

```bash
uv run python -c "import pandas as pd; df = pd.read_csv('data/raw/dataset.csv'); \
df.sample(n=min(len(df), 5000), random_state=42).to_csv('data/sample.csv', index=False)"
```

Для классификации с редким классом берите стратифицированную выборку, иначе в CI
в sample может не оказаться ни одного положительного примера.
