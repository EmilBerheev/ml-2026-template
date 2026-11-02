"""Генератор демо-датасета «отток клиентов связи» (синтетика, CC0).

Нужен, чтобы шаблон работал сразу после клонирования. Для сдачи замените его своим вариантом.

    uv run python -m mlproject.make_sample                       # data/sample.csv, 5000 строк
    uv run python -m mlproject.make_sample --rows 20000 --out data/raw/dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACTS = ["month-to-month", "one-year", "two-year"]
PAYMENTS = ["card", "bank_transfer", "e_wallet", "cash"]
INTERNET = ["dsl", "fiber", "none"]
REGIONS = ["center", "north", "south", "east", "west"]


def generate(rows: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp("2025-07-01")
    start = pd.Timestamp("2021-01-01")
    days = rng.integers(0, (end - start).days, size=rows)
    signup = start + pd.to_timedelta(days, unit="D")
    tenure = ((end - signup).days // 30).astype(int)

    contract = rng.choice(CONTRACTS, size=rows, p=[0.55, 0.25, 0.20])
    payment = rng.choice(PAYMENTS, size=rows, p=[0.40, 0.25, 0.25, 0.10])
    internet = rng.choice(INTERNET, size=rows, p=[0.35, 0.45, 0.20])
    region = rng.choice(REGIONS, size=rows, p=[0.30, 0.15, 0.20, 0.20, 0.15])
    num_services = rng.integers(0, 7, size=rows)

    base = np.select([internet == "fiber", internet == "dsl"], [70.0, 45.0], default=20.0)
    monthly = np.round(base + num_services * 6 + rng.normal(0, 8, rows), 2).clip(10, 180)
    support = rng.poisson(np.where(internet == "fiber", 1.6, 0.9))
    total = np.round(monthly * np.maximum(tenure, 1) * rng.uniform(0.9, 1.1, rows), 2)

    # Зависимость нарочно нелинейная: пороги и взаимодействия, которые линейная модель
    # улавливает хуже бустинга. Так видно, зачем нужен этап 3.
    logit = (
        -3.1
        + 1.0 * (contract == "month-to-month")
        - 1.0 * (contract == "two-year")
        + 2.6 * (tenure < 6)
        + 0.8 * ((tenure >= 6) & (tenure < 12))
        + 2.8 * ((internet == "fiber") & (support >= 3))
        + 2.0 * (monthly > 110)
        + 0.6 * (payment == "e_wallet")
        - 1.0 * ((num_services >= 3) & (contract != "month-to-month"))
        + 1.0 * ((region == "south") & (contract == "month-to-month"))
        + rng.normal(0, 0.3, rows)
    )
    churn = (rng.uniform(size=rows) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, rows + 1)],
            "signup_date": signup.strftime("%Y-%m-%d"),
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "region": region,
            "support_calls": support,
            "num_services": num_services,
            "churn": churn,
        }
    )
    # немного реальной грязи: пропуски в total_charges
    missing = rng.uniform(size=rows) < 0.015
    df.loc[missing, "total_charges"] = np.nan
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/sample.csv")
    args = parser.parse_args(argv)
    df = generate(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{out}: {len(df)} строк, доля оттока {df['churn'].mean():.3f}")


if __name__ == "__main__":
    main()
