"""Independently computed ground truth for the real datasets.

Everything here is calculated with **pandas**, deliberately not with the DuckDB
path the application uses. That makes the evaluation a genuine cross-check: if the
agent's SQL and this pandas implementation disagree, one of them is wrong and the
eval fails. A ground truth computed by the system under test proves nothing.

All figures come from the committed real files. No expected value is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RETAIL_CSV = DATA_DIR / "online_retail_ii_international.csv"
COUNTRY_CSV = DATA_DIR / "world_bank_country_profile.csv"


@lru_cache(maxsize=1)
def retail() -> pd.DataFrame:
    """The real transaction file, with the one derived column the data lacks."""
    frame = pd.read_csv(RETAIL_CSV)
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    frame["invoicedate"] = pd.to_datetime(frame["invoicedate"])
    frame["line_revenue"] = frame["quantity"] * frame["price"]
    return frame


@lru_cache(maxsize=1)
def countries() -> pd.DataFrame:
    return pd.read_csv(COUNTRY_CSV)


@dataclass(frozen=True)
class Truth:
    """One verified fact plus how it was derived, for the eval report."""

    key: str
    value: Any
    derivation: str


def _truth(key: str, value: Any, derivation: str) -> Truth:
    return Truth(key=key, value=value, derivation=derivation)


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def top_country_by_revenue() -> Truth:
    totals = retail().groupby("country")["line_revenue"].sum().sort_values(ascending=False)
    return _truth(
        "top_country_by_revenue",
        {"country": totals.index[0], "revenue": round(float(totals.iloc[0]), 2)},
        "pandas: groupby(country).sum(quantity*price), descending",
    )


def top_countries_by_revenue(n: int = 5) -> Truth:
    totals = retail().groupby("country")["line_revenue"].sum().sort_values(ascending=False).head(n)
    return _truth(
        "top_countries_by_revenue",
        [{"country": c, "revenue": round(float(v), 2)} for c, v in totals.items()],
        f"pandas: top {n} countries by sum(quantity*price)",
    )


def top_customers_by_revenue(n: int = 5) -> Truth:
    frame = retail().dropna(subset=["customer_id"])
    totals = frame.groupby("customer_id")["line_revenue"].sum().sort_values(ascending=False).head(n)
    return _truth(
        "top_customers_by_revenue",
        [{"customer_id": int(c), "revenue": round(float(v), 2)} for c, v in totals.items()],
        f"pandas: top {n} customer_id by sum(quantity*price), nulls dropped",
    )


def monthly_revenue() -> Truth:
    frame = retail()
    series = frame.groupby(frame["invoicedate"].dt.to_period("M"))["line_revenue"].sum()
    best = series.idxmax()
    return _truth(
        "monthly_revenue",
        {
            "periods": int(len(series)),
            "first_period": str(series.index[0]),
            "last_period": str(series.index[-1]),
            "best_period": str(best),
            "best_value": round(float(series.loc[best]), 2),
        },
        "pandas: resample invoicedate to month, sum(quantity*price)",
    )


def revenue_by_world_bank_region() -> Truth:
    merged = retail().merge(countries(), on="country", how="inner")
    totals = merged.groupby("world_bank_region")["line_revenue"].sum().sort_values(ascending=False)
    return _truth(
        "revenue_by_world_bank_region",
        {
            "top_region": totals.index[0],
            "top_revenue": round(float(totals.iloc[0]), 2),
            "regions": int(len(totals)),
            "matched_rows": int(len(merged)),
            "unmatched_countries": sorted(
                set(retail()["country"]) - set(countries()["country"])
            ),
        },
        "pandas: inner join on country, groupby(world_bank_region).sum(quantity*price)",
    )


def worst_products_by_revenue(n: int = 5) -> Truth:
    frame = retail().dropna(subset=["description"])
    totals = frame.groupby("description")["line_revenue"].sum().sort_values()
    return _truth(
        "worst_products_by_revenue",
        [{"description": d, "revenue": round(float(v), 2)} for d, v in totals.head(n).items()],
        f"pandas: bottom {n} descriptions by sum(quantity*price)",
    )


def largest_quantity_outlier() -> Truth:
    frame = retail()
    row = frame.loc[frame["quantity"].idxmax()]
    q1, q3 = frame["quantity"].quantile([0.25, 0.75])
    iqr = q3 - q1
    return _truth(
        "largest_quantity_outlier",
        {
            "quantity": int(row["quantity"]),
            "description": str(row["description"]),
            "invoice": str(row["invoice"]),
            "upper_fence": round(float(q3 + 1.5 * iqr), 2),
        },
        "pandas: max(quantity) plus Tukey upper fence from quantiles",
    )


def returns_summary() -> Truth:
    frame = retail()
    negatives = frame[frame["quantity"] < 0]
    return _truth(
        "returns_summary",
        {
            "return_rows": int(len(negatives)),
            "return_value": round(float(negatives["line_revenue"].sum()), 2),
            "credit_note_invoices": int(
                frame.loc[frame["invoice"].astype(str).str.startswith("C"), "invoice"].nunique()
            ),
        },
        "pandas: rows where quantity < 0, and invoices prefixed 'C'",
    )


def dataset_shape() -> Truth:
    frame = retail()
    return _truth(
        "dataset_shape",
        {
            "rows": int(len(frame)),
            "countries": int(frame["country"].nunique()),
            "duplicate_rows": int(frame.duplicated(subset=frame.columns[:8]).sum()),
            "null_customer_ids": int(frame["customer_id"].isna().sum()),
            "zero_price_rows": int((frame["price"] == 0).sum()),
        },
        "pandas: shape, nunique, duplicated, isna counts",
    )


ALL_TRUTHS = (
    dataset_shape,
    top_country_by_revenue,
    top_countries_by_revenue,
    top_customers_by_revenue,
    monthly_revenue,
    revenue_by_world_bank_region,
    worst_products_by_revenue,
    largest_quantity_outlier,
    returns_summary,
)


def compute_all() -> dict[str, Truth]:
    return {fn().key: fn() for fn in ALL_TRUTHS}


if __name__ == "__main__":
    import json

    for key, truth in compute_all().items():
        print(f"\n{key}\n  {truth.derivation}\n  {json.dumps(truth.value, indent=2, default=str)}")
