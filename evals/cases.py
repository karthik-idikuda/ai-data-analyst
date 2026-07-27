"""The golden question set.

Every expected value is computed from the real data by :mod:`evals.ground_truth`
(pandas), never hard-coded, so the set stays correct if the data is refreshed.

The set intentionally includes cases the system should *refuse*: the retail file
has no cost, margin or marketing-spend column, and a good analyst says so rather
than substituting a proxy. An eval suite that only contains answerable questions
rewards a model for guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals import ground_truth as gt
from evals.checks import (
    Check,
    ContainsText,
    ContainsTruthNumber,
    ContainsTruthText,
    HasArtifact,
    ReasoningPresent,
    Rejects,
    SqlContains,
    SqlExecuted,
)


@dataclass
class EvalCase:
    id: str
    question: str
    truth_key: str | None
    checks: list[Check]
    tags: list[str] = field(default_factory=list)
    # Follow-up asked in the same session, to test conversational memory.
    follow_up: str | None = None
    follow_up_checks: list[Check] = field(default_factory=list)


GOLDEN_SET: list[EvalCase] = [
    # ---------------------------------------------------------------- ranking
    EvalCase(
        id="top_country_revenue",
        question="Which country generated the highest revenue?",
        truth_key="top_country_by_revenue",
        checks=[
            ContainsTruthText("country"),
            ContainsTruthNumber("revenue", rel_tolerance=0.02),
            SqlContains(("quantity", "price", "country")),
            SqlExecuted(),
            ReasoningPresent(),
        ],
        tags=["aggregation", "derived-metric", "assignment-example"],
        follow_up="And which country was second?",
        follow_up_checks=[ContainsText("netherlands"), SqlExecuted()],
    ),
    EvalCase(
        id="top_five_customers",
        question="What are the top five customers by revenue?",
        truth_key="top_customers_by_revenue",
        checks=[
            ContainsTruthText("0.customer_id"),
            ContainsTruthNumber("0.revenue", rel_tolerance=0.02),
            ContainsTruthText("1.customer_id"),
            SqlContains(("customer_id", "limit")),
            SqlExecuted(),
        ],
        tags=["ranking", "assignment-example"],
    ),
    EvalCase(
        id="underperforming_products",
        question="Which products are underperforming?",
        truth_key="worst_products_by_revenue",
        checks=[SqlContains(("description",)), SqlExecuted(), ReasoningPresent()],
        tags=["open-ended", "assignment-example"],
    ),
    # ------------------------------------------------------------------ trend
    EvalCase(
        id="monthly_trend",
        question="Show monthly sales trends.",
        truth_key="monthly_revenue",
        checks=[
            HasArtifact("chart"),
            SqlContains(("invoicedate",)),
            SqlExecuted(),
        ],
        tags=["chart", "time-series", "assignment-example"],
    ),
    EvalCase(
        id="best_month",
        question="Which single month had the highest revenue, and how much was it?",
        truth_key="monthly_revenue",
        checks=[
            ContainsTruthText("best_period"),
            ContainsTruthNumber("best_value", rel_tolerance=0.02),
            SqlExecuted(),
        ],
        tags=["time-series"],
    ),
    # ------------------------------------------------------------- multi-file
    EvalCase(
        id="revenue_by_region",
        question=(
            "Join the retail transactions to the World Bank country profile and tell me "
            "which World Bank region generated the highest revenue."
        ),
        truth_key="revenue_by_world_bank_region",
        checks=[
            ContainsTruthText("top_region"),
            SqlContains(("join", "world_bank_region")),
            SqlExecuted(),
        ],
        tags=["multi-file", "join"],
    ),
    # -------------------------------------------------------------- anomalies
    EvalCase(
        id="anomalies_explained",
        question="Detect anomalies in the dataset and explain why they were flagged.",
        truth_key="largest_quantity_outlier",
        checks=[
            HasArtifact("anomaly"),
            ContainsText("iqr", where="all"),
            ReasoningPresent(),
        ],
        tags=["anomaly", "assignment-example"],
    ),
    EvalCase(
        id="returns_are_understood",
        question="There are negative quantities in this data. What are they, and how many rows?",
        truth_key="returns_summary",
        checks=[ContainsTruthNumber("return_rows", rel_tolerance=0.02), SqlExecuted()],
        tags=["data-understanding"],
    ),
    # ------------------------------------------------------------------- code
    EvalCase(
        id="generate_sql",
        question="Generate the SQL for revenue by country, but don't run it.",
        truth_key=None,
        checks=[HasArtifact("code")],
        tags=["code-generation", "assignment-example"],
    ),
    # ----------------------------------------------------------- data quality
    EvalCase(
        id="data_quality",
        question="How reliable is this data? Give me a data-quality assessment.",
        truth_key="dataset_shape",
        checks=[HasArtifact("quality"), ContainsText("duplicate", where="all")],
        tags=["data-quality"],
    ),
    # ----------------------------------------------- must refuse, not invent
    EvalCase(
        id="refuse_missing_profit",
        question="What was the profit margin by country last year?",
        truth_key=None,
        checks=[Rejects(forbidden=("profit_margin", "margin column", "cost column"))],
        tags=["honesty", "refusal"],
    ),
    EvalCase(
        id="refuse_missing_marketing",
        question="How much did we spend on marketing per channel?",
        truth_key=None,
        checks=[Rejects(forbidden=("marketing_spend", "channel column"))],
        tags=["honesty", "refusal"],
    ),
    EvalCase(
        id="refuse_uk_data",
        question="How much revenue came from the United Kingdom?",
        truth_key=None,
        checks=[Rejects()],
        tags=["honesty", "refusal", "dataset-scope"],
    ),
]


def by_tag(tag: str) -> list[EvalCase]:
    return [case for case in GOLDEN_SET if tag in case.tags]


def truths() -> dict[str, gt.Truth]:
    return gt.compute_all()
