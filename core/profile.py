"""Dataset profiling, role inference and data-quality checks.

Why this module matters: evaluations of LLM-generated SQL consistently find that
failures are dominated by schema hallucination and wrong joins rather than
syntax, and that supplying a semantic/schema context layer moves accuracy from
roughly half to the high eighties/nineties. So the profile is not decoration —
it is the primary accuracy mechanism in this application. Every column the model
is allowed to reference is described here with type, cardinality, null rate,
range and real sample values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import get_settings
from .ingest import LoadedFile
from .models import (
    ColumnProfile,
    ColumnRole,
    DataQualityIssue,
    DatasetProfile,
    JoinHint,
)
from .observability import get_logger

log = get_logger(__name__)

_ID_HINTS = ("id", "code", "uuid", "guid", "key", "number", "no", "ref", "sku")
_MEASURE_HINTS = (
    "amount", "revenue", "sales", "price", "cost", "qty", "quantity", "total",
    "profit", "margin", "discount", "count", "value", "spend", "budget", "units",
    "score", "rate", "fee", "balance", "volume", "weight",
)


def _duckdb_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if isinstance(dtype, pd.CategoricalDtype):
        return "VARCHAR"
    return "VARCHAR"


def infer_role(series: pd.Series, name: str, row_count: int) -> ColumnRole:
    """Infer the analytical role of a column.

    Order matters. Temporal and boolean are unambiguous from dtype. For numerics
    we distinguish identifiers (near-unique, or an id-ish name) from measures,
    because summing an order_id is a classic text-to-SQL failure.
    """
    lname = name.lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.TEMPORAL
    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.BOOLEAN

    non_null = series.dropna()
    if non_null.empty:
        return ColumnRole.UNKNOWN
    distinct = int(non_null.nunique())
    distinct_ratio = distinct / max(len(non_null), 1)
    name_is_id = any(
        lname == h or lname.endswith(f"_{h}") or lname.startswith(f"{h}_") for h in _ID_HINTS
    )

    if pd.api.types.is_numeric_dtype(series):
        if name_is_id:
            return ColumnRole.IDENTIFIER
        if distinct_ratio > 0.95 and row_count > 20 and pd.api.types.is_integer_dtype(series):
            return ColumnRole.IDENTIFIER
        if any(h in lname for h in _MEASURE_HINTS):
            return ColumnRole.MEASURE
        # Small-cardinality integers behave like categories (year, rating, flag).
        if pd.api.types.is_integer_dtype(series) and distinct <= 12 and row_count > 50:
            return ColumnRole.DIMENSION
        return ColumnRole.MEASURE

    # Text-like
    if name_is_id and distinct_ratio > 0.5:
        return ColumnRole.IDENTIFIER
    if distinct_ratio > 0.95 and row_count > 50:
        return ColumnRole.IDENTIFIER
    return ColumnRole.DIMENSION


def _sample_values(series: pd.Series, k: int = 5) -> list[str]:
    non_null = series.dropna()
    if non_null.empty:
        return []
    uniques = pd.Series(non_null.unique())
    take = uniques.head(k)
    out = []
    for v in take:
        if isinstance(v, (pd.Timestamp,)):
            out.append(v.isoformat(sep=" ")[:19])
        elif isinstance(v, float):
            out.append(f"{v:g}")
        else:
            out.append(str(v)[:60])
    return out


def profile_column(series: pd.Series, name: str, row_count: int) -> ColumnProfile:
    non_null = series.dropna()
    null_count = int(series.isna().sum())
    distinct = int(non_null.nunique())
    role = infer_role(series, name, row_count)

    prof = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        duckdb_type=_duckdb_type(series),
        role=role,
        null_count=null_count,
        null_pct=round(100 * null_count / max(row_count, 1), 2),
        distinct_count=distinct,
        distinct_pct=round(100 * distinct / max(row_count, 1), 2),
        sample_values=_sample_values(series),
    )

    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        if not non_null.empty:
            desc = non_null.astype("float64")
            prof.min = float(desc.min())
            prof.max = float(desc.max())
            prof.mean = float(desc.mean())
            prof.std = float(desc.std()) if len(desc) > 1 else 0.0
            prof.p25 = float(desc.quantile(0.25))
            prof.p50 = float(desc.quantile(0.50))
            prof.p75 = float(desc.quantile(0.75))
    elif pd.api.types.is_datetime64_any_dtype(series):
        if not non_null.empty:
            prof.min_date = pd.Timestamp(non_null.min()).isoformat(sep=" ")[:19]
            prof.max_date = pd.Timestamp(non_null.max()).isoformat(sep=" ")[:19]

    if role in (ColumnRole.DIMENSION, ColumnRole.BOOLEAN) and distinct <= 50 and not non_null.empty:
        counts = non_null.value_counts().head(10)
        prof.top_values = [
            {"value": str(idx)[:60], "count": int(cnt), "pct": round(100 * cnt / row_count, 2)}
            for idx, cnt in counts.items()
        ]
    return prof


def _quality_checks(frame: pd.DataFrame, cols: list[ColumnProfile]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    rows = len(frame)

    for col in cols:
        if col.null_pct >= 50:
            issues.append(
                DataQualityIssue(
                    severity="warning", kind="high_null_rate", column=col.name,
                    message=f"'{col.name}' is {col.null_pct:.0f}% null — aggregates on it will be unreliable.",
                )
            )
        elif col.null_pct > 0:
            issues.append(
                DataQualityIssue(
                    severity="info", kind="nulls", column=col.name,
                    message=f"'{col.name}' has {col.null_count} null value(s) ({col.null_pct:.1f}%).",
                )
            )
        if col.distinct_count == 1 and rows > 1:
            issues.append(
                DataQualityIssue(
                    severity="info", kind="constant_column", column=col.name,
                    message=f"'{col.name}' has a single distinct value; it cannot differentiate rows.",
                )
            )
        if col.role == ColumnRole.MEASURE and col.min is not None and col.min < 0:
            neg = int((frame[col.name] < 0).sum())
            issues.append(
                DataQualityIssue(
                    severity="info", kind="negative_values", column=col.name,
                    message=f"'{col.name}' contains {neg} negative value(s) (min {col.min:,.2f}) — could be returns or data entry errors.",
                )
            )
        if col.role == ColumnRole.DIMENSION and col.distinct_count > 1:
            # Case/whitespace variants of the same label distort GROUP BY results.
            series = frame[col.name].dropna().astype(str)
            normalised = series.str.strip().str.casefold().nunique()
            if normalised < series.nunique():
                issues.append(
                    DataQualityIssue(
                        severity="warning", kind="inconsistent_casing", column=col.name,
                        message=(
                            f"'{col.name}' has {series.nunique() - normalised} label(s) that differ only by "
                            "case or whitespace; GROUP BY will split them."
                        ),
                    )
                )

    dupes = int(frame.duplicated().sum())
    if dupes:
        issues.append(
            DataQualityIssue(
                severity="warning", kind="duplicate_rows",
                message=f"{dupes} fully duplicated row(s) ({100 * dupes / max(rows, 1):.1f}%).",
            )
        )
    if rows < 10:
        issues.append(
            DataQualityIssue(
                severity="warning", kind="small_dataset",
                message=f"Only {rows} row(s); statistical results including anomaly detection will not be meaningful.",
            )
        )
    if not any(c.role == ColumnRole.MEASURE for c in cols):
        issues.append(
            DataQualityIssue(
                severity="info", kind="no_measures",
                message="No numeric measure column detected; quantitative questions may not be answerable.",
            )
        )
    return issues


def build_profile(loaded: LoadedFile) -> DatasetProfile:
    """Profile a loaded file. Sampling keeps this bounded on very wide/long files."""
    settings = get_settings()
    frame = loaded.frame
    rows = len(frame)
    sample = frame if rows <= settings.profile_sample_rows else frame.sample(
        settings.profile_sample_rows, random_state=7
    )

    cols = [profile_column(sample[c], str(c), len(sample)) for c in sample.columns]
    issues = list(loaded.issues) + _quality_checks(sample, cols)

    profile = DatasetProfile(
        table=loaded.table,
        source_name=loaded.source_name,
        row_count=rows,
        column_count=len(frame.columns),
        columns=cols,
        duplicate_row_count=int(frame.duplicated().sum()),
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        issues=issues,
    )
    if rows > settings.profile_sample_rows:
        profile.issues.append(
            DataQualityIssue(
                severity="info", kind="sampled_profile",
                message=f"Profile computed on a {settings.profile_sample_rows:,}-row sample of {rows:,} rows. Queries still run on all rows.",
            )
        )
    log.info("profile.built", table=profile.table, rows=rows, cols=len(cols), issues=len(profile.issues))
    return profile


def detect_join_hints(
    frames: dict[str, pd.DataFrame], profiles: dict[str, DatasetProfile]
) -> list[JoinHint]:
    """Find likely join keys between loaded tables.

    Two signals, both cheap and both verifiable: identical/similar column names,
    and actual value overlap. Overlap is what we report to the model, so it can
    tell a real key from a coincidental name match.
    """
    hints: list[JoinHint] = []
    tables = sorted(frames)
    for i, left in enumerate(tables):
        for right in tables[i + 1 :]:
            lp, rp = profiles[left], profiles[right]
            for lcol in lp.columns:
                if lcol.role not in (ColumnRole.IDENTIFIER, ColumnRole.DIMENSION):
                    continue
                for rcol in rp.columns:
                    if rcol.role not in (ColumnRole.IDENTIFIER, ColumnRole.DIMENSION):
                        continue
                    name_match = lcol.name == rcol.name or (
                        lcol.name.rstrip("s") == rcol.name.rstrip("s")
                    )
                    if not name_match:
                        continue
                    lvals = set(frames[left][lcol.name].dropna().astype(str).unique()[:20_000])
                    rvals = set(frames[right][rcol.name].dropna().astype(str).unique()[:20_000])
                    if not lvals or not rvals:
                        continue
                    overlap = len(lvals & rvals) / min(len(lvals), len(rvals))
                    if overlap < 0.30:
                        continue
                    side = "one-to-many" if lcol.distinct_pct > rcol.distinct_pct else "many-to-one"
                    hints.append(
                        JoinHint(
                            left_table=left, left_column=lcol.name,
                            right_table=right, right_column=rcol.name,
                            overlap_pct=round(100 * overlap, 1),
                            reason=f"matching name and {100 * overlap:.0f}% value overlap ({side})",
                        )
                    )
    hints.sort(key=lambda h: h.overlap_pct, reverse=True)
    return hints[:12]


def quality_score(profile: DatasetProfile) -> dict[str, object]:
    """A single 0-100 completeness/consistency score plus its components."""
    errors = sum(1 for i in profile.issues if i.severity == "error")
    warnings = sum(1 for i in profile.issues if i.severity == "warning")
    total_cells = max(profile.row_count * profile.column_count, 1)
    null_cells = sum(c.null_count for c in profile.columns)
    completeness = 100 * (1 - null_cells / total_cells)
    uniqueness = 100 * (1 - profile.duplicate_row_count / max(profile.row_count, 1))
    penalty = min(30.0, 10.0 * errors + 3.0 * warnings)
    score = max(0.0, 0.5 * completeness + 0.3 * uniqueness + 0.2 * 100 - penalty)
    return {
        "score": round(min(score, 100.0), 1),
        "completeness_pct": round(completeness, 2),
        "uniqueness_pct": round(uniqueness, 2),
        "errors": errors,
        "warnings": warnings,
        "null_cells": int(null_cells),
        "total_cells": int(total_cells),
    }
