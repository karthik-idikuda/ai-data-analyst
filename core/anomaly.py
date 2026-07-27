"""Deterministic anomaly detection with numeric, reproducible explanations.

Design decision: **the LLM does not find anomalies.** Asking a language model to
eyeball outliers produces confident, unreproducible answers. Instead four
statistical methods run in pandas/numpy/scikit-learn, each anomaly carries the
method, the threshold and the observed value that breached it, and the LLM's only
job is to turn that record into business language.

Methods
-------
``iqr``
    Tukey's fence: outside ``Q1 - k*IQR`` or ``Q3 + k*IQR`` (default k=1.5).
    Distribution-free, the standard boxplot rule.
``robust_zscore``
    Deviation from the median scaled by MAD (``0.6745*(x-median)/MAD``), flagged
    above 3.5. Preferred over a plain z-score because mean and standard deviation
    are themselves dragged by the outliers you are hunting.
``isolation_forest``
    Multivariate: catches rows that are unremarkable per column but implausible
    in combination (small quantity at a huge revenue, say).
``seasonal_residual``
    For a dated series: STL decomposition, then robust-z on the residual, so a
    genuine December peak is not flagged while an off-trend spike is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Anomaly, AnomalyMethod, AnomalyReport, ColumnRole, DatasetProfile
from .observability import get_logger

log = get_logger(__name__)

MAD_SCALE = 0.6745  # makes MAD a consistent estimator of sigma for normal data
DEFAULT_IQR_K = 1.5
DEFAULT_Z_THRESHOLD = 3.5
MIN_ROWS_UNIVARIATE = 12
MIN_ROWS_SEASONAL = 24


def _label_for(frame: pd.DataFrame, profile: DatasetProfile, idx: int) -> str | None:
    """A human-readable identifier for a flagged row."""
    preferred = [
        c.name
        for c in profile.columns
        if c.role in (ColumnRole.IDENTIFIER, ColumnRole.DIMENSION, ColumnRole.TEMPORAL)
    ]
    parts = []
    for name in preferred[:3]:
        if name in frame.columns:
            value = frame.at[idx, name]
            if pd.notna(value):
                parts.append(f"{name}={value}")
    return ", ".join(parts) if parts else None


def _row_context(frame: pd.DataFrame, profile: DatasetProfile, idx: int, exclude: str) -> dict:
    ctx: dict = {}
    for col in profile.columns:
        if col.name == exclude or col.name not in frame.columns:
            continue
        if col.role in (ColumnRole.DIMENSION, ColumnRole.TEMPORAL, ColumnRole.IDENTIFIER):
            value = frame.at[idx, col.name]
            if pd.notna(value):
                ctx[col.name] = str(value)
        if len(ctx) >= 5:
            break
    return ctx


def detect_iqr(series: pd.Series, k: float = DEFAULT_IQR_K) -> tuple[pd.Series, float, float, dict]:
    clean = series.dropna().astype("float64")
    q1, q3 = float(clean.quantile(0.25)), float(clean.quantile(0.75))
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    mask = (series < low) | (series > high)
    stats = {"q1": q1, "q3": q3, "iqr": iqr, "k": k, "lower_fence": low, "upper_fence": high}
    return mask.fillna(False), low, high, stats


def detect_robust_z(
    series: pd.Series, threshold: float = DEFAULT_Z_THRESHOLD
) -> tuple[pd.Series, pd.Series, dict]:
    clean = series.dropna().astype("float64")
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    if mad == 0:
        # Degenerate spread: fall back to a scaled mean-absolute deviation so a
        # column of mostly-identical values does not flag every distinct value.
        mad = float((clean - median).abs().mean())
    if mad == 0:
        zeros = pd.Series(0.0, index=series.index)
        return pd.Series(False, index=series.index), zeros, {"median": median, "mad": 0.0}
    scores = (MAD_SCALE * (series.astype("float64") - median) / mad).abs()
    stats = {
        "median": median,
        "mad": mad,
        "threshold": threshold,
        "implied_lower": median - threshold * mad / MAD_SCALE,
        "implied_upper": median + threshold * mad / MAD_SCALE,
    }
    return (scores > threshold).fillna(False), scores.fillna(0.0), stats


def detect_isolation_forest(
    frame: pd.DataFrame, columns: list[str], contamination: float = 0.02
) -> tuple[pd.Series, pd.Series] | None:
    """Multivariate detection. Returns (mask, score) or None when not applicable."""
    from sklearn.ensemble import IsolationForest
    from threadpoolctl import threadpool_limits

    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna()
    if len(numeric) < 30 or len(columns) < 2:
        return None

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,  # fixed seed: the same data must yield the same anomalies
        n_jobs=1,
    )
    # Pin native thread pools for the duration of the fit.
    #
    # This is not premature caution: running this inside Streamlit's script thread
    # segfaulted the whole process on macOS. Streamlit pulls in pyarrow, which ships
    # its own OpenMP runtime alongside the one scikit-learn/scipy use, and two
    # OpenMP runtimes in one process is undefined behaviour. Confining the fit to a
    # single thread removes the interaction. The dataset sizes here are small enough
    # that the lost parallelism is not measurable.
    with threadpool_limits(limits=1):
        labels = model.fit_predict(numeric.values)
        raw_scores = -model.score_samples(numeric.values)  # higher == more anomalous

    mask = pd.Series(False, index=frame.index)
    scores = pd.Series(0.0, index=frame.index)
    mask.loc[numeric.index] = labels == -1
    scores.loc[numeric.index] = raw_scores
    return mask, scores


def detect_seasonal_residual(
    frame: pd.DataFrame, date_col: str, value_col: str, freq: str = "ME"
) -> tuple[pd.DataFrame, dict] | None:
    """STL on an aggregated series; robust-z on the residual."""
    from statsmodels.tsa.seasonal import STL
    from threadpoolctl import threadpool_limits

    work = frame[[date_col, value_col]].dropna()
    if work.empty:
        return None
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    series = (
        work.set_index(date_col)[value_col]
        .astype("float64")
        .resample(freq)
        .sum()
        .dropna()
    )
    if len(series) < MIN_ROWS_SEASONAL:
        return None

    period = 12 if freq in ("M", "ME") else (7 if freq == "D" else 4)
    if len(series) < 2 * period:
        return None
    try:
        # Same OpenMP reasoning as detect_isolation_forest: STL drops into LAPACK,
        # which is the other native thread pool in this process.
        with threadpool_limits(limits=1):
            result = STL(series, period=period, robust=True).fit()
    except Exception as exc:  # noqa: BLE001 - statsmodels raises many types
        log.warning("anomaly.stl_failed", error=str(exc))
        return None

    resid = result.resid
    median = float(resid.median())
    mad = float((resid - median).abs().median()) or float((resid - median).abs().mean())
    if not mad:
        return None
    z = (MAD_SCALE * (resid - median) / mad).abs()
    out = pd.DataFrame(
        {
            "period": series.index.strftime("%Y-%m-%d"),
            "actual": series.values,
            "expected": (result.trend + result.seasonal).values,
            "residual": resid.values,
            "z": z.values,
        }
    )
    stats = {"period_length": period, "residual_median": median, "residual_mad": mad, "freq": freq}
    return out[out["z"] > DEFAULT_Z_THRESHOLD], stats


def analyse(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    *,
    columns: list[str] | None = None,
    max_per_method: int = 15,
    iqr_k: float = DEFAULT_IQR_K,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    include_multivariate: bool = True,
) -> AnomalyReport:
    """Run every applicable method and return a consolidated, explained report."""
    measures = [c.name for c in profile.columns_by_role(ColumnRole.MEASURE)]
    if columns:
        requested = [c for c in columns if c in frame.columns]
        unknown = [c for c in columns if c not in frame.columns]
        targets = requested
    else:
        targets = measures
        unknown = []

    notes: list[str] = []
    if unknown:
        notes.append(f"Ignored unknown column(s): {', '.join(unknown)}.")
    if not targets:
        notes.append("No numeric measure columns are available, so no anomaly test could run.")
        return AnomalyReport(
            table=profile.table, columns_tested=[], methods_used=[],
            rows_tested=len(frame), anomalies=[], notes=notes,
        )

    numeric_targets = [
        c for c in targets if pd.api.types.is_numeric_dtype(frame[c])
    ]
    skipped = sorted(set(targets) - set(numeric_targets))
    if skipped:
        notes.append(f"Skipped non-numeric column(s): {', '.join(skipped)}.")
    if not numeric_targets:
        return AnomalyReport(
            table=profile.table, columns_tested=[], methods_used=[],
            rows_tested=len(frame), anomalies=[], notes=notes,
        )

    anomalies: list[Anomaly] = []
    methods: list[AnomalyMethod] = []
    summary: dict[str, dict] = {}

    if len(frame) < MIN_ROWS_UNIVARIATE:
        notes.append(
            f"Only {len(frame)} rows: below the {MIN_ROWS_UNIVARIATE}-row minimum for reliable "
            "univariate outlier detection. Results are indicative only."
        )

    for col in numeric_targets:
        series = frame[col]
        if series.dropna().nunique() < 3:
            notes.append(f"'{col}' has fewer than 3 distinct values; skipped.")
            continue

        iqr_mask, low, high, iqr_stats = detect_iqr(series, k=iqr_k)
        z_mask, z_scores, z_stats = detect_robust_z(series, threshold=z_threshold)
        summary[col] = {"iqr": iqr_stats, "robust_zscore": z_stats}

        # Real data repeats the same extreme value across many rows: the UCI retail
        # file has dozens of 'Manual' adjustment lines at an identical price. Listing
        # each one buries every other finding, so report distinct values per method
        # and column and say how many rows shared each.
        seen_values: set[float] = set()
        repeat_counts = series.value_counts()

        if iqr_mask.any():
            if AnomalyMethod.IQR not in methods:
                methods.append(AnomalyMethod.IQR)
            flagged = series[iqr_mask].abs().sort_values(ascending=False).index
            for idx in flagged:
                if len([a for a in anomalies if a.column == col and a.method == AnomalyMethod.IQR]) >= max_per_method:
                    break
                value = float(series.at[idx])
                if value in seen_values:
                    continue
                seen_values.add(value)
                shared = int(repeat_counts.get(series.at[idx], 1))
                direction = "high" if value > high else "low"
                fence = high if direction == "high" else low
                distance = abs(value - fence)
                anomalies.append(
                    Anomaly(
                        method=AnomalyMethod.IQR,
                        column=col,
                        row_index=int(idx),
                        value=value,
                        score=round(distance / (iqr_stats["iqr"] or 1), 3),
                        direction=direction,
                        threshold_low=low,
                        threshold_high=high,
                        label=_label_for(frame, profile, idx),
                        reason=(
                            f"{col}={value:,.2f} falls {direction_word(direction)} the Tukey fence "
                            f"[{low:,.2f}, {high:,.2f}] (Q1={iqr_stats['q1']:,.2f}, "
                            f"Q3={iqr_stats['q3']:,.2f}, IQR={iqr_stats['iqr']:,.2f}, k={iqr_k}); "
                            f"it is {distance:,.2f} beyond the fence."
                            + (f" This exact value appears in {shared:,} rows." if shared > 1 else "")
                        ),
                        context=_row_context(frame, profile, idx, col),
                    )
                )

        if z_mask.any():
            if AnomalyMethod.ROBUST_Z not in methods:
                methods.append(AnomalyMethod.ROBUST_Z)
            ranked = z_scores[z_mask].sort_values(ascending=False).index
            added = 0
            for idx in ranked:
                if added >= max_per_method:
                    break
                value = float(series.at[idx])
                if value in seen_values:
                    continue  # already reported for this column, by either method
                seen_values.add(value)
                shared = int(repeat_counts.get(series.at[idx], 1))
                anomalies.append(
                    Anomaly(
                        method=AnomalyMethod.ROBUST_Z,
                        column=col,
                        row_index=int(idx),
                        value=value,
                        score=round(float(z_scores.at[idx]), 3),
                        direction="high" if value > z_stats["median"] else "low",
                        threshold_low=z_stats.get("implied_lower"),
                        threshold_high=z_stats.get("implied_upper"),
                        label=_label_for(frame, profile, idx),
                        reason=(
                            f"{col}={value:,.2f} has a robust z-score of {z_scores.at[idx]:.2f} "
                            f"against median {z_stats['median']:,.2f} and MAD {z_stats['mad']:,.2f}, "
                            f"above the {z_threshold} threshold."
                            + (f" This exact value appears in {shared:,} rows." if shared > 1 else "")
                        ),
                        context=_row_context(frame, profile, idx, col),
                    )
                )
                added += 1

    # ---- multivariate -----------------------------------------------------
    if include_multivariate and len(numeric_targets) >= 2:
        mv = detect_isolation_forest(frame, numeric_targets)
        if mv is not None:
            mask, scores = mv
            if mask.any():
                methods.append(AnomalyMethod.ISOLATION_FOREST)
                top = scores[mask].sort_values(ascending=False).index[:max_per_method]
                for idx in top:
                    values = {c: float(frame.at[idx, c]) for c in numeric_targets if pd.notna(frame.at[idx, c])}
                    detail = ", ".join(f"{k}={v:,.2f}" for k, v in list(values.items())[:4])
                    anomalies.append(
                        Anomaly(
                            method=AnomalyMethod.ISOLATION_FOREST,
                            column="+".join(numeric_targets[:4]),
                            row_index=int(idx),
                            value=None,
                            score=round(float(scores.at[idx]), 4),
                            direction="multivariate",
                            label=_label_for(frame, profile, idx),
                            reason=(
                                "Isolation Forest isolated this row in few splits "
                                f"(score {scores.at[idx]:.4f}, top 2% of the distribution). The value "
                                f"combination is unusual even though individual fields may look normal: {detail}."
                            ),
                            context=_row_context(frame, profile, idx, ""),
                        )
                    )
        else:
            notes.append(
                "Isolation Forest needs at least 30 complete rows across 2+ numeric columns; skipped."
            )

    # ---- seasonal ---------------------------------------------------------
    temporal = profile.columns_by_role(ColumnRole.TEMPORAL)
    if temporal and numeric_targets:
        date_col, value_col = temporal[0].name, numeric_targets[0]
        seasonal = detect_seasonal_residual(frame, date_col, value_col)
        if seasonal is not None:
            flagged, stats = seasonal
            summary["seasonal"] = stats
            if not flagged.empty:
                methods.append(AnomalyMethod.SEASONAL_RESIDUAL)
                for _, row in flagged.head(max_per_method).iterrows():
                    gap = row["actual"] - row["expected"]
                    pct = 100 * gap / row["expected"] if row["expected"] else float("nan")
                    anomalies.append(
                        Anomaly(
                            method=AnomalyMethod.SEASONAL_RESIDUAL,
                            column=value_col,
                            row_index=-1,  # aggregated period, not a source row
                            value=float(row["actual"]),
                            score=round(float(row["z"]), 3),
                            direction="high" if gap > 0 else "low",
                            label=f"{date_col} period {row['period']}",
                            reason=(
                                f"For the month starting {row['period']}, {value_col} totalled "
                                f"{row['actual']:,.2f} against an STL trend+seasonal expectation of "
                                f"{row['expected']:,.2f} ({pct:+.1f}%). The residual's robust z-score is "
                                f"{row['z']:.2f}, so this is off-pattern rather than ordinary seasonality."
                            ),
                            context={"period": str(row["period"]), "expected": float(row["expected"])},
                        )
                    )
        else:
            notes.append(
                f"Seasonal decomposition needs {MIN_ROWS_SEASONAL}+ monthly periods; skipped."
            )

    anomalies.sort(key=lambda a: a.score, reverse=True)
    log.info(
        "anomaly.done", table=profile.table, tested=len(numeric_targets),
        found=len(anomalies), methods=[m.value for m in methods],
    )
    return AnomalyReport(
        table=profile.table,
        columns_tested=numeric_targets,
        methods_used=methods,
        rows_tested=len(frame),
        anomalies=anomalies,
        summary_stats=summary,
        notes=notes,
    )


def direction_word(direction: str) -> str:
    return "above" if direction == "high" else "below"
