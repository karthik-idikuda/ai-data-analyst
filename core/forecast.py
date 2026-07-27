"""Time-series forecasting.

Deliberately conservative. Two models, both interpretable, chosen by how much
history exists:

* **Holt-Winters exponential smoothing** (additive trend + additive seasonality)
  once there are at least two full seasonal cycles.
* **OLS linear trend** otherwise, with prediction intervals from the residual
  standard error.

Both report an in-sample MAPE so the caller can see how much to trust the line,
and both attach a note when history is thin. Nothing here is an LLM output —
the model only narrates the numbers it is given.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .errors import ToolError
from .models import ForecastPoint, ForecastResult
from .observability import get_logger

log = get_logger(__name__)

FREQ_ALIASES = {
    "daily": "D", "day": "D", "d": "D",
    "weekly": "W", "week": "W", "w": "W",
    "monthly": "ME", "month": "ME", "m": "ME", "me": "ME",
    "quarterly": "QE", "quarter": "QE", "q": "QE", "qe": "QE",
    "yearly": "YE", "annual": "YE", "year": "YE", "y": "YE", "ye": "YE",
}
SEASON_LENGTH = {"D": 7, "W": 52, "ME": 12, "QE": 4, "YE": 1}


def _normalise_freq(freq: str) -> str:
    key = (freq or "monthly").strip().lower()
    return FREQ_ALIASES.get(key, "ME")


def _label(index: pd.DatetimeIndex, freq: str) -> list[str]:
    if freq == "ME":
        return [d.strftime("%Y-%m") for d in index]
    if freq == "QE":
        return [f"{d.year}-Q{((d.month - 1) // 3) + 1}" for d in index]
    if freq == "YE":
        return [str(d.year) for d in index]
    return [d.strftime("%Y-%m-%d") for d in index]


def _mape(actual: np.ndarray, fitted: np.ndarray) -> float | None:
    mask = actual != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100)


def forecast_series(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    *,
    periods: int = 6,
    freq: str = "monthly",
    agg: str = "sum",
    table: str = "",
) -> ForecastResult:
    if date_column not in frame.columns:
        raise ToolError(f"Column '{date_column}' does not exist.")
    if value_column not in frame.columns:
        raise ToolError(f"Column '{value_column}' does not exist.")
    if not pd.api.types.is_numeric_dtype(frame[value_column]):
        raise ToolError(f"Column '{value_column}' is not numeric, so it cannot be forecast.")

    periods = max(1, min(int(periods), 36))
    pandas_freq = _normalise_freq(freq)
    notes: list[str] = []

    work = frame[[date_column, value_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    dropped = int(work[date_column].isna().sum())
    work = work.dropna(subset=[date_column])
    if dropped:
        notes.append(f"Dropped {dropped} row(s) with an unparsable {date_column}.")
    if work.empty:
        raise ToolError(f"No usable dates in '{date_column}'.")

    aggregator = {"sum": "sum", "mean": "mean", "avg": "mean", "count": "count", "max": "max", "min": "min"}
    how = aggregator.get(agg.lower(), "sum")
    series = (
        work.set_index(date_column)[value_column]
        .astype("float64")
        .resample(pandas_freq)
        .agg(how)
    )
    gaps = int(series.isna().sum())
    if gaps:
        series = series.interpolate(limit_direction="both")
        notes.append(f"{gaps} empty period(s) were linearly interpolated.")
    series = series.dropna()

    if len(series) < 4:
        raise ToolError(
            f"Only {len(series)} period(s) of history after resampling to {freq}; "
            "at least 4 are needed to forecast."
        )

    season = SEASON_LENGTH.get(pandas_freq, 12)
    use_hw = len(series) >= 2 * season and season > 1

    if use_hw:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    series, trend="add", seasonal="add",
                    seasonal_periods=season, initialization_method="estimated",
                ).fit(optimized=True)
                fitted = np.asarray(model.fittedvalues, dtype="float64")
                predicted = np.asarray(model.forecast(periods), dtype="float64")
                resid_std = float(np.std(series.values - fitted, ddof=1))
                method = f"Holt-Winters exponential smoothing (additive trend + additive seasonality, period={season})"
            except Exception as exc:  # noqa: BLE001
                log.warning("forecast.hw_failed", error=str(exc))
                use_hw = False
                notes.append("Holt-Winters failed to converge; fell back to a linear trend.")

    if not use_hw:
        x = np.arange(len(series), dtype="float64")
        y = series.values.astype("float64")
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        future_x = np.arange(len(series), len(series) + periods, dtype="float64")
        predicted = slope * future_x + intercept
        dof = max(len(series) - 2, 1)
        resid_std = float(np.sqrt(np.sum((y - fitted) ** 2) / dof))
        method = "Ordinary least squares linear trend"
        if season > 1 and len(series) < 2 * season:
            notes.append(
                f"Only {len(series)} periods of history — fewer than the {2 * season} needed to "
                "estimate seasonality, so the forecast is trend-only."
            )

    last = series.index[-1]
    future_index = pd.date_range(start=last, periods=periods + 1, freq=pandas_freq)[1:]
    labels = _label(future_index, pandas_freq)

    # 95% interval, widened with the square root of the horizon to reflect
    # compounding uncertainty further out.
    points = [
        ForecastPoint(
            period=labels[i],
            forecast=round(float(predicted[i]), 4),
            lower=round(float(predicted[i] - 1.96 * resid_std * np.sqrt(i + 1)), 4),
            upper=round(float(predicted[i] + 1.96 * resid_std * np.sqrt(i + 1)), 4),
        )
        for i in range(periods)
    ]

    history = [
        {"period": label, "value": round(float(value), 4)}
        for label, value in zip(_label(series.index, pandas_freq), series.values)
    ]
    mape = _mape(series.values.astype("float64"), np.asarray(fitted, dtype="float64"))
    if mape is not None and mape > 30:
        notes.append(
            f"In-sample MAPE is {mape:.1f}% — the history is noisy, so treat the forecast as a rough direction."
        )

    log.info("forecast.done", table=table, method=method, periods=periods, history=len(series))
    return ForecastResult(
        table=table,
        date_column=date_column,
        value_column=value_column,
        freq=pandas_freq,
        method=method,
        history=history,
        points=points,
        in_sample_mape=round(mape, 2) if mape is not None else None,
        notes=notes,
    )
