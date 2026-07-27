"""CSV ingestion and validation.

Real-world CSVs are hostile: BOMs, cp1252 bytes, semicolon delimiters, duplicate
headers, blank unnamed columns, thousands separators, mixed date formats. This
module normalises all of that into a clean DataFrame plus a list of validation
issues, and raises a typed error when a file is genuinely unusable.

No silent coercion: every transformation we apply is recorded as a
`DataQualityIssue` so the user (and the LLM) knows what happened.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import get_settings
from .errors import EmptyFileError, FileTooLargeError, IngestionError, UnsupportedFileError
from .models import DataQualityIssue
from .observability import get_logger

log = get_logger(__name__)

ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt"}
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_SQL_IDENT = re.compile(r"[^0-9a-zA-Z_]+")
_NUMERIC_LIKE = re.compile(r"^-?[\d,]*\.?\d+%?$")
_CURRENCY = re.compile(r"^\s*[-+]?[$£€₹¥]\s*[\d,]*\.?\d+\s*$")


@dataclass
class LoadedFile:
    """A validated, normalised dataset ready to be registered in DuckDB."""

    table: str
    source_name: str
    frame: pd.DataFrame
    issues: list[DataQualityIssue]
    raw_bytes: int
    delimiter: str
    encoding: str


def sanitize_identifier(name: str, *, fallback: str = "col") -> str:
    """Turn an arbitrary string into a safe, lower-case SQL identifier.

    This is a defence layer, not a convenience: sanitised identifiers mean the
    guard's table/column allow-list can never be defeated by a crafted CSV
    header such as ``"a"; DROP TABLE x; --``.
    """
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = text.strip().lower().replace(" ", "_").replace("-", "_")
    text = _SQL_IDENT.sub("_", text).strip("_")
    text = re.sub(r"_{2,}", "_", text)
    if not text or text[0].isdigit():
        text = f"{fallback}_{text}" if text else fallback
    return text[:63]


def _dedupe(names: list[str]) -> tuple[list[str], list[str]]:
    seen: dict[str, int] = {}
    out: list[str] = []
    renamed: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            new = f"{name}_{seen[name]}"
            renamed.append(f"{name} -> {new}")
            out.append(new)
        else:
            seen[name] = 0
            out.append(name)
    return out, renamed


def _detect_encoding(raw: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            raw[:200_000].decode(enc)
        except UnicodeDecodeError:
            continue
        return enc
    raise UnsupportedFileError(
        "Could not decode the file as text.",
        detail=f"Tried encodings: {', '.join(_ENCODINGS)}. Is this really a CSV?",
    )


def _detect_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer is unreliable on single-column or quote-heavy files; fall back
        # to whichever candidate appears most consistently on the first lines.
        lines = [ln for ln in sample.splitlines()[:20] if ln.strip()]
        best, best_score = ",", -1.0
        for cand in (",", ";", "\t", "|"):
            counts = [ln.count(cand) for ln in lines]
            if not counts or max(counts) == 0:
                continue
            consistency = counts.count(counts[0]) / len(counts)
            score = consistency * min(max(counts), 50)
            if score > best_score:
                best, best_score = cand, score
        return best


# Tokens that genuinely mean "missing" and may therefore fail numeric conversion
# without that implying the column is categorical.
_NULLISH = frozenset(
    {"", "-", "--", "na", "n/a", "n.a.", "none", "null", "nil", "nan", "unknown", "tbd", "?", "."}
)


def _clean_numeric_series(series: pd.Series) -> tuple[pd.Series, str] | None:
    """Try to convert an object column of formatted numbers into floats.

    Handles ``1,234.5``, ``$1,234``, ``45%``, ``(123)`` negatives.

    Strictness matters here, and it is not theoretical. The UCI Online Retail II
    file has an ``Invoice`` column where ~97% of values are numeric
    (``489434``) and the rest are credit notes (``C489449``). A "convert if 95%
    of values parse" rule silently turns every credit note into NULL — real data
    loss that then corrupts every downstream aggregate. So a column is only
    coerced when **every** non-null value either converts or is a recognised
    missing-value token; otherwise it stays text and nothing is lost.
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return None

    is_pct = non_null.str.endswith("%").mean() > 0.9
    is_currency = non_null.str.match(_CURRENCY).mean() > 0.9

    stripped = non_null.str.replace(r"[$£€₹¥%,\s]", "", regex=True)
    parses = stripped.str.match(r"^\(?-?\d*\.?\d+\)?$")
    if parses.all():
        pass  # clean numeric column
    else:
        failures = non_null[~parses]
        # Tolerate only explicit missing-value markers, never real content.
        if not failures.str.lower().isin(_NULLISH).all():
            return None
        if parses.mean() < 0.5:
            return None

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[$£€₹¥%,\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    cleaned = cleaned.where(~cleaned.str.lower().isin(_NULLISH), other=None)
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Final guard: conversion must not have destroyed anything unexpectedly.
    expected_valid = int(parses.sum())
    if int(converted.notna().sum()) < expected_valid:
        return None

    if is_pct:
        return converted / 100.0, "percent strings converted to fractions"
    if is_currency:
        return converted, "currency symbols and separators stripped"
    return converted, "thousands separators stripped"


def _maybe_datetime(series: pd.Series) -> pd.Series | None:
    """Convert an object column to datetime only when it is unambiguous enough.

    ``pd.to_datetime`` with ``format="mixed"`` is intentionally avoided: it will
    happily parse ``"12"`` or ``"Region A"``. We require a date-ish shape first.
    """
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if len(non_null) < 3:
        return None
    shape = re.compile(
        r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
        r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
        r"|^\d{1,2}[- ][A-Za-z]{3,9}[- ]\d{2,4}$"
        r"|^[A-Za-z]{3,9} \d{1,2},? \d{4}$"
    )
    if non_null.str.match(shape).mean() < 0.9:
        return None
    converted = pd.to_datetime(series, errors="coerce", format="ISO8601")
    if converted.notna().sum() < len(non_null) * 0.995:
        converted = pd.to_datetime(series, errors="coerce", dayfirst=False)
    # Same principle as numeric coercion: near-total success only, so a column of
    # mostly-dates-plus-real-text is never silently gutted.
    if converted.notna().sum() < len(non_null) * 0.995:
        return None
    return converted


def load_csv_bytes(raw: bytes, source_name: str, *, table: str | None = None) -> LoadedFile:
    """Parse, normalise and validate a CSV supplied as bytes."""
    settings = get_settings()
    issues: list[DataQualityIssue] = []

    suffix = Path(source_name).suffix.lower()
    if suffix and suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedFileError(
            f"'{suffix}' files are not supported.",
            detail=f"Supported extensions: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    size_mb = len(raw) / 1_048_576
    if size_mb > settings.max_upload_mb:
        raise FileTooLargeError(
            f"'{source_name}' is {size_mb:.1f} MB, above the {settings.max_upload_mb:.0f} MB limit.",
            detail="Pre-aggregate the file or raise MAX_UPLOAD_MB.",
        )
    if not raw.strip():
        raise EmptyFileError(f"'{source_name}' is empty.")

    encoding = _detect_encoding(raw)
    if encoding not in ("utf-8", "utf-8-sig"):
        issues.append(
            DataQualityIssue(
                severity="info",
                kind="encoding",
                message=f"File decoded as {encoding} (not UTF-8); characters were transliterated where needed.",
            )
        )

    text = raw.decode(encoding, errors="replace")
    delimiter = _detect_delimiter(text[:65_536])

    # Read the header line ourselves before pandas gets it. pandas silently renames
    # repeated headers to `id`, `id.1`, `id.2`, so by the time we see the frame the
    # duplication is invisible — and a duplicated header usually means the file was
    # assembled by hand and deserves a warning.
    raw_header: list[str] = []
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if first_line:
        try:
            raw_header = next(csv.reader([first_line], delimiter=delimiter))
        except (csv.Error, StopIteration):
            raw_header = []
    duplicated_headers = sorted(
        {h.strip() for h in raw_header if h.strip() and raw_header.count(h) > 1}
    )

    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            engine="python",
            skip_blank_lines=True,
            on_bad_lines="warn",
        )
    except pd.errors.EmptyDataError as exc:
        raise EmptyFileError(f"'{source_name}' contains no parsable rows.") from exc
    except pd.errors.ParserError as exc:
        raise IngestionError(
            f"Could not parse '{source_name}' as delimited text.",
            detail=str(exc)[:400],
        ) from exc

    if frame.empty:
        raise EmptyFileError(
            f"'{source_name}' has headers but no data rows.",
            detail="At least one data row is required.",
        )

    # --- headers -----------------------------------------------------------
    original_cols = [str(c) for c in frame.columns]
    unnamed = [c for c in original_cols if c.startswith("Unnamed:")]
    if unnamed:
        # Drop index-like columns that are entirely null, keep the rest.
        droppable = [c for c in unnamed if frame[c].isna().all()]
        if droppable:
            frame = frame.drop(columns=droppable)
            issues.append(
                DataQualityIssue(
                    severity="info",
                    kind="dropped_column",
                    message=f"Dropped {len(droppable)} unnamed all-null column(s).",
                )
            )

    safe_cols = [sanitize_identifier(c, fallback=f"col_{i}") for i, c in enumerate(frame.columns)]
    safe_cols, renamed = _dedupe(safe_cols)
    rename_map = dict(zip(frame.columns, safe_cols))
    changed = [f"{o} -> {n}" for o, n in rename_map.items() if str(o) != n]
    frame.columns = safe_cols
    if changed:
        issues.append(
            DataQualityIssue(
                severity="info",
                kind="renamed_column",
                message="Normalised column names for SQL: " + "; ".join(changed[:12])
                + ("…" if len(changed) > 12 else ""),
            )
        )
    if renamed or duplicated_headers:
        detail = "; ".join(renamed) if renamed else ", ".join(duplicated_headers)
        issues.append(
            DataQualityIssue(
                severity="warning",
                kind="duplicate_header",
                message=(
                    f"The header row repeats {len(duplicated_headers) or len(renamed)} name(s) "
                    f"({detail}). Each occurrence was given a distinct suffixed name, but check "
                    "which one you actually want."
                ),
            )
        )

    # --- drop fully empty rows/cols ----------------------------------------
    before = len(frame)
    frame = frame.dropna(how="all")
    if len(frame) < before:
        issues.append(
            DataQualityIssue(
                severity="info",
                kind="dropped_row",
                message=f"Dropped {before - len(frame)} completely blank row(s).",
            )
        )

    empty_cols = [c for c in frame.columns if frame[c].isna().all()]
    for col in empty_cols:
        issues.append(
            DataQualityIssue(
                severity="warning",
                kind="empty_column",
                column=col,
                message=f"Column '{col}' is entirely null and will not be useful for analysis.",
            )
        )

    # --- type normalisation -------------------------------------------------
    for col in frame.columns:
        if frame[col].dtype != object:
            continue
        stripped = frame[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        if not stripped.equals(frame[col]):
            frame[col] = stripped
            issues.append(
                DataQualityIssue(
                    severity="info", kind="whitespace", column=col,
                    message=f"Trimmed surrounding whitespace in '{col}'.",
                )
            )

        numeric = _clean_numeric_series(frame[col])
        if numeric is not None:
            frame[col], note = numeric
            issues.append(
                DataQualityIssue(
                    severity="info", kind="coerced_numeric", column=col,
                    message=f"Column '{col}' parsed as numeric ({note}).",
                )
            )
            continue

        dates = _maybe_datetime(frame[col])
        if dates is not None:
            unparsed = int(dates.isna().sum() - frame[col].isna().sum())
            frame[col] = dates
            msg = f"Column '{col}' parsed as datetime."
            if unparsed > 0:
                msg += f" {unparsed} value(s) could not be parsed and became null."
            issues.append(
                DataQualityIssue(
                    severity="warning" if unparsed else "info",
                    kind="coerced_datetime", column=col, message=msg,
                )
            )

    frame = frame.reset_index(drop=True)
    table_name = sanitize_identifier(table or Path(source_name).stem, fallback="dataset")

    log.info(
        "ingest.ok",
        source=source_name,
        table=table_name,
        rows=len(frame),
        cols=len(frame.columns),
        delimiter=delimiter,
        encoding=encoding,
        issues=len(issues),
    )
    return LoadedFile(
        table=table_name,
        source_name=source_name,
        frame=frame,
        issues=issues,
        raw_bytes=len(raw),
        delimiter=delimiter,
        encoding=encoding,
    )


def load_csv_path(path: str | Path, *, table: str | None = None) -> LoadedFile:
    p = Path(path)
    if not p.exists():
        raise IngestionError(f"File not found: {p}")
    return load_csv_bytes(p.read_bytes(), p.name, table=table)
