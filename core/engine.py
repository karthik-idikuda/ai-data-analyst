"""DuckDB execution engine and session state.

DuckDB is used for three reasons: it queries in-memory DataFrames with no load
step, it is an in-process sandbox so a bad generated query cannot reach a real
warehouse, and it makes multi-file joins free.

Defence in depth around query execution:

* **Guard first** — every statement passes :mod:`core.guard` before arriving here.
* **Engine hardening** — the connection sets ``enable_external_access=false`` and
  then ``lock_configuration=true``, so even a query that somehow slipped past the
  guard cannot read a file, reach the network, or re-enable those capabilities.
* **Row cap** — the guard rewrites the statement with a LIMIT.
* **Timeout** — queries run on a worker thread and the connection is interrupted
  if they exceed the budget, so a cartesian join cannot hang the app.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import get_settings
from .errors import (
    DatasetNotFoundError,
    IngestionError,
    QueryExecutionError,
    QueryTimeoutError,
)
from .guard import GuardResult, validate_sql
from .ingest import LoadedFile, load_csv_bytes, load_csv_path, sanitize_identifier
from .models import (
    ChatMessage,
    ColumnRole,
    DatasetProfile,
    JoinHint,
    QueryResult,
)
from .profile import build_profile, detect_join_hints, quality_score
from .observability import get_logger

log = get_logger(__name__)


@dataclass
class Dataset:
    table: str
    source_name: str
    frame: pd.DataFrame
    profile: DatasetProfile

    @property
    def quality(self) -> dict[str, Any]:
        return quality_score(self.profile)


class DataSession:
    """One user session: its datasets, its DuckDB connection, its chat history.

    Thread-safe for the access patterns the API uses (a lock around query
    execution and registration); DuckDB connections are not safe for concurrent
    use from multiple threads.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.datasets: dict[str, Dataset] = {}
        self.join_hints: list[JoinHint] = []
        self.history: list[ChatMessage] = []
        self.created_at = time.time()
        self._lock = threading.RLock()
        self._con = duckdb.connect(database=":memory:")
        self._harden_connection()

    # ------------------------------------------------------------------ setup
    def _harden_connection(self) -> None:
        """Disable every capability that could escape the in-memory sandbox."""
        hardening = [
            "SET enable_external_access=false",
            "SET allow_unsigned_extensions=false",
            "SET autoinstall_known_extensions=false",
            "SET autoload_known_extensions=false",
            f"SET memory_limit='{2}GB'",
            "SET threads=4",
        ]
        for stmt in hardening:
            try:
                self._con.execute(stmt)
            except duckdb.Error as exc:  # pragma: no cover - version dependent
                log.warning("engine.harden_failed", statement=stmt, error=str(exc))
        try:
            # Must be last: makes all of the above immutable for this connection.
            self._con.execute("SET lock_configuration=true")
        except duckdb.Error as exc:  # pragma: no cover
            log.warning("engine.lock_failed", error=str(exc))
        log.info("engine.hardened", session=self.session_id)

    # --------------------------------------------------------------- datasets
    @property
    def table_names(self) -> list[str]:
        return sorted(self.datasets)

    def _unique_table_name(self, base: str) -> str:
        name = sanitize_identifier(base, fallback="dataset")
        if name not in self.datasets:
            return name
        i = 2
        while f"{name}_{i}" in self.datasets:
            i += 1
        return f"{name}_{i}"

    def add_loaded_file(self, loaded: LoadedFile) -> Dataset:
        settings = get_settings()
        with self._lock:
            if len(self.datasets) >= settings.max_files_per_session:
                raise IngestionError(
                    f"Session limit of {settings.max_files_per_session} files reached.",
                    detail="Remove a dataset before adding another.",
                )
            table = self._unique_table_name(loaded.table)
            loaded.table = table
            profile = build_profile(loaded)
            profile.table = table
            dataset = Dataset(
                table=table,
                source_name=loaded.source_name,
                frame=loaded.frame,
                profile=profile,
            )
            self.datasets[table] = dataset
            # DuckDB reads the DataFrame straight out of the Python namespace.
            self._con.register(table, loaded.frame)
            self._refresh_join_hints()
            log.info(
                "session.dataset_added",
                session=self.session_id, table=table, rows=profile.row_count,
            )
            return dataset

    def add_csv_bytes(self, raw: bytes, source_name: str) -> Dataset:
        return self.add_loaded_file(load_csv_bytes(raw, source_name))

    def add_csv_path(self, path: str | Path) -> Dataset:
        return self.add_loaded_file(load_csv_path(path))

    def remove_dataset(self, table: str) -> None:
        with self._lock:
            if table not in self.datasets:
                raise DatasetNotFoundError(f"No dataset named '{table}'.")
            del self.datasets[table]
            try:
                self._con.unregister(table)
            except duckdb.Error:
                pass
            self._refresh_join_hints()

    def get_dataset(self, table: str) -> Dataset:
        ds = self.datasets.get(table) or self.datasets.get(sanitize_identifier(table))
        if ds is None:
            raise DatasetNotFoundError(
                f"No dataset named '{table}'.",
                detail=f"Loaded tables: {', '.join(self.table_names) or '(none)'}",
            )
        return ds

    def default_table(self) -> str:
        """The table to use when the user does not name one: the fact table.

        Ranking, in order: has a date column, then row count, then number of
        measures. Counting measures first looks reasonable and is wrong on real
        data — the World Bank country reference has more numeric columns (GDP,
        population, per year) than the 86k-row transaction table it describes, so
        a measure-count heuristic picks the lookup table. A transaction table is
        identified by being dated and long, which is what this scores.
        """
        if not self.datasets:
            raise DatasetNotFoundError("No datasets have been uploaded yet.")

        def key(ds: Dataset) -> tuple[int, int, int]:
            has_date = 1 if ds.profile.columns_by_role(ColumnRole.TEMPORAL) else 0
            measures = len(ds.profile.columns_by_role(ColumnRole.MEASURE))
            return (has_date, ds.profile.row_count, measures)

        return max(self.datasets.values(), key=key).table

    def _refresh_join_hints(self) -> None:
        if len(self.datasets) < 2:
            self.join_hints = []
            return
        frames = {t: d.frame for t, d in self.datasets.items()}
        profiles = {t: d.profile for t, d in self.datasets.items()}
        self.join_hints = detect_join_hints(frames, profiles)

    # ------------------------------------------------------------------ query
    def execute_sql(self, sql: str, *, max_rows: int | None = None) -> tuple[QueryResult, GuardResult]:
        """Validate then execute a statement. Returns the result and guard report."""
        settings = get_settings()
        cap = max_rows or settings.max_result_rows
        guard = validate_sql(sql, self.table_names, max_rows=cap)
        result = self._run_guarded(guard.sql, cap)
        return result, guard

    def _run_guarded(self, sql: str, cap: int) -> QueryResult:
        settings = get_settings()
        started = time.perf_counter()
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["frame"] = self._con.execute(sql).fetch_df()
            except BaseException as exc:  # noqa: BLE001 - forwarded to caller
                box["error"] = exc

        with self._lock:
            worker = threading.Thread(target=work, daemon=True, name="duckdb-query")
            worker.start()
            worker.join(timeout=settings.query_timeout_s)
            if worker.is_alive():
                try:
                    self._con.interrupt()
                except duckdb.Error:
                    pass
                worker.join(timeout=5)
                raise QueryTimeoutError(
                    f"Query exceeded the {settings.query_timeout_s:.0f}s time budget and was cancelled.",
                    detail="Add a filter, aggregate earlier, or narrow the join.",
                )

        if "error" in box:
            exc = box["error"]
            raise QueryExecutionError(
                "The database rejected the query.",
                detail=str(exc)[:600],
            ) from exc

        frame: pd.DataFrame = box["frame"]
        duration_ms = (time.perf_counter() - started) * 1000
        truncated = len(frame) >= cap
        return QueryResult(
            sql=sql,
            columns=[str(c) for c in frame.columns],
            rows=_jsonable_rows(frame),
            row_count=len(frame),
            truncated=truncated,
            duration_ms=duration_ms,
        )

    def run_dataframe_query(self, sql: str, *, max_rows: int | None = None) -> pd.DataFrame:
        """Execute guarded SQL and return a DataFrame (for internal analytics)."""
        settings = get_settings()
        cap = max_rows or settings.max_result_rows
        guard = validate_sql(sql, self.table_names, max_rows=cap)
        with self._lock:
            try:
                return self._con.execute(guard.sql).fetch_df()
            except duckdb.Error as exc:
                raise QueryExecutionError(
                    "The database rejected the query.", detail=str(exc)[:600]
                ) from exc

    # ---------------------------------------------------------------- history
    def add_message(self, message: ChatMessage) -> None:
        self.history.append(message)

    def recent_history(self, turns: int | None = None) -> list[ChatMessage]:
        n = turns if turns is not None else get_settings().history_turns
        return self.history[-(2 * n) :] if n else []

    def close(self) -> None:
        with self._lock:
            try:
                self._con.close()
            except duckdb.Error:
                pass

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "datasets": [
                {
                    "table": d.table,
                    "source_name": d.source_name,
                    "rows": d.profile.row_count,
                    "columns": d.profile.column_count,
                    "quality": d.quality,
                }
                for d in self.datasets.values()
            ],
            "join_hints": [h.model_dump() for h in self.join_hints],
            "messages": len(self.history),
        }


def _jsonable_rows(frame: pd.DataFrame) -> list[list[Any]]:
    """Convert a DataFrame to JSON-serialisable rows.

    pandas/numpy scalars (int64, Timestamp, NaT, NaN, Decimal) are not JSON
    serialisable, and letting them reach FastAPI produces opaque 500s.
    """
    if frame.empty:
        return []
    out: list[list[Any]] = []
    converted = frame.copy()
    for col in converted.columns:
        if pd.api.types.is_datetime64_any_dtype(converted[col]):
            converted[col] = converted[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        elif str(converted[col].dtype) == "object":
            converted[col] = converted[col].map(
                lambda v: v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
            )
    converted = converted.astype(object).where(pd.notna(converted), None)
    for row in converted.itertuples(index=False, name=None):
        out.append([_scalar(v) for v in row])
    return out


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Session registry
# --------------------------------------------------------------------------- #
class SessionStore:
    """In-memory session registry with LRU-style eviction.

    Documented limitation: sessions live in the process, so the API is
    single-instance. Swapping this for Redis plus object storage is the change
    required to run more than one replica.
    """

    def __init__(self, max_sessions: int = 64) -> None:
        self._sessions: dict[str, DataSession] = {}
        self._max = max_sessions
        self._lock = threading.Lock()

    def create(self) -> DataSession:
        with self._lock:
            self._evict_if_needed()
            session = DataSession()
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> DataSession | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> DataSession:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.close()

    def _evict_if_needed(self) -> None:
        while len(self._sessions) >= self._max:
            oldest = min(self._sessions.values(), key=lambda s: s.created_at)
            self._sessions.pop(oldest.session_id, None)
            oldest.close()
            log.info("session.evicted", session=oldest.session_id)


SESSIONS = SessionStore()
