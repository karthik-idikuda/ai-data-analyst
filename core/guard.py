"""The SQL guard: a deterministic control layer in front of query execution.

The design principle: **the LLM may propose SQL, but it never executes SQL.**
Generated text is parsed into an AST with sqlglot and must survive every check
below before DuckDB sees it. This is the standard mitigation for the known
failure mode where a text-to-SQL system executes model output directly and
inherits both injection risk and destructive-statement risk.

Checks performed, in order:
  1. Non-empty, single statement (blocks ``SELECT 1; DROP TABLE t``).
  2. Parses as valid DuckDB SQL.
  3. Root node is a read-only projection (SELECT / WITH / UNION only).
  4. No DDL, DML or side-effecting command nodes anywhere in the tree.
  5. Every referenced table is on the session allow-list (CTE aliases excepted).
  6. No banned function (filesystem, network, shell, extension loading).
  7. A LIMIT is enforced on the outermost query.

The guard returns a *rewritten* statement — the string that actually runs — so
what the UI shows the user is what the database executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .errors import UnsafeQueryError
from .observability import get_logger

log = get_logger(__name__)

DIALECT = "duckdb"

# Any of these node types anywhere in the tree is an immediate rejection.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Copy, exp.Attach, exp.Detach,
    exp.Command, exp.Set, exp.Grant, exp.Use, exp.Transaction, exp.Commit,
    exp.Rollback, exp.AlterColumn, exp.Pragma if hasattr(exp, "Pragma") else exp.Command,
)

# Function / table-function names that would let a query touch anything other
# than the registered in-memory tables.
_BANNED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
        "read_ndjson", "read_ndjson_auto", "read_text", "read_blob", "parquet_scan",
        "csv_scan", "json_scan", "glob", "sniff_csv", "delta_scan", "iceberg_scan",
        "postgres_scan", "postgres_query", "mysql_scan", "mysql_query", "sqlite_scan",
        "sqlite_query", "duckdb_query", "shell", "system", "getenv", "load_extension",
        "install_extension", "gen_random_uuid_v7", "read_blob_auto", "httpfs",
        "url_encode_file", "write_csv", "write_parquet", "copy_to", "read_arrow",
    }
)

# Statement keywords that must not appear even inside strings-free SQL text.
# Belt-and-braces check for constructs sqlglot may parse into a generic node.
_BANNED_KEYWORDS: tuple[str, ...] = (
    "attach", "detach", "install", "load ", "export ", "import ",
    "pragma", "call ", "checkpoint", "vacuum", "copy ",
)


@dataclass
class GuardResult:
    sql: str                       # the rewritten SQL that will execute
    original_sql: str
    tables: list[str] = field(default_factory=list)
    limit_applied: int | None = None
    warnings: list[str] = field(default_factory=list)


def _collect_cte_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias_or_name
        if alias:
            names.add(alias.lower())
    return names


def _strip_string_literals(sql: str) -> str:
    """Remove quoted literals so keyword scanning cannot be fooled by data."""
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def validate_sql(
    sql: str,
    allowed_tables: set[str] | list[str],
    *,
    max_rows: int = 5_000,
) -> GuardResult:
    """Validate and rewrite a candidate SQL statement.

    Raises:
        UnsafeQueryError: with a message safe to show the user and to feed back
            to the model as a repair hint.
    """
    allowed = {t.lower() for t in allowed_tables}
    raw = (sql or "").strip().rstrip(";").strip()
    if not raw:
        raise UnsafeQueryError("Empty SQL statement.")

    scrubbed = _strip_string_literals(raw).lower()
    for kw in _BANNED_KEYWORDS:
        if kw in scrubbed:
            raise UnsafeQueryError(
                f"Statement rejected: '{kw.strip().upper()}' is not permitted.",
                detail="Only read-only SELECT queries over the loaded tables are allowed.",
            )

    try:
        statements = [s for s in sqlglot.parse(raw, read=DIALECT) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise UnsafeQueryError(
            "The SQL could not be parsed.",
            detail=str(exc)[:400],
        ) from exc

    if len(statements) != 1:
        raise UnsafeQueryError(
            f"Expected exactly one statement, found {len(statements)}.",
            detail="Multiple statements are rejected to prevent statement injection.",
        )

    tree = statements[0]

    if not isinstance(tree, (exp.Select, exp.Union, exp.Subquery, exp.With, exp.Except, exp.Intersect)):
        raise UnsafeQueryError(
            f"Only SELECT queries are allowed; got {type(tree).__name__.upper()}.",
        )

    for node_type in _FORBIDDEN_NODES:
        found = tree.find(node_type)
        if found is not None:
            raise UnsafeQueryError(
                f"Statement rejected: {node_type.__name__.upper()} is not permitted.",
                detail="This engine is strictly read-only.",
            )

    # --- function allow-list ------------------------------------------------
    for func in tree.find_all(exp.Anonymous):
        name = (func.name or "").lower()
        if name in _BANNED_FUNCTIONS:
            raise UnsafeQueryError(
                f"Function '{name}' is not permitted.",
                detail="Filesystem, network and extension functions are blocked.",
            )
    for func in tree.find_all(exp.Func):
        name = (func.sql_name() or "").lower()
        if name in _BANNED_FUNCTIONS:
            raise UnsafeQueryError(f"Function '{name}' is not permitted.")

    # --- table allow-list ---------------------------------------------------
    cte_names = _collect_cte_names(tree)
    referenced: list[str] = []
    for table in tree.find_all(exp.Table):
        if table.db or table.catalog:
            raise UnsafeQueryError(
                "Schema- or catalog-qualified table names are not permitted.",
                detail=f"Reference tables directly by name: {', '.join(sorted(allowed))}",
            )
        name = (table.name or "").lower()
        if not name or name in cte_names:
            continue
        if name not in allowed:
            raise UnsafeQueryError(
                f"Unknown table '{table.name}'.",
                detail=f"Available tables: {', '.join(sorted(allowed)) or '(none loaded)'}",
            )
        if name not in referenced:
            referenced.append(name)

    if not referenced:
        raise UnsafeQueryError(
            "The query does not reference any loaded table.",
            detail=f"Available tables: {', '.join(sorted(allowed)) or '(none loaded)'}",
        )

    # --- enforce a row cap on the outermost query ---------------------------
    warnings: list[str] = []
    limit_applied: int | None = None
    if isinstance(tree, exp.Select) or isinstance(tree, (exp.Union, exp.Except, exp.Intersect)):
        existing = tree.args.get("limit")
        if existing is None:
            tree = tree.limit(max_rows)
            limit_applied = max_rows
        else:
            try:
                current = int(existing.expression.this)
                if current > max_rows:
                    tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                    limit_applied = max_rows
                    warnings.append(
                        f"LIMIT reduced from {current:,} to the {max_rows:,}-row cap."
                    )
            except (AttributeError, TypeError, ValueError):
                # Non-literal LIMIT (expression/parameter): replace with the cap.
                tree.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                limit_applied = max_rows

    final_sql = tree.sql(dialect=DIALECT, pretty=True)
    log.info("guard.pass", tables=referenced, limit_applied=limit_applied)
    return GuardResult(
        sql=final_sql,
        original_sql=raw,
        tables=referenced,
        limit_applied=limit_applied,
        warnings=warnings,
    )


def format_sql(sql: str) -> str:
    """Pretty-print SQL for display; returns the input unchanged if unparsable."""
    try:
        return sqlglot.transpile(sql, read=DIALECT, write=DIALECT, pretty=True)[0]
    except Exception:  # noqa: BLE001 - formatting must never break a response
        return sql
