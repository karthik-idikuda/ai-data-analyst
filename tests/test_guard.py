"""SQL guard tests.

This is the security boundary, so it gets the most adversarial coverage in the
suite. Each rejection case corresponds to a real way an LLM-driven SQL system
gets compromised or breaks: statement chaining, DDL/DML slipping through,
filesystem-reading table functions, and unbounded result sets.
"""

from __future__ import annotations

import pytest

from core.errors import UnsafeQueryError
from core.guard import format_sql, validate_sql

TABLES = {"sales", "customers"}


# --------------------------------------------------------------------------- #
# Accepted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM sales",
        "select country, sum(quantity) from sales group by 1 order by 2 desc",
        "SELECT s.country FROM sales s JOIN customers c ON s.country = c.country",
        "WITH totals AS (SELECT country, SUM(quantity) q FROM sales GROUP BY 1) "
        "SELECT * FROM totals WHERE q > 10",
        "SELECT country FROM sales UNION SELECT country FROM customers",
        "SELECT country, COUNT(*) FROM sales WHERE quantity > 0 GROUP BY country HAVING COUNT(*) > 5",
        "SELECT date_trunc('month', invoicedate) m, SUM(quantity) FROM sales GROUP BY 1 ORDER BY 1",
        "SELECT * FROM (SELECT * FROM sales LIMIT 10)",
        # A literal containing a banned keyword must not trip the scanner.
        "SELECT * FROM sales WHERE description = 'ATTACH THE LABEL'",
    ],
)
def test_accepts_read_only_queries(sql: str) -> None:
    result = validate_sql(sql, TABLES)
    assert result.sql
    assert set(result.tables) <= {t.lower() for t in TABLES}


def test_cte_alias_is_not_treated_as_unknown_table() -> None:
    result = validate_sql(
        "WITH ranked AS (SELECT country FROM sales) SELECT * FROM ranked", TABLES
    )
    assert result.tables == ["sales"]


# --------------------------------------------------------------------------- #
# Rejected — statement shape
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE sales",
        "SELECT * FROM sales; SELECT * FROM customers",
    ],
)
def test_rejects_statement_chaining(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql, TABLES)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "UPDATE sales SET quantity = 0",
        "INSERT INTO sales VALUES (1)",
        "CREATE TABLE evil AS SELECT * FROM sales",
        "ALTER TABLE sales ADD COLUMN x INT",
        "TRUNCATE sales",
        "COPY sales TO '/tmp/leak.csv'",
        "ATTACH '/etc/passwd' AS leak",
        "INSTALL httpfs",
        "PRAGMA database_list",
        "SET memory_limit='99GB'",
    ],
)
def test_rejects_write_and_side_effecting_statements(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql, TABLES)


def test_rejects_empty_statement() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("   ", TABLES)


def test_rejects_unparsable_sql() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT FROM WHERE ORDER", TABLES)


# --------------------------------------------------------------------------- #
# Rejected — reaching outside the sandbox
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
        "SELECT * FROM glob('/**')",
        "SELECT * FROM read_json_auto('http://evil.example/x.json')",
        "SELECT getenv('LLM_API_KEY')",
        "SELECT * FROM postgres_scan('host=x', 'public', 'users')",
    ],
)
def test_rejects_filesystem_and_network_functions(sql: str) -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql, TABLES)


def test_rejects_unknown_table_and_names_the_available_ones() -> None:
    with pytest.raises(UnsafeQueryError) as exc:
        validate_sql("SELECT * FROM secrets", TABLES)
    assert "secrets" in exc.value.message
    assert "sales" in (exc.value.detail or "")


def test_rejects_schema_qualified_names() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT * FROM main.sales", TABLES)


def test_rejects_query_referencing_no_table() -> None:
    with pytest.raises(UnsafeQueryError):
        validate_sql("SELECT 1 + 1", TABLES)


def test_crafted_identifier_cannot_smuggle_a_second_statement() -> None:
    """Column names are sanitised at ingest, but the guard must hold regardless."""
    with pytest.raises(UnsafeQueryError):
        validate_sql('SELECT "a"; DROP TABLE sales; --" FROM sales', TABLES)


# --------------------------------------------------------------------------- #
# Row cap
# --------------------------------------------------------------------------- #
def test_adds_limit_when_absent() -> None:
    result = validate_sql("SELECT * FROM sales", TABLES, max_rows=100)
    assert result.limit_applied == 100
    assert "LIMIT 100" in result.sql.upper()


def test_keeps_a_smaller_user_limit() -> None:
    result = validate_sql("SELECT * FROM sales LIMIT 5", TABLES, max_rows=100)
    assert result.limit_applied is None
    assert "LIMIT 5" in result.sql.upper()


def test_reduces_an_oversized_limit_and_warns() -> None:
    result = validate_sql("SELECT * FROM sales LIMIT 999999", TABLES, max_rows=100)
    assert result.limit_applied == 100
    assert result.warnings
    assert "LIMIT 100" in result.sql.upper()


def test_format_sql_never_raises_on_garbage() -> None:
    assert format_sql("not ) valid ( sql") == "not ) valid ( sql"
