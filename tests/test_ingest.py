"""Ingestion tests.

The headline cases are regressions found by running the real UCI file through an
earlier version of this module: a 95%-tolerance numeric coercion silently turned
every credit-note invoice (``C489449``) into NULL. Those tests exist so it cannot
come back.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.errors import EmptyFileError, FileTooLargeError, IngestionError, UnsupportedFileError
from core.ingest import load_csv_bytes, load_csv_path, sanitize_identifier
from tests.conftest import requires_real_data


# --------------------------------------------------------------------------- #
# Identifier sanitisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Customer ID", "customer_id"),
        ("Total Revenue (£)", "total_revenue"),
        ("  spaced  ", "spaced"),
        ("2024", "col_2024"),
        ("Ünïcodé", "unicode"),
        ("a--b__c", "a_b_c"),
        ('"a"; DROP TABLE x; --', "a_drop_table_x"),
        ("", "col"),
    ],
)
def test_sanitize_identifier(raw: str, expected: str) -> None:
    assert sanitize_identifier(raw) == expected


# --------------------------------------------------------------------------- #
# Real file
# --------------------------------------------------------------------------- #
@requires_real_data
def test_loads_real_retail_file(retail_path) -> None:
    loaded = load_csv_path(retail_path)
    assert loaded.table == "online_retail_ii_international"
    assert loaded.frame.shape[0] == 86_041
    assert list(loaded.frame.columns) == [
        "invoice", "stockcode", "description", "quantity",
        "invoicedate", "price", "customer_id", "country",
    ]


@requires_real_data
def test_credit_note_invoices_survive_ingestion(retail_path) -> None:
    """Regression: ``Invoice`` mixes 489434 with C489449 (credit notes).

    A lenient numeric coercion nulls the C-prefixed values. They must stay text.
    """
    loaded = load_csv_path(retail_path)
    invoices = loaded.frame["invoice"]
    assert invoices.dtype == object, "invoice must stay text, not be coerced to float"
    assert invoices.isna().sum() == 0, "no invoice value may be lost to coercion"
    credit_notes = invoices[invoices.astype(str).str.startswith("C")]
    assert len(credit_notes) > 0, "the real file contains credit notes"


@requires_real_data
def test_real_dates_and_measures_are_typed(retail_path) -> None:
    loaded = load_csv_path(retail_path)
    assert pd.api.types.is_datetime64_any_dtype(loaded.frame["invoicedate"])
    assert pd.api.types.is_numeric_dtype(loaded.frame["quantity"])
    assert pd.api.types.is_numeric_dtype(loaded.frame["price"])
    # Real returns data: negative quantities must be preserved, not clipped.
    assert (loaded.frame["quantity"] < 0).sum() > 0


@requires_real_data
def test_real_country_reference_file(country_path) -> None:
    loaded = load_csv_path(country_path)
    assert loaded.frame.shape[0] > 200
    assert "world_bank_region" in loaded.frame.columns


# --------------------------------------------------------------------------- #
# Type coercion rules
# --------------------------------------------------------------------------- #
def test_mixed_alphanumeric_column_is_not_coerced() -> None:
    csv = b"code,value\n100,1\n101,2\nC102,3\n103,4\n" + b"".join(
        f"{i},5\n".encode() for i in range(200, 400)
    )
    loaded = load_csv_bytes(csv, "mixed.csv")
    assert loaded.frame["code"].dtype == object
    assert loaded.frame["code"].isna().sum() == 0


def test_currency_and_percent_strings_are_parsed() -> None:
    csv = b'revenue,discount\n"$1,234.50",12%\n"$2,000.00",5%\n"$999.99",0%\n'
    loaded = load_csv_bytes(csv, "money.csv")
    assert loaded.frame["revenue"].tolist() == [1234.50, 2000.00, 999.99]
    assert loaded.frame["discount"].tolist() == pytest.approx([0.12, 0.05, 0.0])


def test_nullish_tokens_do_not_block_numeric_coercion() -> None:
    """``-`` and ``N/A`` are missing-value markers, not evidence of a text column.

    A second column keeps every row alive; a single-column file would have its
    all-null rows dropped, which is correct but would obscure what is under test.
    """
    csv = b"id,amount\n1,10\n2,20\n3,N/A\n4,30\n5,-\n6,40\n"
    loaded = load_csv_bytes(csv, "nulls.csv")
    assert pd.api.types.is_numeric_dtype(loaded.frame["amount"])
    assert len(loaded.frame) == 6
    assert loaded.frame["amount"].isna().sum() == 2


def test_parenthesised_negatives_are_parsed() -> None:
    csv = b"balance\n100\n(50)\n200\n"
    loaded = load_csv_bytes(csv, "neg.csv")
    assert loaded.frame["balance"].tolist() == [100.0, -50.0, 200.0]


def test_date_column_with_real_text_is_left_alone() -> None:
    csv = b"when\n2024-01-01\n2024-01-02\nnot a date at all\nsome other text\n"
    loaded = load_csv_bytes(csv, "dates.csv")
    assert loaded.frame["when"].dtype == object


# --------------------------------------------------------------------------- #
# Structure and encoding
# --------------------------------------------------------------------------- #
def test_semicolon_delimiter_is_detected() -> None:
    csv = b"a;b;c\n1;2;3\n4;5;6\n7;8;9\n"
    loaded = load_csv_bytes(csv, "euro.csv")
    assert loaded.delimiter == ";"
    assert list(loaded.frame.columns) == ["a", "b", "c"]


def test_tab_delimiter_is_detected() -> None:
    loaded = load_csv_bytes(b"a\tb\n1\t2\n3\t4\n", "tabbed.tsv")
    assert loaded.delimiter == "\t"


def test_cp1252_bytes_are_decoded_not_crashed() -> None:
    loaded = load_csv_bytes(b"name,city\nJos\xe9,Z\xfcrich\n", "latin.csv")
    assert loaded.encoding in ("cp1252", "latin-1")
    assert len(loaded.frame) == 1


def test_utf8_bom_is_stripped_from_the_first_header() -> None:
    loaded = load_csv_bytes("region,value\nNorth,1\n".encode("utf-8-sig"), "bom.csv")
    assert list(loaded.frame.columns) == ["region", "value"]


def test_duplicate_headers_are_suffixed_and_reported() -> None:
    loaded = load_csv_bytes(b"id,id,id\n1,2,3\n", "dupes.csv")
    assert list(loaded.frame.columns) == ["id", "id_1", "id_2"]
    assert any(i.kind == "duplicate_header" for i in loaded.issues)


def test_blank_rows_are_dropped_and_reported() -> None:
    loaded = load_csv_bytes(b"a,b\n1,2\n,\n3,4\n", "blank.csv")
    assert len(loaded.frame) == 2
    assert any(i.kind == "dropped_row" for i in loaded.issues)


def test_whitespace_is_trimmed_and_reported() -> None:
    loaded = load_csv_bytes(b"region\n  North  \nSouth\n", "ws.csv")
    assert loaded.frame["region"].tolist() == ["North", "South"]
    assert any(i.kind == "whitespace" for i in loaded.issues)


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_rejects_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        load_csv_bytes(b"", "empty.csv")


def test_rejects_header_only_file() -> None:
    with pytest.raises(EmptyFileError):
        load_csv_bytes(b"a,b,c\n", "headers.csv")


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileError):
        load_csv_bytes(b"a,b\n1,2\n", "data.xlsx")


def test_rejects_oversized_file(monkeypatch) -> None:
    from core import config

    monkeypatch.setattr(config.get_settings(), "max_upload_mb", 0.00001)
    with pytest.raises(FileTooLargeError):
        load_csv_bytes(b"a,b\n1,2\n" * 500, "big.csv")


def test_rejects_binary_content() -> None:
    with pytest.raises((UnsupportedFileError, IngestionError)):
        load_csv_bytes(bytes(range(256)) * 40, "blob.csv")


def test_missing_path_raises_typed_error(tmp_path) -> None:
    with pytest.raises(IngestionError):
        load_csv_path(tmp_path / "nope.csv")
