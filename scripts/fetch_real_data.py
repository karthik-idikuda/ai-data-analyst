#!/usr/bin/env python3
"""Fetch the REAL open datasets this project ships with.

Nothing here is generated, simulated or synthetic. Every row comes from a public
authoritative source and is written out unmodified apart from the transformations
listed explicitly below.

Sources
-------
1. **Online Retail II** — UCI Machine Learning Repository, dataset 502.
   Real transactions from a UK-based online giftware retailer,
   2009-12-01 to 2011-12-09. 1,067,371 rows. Donated by Dr Daqing Chen
   (London South Bank University). Licence: CC BY 4.0.
   https://archive.ics.uci.edu/dataset/502/online+retail+ii

2. **World Bank Open Data** — World Development Indicators.
   - ``NY.GDP.PCAP.CD``  GDP per capita (current US$)
   - ``SP.POP.TOTL``     Population, total
   Plus the accompanying country metadata file, which supplies the official
   World Bank *region* and *income group* for each country.
   Licence: CC BY 4.0.  https://data.worldbank.org

Transformations applied (and nothing else)
------------------------------------------
Online Retail II
  * The two workbook sheets ("Year 2009-2010", "Year 2010-2011") are concatenated
    in order — they are one continuous series split for file-size reasons.
  * Column headers are passed through unchanged; the app's ingestion layer
    normalises them at load time.
  * The committed slice excludes rows where Country == 'United Kingdom' purely so
    the file fits comfortably in a git repository (7 MB instead of 95 MB). The
    excluded rows are real; use ``--full`` to write all 1,067,371 of them.
    NO row is edited, no value is imputed, no outlier is removed.

World Bank
  * The wide year-per-column layout is reduced to the two years that overlap the
    retail data (2010, 2011) and joined to the country metadata file.
  * Empty cells stay empty — no interpolation.

Usage
-----
    python scripts/fetch_real_data.py                 # committed demo slices
    python scripts/fetch_real_data.py --full          # + full 1.07M-row file
    python scripts/fetch_real_data.py --out data
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import httpx
import pandas as pd

UCI_ONLINE_RETAIL_II = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
WORLD_BANK_GDP_PC = "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=csv"
WORLD_BANK_POP = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"

TIMEOUT = httpx.Timeout(300.0, connect=30.0)
RETAIL_YEARS = ("2010", "2011")


def _download(url: str, label: str) -> bytes:
    print(f"  downloading {label} …", flush=True)
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    print(f"    {len(response.content) / 1_048_576:.1f} MB received")
    return response.content


# --------------------------------------------------------------------------- #
# Online Retail II
# --------------------------------------------------------------------------- #
def fetch_online_retail(out_dir: Path, *, write_full: bool) -> pd.DataFrame:
    print("\n[1/2] Online Retail II — UCI Machine Learning Repository (dataset 502)")
    archive = zipfile.ZipFile(io.BytesIO(_download(UCI_ONLINE_RETAIL_II, "online+retail+ii.zip")))
    name = next(n for n in archive.namelist() if n.endswith(".xlsx"))
    print(f"  reading {name} (two sheets, this takes ~60s) …", flush=True)

    workbook = pd.ExcelFile(io.BytesIO(archive.read(name)), engine="openpyxl")
    frames = []
    for sheet in workbook.sheet_names:
        frame = workbook.parse(sheet)
        print(f"    sheet '{sheet}': {len(frame):,} rows")
        frames.append(frame)
    everything = pd.concat(frames, ignore_index=True)

    dates = pd.to_datetime(everything["InvoiceDate"], errors="coerce")
    print(
        f"  total {len(everything):,} rows · {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d} · "
        f"{everything['Country'].nunique()} countries"
    )

    if write_full:
        target = out_dir / "online_retail_ii_full.csv"
        everything.to_csv(target, index=False)
        print(f"  wrote {target} ({target.stat().st_size / 1_048_576:.1f} MB, all rows, unmodified)")

    international = everything[everything["Country"] != "United Kingdom"].copy()
    target = out_dir / "online_retail_ii_international.csv"
    international.to_csv(target, index=False)
    months = pd.to_datetime(international["InvoiceDate"]).dt.to_period("M").nunique()
    print(
        f"  wrote {target} ({target.stat().st_size / 1_048_576:.1f} MB) — "
        f"{len(international):,} unmodified rows, {international['Country'].nunique()} countries, "
        f"{months} months"
    )
    return everything


# --------------------------------------------------------------------------- #
# World Bank
# --------------------------------------------------------------------------- #
def _read_world_bank(payload: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (indicator_frame, country_metadata_frame) from a World Bank zip."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    data_name = next(
        n for n in archive.namelist() if n.startswith("API_") and n.endswith(".csv")
    )
    meta_name = next(
        n for n in archive.namelist() if n.startswith("Metadata_Country") and n.endswith(".csv")
    )
    # The indicator CSV carries four preamble lines before the real header.
    data = pd.read_csv(io.BytesIO(archive.read(data_name)), skiprows=4)
    meta = pd.read_csv(io.BytesIO(archive.read(meta_name)))
    return data, meta


def fetch_world_bank(out_dir: Path) -> pd.DataFrame:
    print("\n[2/2] World Bank Open Data — World Development Indicators")
    gdp, meta = _read_world_bank(_download(WORLD_BANK_GDP_PC, "GDP per capita (NY.GDP.PCAP.CD)"))
    pop, _ = _read_world_bank(_download(WORLD_BANK_POP, "Population (SP.POP.TOTL)"))

    keep = ["Country Name", "Country Code", *RETAIL_YEARS]
    gdp_slice = gdp[[c for c in keep if c in gdp.columns]].rename(
        columns={y: f"gdp_per_capita_usd_{y}" for y in RETAIL_YEARS}
    )
    pop_slice = pop[[c for c in keep if c in pop.columns]].rename(
        columns={y: f"population_{y}" for y in RETAIL_YEARS}
    )

    meta_slice = meta[["Country Code", "Region", "IncomeGroup", "TableName"]].rename(
        columns={
            "Country Code": "country_code",
            "Region": "world_bank_region",
            "IncomeGroup": "income_group",
            "TableName": "country",
        }
    )
    # Rows with no Region are World Bank aggregates (e.g. "Euro area"), not countries.
    aggregates = int(meta_slice["world_bank_region"].isna().sum())
    meta_slice = meta_slice[meta_slice["world_bank_region"].notna()]
    print(f"  dropped {aggregates} World Bank aggregate rows (no region = not a country)")

    merged = (
        meta_slice.merge(
            gdp_slice.rename(columns={"Country Code": "country_code"}).drop(columns=["Country Name"]),
            on="country_code",
            how="left",
        )
        .merge(
            pop_slice.rename(columns={"Country Code": "country_code"}).drop(columns=["Country Name"]),
            on="country_code",
            how="left",
        )
        .sort_values("country")
        .reset_index(drop=True)
    )
    ordered = [
        "country", "country_code", "world_bank_region", "income_group",
        "gdp_per_capita_usd_2010", "gdp_per_capita_usd_2011",
        "population_2010", "population_2011",
    ]
    merged = merged[[c for c in ordered if c in merged.columns]]

    target = out_dir / "world_bank_country_profile.csv"
    merged.to_csv(target, index=False)
    print(
        f"  wrote {target} ({target.stat().st_size / 1024:.0f} KB) — {len(merged)} countries, "
        f"{merged['world_bank_region'].nunique()} World Bank regions"
    )
    return merged


def report_join_overlap(retail: pd.DataFrame, reference: pd.DataFrame) -> None:
    """State the honest truth about how well the two real sources join."""
    retail_countries = set(retail.loc[retail["Country"] != "United Kingdom", "Country"].unique())
    reference_countries = set(reference["country"].unique())
    matched = sorted(retail_countries & reference_countries)
    unmatched = sorted(retail_countries - reference_countries)

    print("\nJoin check: online_retail_ii.country  ->  world_bank_country_profile.country")
    print(f"  {len(matched)} of {len(retail_countries)} retail countries match by exact name.")
    if unmatched:
        print("  Unmatched (the two sources genuinely name these differently, or they are not countries):")
        print("    " + ", ".join(unmatched))
    print(
        "  These are left as-is on purpose. Inventing a name-mapping table would mean adding data\n"
        "  that neither source published. The app measures and reports this overlap itself."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument(
        "--full", action="store_true",
        help="also write the complete 1,067,371-row retail file (~95 MB, not committed to git)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching real open datasets into {out_dir.resolve()}")

    try:
        retail = fetch_online_retail(out_dir, write_full=args.full)
        reference = fetch_world_bank(out_dir)
    except httpx.HTTPError as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print("Check your connection; both sources are public and need no credentials.", file=sys.stderr)
        return 1

    report_join_overlap(retail, reference)
    print("\nDone. Both files are real published data. Upload them in the app to reproduce the demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
