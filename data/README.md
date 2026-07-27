# Datasets

Both files are **real published data**. Nothing here is generated, simulated or
synthetic. Regenerate them at any time with:

```bash
python scripts/fetch_real_data.py          # the two committed files
python scripts/fetch_real_data.py --full   # + the complete 1,067,371-row retail file
```

---

## `online_retail_ii_international.csv`

| | |
|---|---|
| **Source** | [UCI Machine Learning Repository, dataset 502 — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **Donated by** | Dr Daqing Chen, London South Bank University |
| **Licence** | CC BY 4.0 |
| **What it is** | Every transaction recorded by a UK-based, registered non-store online retailer selling all-occasion giftware. Many customers are wholesalers. |
| **Period** | 2009-12-01 → 2011-12-09 (25 monthly periods) |
| **Committed rows** | 86,041 |
| **Countries** | 42 |

### Columns (headers exactly as published)

| column | type | notes |
|---|---|---|
| `Invoice` | text | 6-digit number. A `C` prefix means a **credit note** (a return). |
| `StockCode` | text | Product code. Non-product codes exist: `POST` (postage), `M` (manual adjustment), `D` (discount), `BANK CHARGES`. |
| `Description` | text | Product name. 4,382 are missing in the full file. |
| `Quantity` | integer | **Negative on returns.** Not an error. |
| `InvoiceDate` | timestamp | |
| `Price` | float | Unit price in GBP. Some rows are `0.00`. |
| `Customer ID` | float | Missing on ~23% of the full file (guest / untracked orders). |
| `Country` | text | Customer country. |

### There is no revenue column

Revenue must be computed as `Quantity * Price`. The app detects this and tells
the model explicitly (see `core/semantic.py::derived_metric_hints`), because
otherwise a question about revenue gets silently answered with `SUM(quantity)`.

### Why the UK rows are not committed

`Country = 'United Kingdom'` accounts for 981,330 of the 1,067,371 rows (92%) and
would make the file 95 MB. The committed slice is every **non-UK** row, complete
and unedited, which keeps the repository small while preserving all 25 months and
42 countries. No row is modified, no value imputed, no outlier removed. Use
`--full` for the complete file.

Consequence to be aware of: revenue rankings in this slice are led by **EIRE**
(£615,519.55), not the UK.

### Real imperfections in this data

These are genuine properties of the source, and the app is built to surface
rather than hide them:

- 22,950 negative quantities in the full file (returns and credit notes)
- 6,202 rows priced at 0.00
- 34,335 fully duplicated rows in the full file (1,326 in the committed slice)
- 243,007 missing customer IDs in the full file (2,978 in the slice)
- Mixed-type `Invoice` (numeric plus `C`-prefixed) — a lenient CSV parser destroys
  the credit notes here; `tests/test_ingest.py` guards against that regression
- Adjustment rows with `StockCode = 'M'` and prices up to £6,958.17, which
  legitimately dominate outlier detection

---

## `world_bank_country_profile.csv`

| | |
|---|---|
| **Source** | [World Bank Open Data — World Development Indicators](https://data.worldbank.org) |
| **Indicators** | `NY.GDP.PCAP.CD` (GDP per capita, current US$), `SP.POP.TOTL` (population), plus the official country metadata file |
| **Licence** | CC BY 4.0 |
| **Rows** | 217 countries |
| **Regions** | 7 official World Bank regions |

### Columns

| column | notes |
|---|---|
| `country` | World Bank `TableName` |
| `country_code` | ISO 3166-1 alpha-3 |
| `world_bank_region` | Official World Bank region |
| `income_group` | High / Upper middle / Lower middle / Low income |
| `gdp_per_capita_usd_2010`, `gdp_per_capita_usd_2011` | Years chosen to overlap the retail period |
| `population_2010`, `population_2011` | |

Transformations: the wide year-per-column layout was reduced to 2010 and 2011 and
joined to the metadata file. 47 rows with no region (World Bank aggregates such as
"Euro area") were dropped because they are not countries. Empty cells are left
empty — nothing is interpolated.

---

## Joining the two files

`online_retail_ii_international.country` → `world_bank_country_profile.country`

**33 of 42** retail countries match by exact name. These nine do not:

```
Czech Republic, EIRE, European Community, Hong Kong, Korea,
RSA, USA, Unspecified, West Indies
```

The two organisations genuinely name these differently (`EIRE` vs `Ireland`,
`USA` vs `United States`, `RSA` vs `South Africa`, `Czech Republic` vs `Czechia`),
and two are not countries at all (`Unspecified`, `European Community`).

**No mapping table has been added.** Writing one would mean inserting data that
neither source published. Instead the app measures the overlap itself and reports
it — the join-key detector in `core/profile.py::detect_join_hints` shows this as a
79% value overlap in the UI, so any region-level answer is visibly partial rather
than silently wrong.
