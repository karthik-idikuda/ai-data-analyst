# Usage guide

A plain, click-by-click walkthrough of every screen in the app. If you only need
the fast version, see the [main README](../README.md#step-by-step-walkthrough)
instead — this document goes deeper into each workspace.

---

## 1. Install and run

```bash
git clone <your-repo-url> && cd ai-data-analyst
make setup      # creates .venv, installs requirements.txt, copies .env.example -> .env
make data       # downloads the two real sample datasets
make ui         # starts Streamlit on http://localhost:8501
```

Prefer Docker?

```bash
cp .env.example .env
docker compose up -d
# UI  -> http://localhost:8501
# API -> http://localhost:8000/docs
```

Add an LLM key to `.env` (`LLM_PROVIDER` and `LLM_API_KEY`) for natural-language
chat and narrated insights. Every other feature — upload, validation, profiling,
SQL, charts, anomaly detection, forecasting, the dashboard, and report export —
works without one.

---

## 2. Load a dataset

On first open you will see the landing screen: an upload box in the centre and six
capability cards below it explaining what the app does.

- **Bring your own data.** Drag a CSV, TSV or TXT file (or several) onto the
  upload box. Up to 200 MB per file.
- **Or try the real sample data.** Click **Load sample workspace**. A dialog opens
  showing each bundled dataset by name, description and licence, so you choose
  what gets loaded rather than something loading silently:
  - `online_retail_ii_international.csv` — 86,041 real UK online-retailer
    transactions, Dec 2009 – Dec 2011, 42 countries.
  - `world_bank_country_profile.csv` — 217 countries with region, income group,
    GDP per capita and population.

Once a dataset loads, the sidebar shows the running total of datasets, rows and
average quality score, and the app switches into the main workspace.

---

## 3. The sidebar

The sidebar is present on every workspace and shows, top to bottom:

1. **Navigation** — jump between Overview, Chat, Insights, Quality, Explore and
   Export.
2. **Dataset count** for the current session.
3. **Load another dataset** — resets the workspace back to the upload screen.

---

## 4. Overview — the auto-built dashboard

No language-model call happens here; everything is computed directly from the
data, so it is deterministic and available with zero API key.

- **Dataset selector** — pick which loaded table the dashboard describes.
- **Key figures** — row count, the derived headline measure (e.g. revenue,
  computed as `quantity * price` when the file has no revenue column and the app
  says so explicitly), average, distinct counts and the covered date range.
- **Visual analysis panels** — two to four charts built from the columns actually
  present, each with a **Query** expander showing the exact SQL that produced it.

---

## 5. Chat — ask anything in natural language

This is the natural-language core of the app.

1. If you haven't asked anything yet, you'll see a set of suggested questions
   generated from the columns and relationships that are actually loaded — not a
   fixed script.
2. Type a question, e.g.:
   - *"Which country generated the highest revenue?"*
   - *"Show monthly sales trends."*
   - *"What are the top five customers?"*
   - *"Detect anomalies in the dataset."*
   - *"Generate the SQL for this analysis."*
3. Watch the **status panel** while the agent works — it shows each tool call
   live (`Running run_sql…`, `Running create_chart…`) rather than a generic
   spinner.
4. The answer appears with any charts, tables or code it produced, followed by an
   **"How this answer was produced"** expander:
   - **Steps taken** — the reasoning trail in order.
   - **Statements executed** — the literal SQL or pandas that ran.
   - **Execution trace** — step count, latency, tokens in/out, and a trace ID you
     can quote when reporting an issue.
5. Ask a follow-up in the same session — *"and the second one?"* — and the agent
   resolves it against the previous turn's context.
6. **Clear conversation** resets the chat history without touching your loaded
   datasets.

---

## 6. Insights — measured summaries

1. Choose a dataset.
2. Click **Generate**. Statistics (totals, trend, top segments) are computed
   first via SQL.
3. If an LLM key is configured, the numbers are narrated into plain English,
   streamed token by token. Without a key, you get the deterministic statistical
   summary directly — the app never blocks on a missing key.
4. Expand **Computed statistics (JSON)** to see every number the narration is
   constrained to.

---

## 7. Quality — data health report

1. Pick a dataset from the selector.
2. Review the **quality score** (completeness + row uniqueness), broken down into
   a full findings table: severity, affected column, and the specific issue
   (nulls, duplicates, constant columns, inconsistent casing, and more).
3. Scroll to the **column catalog** for every column's inferred type, analytical
   role (measure / dimension / temporal / identifier), distinct count, null rate,
   value range and real sample values.
4. If multiple files are loaded, **verified relationships** shows any detected
   join keys with the measured percentage overlap — never an invented mapping.

---

## 8. Explore — run SQL or pandas directly

For when you want to skip the LLM and query the data yourself.

- **SQL tab** — write a `SELECT` statement against any loaded table. Every
  statement passes through the same read-only guard the agent uses: no DDL/DML,
  no filesystem access, automatic row caps.
- **pandas tab** — write a single pandas expression. `df` is bound to the main
  table; every other loaded table is available by its own name. Assignments,
  imports and loops are rejected before execution, not caught afterwards.

Both tabs show the row count, execution time, and a **Download CSV** button for
the result.

---

## 9. Export — download the whole session

Pick a format and download a report covering every dataset, its schema, its
quality findings and the full conversation:

| Format | Best for |
|---|---|
| PDF | sharing a finished report |
| Excel workbook | further analysis in a spreadsheet |
| Markdown | pasting into docs or a PR description |
| HTML | viewing in a browser with zero dependencies |

---

## 10. Ending a session

Click **Load another dataset** in the sidebar (UI) at any time to close the
current session and native resources cleanly and return to the landing screen.
Using the API instead? `DELETE /sessions/{id}` does the same thing programmatically.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Chat says no LLM is configured | `LLM_API_KEY` is empty in `.env` | Add a key from Gemini, Groq or OpenAI — all have free tiers |
| A question is refused instead of answered | The data genuinely does not contain that column | This is intentional — the app refuses rather than guesses; check the Quality tab for what's really in the file |
| Upload rejected | File is not valid CSV/TSV/TXT, or is empty/duplicate-headered | The error message names the exact problem; fix the source file and re-upload |
| A chart doesn't render | The underlying query returned no rows or an unusable shape | The table view still shows the data so you can see why |
