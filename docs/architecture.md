# Architecture — AI Data Analyst

> Built for the Digital Back Office AI Engineer Assignment

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interface                               │
│                   Streamlit  (port 8501)                            │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│   │  Chat    │ │Overview  │ │ Insights │ │ Quality  │ │ Explore│  │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘  │
└────────┼────────────┼────────────┼─────────────┼───────────┼───────┘
         │            │            │             │           │
         ▼            ▼            ▼             ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Core Engine                                │
│                                                                     │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────┐    │
│  │  DataSession │  │    Agent      │  │   Deterministic Layer  │    │
│  │             │  │  (LLM loop)   │  │                        │    │
│  │ - datasets  │  │               │  │  ┌──────────────────┐  │    │
│  │ - history   │◄─┤ plan → act    │  │  │  dashboard.py    │  │    │
│  │ - join_hints│  │ → observe     │  │  │  anomaly.py      │  │    │
│  │             │  │ → answer      │  │  │  insights.py     │  │    │
│  └─────────────┘  │               │  │  │  forecast.py     │  │    │
│                   │  ┌──────────┐ │  │  │  profile.py      │  │    │
│  ┌─────────────┐  │  │  Tools   │ │  │  │  reports.py      │  │    │
│  │   Ingest    │  │  │          │ │  │  └──────────────────┘  │    │
│  │             │  │  │ run_sql  │ │  └────────────────────────┘    │
│  │ - CSV parse │  │  │ run_pand │ │                                 │
│  │ - type cast │  │  │ create_  │ │  ┌──────────────────────────┐  │
│  │ - validate  │  │  │   chart  │ │  │    Guard (Read-Only SQL)  │  │
│  └──────┬──────┘  │  │ anomaly  │ │  │                          │  │
│         │         │  │ forecast │ │  │  - AST parse every SQL   │  │
│         ▼         │  └──────────┘ │  │  - blocks DROP/UPDATE/   │  │
│  ┌─────────────┐  └───────┬───────┘  │    INSERT/DELETE/ATTACH  │  │
│  │   DuckDB    │◄─────────┘          │  - row-count cap         │  │
│  │  (in-proc)  │                     └──────────────────────────┘  │
│  └─────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Supporting Services                             │
│                                                                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │  LLM Client   │  │  Answer Cache │  │   Observability       │   │
│  │               │  │               │  │                       │   │
│  │  Gemini/Groq/ │  │  In-memory    │  │  Structured logging   │   │
│  │  OpenAI       │  │  LRU keyed by │  │  per-turn Trace       │   │
│  │  + fallback   │  │  schema + Q   │  │  step timings         │   │
│  │    chain      │  │               │  │  token counts         │   │
│  └───────────────┘  └───────────────┘  └───────────────────────┘   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  FastAPI REST API  (port 8000)  — programmatic access only    │  │
│  │  POST /sessions  · POST /datasets · POST /sql · GET /report   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow — Natural Language Question

```
User types question
        │
        ▼
  Agent.run(session, question)
        │
        ├─ 1. Cache lookup (schema + question hash)
        │         Hit → return cached answer immediately
        │
        ├─ 2. Build system prompt
        │         schema context (column roles, sample values, join hints)
        │         + last 4 conversation turns for continuity
        │
        ├─ 3. LLM call (plan step)
        │         Returns: list of tool calls to make
        │         Fallback chain: gemini-3.5-flash → gemini-3.6-flash → gemini-3.1-flash-lite
        │
        ├─ 4. Tool execution loop
        │    ┌─────────────────────────────────────────────────────┐
        │    │ run_sql       → Guard validates AST → DuckDB runs   │
        │    │ run_pandas    → Restricted exec (no imports/loops)  │
        │    │ create_chart  → ValidatedChartSpec → Plotly figure  │
        │    │ detect_anomaly → IQR + Z-score + Isolation Forest   │
        │    │ forecast      → Exponential Smoothing / ARIMA       │
        │    └─────────────────────────────────────────────────────┘
        │
        ├─ 5. LLM call (answer step)
        │         Narrates over the verified tool outputs
        │         Never invents numbers — only references actual results
        │
        ├─ 6. Store in session history (conversation continuity)
        │
        └─ 7. Cache the answer
                    │
                    ▼
              AgentAnswer → UI renders artifacts (tables, charts, code)
```

---

## Data Flow — CSV Upload

```
User drops file(s) onto uploader
        │
        ▼
  session.add_csv_bytes(raw_bytes, filename)
        │
        ├─ ingest.py: detect encoding → parse CSV/TSV
        ├─ ingest.py: infer column types (numeric, date, categorical, ID)
        ├─ DuckDB: COPY into in-process table
        ├─ profile.py: compute null %, distinct count, sample values, role
        ├─ profile.py: calculate quality score (completeness + uniqueness)
        └─ engine.py: detect join hints across all loaded tables
                    │
                    ▼
              Dataset registered → UI switches to workspace tabs
```

---

## Module Map

| Module | Responsibility |
|---|---|
| `core/engine.py` | `DataSession` — owns DuckDB connection, dataset registry, history |
| `core/agent.py` | LLM plan → act → observe loop with streaming |
| `core/ingest.py` | CSV parsing, type inference, encoding detection |
| `core/profile.py` | Column profiling, quality scoring, issue detection |
| `core/guard.py` | SQL AST validation — blocks all write/DDL statements |
| `core/tools/` | `run_sql`, `run_pandas`, `create_chart`, `detect_anomaly`, `forecast` |
| `core/anomaly.py` | IQR, Z-score, Isolation Forest anomaly detection |
| `core/charts.py` | `ChartSpec` → validated Plotly figure builder |
| `core/dashboard.py` | Deterministic KPI + panel generation from column roles |
| `core/insights.py` | Statistical fact extraction + LLM-narrated summary |
| `core/forecast.py` | Exponential Smoothing / ARIMA time-series forecasting |
| `core/semantic.py` | Schema context builder, question suggestions |
| `core/cache.py` | In-memory LRU answer cache keyed by schema + question |
| `core/reports.py` | PDF, Excel, Markdown, HTML report generation |
| `core/observability.py` | Structured logging, per-turn `Trace` with step timings |
| `core/llm/` | LLM client with provider fallback chain |
| `core/prompts.py` | System and user prompt templates |
| `api/main.py` | FastAPI REST transport layer |
| `ui/app.py` | Streamlit UI — all tabs and rendering logic |
| `ui/theme.py` | Plotly chart theming + SVG icon system |

---

## Technology Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.x |
| REST API | FastAPI + Uvicorn |
| In-process analytics | DuckDB (SQL) + Pandas + NumPy |
| ML / Statistics | scikit-learn (Isolation Forest), statsmodels (ARIMA), scipy |
| Charting | Plotly |
| LLM | Google Gemini / Groq / OpenAI (pluggable, fallback chain) |
| Data validation | Pydantic v2 |
| Containerisation | Docker (multi-stage) + Docker Compose |
| Testing | pytest (262 test functions, 12 test files) |
| Evaluation | Custom golden-set eval framework in `evals/` |

---

## Security Model

- **Read-only SQL:** every statement is parsed into an AST by the Guard before DuckDB runs it. `DROP`, `UPDATE`, `INSERT`, `DELETE`, `ATTACH`, `COPY TO` and any file-system access are blocked at the AST level — not by string matching.
- **Restricted Pandas:** the pandas executor uses `compile()` + `exec()` with a locked global scope — no imports, no loops, no attribute access to filesystem objects.
- **Non-root container:** the Docker image runs as user `analyst` (uid 10001).
- **Row caps:** all queries are capped at configurable row limits to prevent memory exhaustion.
