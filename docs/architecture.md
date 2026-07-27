# Architecture

## The core idea

An LLM is good at turning a vague business question into a precise query. It is
bad at arithmetic, and it will confidently invent a column name. So the model is
allowed to *propose* and *narrate*, and is never allowed to *compute* or *execute*.

```
question ──▶ LLM proposes SQL ──▶ deterministic guard ──▶ DuckDB executes ──▶ LLM narrates result
                                        │
                                    rejects → error text goes back to the model, which retries
```

Every number the user sees came out of a query that is shown to them. The
statistics (anomaly thresholds, forecasts, quality scores) are computed in
pandas/numpy/scikit-learn, never by the model.

## System diagram

```mermaid
flowchart TB
    subgraph clients["Clients"]
        UI["Streamlit UI<br/>ui/app.py"]
        HTTP["HTTP clients<br/>curl · scripts · evals"]
    end

    subgraph api["Transport — api/main.py"]
        REST["FastAPI<br/>REST + SSE · optional X-API-Key"]
    end

    subgraph engine["Engine — core/ (no UI or web dependency)"]
        AGENT["Agent<br/>core/agent.py<br/>bounded plan→act→observe loop"]
        LLM["Provider abstraction<br/>core/llm/<br/>Gemini · Groq · OpenAI · Null"]
        TOOLS["Tool registry<br/>core/tools/"]
        GUARD["SQL guard<br/>core/guard.py<br/>sqlglot AST validation"]
        SEM["Semantic layer<br/>core/semantic.py<br/>schema cards · derived metrics · join keys"]
        DUCK["DuckDB session<br/>core/engine.py<br/>read-only · row cap · timeout"]
        STATS["Deterministic analytics<br/>anomaly · forecast · charts · insights"]
        OBS["Observability<br/>structlog · Trace"]
        CACHE["Answer cache<br/>question + schema fingerprint"]
    end

    subgraph data["Data"]
        INGEST["Ingestion + profiling<br/>core/ingest.py · core/profile.py"]
        CSV[("Real CSVs<br/>UCI Online Retail II<br/>World Bank WDI")]
    end

    UI --> AGENT
    HTTP --> REST
    REST --> AGENT

    AGENT <--> LLM
    AGENT --> TOOLS
    AGENT --> CACHE
    AGENT --> OBS
    AGENT -. system prompt .-> SEM

    TOOLS --> GUARD
    TOOLS --> STATS
    GUARD --> DUCK
    STATS --> DUCK
    SEM --> INGEST
    DUCK --> INGEST
    INGEST --> CSV

    classDef safety fill:#fee2e2,stroke:#dc2626,stroke-width:2px
    classDef deterministic fill:#dcfce7,stroke:#16a34a
    class GUARD safety
    class STATS,INGEST deterministic
```

Red is the security boundary. Green is deterministic — those paths need no API key
and give the same answer every time.

## Request flow for one question

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent
    participant M as LLM
    participant G as Guard
    participant D as DuckDB

    U->>A: "Which country generated the highest revenue?"
    A->>A: cache lookup (question + schema fingerprint)
    A->>M: system prompt with schema cards, roles,<br/>real sample values, derived-metric hints
    M-->>A: tool call: run_sql(SELECT country, SUM(quantity*price) …)
    A->>G: validate
    G->>G: single statement · SELECT only · tables allow-listed<br/>· no file/network functions · LIMIT injected
    G-->>A: rewritten SQL
    A->>D: execute (read-only, 30s budget)
    D-->>A: 5 rows
    A->>M: rows + row count + timing
    M-->>A: prose answer + "Why:" line
    A-->>U: answer · table · SQL · reasoning trail · trace
```

If the guard rejects, or a column does not exist, the error text is returned to
the model as the tool result and it retries with a corrected call. An identical
repeated call is refused, which is the usual way a tool-calling loop spins.

## Why these choices

**DuckDB** queries DataFrames in-process with no load step, makes cross-file joins
free, and is a sandbox by construction — a bad generated query cannot reach a real
warehouse.

**sqlglot** parses to an AST, so validation is structural rather than regex-based.
String scanning is used only as a secondary check, after literals are stripped so
data cannot trip it.

**A hand-written agent loop** instead of a framework. It is ~150 lines with no
hidden control flow, unit-testable with a scripted fake provider, and every
transition lands in a trace. A graph runtime earns its place when you need
checkpointed state, resumable branches or human-in-the-loop pauses; a single
analytics turn needs none of those.

**Raw HTTP for every LLM provider** rather than three vendor SDKs: one dependency,
one upgrade path, and an identical tool-calling contract, so the agent contains no
provider conditionals.

**A rich semantic layer.** Published evaluations of LLM-generated SQL find failures
are dominated by schema hallucination and wrong join paths rather than syntax, and
that supplying explicit schema context is what lifts accuracy substantially. So
the model gets exact names and types, inferred column roles, real sample values,
measured ranges, null rates, verified join keys with overlap percentages, and
explicit arithmetic for metrics the data does not store.

## Layering rule

`core/` imports neither Streamlit nor FastAPI. The UI, the HTTP API and the
evaluation harness are three peer consumers of the same library, which is why the
tests exercise the exact code the app runs.

```
core/          engine        — no UI, no web framework
├── api/       transport     — imports core
├── ui/        presentation  — imports core
└── evals/     evaluation    — imports core
```

## Safety layers around query execution

| Layer | Mechanism | Stops |
|---|---|---|
| 1. Identifier sanitisation | `core/ingest.py` normalises every column name at load | A crafted CSV header injecting SQL |
| 2. AST validation | `core/guard.py` via sqlglot | Statement chaining, DDL/DML, unknown tables, schema-qualified names |
| 3. Function deny-list | `core/guard.py` | `read_csv`, `read_parquet`, `glob`, `postgres_scan`, `getenv`, extension loading |
| 4. Engine hardening | `enable_external_access=false`, then `lock_configuration=true` | Filesystem and network access even if a query slipped through, and re-enabling it |
| 5. Row cap | LIMIT injected/reduced by the guard | Memory exhaustion from an unbounded scan |
| 6. Timeout | Worker thread + `connection.interrupt()` | A cartesian join hanging the app |
| 7. No code execution | Generated pandas is parsed and screened, then **displayed only** | Arbitrary code execution — there is no `exec` anywhere in the app |

## Known limitations

- **Sessions are in-process.** One API replica only; `--workers=1` is set in
  compose for that reason. Redis plus object storage is the change needed to scale
  out.
- **Whole files are held in memory.** Fine to a few million rows on a normal
  machine; beyond that, register Parquet on disk with DuckDB instead.
- **Conversation memory is a rolling window** of recent turns plus truncated
  assistant replies, not a summarising memory. Very long sessions lose early
  context.
- **The answer cache is per process** and cleared on restart.
- **The country join is partial by design** (33 of 42 names match). See
  `data/README.md`.
