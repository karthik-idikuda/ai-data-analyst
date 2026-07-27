"""Session export: Markdown and self-contained HTML.

The report is assembled from the session's real history and artifacts — the same
SQL that ran, the same tables, the same anomaly records. Nothing is regenerated,
so an exported report is a faithful audit of the session.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Any

from .engine import DataSession
from .models import Artifact
from .profile import quality_score


def _table_md(columns: list[str], rows: list[list[Any]], limit: int = 30) -> str:
    if not columns:
        return "_(no columns)_"
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        out.append("| " + " | ".join("" if v is None else str(v).replace("|", "\\|") for v in row) + " |")
    if len(rows) > limit:
        out.append(f"_… {len(rows) - limit} more rows_")
    return "\n".join(out)


def _artifact_md(artifact: Artifact) -> str:
    payload = artifact.payload
    if artifact.kind == "table":
        body = _table_md(payload.get("columns", []), payload.get("rows", []))
        sql = payload.get("sql")
        block = f"**{artifact.title}**\n\n{body}"
        if sql:
            block += f"\n\n```sql\n{sql}\n```"
        return block

    if artifact.kind == "chart":
        spec = payload.get("spec", {})
        body = _table_md(payload.get("columns", []), payload.get("rows", []), limit=20)
        return (
            f"**Chart — {artifact.title}**\n\n"
            f"- type: `{spec.get('type')}`, x: `{spec.get('x')}`, y: `{spec.get('y')}`\n\n"
            f"{body}\n\n```sql\n{payload.get('sql', '')}\n```"
        )

    if artifact.kind == "code":
        return (
            f"**{artifact.title}**\n\n{payload.get('explanation', '')}\n\n"
            f"```{payload.get('language', '')}\n{payload.get('code', '')}\n```\n\n"
            f"_{payload.get('note', '')}_"
        )

    if artifact.kind == "anomaly":
        rows = [
            [
                a.get("method"), a.get("column"), a.get("label") or a.get("row_index"),
                a.get("value"), a.get("score"), a.get("reason"),
            ]
            for a in payload.get("anomalies", [])[:25]
        ]
        header = ["method", "column", "where", "value", "score", "why flagged"]
        notes = payload.get("notes") or []
        block = f"**{artifact.title}** — {len(payload.get('anomalies', []))} finding(s)\n\n"
        block += _table_md(header, rows, limit=25)
        if notes:
            block += "\n\n" + "\n".join(f"- {n}" for n in notes)
        return block

    if artifact.kind == "forecast":
        rows = [[p["period"], p["forecast"], p["lower"], p["upper"]] for p in payload.get("points", [])]
        return (
            f"**{artifact.title}** — {payload.get('method')}\n\n"
            + _table_md(["period", "forecast", "lower 95%", "upper 95%"], rows, limit=36)
        )

    if artifact.kind == "quality":
        score = payload.get("score", {})
        rows = [[i.get("severity"), i.get("column") or "", i.get("message")] for i in payload.get("issues", [])[:30]]
        return (
            f"**{artifact.title}** — score {score.get('score')}/100\n\n"
            + _table_md(["severity", "column", "message"], rows, limit=30)
        )

    if artifact.kind == "profile":
        rows = [
            [c.get("name"), c.get("duckdb_type"), c.get("role"), c.get("distinct_count"), c.get("null_pct")]
            for c in payload.get("columns", [])
        ]
        return (
            f"**{artifact.title}** — {payload.get('row_count', 0):,} rows\n\n"
            + _table_md(["column", "type", "role", "distinct", "null %"], rows, limit=60)
        )

    return f"**{artifact.title}**\n\n```json\n{json.dumps(payload, indent=2, default=str)[:2000]}\n```"


def to_markdown(session: DataSession, *, title: str = "AI Data Analyst — Session Report") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"# {title}", f"_Generated {now} · session `{session.session_id}`_", "", "## Datasets", ""]

    for dataset in session.datasets.values():
        score = quality_score(dataset.profile)
        parts.append(
            f"### `{dataset.table}`\n\n"
            f"- source: `{dataset.source_name}`\n"
            f"- rows: {dataset.profile.row_count:,} · columns: {dataset.profile.column_count}\n"
            f"- data-quality score: {score['score']}/100 "
            f"(completeness {score['completeness_pct']}%, uniqueness {score['uniqueness_pct']}%)\n"
        )
        rows = [
            [c.name, c.duckdb_type, c.role.value, f"{c.distinct_count:,}", f"{c.null_pct}%"]
            for c in dataset.profile.columns
        ]
        parts.append(_table_md(["column", "type", "role", "distinct", "null %"], rows, limit=80))
        parts.append("")

    if session.join_hints:
        parts.append("### Detected join keys\n")
        rows = [
            [f"{h.left_table}.{h.left_column}", f"{h.right_table}.{h.right_column}", f"{h.overlap_pct}%"]
            for h in session.join_hints
        ]
        parts.append(_table_md(["left", "right", "value overlap"], rows))
        parts.append("")

    parts.append("## Conversation\n")
    if not session.history:
        parts.append("_No questions were asked in this session._")

    turn = 0
    for message in session.history:
        if message.role == "user":
            turn += 1
            parts.append(f"### Q{turn}. {message.content}\n")
            continue
        parts.append(message.content + "\n")
        if message.reasoning:
            parts.append("<details><summary>Reasoning trail</summary>\n")
            parts.extend(f"{i + 1}. {r}" for i, r in enumerate(message.reasoning))
            parts.append("\n</details>\n")
        for artifact in message.artifacts:
            parts.append(_artifact_md(artifact))
            parts.append("")

    all_sql = [s for m in session.history for s in m.sql_executed]
    if all_sql:
        parts.append("## Every SQL statement executed\n")
        for i, sql in enumerate(all_sql, start=1):
            parts.append(f"{i}.\n```sql\n{sql}\n```")

    parts.append("\n---\n")
    parts.append(
        "_All figures in this report came from SQL executed against the uploaded data. "
        "Generated code shown here was validated but not executed._"
    )
    return "\n".join(parts)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font: 15px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 900px; margin: 2.5rem auto; padding: 0 1.25rem; color: #1f2933; }}
  h1 {{ font-size: 1.7rem; border-bottom: 2px solid #2563eb; padding-bottom: .4rem; }}
  h2 {{ margin-top: 2.25rem; font-size: 1.3rem; color: #1e40af; }}
  h3 {{ margin-top: 1.5rem; font-size: 1.05rem; }}
  pre {{ background: #f6f8fa; padding: .85rem; border-radius: 6px; overflow-x: auto;
         border: 1px solid #e5e7eb; font-size: 13px; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  table {{ border-collapse: collapse; width: 100%; margin: .85rem 0; font-size: 13.5px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: left; }}
  th {{ background: #f3f4f6; }}
  .meta {{ color: #6b7280; font-size: 13px; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {now} · session {session_id}</p>
<pre>{body}</pre>
</body></html>"""


def to_excel(session: DataSession) -> bytes:
    """Export the session as a multi-sheet .xlsx workbook.

    Sheets: an overview, one sheet per dataset profile, one sheet per result table
    or chart dataset produced during the conversation, and a sheet listing every
    SQL statement that was executed. Excel is where a business reviewer will
    actually continue the analysis, so the numbers are written as numbers rather
    than as formatted strings.
    """
    import io

    import pandas as pd

    buffer = io.BytesIO()
    used_names: set[str] = set()

    def sheet_name(base: str) -> str:
        # Excel: max 31 chars, and these characters are forbidden.
        clean = re.sub(r"[\[\]:*?/\\]", "_", base)[:31] or "sheet"
        candidate, i = clean, 2
        while candidate.lower() in used_names:
            suffix = f"_{i}"
            candidate = clean[: 31 - len(suffix)] + suffix
            i += 1
        used_names.add(candidate.lower())
        return candidate

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        overview = pd.DataFrame(
            [
                {
                    "table": d.table,
                    "source_file": d.source_name,
                    "rows": d.profile.row_count,
                    "columns": d.profile.column_count,
                    "quality_score": quality_score(d.profile)["score"],
                    "duplicate_rows": d.profile.duplicate_row_count,
                }
                for d in session.datasets.values()
            ]
        )
        if overview.empty:
            overview = pd.DataFrame([{"note": "No datasets were loaded."}])
        overview.to_excel(writer, sheet_name=sheet_name("Overview"), index=False)

        for dataset in session.datasets.values():
            profile_frame = pd.DataFrame(
                [
                    {
                        "column": c.name,
                        "type": c.duckdb_type,
                        "role": c.role.value,
                        "distinct": c.distinct_count,
                        "null_count": c.null_count,
                        "null_pct": c.null_pct,
                        "min": c.min if c.min is not None else c.min_date,
                        "max": c.max if c.max is not None else c.max_date,
                        "mean": c.mean,
                        "median": c.p50,
                        "examples": ", ".join(c.sample_values[:3]),
                    }
                    for c in dataset.profile.columns
                ]
            )
            profile_frame.to_excel(
                writer, sheet_name=sheet_name(f"schema_{dataset.table}"), index=False
            )

            issues = pd.DataFrame(
                [
                    {"severity": i.severity, "kind": i.kind, "column": i.column or "", "message": i.message}
                    for i in dataset.profile.issues
                ]
            )
            if not issues.empty:
                issues.to_excel(
                    writer, sheet_name=sheet_name(f"quality_{dataset.table}"), index=False
                )

        turn = 0
        for message in session.history:
            if message.role == "user":
                turn += 1
                continue
            for index, artifact in enumerate(message.artifacts, start=1):
                payload = artifact.payload
                if artifact.kind in ("table", "chart") and payload.get("columns"):
                    frame = pd.DataFrame(payload.get("rows", []), columns=payload["columns"])
                elif artifact.kind == "anomaly":
                    frame = pd.DataFrame(payload.get("anomalies", []))
                elif artifact.kind == "forecast":
                    frame = pd.DataFrame(payload.get("points", []))
                else:
                    continue
                if frame.empty:
                    continue
                frame.to_excel(
                    writer, sheet_name=sheet_name(f"Q{turn}_{index}_{artifact.kind}"), index=False
                )

        statements = [s for m in session.history for s in m.sql_executed]
        sql_frame = pd.DataFrame(
            {"n": range(1, len(statements) + 1), "sql": statements}
            if statements
            else {"n": [], "sql": []}
        )
        if statements:
            sql_frame.to_excel(writer, sheet_name=sheet_name("SQL executed"), index=False)

    return buffer.getvalue()


def to_html(session: DataSession, *, title: str = "AI Data Analyst — Session Report") -> str:
    """Self-contained HTML. The Markdown body is escaped, not rendered, so the
    export has zero dependencies and cannot inject anything into a browser."""
    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        session_id=html.escape(session.session_id),
        body=html.escape(to_markdown(session, title=title)),
    )
