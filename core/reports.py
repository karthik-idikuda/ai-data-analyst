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


def to_pdf(session: DataSession, *, title: str = "AI Data Analyst — Session Report") -> bytes:
    """Render the session as a paginated PDF.

    Built with ReportLab's Platypus flowables rather than a headless browser, so the
    container stays small and the export has no system dependencies. Content is the
    same audit trail as the Markdown export: datasets, detected joins, every
    question and answer, and every SQL statement that executed.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table as PdfTable,
        TableStyle,
    )
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        author="AI Data Analyst",
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    )

    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=19, spaceAfter=4,
                                textColor=colors.HexColor("#1e3a8a")),
        "meta": ParagraphStyle("m", parent=base["Normal"], fontSize=8,
                               textColor=colors.HexColor("#6b7280"), spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=13, spaceBefore=12,
                             spaceAfter=5, textColor=colors.HexColor("#1e40af")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10.5, spaceBefore=8,
                             spaceAfter=3),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9, leading=12.5,
                               alignment=TA_LEFT),
        "code": ParagraphStyle("c", parent=base["Code"], fontSize=7.4, leading=9.4,
                               backColor=colors.HexColor("#f3f4f6"),
                               borderPadding=4, spaceBefore=3, spaceAfter=6),
    }

    def para(text: str, style: str = "body") -> Paragraph:
        return Paragraph(html.escape(str(text)), styles[style])

    def code_block(text: str) -> Paragraph:
        # Platypus needs <br/> for line breaks and no tabs.
        safe = html.escape(str(text)).replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;")
        return Paragraph(safe, styles["code"])

    def grid(header: list[str], rows: list[list[Any]], widths: list[float] | None = None) -> PdfTable:
        data = [[Paragraph(f"<b>{html.escape(str(h))}</b>", styles["body"]) for h in header]]
        for row in rows:
            data.append([
                Paragraph(html.escape("" if v is None else str(v))[:300], styles["body"]) for v in row
            ])
        table = PdfTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f9fafb")]),
                ]
            )
        )
        return table

    story: list[Any] = [
        para(title, "title"),
        para(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"session {session.session_id}",
            "meta",
        ),
    ]

    story.append(para("Datasets", "h2"))
    if not session.datasets:
        story.append(para("No datasets were loaded in this session."))
    for dataset in session.datasets.values():
        score = quality_score(dataset.profile)
        story.append(para(f"{dataset.table}  (from {dataset.source_name})", "h3"))
        story.append(
            para(
                f"{dataset.profile.row_count:,} rows · {dataset.profile.column_count} columns · "
                f"quality {score['score']}/100 "
                f"(completeness {score['completeness_pct']}%, uniqueness {score['uniqueness_pct']}%)"
            )
        )
        story.append(Spacer(1, 3))
        story.append(
            grid(
                ["column", "type", "role", "distinct", "null %"],
                [
                    [c.name, c.duckdb_type, c.role.value, f"{c.distinct_count:,}", f"{c.null_pct}%"]
                    for c in dataset.profile.columns
                ],
                widths=[52 * mm, 24 * mm, 26 * mm, 24 * mm, 20 * mm],
            )
        )
        warnings = [i for i in dataset.profile.issues if i.severity in ("warning", "error")]
        if warnings:
            story.append(Spacer(1, 4))
            story.append(para("Data-quality flags:", "h3"))
            for issue in warnings[:10]:
                story.append(para(f"• {issue.message}"))

    if session.join_hints:
        story.append(para("Detected join keys", "h2"))
        story.append(
            grid(
                ["left", "right", "value overlap"],
                [
                    [f"{h.left_table}.{h.left_column}", f"{h.right_table}.{h.right_column}",
                     f"{h.overlap_pct}%"]
                    for h in session.join_hints
                ],
            )
        )

    story.append(PageBreak())
    story.append(para("Conversation", "h2"))
    if not session.history:
        story.append(para("No questions were asked in this session."))

    turn = 0
    for message in session.history:
        if message.role == "user":
            turn += 1
            story.append(para(f"Q{turn}. {message.content}", "h3"))
            continue

        for line in message.content.split("\n"):
            stripped = line.strip()
            if stripped:
                story.append(para(stripped.replace("**", "").replace("`", "")))

        if message.reasoning:
            story.append(Spacer(1, 3))
            story.append(para("Reasoning", "h3"))
            for i, step in enumerate(message.reasoning, start=1):
                story.append(para(f"{i}. {step}"))

        for artifact in message.artifacts:
            payload = artifact.payload
            if artifact.kind in ("table", "chart") and payload.get("columns"):
                story.append(Spacer(1, 3))
                story.append(para(artifact.title, "h3"))
                story.append(grid(payload["columns"], payload.get("rows", [])[:20]))
            elif artifact.kind == "anomaly":
                items = payload.get("anomalies", [])[:12]
                if items:
                    story.append(Spacer(1, 3))
                    story.append(para(f"{artifact.title} — {len(payload.get('anomalies', []))} findings", "h3"))
                    story.append(
                        grid(
                            ["method", "column", "value", "why flagged"],
                            [[a["method"], a["column"], a.get("value"), a["reason"]] for a in items],
                            widths=[26 * mm, 24 * mm, 22 * mm, 100 * mm],
                        )
                    )
            elif artifact.kind == "forecast":
                points = payload.get("points", [])
                if points:
                    story.append(Spacer(1, 3))
                    story.append(para(f"{artifact.title} — {payload.get('method')}", "h3"))
                    story.append(
                        grid(
                            ["period", "forecast", "lower 95%", "upper 95%"],
                            [[p["period"], f"{p['forecast']:,.2f}", f"{p['lower']:,.2f}",
                              f"{p['upper']:,.2f}"] for p in points],
                        )
                    )
            elif artifact.kind == "code":
                story.append(Spacer(1, 3))
                story.append(para(artifact.title, "h3"))
                story.append(code_block(payload.get("code", "")))

        if message.sql_executed:
            story.append(para("SQL executed", "h3"))
            for statement in message.sql_executed:
                story.append(code_block(statement))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))
    story.append(
        para(
            "All figures in this report came from SQL or a restricted pandas expression "
            "executed against the uploaded data.",
            "meta",
        )
    )

    doc.build(story)
    return buffer.getvalue()


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
