"""
Screenshot all 4 required UI states using Playwright.
Run with: .venv/bin/python scripts/take_screenshots.py
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8501"

def wait_for_streamlit(page, timeout=20000):
    """Wait until Streamlit has finished rendering."""
    try:
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=timeout)
    except Exception:
        pass
    time.sleep(2)  # extra settle time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})

    # ── 00: Hero / Landing page ──────────────────────────────────────────
    page = ctx.new_page()
    page.goto(BASE, wait_until="networkidle")
    wait_for_streamlit(page)
    page.screenshot(path=str(OUT / "00-landing.png"), full_page=False)
    print("ok  00-landing.png")

    # ── 01: Load sample data, show workspace + suggestions ─────────────
    # Click "Load sample workspace"
    try:
        page.get_by_text("Load sample workspace").first.click()
        time.sleep(6)  # wait for data to load & rerun
        wait_for_streamlit(page)
    except Exception as e:
        print(f"  warn  Could not click sample button: {e}")
    page.screenshot(path=str(OUT / "01-uploaded-suggestions.png"), full_page=False)
    print("ok  01-uploaded-suggestions.png")

    # ── 02: Chat tab — type a question and wait for chart answer ────────
    try:
        # The chat tab is already active
        chat_input = page.locator("textarea[placeholder='Ask a question about your data']")
        chat_input.fill("Which country generated the highest revenue?")
        chat_input.press("Enter")
        time.sleep(15)  # wait for LLM response
        wait_for_streamlit(page)
    except Exception as e:
        print(f"  warn  Chat input: {e}")
    page.screenshot(path=str(OUT / "02-chat-answer-chart.png"), full_page=False)
    print("ok  02-chat-answer-chart.png")

    # ── 03: Expand "How this answer was produced" ────────────────────────
    try:
        page.get_by_text("How this answer was produced").first.click()
        time.sleep(1)
    except Exception as e:
        print(f"  warn  Expander: {e}")
    page.screenshot(path=str(OUT / "03-reasoning-sql-trace.png"), full_page=False)
    print("ok  03-reasoning-sql-trace.png")

    browser.close()

print(f"\nAll screenshots saved to: {OUT}")
