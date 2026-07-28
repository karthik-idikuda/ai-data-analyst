#!/usr/bin/env python3
"""Fail if pictographic characters appear in source or docs.

The interface is meant to carry meaning through type, weight and spacing. Emoji
render inconsistently across platforms and fonts and date an interface quickly, so
this is enforced rather than left to discipline.

    python scripts/check_no_icons.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".html", ".css"}
SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache", "data"}

RANGES = (
    (0x1F300, 0x1FAFF),  # emoji, pictographs, supplemental symbols
    (0x1F000, 0x1F2FF),  # mahjong/domino/enclosed alphanumerics
    (0x2600, 0x27BF),    # miscellaneous symbols and dingbats
    (0x2B00, 0x2BFF),    # arrows and geometric shapes extras
    (0xFE0F, 0xFE0F),    # variation selector-16 (emoji presentation)
    (0x1F1E6, 0x1F1FF),  # regional indicators (flags)
)


def is_pictographic(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in RANGES)


def main() -> int:
    offenders: list[str] = []
    scanned = 0

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # This checker necessarily contains the code points it looks for.
        if path.name == Path(__file__).name:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for char in line:
                if is_pictographic(char):
                    name = unicodedata.name(char, f"U+{ord(char):04X}")
                    rel = path.relative_to(ROOT)
                    offenders.append(f"  {rel}:{lineno}  {name}")

    print(f"Scanned {scanned} files.")
    if offenders:
        print(f"Found {len(offenders)} pictographic character(s):")
        for line in offenders[:40]:
            print(line)
        if len(offenders) > 40:
            print(f"  … {len(offenders) - 40} more")
        return 1
    print("No emoji or icon characters found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
