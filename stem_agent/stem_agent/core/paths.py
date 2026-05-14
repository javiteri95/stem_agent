"""
paths.py — Canonical paths anchored to the project root.

Import PROJECT_ROOT from here instead of computing Path(__file__) ad-hoc.
This ensures all file I/O lands inside the stem_agent project directory
regardless of the working directory the user invokes the CLI from.
"""

from pathlib import Path

# stem_agent/stem_agent/core/paths.py  →  up three levels = stem_agent/ project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
