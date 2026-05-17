#!/usr/bin/env python
"""Compatibility entry point for Script2Voice."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from script2voice.cli import main


if __name__ == "__main__":
    main()
