#!/usr/bin/env python3
"""Entry point: python run.py <command> [args]   (see `python run.py --help`)"""

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
