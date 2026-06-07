#!/usr/bin/env python3
"""Download ML artifacts listed in models.manifest.json. Run from repo root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_download import DEFAULT_MANIFEST, download_models


def main() -> int:
    p = argparse.ArgumentParser(description="Download ML models from models.manifest.json")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--required-only",
        action="store_true",
        help="Only download artifacts marked required (default: all with URLs)",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    report = download_models(args.manifest, required_only=args.required_only, quiet=args.quiet)
    if report.optional_missing and not args.quiet:
        print(f"optional not configured: {len(report.optional_missing)}", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
