from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_one_cow(tag_id: str, uwb_dir: Path) -> tuple[pd.DataFrame, str | None]:
    """
    Load UWB CSV for a tag. Uses the **newest** *.csv* in ``uwb_dir / <tag>`` (by mtime).
    ``tag_id`` may be ``T01`` or ``t01``; folder names are matched case-insensitively on Windows.
    """
    base = Path(uwb_dir)
    tid = tag_id.strip().upper()
    folder = base / tid
    if not folder.is_dir():
        folder = base / tag_id.strip()
    if not folder.is_dir():
        return pd.DataFrame(), None

    csvs = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return pd.DataFrame(), None

    path = csvs[0]
    return pd.read_csv(path), str(path.resolve())


def list_tag_ids(uwb_dir: Path) -> list[str]:
    base = Path(uwb_dir)
    if not base.is_dir():
        return []
    out: list[str] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and any(p.glob("*.csv")):
            out.append(p.name)
    return out
