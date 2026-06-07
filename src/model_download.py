"""Download ML artifacts from models.manifest.json (used by CLI and inference API)."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "models.manifest.json"


@dataclass
class DownloadReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


def _is_placeholder_url(url: str) -> bool:
    u = (url or "").strip().lower()
    return not u or "example.com" in u


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Bovitech-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        dest.write_bytes(resp.read())


def download_models(
    manifest_path: Path | None = None,
    *,
    required_only: bool = False,
    quiet: bool = False,
) -> DownloadReport:
    """Download missing artifacts. Required items must succeed or report.failed is set."""
    path = (manifest_path or DEFAULT_MANIFEST).resolve()
    report = DownloadReport()

    if not path.is_file():
        report.failed.append(f"Missing manifest: {path.name}")
        return report

    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data.get("artifacts", []):
        dest = ROOT / item["dest"]
        url = (item.get("url") or "").strip()
        optional = bool(item.get("optional", False))
        required = bool(item.get("required", not optional))

        if required_only and not required:
            continue

        rel = str(dest.relative_to(ROOT))
        if dest.is_file():
            report.skipped.append(rel)
            if not quiet:
                print(f"skip: {rel}")
            continue

        if _is_placeholder_url(url):
            if optional:
                report.optional_missing.append(rel)
                if not quiet:
                    print(f"optional (no URL): {rel}")
            else:
                report.failed.append(f"no URL for required artifact: {rel}")
                if not quiet:
                    print(f"no URL: {rel}", flush=True)
            continue

        if not quiet:
            print(f"download: {rel}")
        try:
            _download(url, dest)
            report.downloaded.append(rel)
        except Exception as exc:
            report.failed.append(f"{rel}: {exc}")
            if not quiet:
                print(f"failed: {rel} — {exc}", flush=True)

    return report


def ensure_models_for_api(*, quiet: bool = False) -> bool:
    """Download required models before serving. Returns False if required artifacts missing."""
    report = download_models(required_only=True, quiet=quiet)
    return report.ok
