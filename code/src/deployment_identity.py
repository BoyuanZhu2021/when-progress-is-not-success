"""Fail-closed verification of an H1 deployment tree.

Formal Gate 1, training, and evaluation must execute exactly the files recorded by
``h1_deploy_mt.py``.  A missing manifest, an unlisted replacement, or any changed byte is a hard
error rather than provenance metadata marked "unavailable".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .model_pins import INJECAGENT_COMMIT


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployment(
    root: str | Path,
    *,
    required_paths: tuple[str, ...] = (),
) -> dict:
    """Load and re-hash every deployed file, rejecting identity or path drift."""
    root_path = Path(root).resolve()
    manifest_path = root_path / "deployment_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(f"deployment manifest unavailable: {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"deployment manifest schema mismatch: {manifest.get('schema_version')!r}"
        )
    if manifest.get("injecagent_commit") != INJECAGENT_COMMIT:
        raise RuntimeError(
            "deployment InjecAgent provenance mismatch: "
            f"expected {INJECAGENT_COMMIT}, got {manifest.get('injecagent_commit')!r}"
        )
    if manifest.get("injecagent_clean") is not True:
        raise RuntimeError("deployment was not produced from a clean InjecAgent checkout")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("deployment manifest has no file hash map")
    missing_required = sorted(set(required_paths) - set(files))
    if missing_required:
        raise RuntimeError(f"deployment manifest omits required files: {missing_required}")

    tree = hashlib.sha256()
    mismatches: list[str] = []
    for relative, expected in sorted(files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("deployment manifest file entries must be string:string")
        candidate = (root_path / relative).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise RuntimeError(f"deployment manifest path escapes root: {relative!r}") from exc
        if not candidate.is_file():
            mismatches.append(f"missing:{relative}")
            continue
        actual = _sha256(candidate)
        if actual != expected:
            mismatches.append(f"sha256:{relative}:{actual}")
        tree.update(f"{relative}\0{expected}\n".encode())
    if mismatches:
        raise RuntimeError(
            "deployed file identity mismatch: " + "; ".join(mismatches[:5])
        )
    actual_tree = tree.hexdigest()
    if manifest.get("deployed_tree_sha256") != actual_tree:
        raise RuntimeError(
            "deployment tree hash mismatch: "
            f"manifest={manifest.get('deployed_tree_sha256')!r} recomputed={actual_tree}"
        )
    return manifest
