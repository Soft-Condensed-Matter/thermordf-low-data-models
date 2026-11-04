#!/usr/bin/env python3
"""Verify the released ThermoRDF repository without fitting any model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not files:
        raise RuntimeError("release_manifest.json contains no file identities.")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected["size_bytes"]):
            raise RuntimeError(f"Size mismatch: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")
    model_files = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "models").rglob("*")
        if path.is_file() and path.suffix in {".npz", ".pt"}
    )
    expected_models = sorted(item["artifact"] for item in manifest["models"].values())
    if model_files != expected_models:
        raise RuntimeError(
            f"Expected exactly the four declared model artifacts; found {model_files}."
        )
    print("STATUS: COMPLETE")
    print("MODEL_ARTIFACTS: 4")
    print(f"SUBSET_ID: {manifest['selection']['subset_id']}")
    print("REPOSITORY_MODIFIED: NO")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
