#!/usr/bin/env python3
"""Audit a complete batch and enforce the visual-publication gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from duck_embody.audit import PASS, audit_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--write-sheets",
        action="store_true",
        help="create missing event-indexed review sheets with pending verdicts",
    )
    args = parser.parse_args(argv)
    result = audit_batch(
        args.batch_dir.resolve(),
        args.manifest.resolve(),
        repo_root=REPO_ROOT,
        audit_dir=args.audit_dir.resolve(),
        write_sheets=args.write_sheets,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"{result['status']}: {len(result['trials'])} trial(s), "
        f"{result['publication_gate']['written']}/"
        f"{result['publication_gate']['expected']} visual verdicts"
    )
    return 0 if result["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
