"""Run the strict machine audit for one trial.

Required usage:
  python scripts/audit_trial.py TRIAL.json --batch-dir RAW_DIR --manifest MANIFEST
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from duck_embody.audit import PASS, audit_trial
from duck_embody.agent.loop import (
    reconstruct_neutral_request,
    request_structure_problems,
)


def audit(path: Path, require_tool_coverage: bool = False) -> int:
    """Compatibility shim for pre-TR.8 callers.

    The CLI deliberately does not use this reduced legacy surface: publication
    requires explicit ``--batch-dir`` and ``--manifest``.  Keeping the function
    lets old unit probes retain their narrow freeze/leak assertions.
    """
    del require_tool_coverage
    document = json.loads(path.read_text(encoding="utf-8"))
    failed = False
    freeze = REPO_ROOT / "results" / "freeze.json"
    raw = REPO_ROOT / "results" / "raw"
    if raw.resolve() in path.resolve().parents and freeze.is_file():
        expected = json.loads(freeze.read_text()).get("config_hash")
        actual = (document.get("config") or {}).get("config_hash")
        ok = actual == expected
        print(
            f"  {'PASS' if ok else 'FAIL'}  "
            "config.config_hash matches results/freeze.json"
        )
        failed |= not ok
    else:
        print("  INFO  outside results/raw/ — freeze-hash check skipped")
    requests = document.get("requests")
    if isinstance(requests, list):
        hash_errors: list[str] = []
        structure_errors: list[str] = []
        for index, request in enumerate(requests):
            try:
                rebuilt = reconstruct_neutral_request(document, index, path.parent)
                if rebuilt["request_sha256"] != request.get("request_sha256"):
                    hash_errors.append(f"request {index}")
                structure_errors.extend(
                    request_structure_problems(document, index, path.parent)
                )
            except Exception as exc:
                hash_errors.append(str(exc))
        print(
            f"  {'PASS' if not hash_errors else 'FAIL'}  "
            "every provider-neutral request reconstructs to its logged hash"
        )
        print(
            f"  {'PASS' if not structure_errors else 'FAIL'}  "
            "model-facing requests derive only from logged harness sources"
        )
        failed |= bool(hash_errors or structure_errors)
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    result = audit_trial(
        args.trial.resolve(),
        batch_dir=args.batch_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        repo_root=REPO_ROOT,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
