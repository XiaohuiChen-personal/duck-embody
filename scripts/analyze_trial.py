"""Replay a batch of trial JSONs through the forensic parser. Read-only.

The generated audit Markdown disagreed with the raw traces because
``scripts/auto_audit.sh`` invented fields (forensics F-08). This script exists so
there is exactly one way to turn `results/raw_*/` into numbers: it calls
``duck_embody.forensics`` and writes the result somewhere that is *not* the raw
directory. The raw trials are evidence (AGENTS.md rule 7) and this refuses to
write inside them.

Run (either interpreter — the module is pure):

    PYTHONDONTWRITEBYTECODE=1 ~/IsaacLab/_isaac_sim/python.sh \\
        scripts/analyze_trial.py results/raw_v5d_r2

    bash scripts/run_tests.sh --version   # for the kit python the tests use

Arguments are trial JSON paths, batch directories, or both; the default is
``results/raw_v5d_r2``. Output goes to ``results/forensics_v5d_r2/`` unless
``--out-dir`` says otherwise, and ``--no-write`` prints the summary only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duck_embody import forensics  # noqa: E402

DEFAULT_BATCH = REPO_ROOT / "results" / "raw_v5d_r2"
DEFAULT_MANIFEST = REPO_ROOT / "results" / "freeze.json"


def collect(paths: list[Path]) -> tuple[list[dict], Path | None]:
    """Load every trial named by ``paths``; return them and the batch directory.

    The batch directory is only well defined when every input came from one
    place — ``visual_audit_status`` and the freeze-matrix check would otherwise
    be asserting against a directory the caller did not analyze.
    """
    documents: list[dict] = []
    directories: set[Path] = set()
    for path in paths:
        if path.is_dir():
            documents.extend(forensics.load_batch(path))
            directories.add(path.resolve())
        else:
            documents.append(forensics.load_trial(path))
            directories.add(path.resolve().parent)
    documents.sort(key=lambda doc: doc["trial_id"])
    batch_dir = next(iter(directories)) if len(directories) == 1 else None
    return documents, batch_dir


def summarize(report: dict) -> str:
    integrity = report["integrity"]
    corrections = report["corrections"]
    lines = [
        f"trials              {integrity['complete_trials']}/{integrity['trials']} complete",
        f"model turns         {integrity['total_turns']}",
        f"config hashes       {len(integrity['config_hashes'])} "
        f"({', '.join(h[:12] for h in integrity['config_hashes'])})",
        f"freeze commits      {len(integrity['freeze_commits'])} "
        f"({', '.join((c or '<none>')[:7] for c in integrity['freeze_commits'])})",
        f"motion calls        {report['motion_calls']} "
        f"({report['bumped_motion_calls']} reported contact, "
        f"{report['counted_bumps']} counted as bumps)",
        f"multi-motion turns  {report['multi_motion_turns']}",
        f"falls               {report['falls']}",
        f"corrections         {corrections['calls']} calls, "
        f"{corrections['accepted']} accepted, {corrections['rejected']} rejected",
        f"correction effect   {corrections['worsened']} worsened / "
        f"{corrections['improved']} improved; error "
        f"{corrections['error_before_sum_m']:.4f} m -> "
        f"{corrections['error_after_sum_m']:.4f} m "
        f"(net {corrections['net_added_error_m']:+.4f} m)",
    ]
    if integrity["manifest"] is not None:
        manifest = integrity["manifest"]
        lines.append(
            f"manifest            config_hash "
            f"{'MATCH' if manifest['config_hash_matches'] else 'MISMATCH'}, "
            f"freeze_commit "
            f"{'MATCH' if manifest['freeze_commit_matches'] else 'MISMATCH'}, "
            f"missing cells {manifest['missing_cells'] or 'none'}"
        )
    if "visual_audits" in report:
        audits = report["visual_audits"]
        lines.append(
            f"visual audits       {len(audits['pending'])}/{audits['total']} pending"
        )
        for name in audits["pending"]:
            lines.append(f"                      pending: {name}")
    denied = report["stage1_success_never_offered_return"]
    lines.append(f"stage-1 v2 successes denied return_home: {denied or 'none'}")
    lines.append("")
    lines.append("correction ledger (effect = error_after - error_before; + is harmful)")
    for row in corrections["ledger"]:
        if not row["accepted"]:
            lines.append(
                f"  {row['trial_id']:<18} {row['stage'][:6]} t{row['turn_idx']:<3} "
                f"REJECTED  place={row['place']!r}"
            )
            continue
        lines.append(
            f"  {row['trial_id']:<18} {row['stage'][:6]} t{row['turn_idx']:<3} "
            f"{row['error_before_m']:.3f} -> {row['error_after_m']:.3f} m "
            f"({row['effect_m']:+.3f})  place={row['place']!r}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_BATCH],
        help="trial JSON files and/or batch directories (default results/raw_v5d_r2)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "forensics_v5d_r2",
        help="where the forensic JSON is written (never inside the raw batch)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="freeze manifest to check the batch against (default results/freeze.json)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the summary, write nothing"
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in (args.paths or [DEFAULT_BATCH])]
    try:
        documents, batch_dir = collect(paths)
    except forensics.ForensicsError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    manifest = None
    if args.manifest and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text())

    report = forensics.batch_report(
        documents, manifest=manifest, batch_dir=batch_dir
    )
    print(summarize(report))

    if args.no_write:
        return 0

    out_dir = args.out_dir.resolve()
    for path in paths:
        source = path.resolve() if path.is_dir() else path.resolve().parent
        if out_dir == source or source in out_dir.parents:
            print(
                f"FAIL: --out-dir {out_dir} is inside the input batch {source}; "
                "raw results are immutable evidence (AGENTS.md rule 7)",
                file=sys.stderr,
            )
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = forensics.as_json(report)
    (out_dir / "batch_summary.json").write_text(
        json.dumps(payload, indent=1, sort_keys=False) + "\n"
    )
    for trial in payload["trials"]:
        (out_dir / f"{trial['trial_id']}.json").write_text(
            json.dumps(trial, indent=1, sort_keys=False) + "\n"
        )
    print(f"\nwrote {out_dir}/batch_summary.json + {len(payload['trials'])} trial files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
