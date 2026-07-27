"""Audit one trial JSON: latency, tool coverage, errors, falls, leaks.

The T3.5 gate is a *transcript* audit as much as a video one (PLAN T3.5): every
tool exercised, no crash, memory block grows, QA lands in the JSON. Doing that by
eye over 20+ turns is how things get missed, so it is a script — and the same
script re-runs over the 12 batch trials at T4.4.

Two of its checks exist because T3.5 found the gap the hard way:

* **per-turn latency**, which doc 06 §8's batch forecast is still guessing at
  (its 5-25 s/turn band is unmeasured). This is the number that turns the
  3-8 h estimate into something you can plan a night around.
* **fall diagnostics present whenever a stage ended in a fall.** The first
  sanity trial recorded ``fell: true`` and nothing else — no height, no tilt,
  no term — and the audit video stops at the chunk boundary BEFORE the topple,
  so neither artifact could establish whether the fall was genuine. A fall ends
  the whole trial, so that is the one event that must never be unauditable.

Run:  ~/IsaacLab/_isaac_sim/python.sh scripts/audit_trial.py results/raw/<trial>.json
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ground-truth substrings that must never appear anywhere the model can read.
#: Field names AND the values themselves — T3.2's review showed a rounded or
#: stringified pose evades a name-only check.
BANNED_IN_MODEL_TEXT = (
    "true_pose",
    "pose_trace",
    "true_displacement",
    "oracle",
    "fall_diagnostics",
    "tilt_deg",
)


def parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def audit(path: Path, require_tool_coverage: bool = False) -> int:
    # require_tool_coverage is for SCRIPTED trials (the S5 mini-trial), whose
    # navigator promises all 12 tools. A real model trial that fell on turn 2
    # legitimately used six — which tools a model chooses is model behaviour,
    # and failing the audit on it would punish the harness for the model's
    # brevity.
    trial = json.loads(path.read_text())
    turns = trial.get("turns", [])
    final = trial.get("final") or {}
    problems: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            problems.append(label)

    print(f"== {trial.get('trial_id', path.stem)} ==")
    print(f"  turns: {len(turns)}   video: {trial.get('video_path')}")

    # -- completeness ------------------------------------------------------
    check(bool(final), "trial has a `final` block (not an infra failure)")
    check("qa" in final and bool(final.get("qa")), "final.qa is populated",
          f"{len(final.get('qa') or [])} answers")
    check(not trial.get("infra_failure"), "no infra failure recorded")

    # -- freeze traceability (T4.3 acceptance: 12/12 under ONE freeze hash) --
    # The runner's mid-batch guard aborts on drift BEFORE the next trial, but
    # this is the post-batch net: every results/raw/ JSON must carry exactly
    # the config_hash results/freeze.json records, or the published table
    # averages trials that ran under different frozen bytes. Scoped to
    # results/raw/ because pre-freeze artifacts (the T3.5 sanity trials,
    # parked in results/logs/) legitimately carry a stale hash there.
    freeze_path = REPO_ROOT / "results" / "freeze.json"
    stored_hash = (trial.get("config") or {}).get("config_hash")
    in_raw = (REPO_ROOT / "results" / "raw") in path.resolve().parents
    if freeze_path.exists() and in_raw:
        frozen_hash = json.loads(freeze_path.read_text()).get("config_hash")
        check(stored_hash == frozen_hash,
              "config.config_hash matches results/freeze.json",
              f"trial {str(stored_hash)[:12]}… vs freeze {str(frozen_hash)[:12]}…")
    elif freeze_path.exists():
        print(f"  INFO  outside results/raw/ — freeze-hash check skipped "
              f"(config_hash {str(stored_hash)[:12]}…)")
    else:
        print("  INFO  no results/freeze.json yet (pre-freeze audit) — "
              "freeze-hash check skipped")

    # -- tool coverage: PLAN T3.5 wants every tool exercised at least once --
    used: dict[str, int] = {}
    errors: list[tuple[int, str, str]] = []
    for t in turns:
        for call in (t.get("model_output") or {}).get("tool_calls", []) or []:
            used[call["name"]] = used.get(call["name"], 0) + 1
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from duck_embody.agent.tools import TOOL_SCHEMAS

        all_tools = [s["name"] for s in TOOL_SCHEMAS]
    except Exception:  # noqa: BLE001 - the audit must still run without kit
        all_tools = sorted(used)
    missing = [n for n in all_tools if n not in used]
    print(f"  tools used: {', '.join(f'{k}x{v}' for k, v in sorted(used.items())) or 'none'}")
    if require_tool_coverage:
        check(not missing, "every tool exercised at least once",
              f"missing: {missing}" if missing else "")
    elif missing:
        print(f"  INFO  tools not exercised by this model: {missing}")

    # -- errors the model was handed ---------------------------------------
    for t in turns:
        for err in (t.get("model_output") or {}).get("parse_errors", []) or []:
            errors.append((t.get("turn_idx", -1), "parse_error", str(err)))
        if (t.get("model_output") or {}).get("nudged"):
            errors.append((t.get("turn_idx", -1), "derailment_nudge", ""))
    if errors:
        print("  errors handed to the model:")
        for idx, kind, detail in errors[:10]:
            print(f"    turn {idx}: {kind} {detail[:70]}")
    check(not [e for e in errors if e[1] == "parse_error"], "no malformed tool calls")

    # -- falls must be auditable -------------------------------------------
    for stage, info in (final.get("stages") or {}).items():
        if info.get("end_reason") == "fall":
            diag = None
            for t in reversed(turns):
                ex = t.get("execution") or {}
                d = ex.get("fall_diagnostics")
                if not d:
                    # The schema carries them PER-CALL (execution.calls[i]);
                    # auditing only the merged turn level made a correctly
                    # instrumented trial read as unauditable.
                    for call in ex.get("calls") or []:
                        if call.get("fall_diagnostics"):
                            d = call["fall_diagnostics"]
                            break
                if d:
                    diag = d
                    break
            check(diag is not None, f"{stage}: fall carries diagnostics")
            if diag:
                fired = [k for k, v in (diag.get("terms") or {}).items() if v]
                print(f"    height {diag.get('height_m')} m (limit {diag.get('height_threshold_m')}) | "
                      f"tilt {diag.get('tilt_deg')} deg (limit {diag.get('tilt_threshold_deg')}) | "
                      f"fired: {fired or 'NONE'}")
                check(bool(fired), f"{stage}: a termination term actually fired",
                      "a fall with no term firing is a harness bug, not a topple")

    # -- per-turn latency ---------------------------------------------------
    stamps = [parse_ts(t["timestamp"]) for t in turns if t.get("timestamp")]
    stamps = [s for s in stamps if s]
    if len(stamps) >= 2:
        deltas = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        deltas = [d for d in deltas if d >= 0]
        print("\n  per-turn wall-clock (API + sim + render):")
        print(f"    n={len(deltas)}  min {min(deltas):.1f}s  median {statistics.median(deltas):.1f}s  "
              f"mean {statistics.mean(deltas):.1f}s  max {max(deltas):.1f}s")
        if len(deltas) >= 4:
            ordered = sorted(deltas)
            print(f"    p90 {ordered[int(0.9 * (len(ordered) - 1))]:.1f}s")
        print(f"    => a 40-turn stage is ~{statistics.median(deltas) * 40 / 60:.0f} min at the median")
    else:
        print("\n  per-turn latency: not enough timestamps")

    # -- leak scan over everything the model could read ----------------------
    model_text = json.dumps(
        [
            {
                "obs": t.get("obs"),
                "model_output": t.get("model_output"),
                "memory_snapshot": t.get("memory_snapshot"),
            }
            for t in turns
        ]
    )
    leaked = [b for b in BANNED_IN_MODEL_TEXT if b in model_text]
    check(not leaked, "no ground-truth field names in the model-visible record",
          f"leaked: {leaked}" if leaked else "")

    # -- cost ---------------------------------------------------------------
    tok = final.get("tokens") or {}
    if tok:
        print(f"\n  tokens: in {tok.get('input_tokens')} / out {tok.get('output_tokens')} / "
              f"cache_read {tok.get('cache_read_tokens')} / cache_write {tok.get('cache_write_tokens')}")
        cost = tok.get("cost_usd_estimate", tok.get("cost_usd", 0))
        print(f"  cost:   ${cost:.4f}")
        check((tok.get("cache_read_tokens") or 0) > 0, "prompt caching is actually being hit",
              "a trial with zero cache reads is paying full rate every turn")

    print(f"\n  {'AUDIT PASS' if not problems else 'AUDIT FAIL: ' + '; '.join(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(max(audit(Path(a)) for a in sys.argv[1:]))
