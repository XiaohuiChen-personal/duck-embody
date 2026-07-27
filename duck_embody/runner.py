"""Resumable batch runner: the 12-trial matrix inside ONE persistent kit session.

doc 06 §7's design, implemented. The matrix (3 models x seeds 101-104) comes
from ``configs/benchmark.yaml`` — read, never hardcoded, for the same reason
``scripts/run_trial.py::frozen_matrix`` reads it: the file is inside the hashed
fairness contract, so the entry point and the contract cannot drift apart, and
enumerating (instead of globbing ``configs/models/*.yaml``) keeps the
out-of-benchmark ``judge.yaml`` from ever occupying a matrix slot.

Four jobs, in the order a batch night meets them:

1. **The freeze manifest** (``--freeze``): enumerate the frozen files
   (``loop.FROZEN_FILES`` — the single enumeration source, doc 06 §2), hash each
   (sha256 over raw bytes, the same bytes ``loop.config_hash`` consumes), and
   write ``results/freeze.json``. Refuses from a dirty tree — and for
   ``--freeze`` "dirty" means ANY tracked file (``git status --porcelain
   -uno``), not just the frozen 15: freeze night is defined as a clean commit,
   and a freeze stamped with a clean sha while ``runner.py`` or ``scoring.py``
   carries uncommitted edits would claim traceability to code no commit
   contains. (Resume keeps the narrower frozen-file scope — see
   ``commit_refusals`` — so a T4.3-branch-(a) fix touching only non-frozen
   code resumes cleanly.)

2. **The guard**: before any trial, recompute every hash and compare against
   ``results/freeze.json`` AND against ``config.config_hash`` in every existing
   result the runner would resume around. Any mismatch is a hard refusal that
   names the drifted file. There is deliberately no ``--force`` flag (doc 06
   §7); the correct responses are "revert the change" or "start a new batch
   directory". MEASURED reason this guard exists at T4.2 and not T4.3: the two
   T3.5 sanity JSONs in ``results/raw/`` carry matrix trial ids with a stale
   ``config_hash`` — without the guard, a matching hash would have silently
   pooled sanity data as a benchmark result, and a mismatched one would have
   been discovered on launch night. The freeze half of the guard RE-RUNS
   before every trial launch and before every infra retry
   (``midbatch_refusals`` — ~15 sha256 reads, microseconds): a startup-only
   check would let a mid-batch edit to a frozen file run trials 6-12 under
   different bytes than 1-5 with nothing firing until a restart, exactly the
   doc 06 §2 recorded incident shape.

3. **Resume** (doc 06 §7, §9.1): a trial is skipped iff its JSON exists, is
   complete (``scoring.is_complete`` — ``final`` present, no ``infra_failure``),
   carries no ``turn_cap_override`` (a smoke-capped run is never a result), and
   its stored ``config_hash`` matches the live tree. An incomplete JSON is moved
   to ``results/incomplete/`` with a timestamp suffix, logged in
   ``results/rerun_log.md``, and the trial reruns from scratch. Model failures
   (cap / fall / wrong ``declare_done``) are complete results and are NEVER
   rerun — "timeouts and cap hits are scored as failures and are never
   selectively retried" (doc 06 §3.2, Locked).

4. **The batch**: rule-1 preflight, ONE ``SimSession.launch()`` (cold start is
   minutes — AGENTS.md rule 1), then trials sequentially with a full state reset
   between them; ``session.close()`` in a ``finally``. Providers are preflighted
   pre-kit and built post-kit (the measured Omit-sentinel hazard, AGENTS.md §5);
   one provider instance per model is reused across its four seeds — providers
   are stateless between sends (the transcript lives in ``EpisodeRunner``), and
   building all of them before trial 1 surfaces a bad key before any money is
   spent. Rate-limit handling is ONLY the SDK-level bounded backoff
   (``build_provider(max_retries=5)``) plus the measured single-request-in-flight
   throughput (``configs/benchmark.yaml: runtime.per_turn_wall_s``); no other
   headroom assumption is on record, so none is relied on (AGENTS.md rule 3).

Run (kit python, unbuffered — kit discards buffered stdout at exit):

    PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p duck_embody/runner.py --dry-run
    PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p duck_embody/runner.py --freeze
    PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p duck_embody/runner.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from duck_embody.agent.loop import FROZEN_FILES, _git, config_hash, freeze_commit
from duck_embody.scoring import is_complete

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The apartment task — the only benchmark task (see run_trial.py::TASKS).
TASK_ID = "DuckEmbody-Apartment-v0"

#: Trial JSONs live at results/raw/ (AGENTS.md §7 + doc 01 §5 — the winners of
#: the doc 06 §7 path reconciliation, updated there in this same change);
#: incomplete/ and the rerun log sit at results/ top level exactly as PLAN T4.2
#: spells them, and freeze.json beside them so the whole results/ tree ships as
#: one self-describing unit.
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "raw"
DEFAULT_VIDEO_DIR = REPO_ROOT / "results" / "videos"
DEFAULT_INCOMPLETE_DIR = REPO_ROOT / "results" / "incomplete"
DEFAULT_RERUN_LOG = REPO_ROOT / "results" / "rerun_log.md"
DEFAULT_FREEZE_JSON = REPO_ROOT / "results" / "freeze.json"

#: Bumped only if the freeze.json layout changes shape — the guard refuses an
#: unknown schema instead of guessing at its fields.
FREEZE_SCHEMA = "duck-embody-freeze-v1"

# Trial statuses (--dry-run vocabulary; the refusing two are uppercase there).
STATUS_PENDING = "pending"
STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete"
STATUS_HASH_DRIFT = "hash-drift"
STATUS_SMOKE_CAPPED = "smoke-capped"

RERUN_LOG_HEADER = """\
# Rerun log — doc 06 §7

Every resume move and infra rerun in the batch, appended by
`duck_embody/runner.py`. It ships with the results: reruns are visible, not
silent. Model failures (cap / fall / wrong `declare_done`) are final results
and never appear here — the only legitimate rerun is a logged infra failure.
T4.3's restart branch, when taken, is also recorded here
(a: fix touches non-frozen code -> keep the freeze commit, resume;
b: fix touches any frozen file -> new freeze commit, new batch directory,
restart from zero).

| trial id | timestamp (UTC) | cause | evidence |
|---|---|---|---|
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def load_matrix(root: Path = REPO_ROOT) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """doc 06 §2's frozen roster and seed set, read from the hashed config.

    Same contract as ``run_trial.py::frozen_matrix`` and for the same reason:
    ``benchmark.yaml`` is inside the fairness contract the guard hashes, so a
    matrix hardcoded here could silently diverge from the one every trial JSON
    claims it ran under.
    """
    import yaml

    raw = yaml.safe_load((root / "configs" / "benchmark.yaml").read_text())
    return tuple(raw["models"]), tuple(int(s) for s in raw["seeds"])


def trial_matrix(
    models: tuple[str, ...], seeds: tuple[int, ...]
) -> list[tuple[str, int, str]]:
    """``(model, seed, trial_id)`` triples, MODEL-major.

    Model-major so one provider instance serves its four seeds back to back:
    consecutive same-model trials keep hitting the prompt cache on the frozen
    system-prompt prefix (doc 06 §8 names caching as the main input-cost
    lever, and the cache has a TTL measured in minutes, not hours). The order
    is a cost optimisation only — the matrix itself is identical either way.
    """
    return [(m, s, f"{m}_seed{s}") for m in models for s in seeds]


# ---------------------------------------------------------------------------
# Freeze manifest (results/freeze.json)
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """sha256 over raw bytes; ``"<missing>"`` for an absent file.

    The sentinel mirrors ``loop.config_hash``'s ``b"<missing>"`` so a deleted
    frozen file shows up as a NAMED drift in the guard's refusal instead of a
    traceback that names nothing.
    """
    if not path.exists():
        return "<missing>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_manifest(
    root: Path = REPO_ROOT, files: tuple[str, ...] = FROZEN_FILES
) -> dict:
    """The results/freeze.json document (doc 06 §2/§7 — schema pinned in §7).

    ``files`` stays in FROZEN_FILES order because that order is an input to
    ``config_hash`` (paths are hashed alongside contents, so a reorder IS a
    change); serialising it in the same order keeps the file diffable against
    the code that defines it. The combined ``config_hash`` is computed by
    ``loop.config_hash`` itself, never re-implemented here — two hash
    implementations is how the guard and the trial JSONs start disagreeing.
    """
    models, seeds = load_matrix(root)
    return {
        "schema": FREEZE_SCHEMA,
        "frozen_at": _utc_now(),
        "freeze_commit": freeze_commit(root, files),
        "config_hash": config_hash(files, root),
        "matrix": {"models": list(models), "seeds": [int(s) for s in seeds]},
        "files": {rel: file_sha256(root / rel) for rel in files},
    }


def write_freeze(
    root: Path = REPO_ROOT, files: tuple[str, ...] = FROZEN_FILES
) -> Path:
    path = root / "results" / "freeze.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(freeze_manifest(root, files), indent=2) + "\n", encoding="utf-8"
    )
    return path


def read_freeze(root: Path = REPO_ROOT) -> dict | None:
    """The stored manifest, or None if it does not exist. Raises on corrupt
    JSON — the caller turns that into a refusal, not a guess."""
    path = root / "results" / "freeze.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_refusals(
    manifest: dict, root: Path = REPO_ROOT, files: tuple[str, ...] = FROZEN_FILES
) -> list[str]:
    """Live tree vs stored freeze — every reason to hard-refuse, each naming
    its file (doc 06 §7: "a message naming the changed file")."""
    out: list[str] = []
    if manifest.get("schema") != FREEZE_SCHEMA:
        out.append(
            f"results/freeze.json schema is {manifest.get('schema')!r}, expected "
            f"{FREEZE_SCHEMA!r} — refusing to interpret an unknown freeze format"
        )
        return out
    recorded = manifest.get("files")
    if not isinstance(recorded, dict) or not recorded:
        out.append("results/freeze.json carries no `files` map — re-freeze")
        return out

    # File-SET drift first: a manifest change (FROZEN_FILES edited) invalidates
    # the batch even if every still-listed file is byte-identical, because the
    # contract itself changed. Editing FROZEN_FILES also edits loop.py, so the
    # per-file check below would fire too — but this message says WHY.
    for rel in sorted(set(recorded) - set(files)):
        out.append(
            f"results/freeze.json freezes {rel}, which FROZEN_FILES no longer "
            "lists — the manifest changed since the freeze; new freeze commit, "
            "new batch directory (doc 06 §2: all prior trials invalid)"
        )
    for rel in sorted(set(files) - set(recorded)):
        out.append(
            f"frozen file absent from results/freeze.json: {rel} — the freeze "
            "predates a manifest change; new freeze commit, new batch directory"
        )

    for rel in files:
        if rel not in recorded:
            continue
        live = file_sha256(root / rel)
        if live != recorded[rel]:
            out.append(
                f"frozen file changed since the freeze: {rel} "
                f"(sha256 {str(recorded[rel])[:12]}… -> {live[:12]}…). Revert the "
                "change or start a new batch directory — there is deliberately "
                "no --force (doc 06 §7)"
            )

    # Belt and braces: identical file set + identical per-file bytes implies an
    # identical combined hash, so this can only fire on an internally
    # inconsistent (hand-edited) freeze.json — refuse rather than trust either
    # number.
    if not out and manifest.get("config_hash") != config_hash(files, root):
        out.append(
            "results/freeze.json `config_hash` disagrees with its own per-file "
            "hashes — the file was edited by hand; re-freeze"
        )
    return out


def commit_refusals(
    root: Path = REPO_ROOT, files: tuple[str, ...] = FROZEN_FILES
) -> list[str]:
    """The dirty/unknown-commit half of doc 06 §2's Locked box.

    ``loop.freeze_commit`` deliberately never raises ("a trial that ran is
    worth logging even from a tree with no git") and its docstring hands the
    hard refusal to THIS runner. Note what is NOT checked: whether HEAD still
    equals the recorded freeze commit. T4.3 branch (a) — a canary fix touching
    only non-frozen code — legitimately moves HEAD while every frozen byte is
    unchanged, and the per-file hashes above already prove content equality,
    which is the fairness-relevant meaning of "differ from that commit".
    """
    value = freeze_commit(root, files)
    if value == "unknown":
        return [
            "freeze commit is unknown (no git repository found) — doc 06 §2 "
            "requires every result traceable to the exact frozen commit"
        ]
    if value.endswith("-dirty"):
        return [
            "frozen files carry UNCOMMITTED changes (freeze_commit would record "
            f"'{value}') — commit or revert before running; a batch from a "
            "dirty tree ran code no commit contains (AGENTS.md §5)"
        ]
    return []


def freeze_tree_refusals(root: Path = REPO_ROOT) -> list[str]:
    """``--freeze`` ONLY: refuse when ANY tracked file is dirty, not just the
    frozen 15.

    ``commit_refusals`` scopes its ``git status`` to FROZEN_FILES because
    resume must stay legal after a T4.3-branch-(a) fix to non-frozen code. But
    at freeze time that scope is a hole: a dirty ``runner.py`` / ``scoring.py``
    would yield a ``freeze.json`` stamped with a clean sha while the batch
    executors run code no commit contains — the same failure
    ``commit_refusals``'s own message warns about, one directory over. Freeze
    night is defined as a clean commit (PLAN T4.3: "freeze commit;
    ``runner.py --freeze``"), so here the whole tracked tree must be clean.

    ``-uno`` deliberately ignores untracked files: this tree carries scratch
    artifacts (AGENTS.md §5), and an untracked file cannot be code a commit
    *claims* to contain. A no-git tree returns [] — ``commit_refusals`` already
    hard-refuses that case with a better message.
    """
    out = _git(root, "status", "--porcelain", "-uno")
    if not out or not out.strip():
        return []
    # Porcelain rows are `XY <path>` — slice the 3-char prefix per LINE (a
    # whole-string strip would eat the first line's status column).
    dirty = sorted(line[3:].strip() for line in out.splitlines() if line.strip())
    return [
        "the tree carries uncommitted TRACKED changes: "
        + ", ".join(dirty)
        + " — --freeze requires a fully clean commit (frozen or not: a dirty "
        "runner/scorer at freeze time runs code no commit contains); commit "
        "or revert first. Resume is narrower by design (doc 06 §7)."
    ]


# ---------------------------------------------------------------------------
# Resume classification (doc 06 §7, §9.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialPlan:
    model: str
    seed: int
    trial_id: str
    json_path: Path
    status: str
    detail: str


def classify_trial(json_path: Path, live_hash: str) -> tuple[str, str]:
    """One trial JSON -> (status, human detail). Read-only.

    The completeness half is ``scoring.is_complete`` — the runner and the
    scorer must agree on what "done" means (its docstring names this exact
    call site). "Validates against the schema" (doc 06 §7) is deliberately NOT
    a full doc 06 §4 conformance pass here: a validator strict enough to
    reject a paid, complete trial would move it to results/incomplete/ and
    RERUN it — spending real money on a schema nitpick. Full conformance is
    ``scripts/audit_trial.py``'s job, after the batch, where a failure costs a
    re-read instead of a re-run. Pinned in doc 06 §7 (this change).
    """
    if not json_path.exists():
        return STATUS_PENDING, "no JSON yet"
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        # TrialLog writes through a temp file + os.replace, so a torn file
        # should be impossible — but a resume gate that crashed on one would
        # wedge the whole batch on the artifact it exists to route around.
        return STATUS_INCOMPLETE, f"unparseable JSON ({error})"
    config = document.get("config") or {}
    # A smoke-capped run (run_trial.py --max-turns) records turn_cap_override
    # precisely so it "can never be mistaken for a result" — skipping it as
    # complete would pool a 5-turn smoke as a benchmark trial, and silently
    # rerunning it would shred a file the operator parked there on purpose.
    # Neither is the runner's call to make: hard refuse (see plan_refusals).
    # Checked BEFORE completeness: a smoke run that infra-crashed (override
    # present, no `final`) is still the operator's parked file, and retiring
    # it as merely INCOMPLETE would silently spend a full paid trial in a slot
    # the operator earmarked for smoke — money spent without the refusal this
    # marker exists to force.
    if config.get("turn_cap_override") is not None:
        return (
            STATUS_SMOKE_CAPPED,
            f"config.turn_cap_override={config['turn_cap_override']} — smoke run, not a result",
        )
    if not is_complete(document):
        if "infra_failure" in document:
            first = str(document["infra_failure"]).strip().splitlines()
            return STATUS_INCOMPLETE, f"infra_failure: {first[-1][:120] if first else ''}"
        return STATUS_INCOMPLETE, "no `final` block"
    stored = config.get("config_hash")
    if stored != live_hash:
        return (
            STATUS_HASH_DRIFT,
            f"stored config_hash {str(stored)[:12]}… != live {live_hash[:12]}…",
        )
    return STATUS_COMPLETE, "final present, config_hash matches"


def plan_batch(
    models: tuple[str, ...],
    seeds: tuple[int, ...],
    raw_dir: Path,
    live_hash: str,
) -> list[TrialPlan]:
    plans = []
    for model, seed, trial_id in trial_matrix(models, seeds):
        json_path = raw_dir / f"{trial_id}.json"
        status, detail = classify_trial(json_path, live_hash)
        plans.append(
            TrialPlan(
                model=model,
                seed=seed,
                trial_id=trial_id,
                json_path=json_path,
                status=status,
                detail=detail,
            )
        )
    return plans


def foreign_trials(raw_dir: Path, plans: list[TrialPlan]) -> list[Path]:
    """JSONs in the results dir that belong to NO matrix slot.

    The judge hazard, one directory later: ``frozen_matrix``'s guard keeps
    ``--model judge`` from ever running, but a benchmark-shaped
    ``judge_seed101.json`` already sitting in results/raw/ would still be
    folded in by any glob-based aggregator (T4.4). Named here so the operator
    moves it before the batch, not after the figures.
    """
    if not raw_dir.exists():
        return []
    known = {plan.json_path.name for plan in plans}
    return sorted(
        path for path in raw_dir.glob("*.json") if path.name not in known
    )


def plan_refusals(plans: list[TrialPlan], foreign: list[Path] = ()) -> list[str]:
    """The startup guard over existing results (doc 06 §7's second paragraph).

    HASH_DRIFT refuses the WHOLE batch, never reruns: a complete result under
    a different frozen configuration means either the tree changed (revert it)
    or the result predates the freeze (move it aside) — rerunning would
    silently discard a paid trial, and skipping would pool incomparable data.
    """
    out: list[str] = []
    for plan in plans:
        if plan.status == STATUS_HASH_DRIFT:
            out.append(
                f"{plan.json_path}: complete result under a DIFFERENT frozen "
                f"configuration ({plan.detail}). If it is a pre-freeze artifact "
                "(e.g. a T3.5 sanity trial), move it out of the results dir "
                "(results/logs/ keeps it citable); if a frozen file changed "
                "since it ran, revert the change or start a new batch "
                "directory. No --force exists (doc 06 §7)."
            )
        elif plan.status == STATUS_SMOKE_CAPPED:
            out.append(
                f"{plan.json_path}: {plan.detail}. It occupies matrix slot "
                f"{plan.trial_id}; move it out of the results dir before the "
                "batch."
            )
    for path in foreign:
        out.append(
            f"{path}: trial JSON outside the frozen matrix — a glob-based "
            "aggregator would fold it into the comparison; move it aside."
        )
    return out


def format_dry_run(plans: list[TrialPlan]) -> str:
    lines = []
    for index, plan in enumerate(plans, 1):
        status = (
            plan.status.upper()
            if plan.status in (STATUS_HASH_DRIFT, STATUS_SMOKE_CAPPED)
            else plan.status
        )
        lines.append(
            f"  [{index:2d}/{len(plans)}] {plan.trial_id:<18} {status:<13} {plan.detail}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# results/incomplete/ + results/rerun_log.md
# ---------------------------------------------------------------------------


def ensure_rerun_log(path: Path) -> None:
    """Write the header if the log does not exist — via temp + ``os.replace``.

    Atomic for the same reason ``TrialLog`` flushes through ``os.replace``: a
    crash mid-``write_text`` would leave a half-header that ``exists()`` then
    treats as done forever, so the table's contract line (``| trial id | …``)
    would never arrive and every later row would be orphaned.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.name + ".tmp")
        tmp.write_text(RERUN_LOG_HEADER, encoding="utf-8")
        os.replace(tmp, path)


def append_rerun_log(path: Path, trial_id: str, cause: str, evidence: str) -> None:
    """One markdown table row per move — the log ships with the results.

    Pipes and newlines in the cause are flattened so a traceback fragment
    cannot break the table a reader (or T4.5's report) parses. The append is
    torn-tolerant: if a previous append died mid-row (power cut — the log is
    written across an hours-long unattended batch), the file ends without a
    newline and a naive append would FUSE two rows into one line, silently
    shifting every cell a parser reads; a leading newline restores the table
    instead, leaving the torn fragment visible as its own (malformed) line.
    """
    ensure_rerun_log(path)
    flat = " ".join(str(cause).split()).replace("|", "\\|")[:200]
    torn = path.stat().st_size > 0
    if torn:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            torn = handle.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            ("\n" if torn else "")
            + f"| {trial_id} | {_utc_now()} | {flat} | {evidence} |\n"
        )


def retire_incomplete(
    json_path: Path,
    *,
    incomplete_dir: Path,
    rerun_log_path: Path,
    cause: str,
) -> Path:
    """Move a partial trial JSON aside and log the move (doc 06 §7).

    The frames directory moves WITH the JSON when it exists: the preserved
    JSON's ``obs.frame_paths`` reference frames that the rerun's ``TrialLog``
    is about to WIPE (its constructor clears ``frames/<trial_id>/`` — the T3.5
    accumulation bug), so leaving them behind preserves a file whose evidence
    is destroyed seconds later.

    The log row is appended BEFORE the move: a crash between the two must not
    produce an unlogged retirement — a logged move that then failed is visible
    (the row's evidence path does not exist, the source file is still in
    place) and re-attempted on resume; an unlogged move is exactly the
    "silent rerun" doc 06 §7 forbids.
    """
    incomplete_dir.mkdir(parents=True, exist_ok=True)
    trial_id = json_path.stem
    stamp = _stamp()
    dest = incomplete_dir / f"{trial_id}.{stamp}.json"
    attempt = 1
    while dest.exists():
        # Two retirements of one trial inside one second (an instant infra
        # failure being retried) must not overwrite the first's evidence.
        attempt += 1
        dest = incomplete_dir / f"{trial_id}.{stamp}-{attempt}.json"

    try:
        evidence = str(dest.relative_to(rerun_log_path.parent.parent))
    except ValueError:
        evidence = str(dest)
    append_rerun_log(rerun_log_path, trial_id, cause, evidence)

    shutil.move(str(json_path), str(dest))
    frames_dir = json_path.parent / "frames" / trial_id
    if frames_dir.is_dir():
        frames_dest = incomplete_dir / "frames" / dest.stem
        frames_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(frames_dir), str(frames_dest))
    return dest


# ---------------------------------------------------------------------------
# The per-trial body — shared with scripts/run_trial.py (SPEC: factored, not
# duplicated; the T3.5-proven single-trial path and the batch runner must run
# byte-identical trial logic or the batch measures a different harness than
# the one the gate passed)
# ---------------------------------------------------------------------------


@dataclass
class TrialOutcome:
    trial_id: str
    json_path: Path
    document: dict
    final: dict | None
    video_path: str | None
    infra_detail: str | None
    #: KeyboardInterrupt/SystemExit — an operator abort, not an infra fault.
    #: The batch must stop (and release the GPU), not retry.
    interrupted: bool


def announce(record: dict) -> None:
    """One line per model turn — the unattended log-tail view of a trial."""
    names = ", ".join(c["name"] for c in record["model_output"]["tool_calls"]) or "(none)"
    print(
        f"  [{record['stage']} t{record['turn_idx']:02d}] {names}"
        f"  |  {record['execution']['result']}"
        f"  |  budget {record['budget']['stage_turns_used']}"
        f"/{record['budget']['stage_turn_cap']} turns,"
        f" {record['budget']['stage_policy_seconds_used']:.1f}"
        f"/{record['budget']['stage_policy_seconds_cap']:g} s",
        flush=True,
    )


def run_one_trial(
    session,
    *,
    model_name: str,
    cfg,
    provider,
    seed: int,
    out_dir: Path,
    video_dir: Path,
    video_every_n: int = 1,
    no_video: bool = False,
    max_turns: int | None = None,
    on_turn=None,
) -> TrialOutcome:
    """One trial against an ALREADY-RUNNING session.

    Everything per-trial is re-created fresh here; everything persistent is
    reset in place — the split scenario_s5 proved in the T3.5 gate:

    * ``session.reset(seed, SpawnPose)`` rewrites the reset_base event on both
      the cfg AND the live event term (the term caches its params — updating
      only cfg silently respawns at the previous seed's pose), reseeds, resets
      playback (clears ``_fell``/diagnostics so a fallback read cannot serve
      the previous trial's fall to this one), and settles.
    * ``HeadCamera`` + ``warmup()`` after every reset (5 renders cost
      milliseconds; one gray first observation poisons the opening room guess).
    * Fresh ``Memory``/``PositionIntegrator``/``Counters``/``ToolContext`` —
      the integrator's spawn anchor is the ONE ground-truth input (doc 05 §5.1).
    * Fresh ``TrialLog`` (wipes its frames dir) and ``Recorder`` (wipes its
      frame dir); ``attach_recorder``'s ``detach()`` runs on every exit path —
      a second attach without detach would nest recorders and feed frames to a
      dead one.

    The caller owns launch/close and the provider (built AFTER kit — the Omit
    hazard); this function owns the doc 05 §8 infra boundary and the
    scoring-before-audit artifact order.
    """
    import traceback

    from duck_embody.agent.loop import EpisodeRunner, TrialLog, redact_secrets
    from duck_embody.agent.memory import Counters, Memory, PositionIntegrator
    from duck_embody.agent.tools import ToolContext
    from duck_embody.env.camera import HeadCamera
    from duck_embody.sim.recorder import Recorder, attach_recorder
    from duck_embody.sim.session import SpawnPose
    from duck_embody.tasks.find_kitchen import spawn_for_seed, stage_specs

    out_dir = Path(out_dir)
    video_dir = Path(video_dir)
    trial_id = f"{model_name}_seed{seed}"
    json_path = out_dir / f"{trial_id}.json"
    spawn_xy, spawn_heading = spawn_for_seed(seed)

    # The log is constructed BEFORE any sim setup so the infra boundary below
    # always has a JSON to record into: a fault in session.reset / camera
    # warmup / recorder attach is an infra failure like any other and must
    # take the same retire+log+retry path, not abort the batch with a bare
    # traceback (a startup-phase render fault is exactly the transient
    # --infra-retries exists for). A failure in THIS constructor is a
    # results-disk fault: no money has been spent, there is nowhere to record
    # infra anyway, and pressing on against a broken results tree would lose
    # every later artifact — so it alone propagates.
    log = TrialLog(
        json_path,
        trial_id=trial_id,
        model_id=cfg.model_id,
        model_name=model_name,
        seed=seed,
        spawn_xy=spawn_xy,
        spawn_heading_deg=spawn_heading,
    )
    if max_turns is not None:
        # Recorded so a capped smoke run can never be mistaken for a result —
        # the batch runner's resume gate hard-refuses on this field.
        log.document["config"]["turn_cap_override"] = max_turns
        log.flush()

    recorder = None
    detach = None
    final = None
    infra_detail = None
    interrupted = False
    video_rel = None
    try:
        session.reset(
            seed=seed, spawn=SpawnPose(spawn_xy[0], spawn_xy[1], spawn_heading)
        )
        camera = HeadCamera(session.env)
        camera.warmup()

        if not no_video:
            # hide_ceiling=True is not optional in the apartment: the chase
            # camera sits above the 0.7 m walls — with the roof on, every
            # audit frame is a photo of the roof (T2.4, recorder.py's own
            # docstring).
            recorder = Recorder(
                video_dir / trial_id, fps=25, every_n=video_every_n, hide_ceiling=True
            )
            detach = attach_recorder(session.playback, session.env.unwrapped, recorder)

        counters = Counters()
        if max_turns is not None:
            counters.turn_cap = max_turns
        context = ToolContext(
            playback=session.playback,
            camera=camera,
            memory=Memory(),
            # The seed's spawn coordinates are the integrator's t=0 anchor —
            # the ONE thing the dead-reckoned estimate takes from ground truth.
            integrator=PositionIntegrator(*spawn_xy),
            counters=counters,
        )
        runner = EpisodeRunner(
            provider=provider,
            context=context,
            stages=stage_specs(seed),
            log=log,
            on_turn=on_turn,
        )
        final = runner.run()
    except BaseException as exc:  # noqa: BLE001 — deliberate, see below
        # doc 05 §8's INFRA path, taken at the only place it is allowed to be
        # taken: the trial boundary — which starts at session.reset, not at
        # runner.run(): a reset/warmup/attach fault bypassing this handler
        # would skip the rerun log and --infra-retries and kill an unattended
        # batch on a possibly-transient fault. Catching inside the loop would
        # launder a render error or a physics NaN into a model-visible result;
        # here the JSON keeps NO `final` block, so the resume check rejects it
        # and the trial reruns whole.
        #
        # `BaseException`, not `Exception`: a Ctrl-C that skipped the caller's
        # `session.close()` would leave a kit process holding the machine's
        # single GPU (AGENTS.md rule 1). The interrupt is flagged, not
        # swallowed silently — the batch runner aborts on it instead of
        # "retrying" the operator's own abort as if it were an infra fault.
        #
        # The traceback is scrubbed (`redact_secrets`) before it is stored OR
        # printed: it is third-party exception text and this JSON is committed
        # to a public repo (AGENTS.md rules 6 and 7).
        infra_detail = redact_secrets(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )
        log.note_infra_failure(infra_detail)
        interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
        print("\n  INFRA FAILURE — trial is incomplete and must be rerun whole:")
        print(infra_detail)

    # --- the SCORING artifact first ----------------------------------------
    #
    # `log.finish(final)` runs before any video work and inside no other
    # guard. A completed episode is 40-80 turns of paid API plus the QA
    # exchange; if an ffmpeg fault could pre-empt this line the JSON would
    # hold every turn and no `final` — byte-for-byte an infra failure, and
    # the resume check would rerun a finished, paid trial.
    if final is not None:
        log.finish(final)

    # --- then the rule-11 audit artifacts, guarded separately ---------------
    try:
        if detach is not None:
            detach()
        if recorder is not None and not interrupted:
            # Skipped on interrupt: encoding thousands of PNGs after a Ctrl-C
            # holds the GPU for minutes to produce evidence of a trial that
            # reruns whole anyway. The frames are deleted with the recorder
            # dir on the rerun's wipe.
            mp4 = recorder.encode()
            if mp4 is not None:
                recorder.filmstrip(mp4)
                try:
                    video_rel = str(mp4.relative_to(REPO_ROOT))
                except ValueError:
                    # video dir outside the repo: absolute is still a usable
                    # pointer, and raising here would cost the trial for a
                    # path preference.
                    video_rel = str(mp4)
    except Exception:  # noqa: BLE001 — evidence failure, not a trial failure
        print("\n  WARNING: video artifacts failed; the trial result stands:")
        traceback.print_exc()
    log.set_video(video_rel)

    return TrialOutcome(
        trial_id=trial_id,
        json_path=json_path,
        document=log.document,
        final=final,
        video_path=video_rel,
        infra_detail=infra_detail,
        interrupted=interrupted,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    """Argument parser. Built outside ``main`` so ``--help`` needs no kit.

    Deliberately NO ``--force`` and NO per-trial selection: doc 06 §7 forbids
    the former, and §3.2 (Locked) forbids selective reruns — the only unit the
    runner knows is "the whole remaining matrix".
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="runner.py",
        description="Run/resume the frozen Duck Embody benchmark matrix "
        "(3 models x 4 seeds) in ONE persistent kit session.",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="write results/freeze.json from the current (clean) tree and exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list every trial with its resume status and the guard verdict; "
             "touch nothing, launch nothing",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--checkpoint", default=None,
                        help="policy .pt (default: policy/model_2999.pt)")
    parser.add_argument("--video-every-n", type=int, default=1,
                        help="grab every Nth recording chunk (1 = 25 fps)")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--headed", action="store_true", help="run with a viewport")
    parser.add_argument(
        "--infra-retries", type=int, default=1,
        help="in-run retries per trial after an INFRA failure before the batch "
             "aborts (model failures are final and never retried)",
    )
    return parser


def _print_refusals(refusals: list[str]) -> None:
    print("FATAL: the freeze guard refuses to run (doc 06 §7):")
    for reason in refusals:
        print(f"  - {reason}")
    print("There is deliberately no --force flag.")


def cmd_freeze(root: Path = REPO_ROOT) -> int:
    missing = [rel for rel in FROZEN_FILES if not (root / rel).exists()]
    if missing:
        print(f"FATAL: frozen files missing from disk: {missing}")
        return 2
    # --freeze demands a FULLY clean tracked tree (freeze_tree_refusals), not
    # just clean frozen files: the freeze commit is what every trial JSON
    # claims traceability to, and the runner/scorer executing that night are
    # part of the claim even though they are not part of the fairness hash.
    refusals = commit_refusals(root) + freeze_tree_refusals(root)
    if refusals:
        _print_refusals(refusals)
        return 2
    path = write_freeze(root)
    manifest = read_freeze(root)
    print(f"wrote {path}")
    print(f"  freeze_commit : {manifest['freeze_commit']}")
    print(f"  config_hash   : {manifest['config_hash']}")
    print(f"  files frozen  : {len(manifest['files'])}")
    print(f"  matrix        : {manifest['matrix']}")
    return 0


def _startup_refusals(
    root: Path, plans: list[TrialPlan], raw_dir: Path
) -> list[str]:
    """Everything that must be true before a single trial runs or resumes."""
    refusals: list[str] = []
    try:
        manifest = read_freeze(root)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        manifest = None
        refusals.append(f"results/freeze.json is unparseable ({error}) — re-freeze")
    else:
        if manifest is None:
            refusals.append(
                "results/freeze.json does not exist — run `runner.py --freeze` "
                "at the freeze commit first (doc 06 §2/§7, PLAN T4.3)"
            )
    if manifest is not None:
        refusals += freeze_refusals(manifest, root)
    refusals += commit_refusals(root)
    refusals += plan_refusals(plans, foreign_trials(raw_dir, plans))
    return refusals


def midbatch_refusals(root: Path = REPO_ROOT) -> list[str]:
    """The freeze half of the guard, RE-RUN before every trial launch.

    A startup-only guard leaves the whole unattended batch exposed: an edit to
    a frozen file during trial 6 of an hours-long night (the doc 06 §2
    recorded incident shape — AGENTS.md §5: this tree "always carries
    uncommitted work") would run trials 6-12 under different frozen bytes, and
    the only detection would be a LATER restart noticing the drift, which a
    batch that finishes in one go never gets. Recomputing here is ~15 sha256
    reads plus one ``git status`` — microseconds against a multi-minute trial.

    ``freeze.json`` is re-READ too (not reused from startup) so a mid-batch
    edit or deletion of the manifest itself also refuses. Plan-level statuses
    are deliberately NOT re-checked: earlier trials this batch legitimately
    wrote new complete JSONs.
    """
    try:
        manifest = read_freeze(root)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return [f"results/freeze.json became unparseable mid-batch ({error})"]
    if manifest is None:
        return ["results/freeze.json disappeared mid-batch — refusing to continue"]
    return freeze_refusals(manifest, root) + commit_refusals(root)


def _abort_on_midbatch_drift(root: Path, prefix: str) -> bool:
    """True (and prints the refusal) iff the frozen tree drifted mid-batch."""
    refusals = midbatch_refusals(root)
    if not refusals:
        return False
    print(
        f"{prefix} FATAL: the frozen configuration DRIFTED MID-BATCH — "
        "aborting before this trial (doc 06 §7). Completed trials stand and "
        "resume will skip them once the drift is reverted:",
        flush=True,
    )
    for reason in refusals:
        print(f"  - {reason}", flush=True)
    print("There is deliberately no --force flag.", flush=True)
    return True


def cmd_dry_run(root: Path = REPO_ROOT, raw_dir: Path | None = None) -> int:
    raw_dir = Path(raw_dir) if raw_dir is not None else root / "results" / "raw"
    models, seeds = load_matrix(root)
    live = config_hash(FROZEN_FILES, root)
    plans = plan_batch(models, seeds, raw_dir, live)
    print(
        f"== dry run: {len(models)} models x {len(seeds)} seeds = "
        f"{len(plans)} trials ==",
    )
    print(f"  live config_hash : {live}")
    try:
        manifest = read_freeze(root)
    except (json.JSONDecodeError, UnicodeDecodeError):
        manifest = {"config_hash": "<unparseable>"}
    print(
        "  freeze.json      : "
        + (
            f"config_hash {manifest.get('config_hash')}"
            if manifest is not None
            else "MISSING (run --freeze at the freeze commit)"
        )
    )
    print(format_dry_run(plans))
    pending = sum(p.status in (STATUS_PENDING, STATUS_INCOMPLETE) for p in plans)
    skips = sum(p.status == STATUS_COMPLETE for p in plans)
    print(f"  would run {pending}, skip {skips} — nothing touched")
    refusals = _startup_refusals(root, plans, raw_dir)
    if refusals:
        _print_refusals(refusals)
        return 2
    return 0


def cmd_run(args, root: Path = REPO_ROOT) -> int:
    raw_dir = Path(args.out_dir)
    video_dir = Path(args.video_dir)
    incomplete_dir = root / "results" / "incomplete"
    rerun_log = root / "results" / "rerun_log.md"

    models, seeds = load_matrix(root)
    live = config_hash(FROZEN_FILES, root)
    plans = plan_batch(models, seeds, raw_dir, live)

    # Every guard, BEFORE the multi-minute cold start and before any file is
    # touched: a refusal after launch would waste the cold start, and a
    # refusal after a retirement would have already moved someone's file.
    refusals = _startup_refusals(root, plans, raw_dir)
    if refusals:
        _print_refusals(refusals)
        return 2

    # Fail fast on a missing key or an unknown provider for EVERY model in the
    # matrix — WITHOUT importing the vendor SDKs (the Omit hazard): a bad
    # OPENAI_API_KEY must surface now, not eight paid trials into the night.
    from duck_embody.agent.providers.base import build_provider, preflight_provider

    cfgs = {name: preflight_provider(name) for name in models}

    # Stale-bytecode guard (AGENTS.md §5): isaaclab.sh -p does not set
    # PYTHONDONTWRITEBYTECODE, and a same-second same-size edit silently runs
    # the OLD module — cheap to assert, expensive to discover from a batch.
    from duck_embody.sim.policy_wrapper import ExecResult as _ExecResult

    if "fall_diagnostics" not in _ExecResult.__dataclass_fields__:
        print("FATAL: loaded policy_wrapper is STALE (no fall_diagnostics). "
              "Clear __pycache__ and re-run.")
        return 2

    # AGENTS.md rule 1, automated: refuse to launch beside another GPU/kit job
    # (preflight.py's own docstring names this runner as a mandatory caller).
    from duck_embody.sim.preflight import format_refusal, rule1_violations

    violations = rule1_violations()
    if violations:
        print(format_refusal(violations))
        return 2

    from duck_embody.sim.session import SimSession

    total = len(plans)
    print(f"== batch: {total} trials, one kit session ==", flush=True)
    session = SimSession.launch(
        task_id=TASK_ID, checkpoint=args.checkpoint, headless=not args.headed
    )
    try:
        # Built AFTER kit (the measured Omit-sentinel hazard) and ONCE per
        # model: providers are stateless between sends, and a broken client
        # config fails here — before trial 1 — rather than mid-batch.
        providers = {name: build_provider(name) for name in models}

        completed = 0
        skipped = 0
        for index, plan in enumerate(plans, 1):
            prefix = f"[{index:2d}/{total}] {plan.trial_id}"
            if plan.status == STATUS_COMPLETE:
                print(f"{prefix} SKIP ({plan.detail})", flush=True)
                skipped += 1
                continue
            # The freeze guard again, before EVERY launch — and before the
            # retirement below, so a drifted tree moves nobody's file. See
            # midbatch_refusals for why startup-only is not enough.
            if _abort_on_midbatch_drift(root, prefix):
                return 2
            if plan.status == STATUS_INCOMPLETE:
                dest = retire_incomplete(
                    plan.json_path,
                    incomplete_dir=incomplete_dir,
                    rerun_log_path=rerun_log,
                    cause=plan.detail,
                )
                print(f"{prefix} retired incomplete JSON -> {dest}", flush=True)

            attempts = 0
            while True:
                attempts += 1
                if attempts > 1 and _abort_on_midbatch_drift(root, prefix):
                    # An infra retry is a launch too — the drift may have
                    # landed during the failed attempt.
                    return 2
                print(f"{prefix} START attempt {attempts} {_utc_now()}", flush=True)
                outcome = run_one_trial(
                    session,
                    model_name=plan.model,
                    cfg=cfgs[plan.model],
                    provider=providers[plan.model],
                    seed=plan.seed,
                    out_dir=raw_dir,
                    video_dir=video_dir,
                    video_every_n=args.video_every_n,
                    no_video=args.no_video,
                    on_turn=announce,
                )
                if outcome.interrupted:
                    retire_incomplete(
                        outcome.json_path,
                        incomplete_dir=incomplete_dir,
                        rerun_log_path=rerun_log,
                        cause="operator interrupt — batch aborted, trial reruns on resume",
                    )
                    print(f"{prefix} INTERRUPTED — aborting the batch", flush=True)
                    return 1
                if outcome.final is not None:
                    final = outcome.final
                    outcomes = " ".join(
                        f"{stage}={verdict}" for stage, verdict in final["outcome"].items()
                    )
                    cost = final["tokens"]["cost_usd_estimate"]
                    print(
                        f"{prefix} END {outcomes} turns={len(outcome.document['turns'])} "
                        f"bumps={final['bumps']} cost=${cost:.4f} {_utc_now()}",
                        flush=True,
                    )
                    completed += 1
                    break
                dest = retire_incomplete(
                    outcome.json_path,
                    incomplete_dir=incomplete_dir,
                    rerun_log_path=rerun_log,
                    cause=f"infra failure (attempt {attempts}): "
                          + ((outcome.infra_detail or "").strip().splitlines() or ["unknown"])[-1],
                )
                print(f"{prefix} INFRA FAILURE — partial JSON -> {dest}", flush=True)
                if attempts > args.infra_retries:
                    # Two consecutive infra failures on ONE trial is a broken
                    # harness, not bad luck. An unattended batch that pressed
                    # on would burn the remaining budget against a possibly
                    # degraded sim — every later trial suspect, none loudly
                    # failed. Abort; resume skips the finished trials.
                    print(
                        f"{prefix} still failing after {attempts} attempts — "
                        "aborting the batch (fix the cause, then re-run to "
                        "resume; completed trials are skipped)",
                        flush=True,
                    )
                    return 1
        print(
            f"== batch done: {completed} run, {skipped} skipped, "
            f"{total - completed - skipped} not reached ==",
            flush=True,
        )
        return 0
    finally:
        # `finally`, so the kit process is released on EVERY path. A surviving
        # kit process holds the machine's single GPU and the rerun cannot
        # start at all (rule 1).
        print("  closing app (nothing after this line runs)", flush=True)
        session.close()


def main() -> int:
    # Parsed BEFORE anything launches: AppLauncher inside SimSession.launch()
    # parses sys.argv for its own flags and would choke on ours. Stripping
    # them afterwards leaves kit exactly the argv it expects.
    args, kit_argv = build_parser().parse_known_args()
    sys.argv = [sys.argv[0], *kit_argv]

    if args.freeze and args.dry_run:
        print("FATAL: --freeze and --dry-run are separate commands")
        return 2
    if args.freeze:
        return cmd_freeze(REPO_ROOT)
    if args.dry_run:
        return cmd_dry_run(REPO_ROOT, Path(args.out_dir))
    return cmd_run(args, REPO_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
