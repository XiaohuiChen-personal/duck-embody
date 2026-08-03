"""Strict, read-only benchmark audit gates.

Every required check has three states: PASS, FAIL, or INCOMPLETE.  INCOMPLETE
means the evidence needed to run the check was not recorded; it can never be
promoted to PASS.  This distinction is essential for historical batches that
predate request journals and write-once manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from duck_embody import forensics, scoring
from duck_embody.agent.loop import (
    reconstruct_neutral_request,
    request_structure_problems,
)
from duck_embody.runner import BATCH_MANIFEST_SCHEMA, manifest_sha256

PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
VERDICT_FIELDS = (
    "locomotion",
    "upright_trunk",
    "alternating_feet_ground_clearance",
    "no_drag_glide_crawl_dither",
    "collision_no_teleport",
    "rooms_recognizable",
    "metric_video_consistent",
    "reviewer",
    "reviewed_at",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""


def _check(name: str, ok: bool, detail: str = "") -> Check:
    return Check(name, PASS if ok else FAIL, detail)


def _missing(name: str, detail: str) -> Check:
    return Check(name, INCOMPLETE, detail)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(repo: Path, batch_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    batch_relative = batch_dir / path
    return batch_relative if batch_relative.exists() else repo / path


def _nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in _nonfinite_paths(child, f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in _nonfinite_paths(child, f"{path}[{index}]")
        ]
    return []


def _manifest_checks(document: dict, manifest: dict) -> list[Check]:
    if manifest.get("schema") != BATCH_MANIFEST_SCHEMA:
        return [
            _missing(
                "write-once manifest SHA",
                f"legacy/unknown schema {manifest.get('schema')!r}",
            ),
            _missing(
                "trial manifest SHA match",
                "trial predates write-once batch manifests",
            ),
        ]
    stored = manifest.get("manifest_sha256")
    actual = manifest_sha256(manifest)
    return [
        _check(
            "write-once manifest SHA",
            isinstance(stored, str) and stored == actual,
            f"stored={stored} actual={actual}",
        ),
        _check(
            "trial manifest SHA match",
            (document.get("config") or {}).get("batch_manifest_sha256") == stored,
            f"trial={(document.get('config') or {}).get('batch_manifest_sha256')} manifest={stored}",
        ),
    ]


def _request_checks(document: dict, batch_dir: Path) -> list[Check]:
    requests = document.get("requests")
    turns = document.get("turns") or []
    if not isinstance(requests, list):
        return [
            _missing("request reconstruction", "no request journal"),
            _missing("frame files and hashes", "no request journal"),
        ]
    expected = len(turns) + (1 if (document.get("final") or {}).get("qa_request_index") is not None else 0)
    errors: list[str] = []
    frame_errors: list[str] = []
    image_count = 0
    for index, request in enumerate(requests):
        try:
            rebuilt = reconstruct_neutral_request(document, index, batch_dir)
            if rebuilt["request_sha256"] != request.get("request_sha256"):
                errors.append(f"request {index}: hash mismatch")
            errors.extend(
                f"request {index}: {problem}"
                for problem in request_structure_problems(document, index, batch_dir)
            )
        except Exception as exc:  # audit must retain every diagnosis
            errors.append(f"request {index}: {exc}")
        for message in request.get("messages") or []:
            blocks = message.get("blocks") or []
            for block in blocks:
                images = [block] if block.get("type") == "image" else block.get("images") or []
                for image in images:
                    image_count += 1
                    frame = batch_dir / str(image.get("frame_path", ""))
                    digest = image.get("sha256")
                    if not frame.is_file():
                        frame_errors.append(f"missing {frame}")
                    elif not isinstance(digest, str):
                        frame_errors.append(f"{frame}: missing sha256")
                    elif _sha256(frame) != digest:
                        frame_errors.append(f"{frame}: sha256 mismatch")
    if len(requests) != expected:
        errors.append(f"{len(requests)} requests for {len(turns)} turns; expected {expected}")
    return [
        _check(
            "request reconstruction",
            not errors,
            errors[0] if errors else f"{len(requests)} request(s)",
        ),
        _check(
            "frame files and hashes",
            image_count > 0 and not frame_errors,
            frame_errors[0] if frame_errors else f"{image_count} hashed image(s)",
        ),
    ]


def _media_checks(document: dict, repo: Path, batch_dir: Path) -> list[Check]:
    value = document.get("video_path")
    if not isinstance(value, str):
        return [_missing("video and filmstrip", "video_path missing")]
    video = _resolve(repo, batch_dir, value)
    filmstrip = video.with_name(f"{video.stem}_filmstrip.png")
    return [
        _check(
            "video and filmstrip",
            video.is_file() and video.stat().st_size > 0
            and filmstrip.is_file() and filmstrip.stat().st_size > 0,
            f"video={video} filmstrip={filmstrip}",
        )
    ]


def _scorer_checks(document: dict) -> tuple[list[Check], dict[str, Any] | None]:
    try:
        replay = scoring.score_trial(document).as_dict()
        mismatches: list[str] = []
        for stage, raw in ((document.get("final") or {}).get("stages") or {}).items():
            stored = raw.get("score") or {}
            expected = (
                scoring.stage_success(document, stage)
                if scoring.trial_success_criterion(document)
                == scoring.CRITERION_V2_ANY_COUNTER
                else scoring.stage_success_preregistered(document, stage)
            )
            if "success" in stored and bool(stored["success"]) != expected:
                mismatches.append(f"{stage}.score.success")
        nonfinite = _nonfinite_paths(replay)
        return (
            [
                _check(
                    "scorer replay",
                    not mismatches,
                    f"mismatch: {mismatches}" if mismatches else "canonical scorer completed",
                ),
                _check(
                    "no NaN or Infinity",
                    not (_nonfinite_paths(document) + nonfinite),
                    (_nonfinite_paths(document) + nonfinite)[0]
                    if (_nonfinite_paths(document) + nonfinite)
                    else "",
                ),
            ],
            replay,
        )
    except Exception as exc:
        return (
            [
                _check("scorer replay", False, str(exc)),
                _missing("no NaN or Infinity", "scorer replay did not complete"),
            ],
            None,
        )


def _model_usage_checks(document: dict) -> list[Check]:
    turns = document.get("turns") or []
    config = document.get("config") or {}
    configured_alias = config.get("model_config")
    logged_resolved = config.get("resolved_model")
    missing_model: list[int] = []
    inconsistent: list[str] = []
    resolved_ids: set[str] = set()
    missing_usage: list[int] = []
    normalized = {
        "input_tokens_total",
        "input_tokens_uncached",
        "output_tokens_total",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd_estimate",
        "pricing_version",
        "pricing_source",
    }
    for index, turn in enumerate(turns):
        metadata = turn.get("response_metadata") or {}
        resolved = metadata.get("resolved_model_id")
        if not resolved:
            missing_model.append(index)
        else:
            resolved_ids.add(str(resolved))
        alias = metadata.get("configured_alias")
        if configured_alias and alias != configured_alias:
            inconsistent.append(f"turn {index}: alias {alias} != {configured_alias}")
        usage = turn.get("usage")
        if not isinstance(usage, dict) or not normalized.issubset(usage):
            missing_usage.append(index)
        if not isinstance(metadata.get("provider_usage"), dict):
            missing_usage.append(index)
    final_tokens = (document.get("final") or {}).get("tokens")
    if not isinstance(final_tokens, dict) or not normalized.issubset(final_tokens):
        missing_usage.append(-1)
    return [
        (
            _missing("resolved model consistency", f"missing on turns {missing_model[:5]}")
            if missing_model
            else _check(
                "resolved model consistency",
                not inconsistent
                and len(resolved_ids) == 1
                and logged_resolved in resolved_ids,
                (
                    inconsistent[0]
                    if inconsistent
                    else f"turns={sorted(resolved_ids)} config={logged_resolved}"
                ),
            )
        ),
        (
            _missing("provider usage completeness", f"legacy/missing fields at {missing_usage[:5]}")
            if missing_usage
            else _check("provider usage completeness", True, f"{len(turns)} turn(s)")
        ),
    ]


def audit_trial(
    trial_path: Path,
    *,
    batch_dir: Path,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run every mandatory machine check for one trial."""
    checks: list[Check] = []
    try:
        document = json.loads(trial_path.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(_check("complete JSON and QA", False, str(exc)))
        return _result(trial_path.stem, checks)
    try:
        forensics.validate_document(document)
        final = document.get("final") or {}
        qa = final.get("qa")
        checks.append(
            _check(
                "complete JSON and QA",
                isinstance(qa, list) and len(qa) == 5,
                f"qa_answers={len(qa) if isinstance(qa, list) else 0}",
            )
        )
    except Exception as exc:
        checks.append(_check("complete JSON and QA", False, str(exc)))
    checks.append(
        _check(
            "no infrastructure failure",
            "infra_failure" not in document and bool(document.get("final")),
            str(document.get("infra_failure", "")),
        )
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks.extend(_manifest_checks(document, manifest))
    except Exception as exc:
        checks.extend(
            [
                _missing("write-once manifest SHA", str(exc)),
                _missing("trial manifest SHA match", str(exc)),
            ]
        )
    checks.extend(_request_checks(document, batch_dir))
    checks.extend(_media_checks(document, repo_root, batch_dir))
    scorer_checks, replay = _scorer_checks(document)
    checks.extend(scorer_checks)
    try:
        correction = forensics.correction_summary([document])
        checks.append(
            _check(
                "corrections accepted/rejected split",
                correction["calls"] == correction["accepted"] + correction["rejected"],
                f"calls={correction['calls']} accepted={correction['accepted']} rejected={correction['rejected']}",
            )
        )
    except Exception as exc:
        checks.append(_check("corrections accepted/rejected split", False, str(exc)))
        correction = None
    try:
        drift = {
            stage: scoring.stage_drift(document, stage).drift_m
            for stage in scoring.STAGES
        }
        checks.append(_check("drift via canonical parser", True, json.dumps(drift)))
    except Exception as exc:
        checks.append(_check("drift via canonical parser", False, str(exc)))
        drift = None
    checks.extend(_model_usage_checks(document))
    result = _result(str(document.get("trial_id", trial_path.stem)), checks)
    result.update(
        {
            "trial_path": str(trial_path),
            "corrections": correction,
            "drift_m": drift,
            "score_replay": replay,
        }
    )
    return result


def _result(trial_id: str, checks: Iterable[Check]) -> dict[str, Any]:
    rows = list(checks)
    status = (
        FAIL
        if any(row.status == FAIL for row in rows)
        else INCOMPLETE
        if any(row.status == INCOMPLETE for row in rows)
        else PASS
    )
    return {
        "trial_id": trial_id,
        "status": status,
        "checks": [asdict(row) for row in rows],
    }


def visual_verdict_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": INCOMPLETE, "missing": list(VERDICT_FIELDS)}
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = []
    for field in VERDICT_FIELDS:
        match = re.search(
            rf"(?mi)^\s*-\s*{re.escape(field)}\s*:\s*(.*?)\s*$", text
        )
        value = match.group(1).strip(" _") if match else ""
        if not value or "pending" in value.lower():
            missing.append(field)
    return {"status": PASS if not missing else INCOMPLETE, "missing": missing}


def visual_sheet(
    document: dict, trial_path: Path, repo_root: Path, *, link_base: Path
) -> str:
    """Create a deterministic event-indexed visual-review worksheet."""
    selected: dict[str, list[tuple[int, str]]] = {
        "spawn": [],
        "doorways": [],
        "contacts": [],
        "corrections": [],
        "kitchen_declare": [],
        "final": [],
    }
    turns = document.get("turns") or []
    correction_turns = {
        (event.stage, event.turn_idx) for event in forensics.correction_events(document)
    }
    previous_room: str | None = None
    last_framed: list[tuple[int, str]] = []
    from duck_embody.env.apartment_layout import room_at

    for index, turn in enumerate(turns):
        frames = (turn.get("obs") or {}).get("frame_paths") or []
        pose = turn.get("true_pose") or {}
        current_room = (
            room_at(float(pose["x"]), float(pose["y"]))
            if "x" in pose and "y" in pose
            else previous_room
        )
        if not frames:
            previous_room = current_room
            continue
        pairs = [(int(turn["global_turn_idx"]), str(frame)) for frame in frames]
        last_framed = pairs
        if index == 0:
            selected["spawn"].extend(pairs)
        status = (turn.get("obs") or {}).get("status") or {}
        if status.get("contact") or status.get("bumped"):
            selected["contacts"].extend(pairs)
        if (turn["stage"], int(turn["turn_idx"])) in correction_turns:
            selected["corrections"].extend(pairs)
        calls = (turn.get("model_output") or {}).get("tool_calls") or []
        names = {call.get("name") for call in calls}
        memory = turn.get("memory_snapshot") or {}
        if "declare_done" in names or str(memory.get("current_room", "")).lower() == "kitchen":
            selected["kitchen_declare"].extend(pairs)
        if previous_room is not None and current_room != previous_room:
            selected["doorways"].extend(pairs)
        previous_room = current_room
    selected["final"].extend(last_framed)
    lines = [f"# {document['trial_id']} — visual audit sheet", ""]
    for label, pairs in selected.items():
        lines.extend([f"## {label.replace('_', ' ').title()}", ""])
        if not pairs:
            lines.append("_No logged frame at this event; review the linked filmstrip/video._")
        for turn, value in pairs:
            target = _resolve(repo_root, trial_path.parent, value)
            relative = os.path.relpath(target.resolve(), link_base.resolve())
            lines.append(
                f"- turn {turn}: [{Path(value).name}]({relative.replace(os.sep, '/')})"
            )
        lines.append("")
    lines.extend(["## Verdict", ""])
    lines.extend(f"- {field}: _pending_" for field in VERDICT_FIELDS)
    return "\n".join(lines) + "\n"


def audit_batch(
    batch_dir: Path,
    manifest_path: Path,
    *,
    repo_root: Path,
    audit_dir: Path,
    write_sheets: bool = False,
) -> dict[str, Any]:
    documents = forensics.load_batch(batch_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, Any]] = []
    for document in documents:
        trial_path = batch_dir / f"{document['trial_id']}.json"
        machine = audit_trial(
            trial_path,
            batch_dir=batch_dir,
            manifest_path=manifest_path,
            repo_root=repo_root,
        )
        sheet = audit_dir / f"{document['trial_id']}.md"
        if write_sheets and not sheet.exists():
            sheet.write_text(
                visual_sheet(
                    document, trial_path, repo_root, link_base=audit_dir
                ),
                encoding="utf-8",
            )
        machine["visual_audit"] = visual_verdict_status(sheet)
        trials.append(machine)
    expected = len((json.loads(manifest_path.read_text()).get("ordered_trials") or []))
    if not expected:
        matrix = json.loads(manifest_path.read_text()).get("matrix") or {}
        expected = len(matrix.get("models") or []) * len(matrix.get("seeds") or [])
    publication = {
        "expected": expected,
        "written": sum(
            trial["visual_audit"]["status"] == PASS for trial in trials
        ),
        "pending": [
            trial["trial_id"]
            for trial in trials
            if trial["visual_audit"]["status"] != PASS
        ],
    }
    status = (
        FAIL
        if any(trial["status"] == FAIL for trial in trials)
        else INCOMPLETE
        if any(trial["status"] == INCOMPLETE for trial in trials)
        or publication["written"] != expected
        else PASS
    )
    return {
        "schema": "duck-embody-batch-audit-v1",
        "status": status,
        "batch_dir": str(batch_dir),
        "manifest_path": str(manifest_path),
        "publication_gate": publication,
        "trials": trials,
    }
