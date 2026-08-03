"""The v5d_r2 forensic baseline, pinned as executable assertions (PLAN TR.0).

Every remediation task after this one claims to fix a defect measured in
``docs/research/V5D_R2_HARNESS_FORENSICS.md``. Those measurements were made once,
by hand, against ``results/raw_v5d_r2``. This file re-derives them from the raw
JSON through ``duck_embody.forensics`` on every test run, so:

* a later parser change that quietly moves the baseline fails here, loudly;
* a fix can be checked against the number it claims to improve;
* the counts stop depending on the audit Markdown, which was wrong (F-08).

**Counts are exact. Floats get recomputation tolerance only.** Widening a pin
because it stopped matching would defeat the file — PLAN TR.0 says to stop and
find out which of the document, the parser, or the artifact changed.

The batch directory is committed evidence; if it is missing these tests fail
rather than skip, because a green suite with no baseline is the failure mode
this file exists to prevent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from duck_embody import forensics
from duck_embody.forensics import ForensicsError

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = REPO_ROOT / "results" / "raw_v5d_r2"
AUDIT_DIR = REPO_ROOT / "results" / "audits_v5d_r2"
MANIFEST = REPO_ROOT / "results" / "freeze.json"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: `results/raw_v5d_r2`, `config_hash 0e9017a84c06…` — the frozen batch every
#: pin below describes.
CONFIG_HASH = "0e9017a84c066a82637d3db9efa874611c542dbb964efa877bc5aa643f417083"


@pytest.fixture(scope="module")
def documents() -> list[dict]:
    assert BATCH_DIR.is_dir(), f"{BATCH_DIR} is missing — the baseline evidence is gone"
    return forensics.load_batch(BATCH_DIR)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def report(documents: list[dict], manifest: dict) -> dict:
    return forensics.batch_report(
        documents, manifest=manifest, batch_dir=BATCH_DIR
    )


# ---------------------------------------------------------------------------
# Pinned baseline facts
# ---------------------------------------------------------------------------


class TestPinnedBaseline:
    """The ten facts PLAN TR.0 freezes, plus what the parser found beside them."""

    def test_twelve_complete_trials(self, report: dict) -> None:
        integrity = report["integrity"]
        assert integrity["trials"] == 12
        assert integrity["complete_trials"] == 12
        assert integrity["incomplete_trial_ids"] == []

    def test_four_hundred_thirty_four_model_turns(self, report: dict) -> None:
        assert report["integrity"]["total_turns"] == 434

    def test_one_config_hash(self, report: dict) -> None:
        integrity = report["integrity"]
        assert integrity["config_hashes"] == [CONFIG_HASH]
        assert integrity["single_config_hash"] is True
        assert integrity["manifest"]["config_hash_matches"] is True
        assert integrity["manifest"]["missing_cells"] == []
        assert integrity["manifest"]["unexpected_trials"] == []

    def test_sixteen_correction_calls_fifteen_accepted(self, report: dict) -> None:
        corrections = report["corrections"]
        assert corrections["calls"] == 16
        assert corrections["accepted"] == 15
        assert corrections["rejected"] == 1

    def test_fourteen_worsened_one_improved(self, report: dict) -> None:
        corrections = report["corrections"]
        assert corrections["worsened"] == 14
        assert corrections["improved"] == 1
        assert corrections["worsened"] + corrections["improved"] == corrections[
            "accepted"
        ]

    def test_net_added_correction_error_is_3_72_m(self, report: dict) -> None:
        corrections = report["corrections"]
        # Forensics § executive conclusion: 2.35 m before -> 6.07 m after.
        assert corrections["error_before_sum_m"] == pytest.approx(2.3499, abs=1e-3)
        assert corrections["error_after_sum_m"] == pytest.approx(6.0697, abs=1e-3)
        assert corrections["net_added_error_m"] == pytest.approx(3.7198, abs=1e-3)

    def test_fifty_two_multi_motion_turns(self, report: dict) -> None:
        assert report["multi_motion_turns"] == 52

    def test_ten_pending_visual_audits(self, report: dict) -> None:
        audits = report["visual_audits"]
        assert audits["total"] == 12
        assert len(audits["pending"]) == 10
        # Pinned as a SET, not just a count: a hand-edit that completes one audit
        # and reverts another would keep the count and change the evidence.
        assert set(audits["pending"]) == {
            "gpt56sol_seed101.md",
            "gpt56sol_seed102.md",
            "gpt56sol_seed103.md",
            "gpt56sol_seed104.md",
            "opus5_seed101.md",
            "opus5_seed102.md",
            "opus5_seed103.md",
            "opus5_seed104.md",
            "sonnet5_seed103.md",
            "sonnet5_seed104.md",
        }
        assert audits["completed"] == ["sonnet5_seed101.md", "sonnet5_seed102.md"]
        assert audits["complete"] is False

    def test_opus5_seed101_is_the_criterion_split(self, documents: list[dict]) -> None:
        """F-02: live gate said failure, the published scorer says success."""
        document = next(d for d in documents if d["trial_id"] == "opus5_seed101")
        outcomes = forensics.published_and_live_outcomes(document)
        stage1 = outcomes["stages"]["find_kitchen"]
        assert stage1["live_outcome"] == "declared_elsewhere"
        assert stage1["live_success"] is False
        assert stage1["live_distance_m"] == pytest.approx(0.3607, abs=1e-4)
        assert stage1["live_radius_m"] == pytest.approx(0.35, abs=1e-9)
        assert stage1["published_success_v2"] is True
        assert stage1["published_success_preregistered"] is False
        assert stage1["criterion_split"] is True
        stage2 = outcomes["stages"]["return_home"]
        assert stage2["ran"] is False
        assert stage2["live_outcome"] == "not_run"
        assert outcomes["stage1_success_never_offered_return"] is True

    def test_opus5_seed101_is_the_only_denied_return(self, report: dict) -> None:
        assert report["stage1_success_never_offered_return"] == ["opus5_seed101"]

    def test_tool_call_counts_match_the_forensic_baseline(self, report: dict) -> None:
        """Forensics § v5d_r2 forensic baseline, re-derived from tool_calls."""
        assert report["tool_call_counts"] == {
            "add_landmark": 22,
            "correct_position": 16,
            "declare_done": 11,
            "get_observation": 121,
            "look_around": 72,
            "mark_exit": 43,
            "move": 151,
            "send_velocity": 33,
            "set_current_room": 26,
            "turn_to_heading": 159,
            "update_plan": 44,
            "update_room": 27,
        }
        assert report["motion_calls"] == 159 + 151 + 33 == 343

    def test_no_falls_and_the_f05_bump_split(self, report: dict) -> None:
        """F-05: 126 published bumps against 198 motion calls reporting contact."""
        assert report["falls"] == 0
        assert report["counted_bumps"] == 126
        assert report["bumped_motion_calls"] == 198


class TestCorrectionLedger:
    """The 16 rows of the forensics § correction-effect ledger, row for row."""

    def test_ledger_matches_the_published_effects(self, report: dict) -> None:
        rows = {
            (row["trial_id"], row["stage"], row["turn_idx"]): row
            for row in report["corrections"]["ledger"]
        }
        assert len(rows) == 16
        expected = {
            ("gpt56sol_seed101", "find_kitchen", 16): +0.737,
            ("gpt56sol_seed103", "find_kitchen", 11): +0.003,
            ("gpt56sol_seed104", "find_kitchen", 29): +0.339,
            ("gpt56sol_seed104", "find_kitchen", 37): +0.056,
            ("opus5_seed102", "find_kitchen", 25): +0.214,
            ("opus5_seed104", "return_home", 3): +0.137,
            ("opus5_seed104", "return_home", 6): +0.015,
            ("opus5_seed104", "return_home", 9): +0.037,
            ("sonnet5_seed101", "find_kitchen", 21): +1.480,
            ("sonnet5_seed103", "find_kitchen", 12): +0.002,
            ("sonnet5_seed103", "find_kitchen", 30): +0.053,
            ("sonnet5_seed104", "find_kitchen", 15): +0.057,
            ("sonnet5_seed104", "find_kitchen", 20): +1.025,
            ("sonnet5_seed104", "find_kitchen", 21): -1.020,
            ("sonnet5_seed104", "find_kitchen", 29): +0.585,
        }
        for key, effect in expected.items():
            assert rows[key]["effect_m"] == pytest.approx(effect, abs=1e-3), key

    def test_worst_regression_is_sonnet5_seed101_t21(self, report: dict) -> None:
        """F-01's headline: a 0.024 m estimate snapped to 1.504 m off."""
        row = next(
            r
            for r in report["corrections"]["ledger"]
            if r["trial_id"] == "sonnet5_seed101" and r["turn_idx"] == 21
        )
        assert row["place"] == "RedRoom@0"
        assert row["error_before_m"] == pytest.approx(0.024, abs=1e-3)
        assert row["error_after_m"] == pytest.approx(1.504, abs=1e-3)
        # The correction was listed BEFORE the turn's motion call, so the true
        # pose is the previous turn's — using the turn's own end pose would
        # measure it against a position reached after the snap.
        assert row["motion_calls_before"] == 0
        assert row["true_xy_source"] == "prior_turn"

    def test_only_improvement_undid_the_preceding_bad_snap(self, report: dict) -> None:
        ledger = report["corrections"]["ledger"]
        improvements = [
            r for r in ledger if r["accepted"] and r["effect_m"] is not None and r["effect_m"] <= 0
        ]
        assert len(improvements) == 1
        row = improvements[0]
        assert (row["trial_id"], row["turn_idx"]) == ("sonnet5_seed104", 21)
        bad = next(
            r
            for r in ledger
            if r["trial_id"] == "sonnet5_seed104" and r["turn_idx"] == 20
        )
        assert row["error_before_m"] == pytest.approx(bad["error_after_m"], abs=1e-6)

    def test_the_one_rejection_is_f10s_blank_place(self, report: dict) -> None:
        rejected = [r for r in report["corrections"]["ledger"] if not r["accepted"]]
        assert len(rejected) == 1
        row = rejected[0]
        assert (row["trial_id"], row["stage"], row["turn_idx"]) == (
            "gpt56sol_seed103",
            "find_kitchen",
            10,
        )
        assert row["place"] == ""
        assert row["old_xy"] is None and row["new_xy"] is None
        assert row["effect_m"] is None


class TestBatchIntegrity:
    """Provenance facts the parser surfaced, pinned so they cannot drift back."""

    def test_batch_spans_two_freeze_commits(self, report: dict) -> None:
        """Discovered by TR.0, extends F-06 — see PLAN TR.0.

        All 12 trials carry one ``config_hash``, so no frozen file changed and
        the fairness contract holds. But ``config.freeze_commit`` is HEAD at
        launch, and HEAD moved mid-batch, so the provenance field alone cannot
        identify the code that ran.
        """
        integrity = report["integrity"]
        assert sorted(integrity["freeze_commits"]) == [
            "74b46c937ec419c5ece20a86282d321f83ec1166",
            "84af3f8089a84c77dcf7e0ab3121e0be955941a2",
        ]
        assert integrity["manifest"]["freeze_commit_matches"] is False
        by_commit: dict[str, list[str]] = {}
        for trial in report["trials"]:
            by_commit.setdefault(trial["freeze_commit"], []).append(trial["trial_id"])
        assert by_commit["84af3f8089a84c77dcf7e0ab3121e0be955941a2"] == [
            "sonnet5_seed101",
            "sonnet5_seed102",
            "sonnet5_seed103",
            "sonnet5_seed104",
        ]
        assert len(by_commit["74b46c937ec419c5ece20a86282d321f83ec1166"]) == 8

    def test_declare_done_is_the_only_undispatched_call(
        self, documents: list[dict]
    ) -> None:
        """The pairing invariant every correction reconstruction depends on."""
        undispatched = [
            call
            for document in documents
            for call in forensics.iter_tool_calls(document)
            if not call.dispatched
        ]
        assert len(undispatched) == 11
        assert {call.name for call in undispatched} == {"declare_done"}

    def test_motion_calls_join_one_to_one_with_execution_records(
        self, documents: list[dict]
    ) -> None:
        for document in documents:
            motion = list(forensics.iter_motion_calls(document))
            listed = [
                call
                for call in forensics.iter_tool_calls(document)
                if call.is_motion and call.dispatched
            ]
            assert len(motion) == len(listed)
            records = sum(
                len(turn["execution"].get("calls") or []) for turn in document["turns"]
            )
            assert len(motion) == records
            for joined, call in zip(motion, listed):
                assert joined.call.name == call.name == joined.execution["tool"]

    def test_visual_audit_status_resolves_the_sibling_directory(self) -> None:
        from_raw = forensics.visual_audit_status(BATCH_DIR)
        from_audits = forensics.visual_audit_status(AUDIT_DIR)
        assert from_raw["pending"] == from_audits["pending"]
        assert Path(from_raw["audit_dir"]).resolve() == AUDIT_DIR.resolve()

    def test_batch_integrity_without_a_manifest_reports_nothing_checked(
        self, documents: list[dict]
    ) -> None:
        integrity = forensics.batch_integrity(documents, None)
        assert integrity["manifest"] is None
        assert integrity["trials"] == 12


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


class TestMalformedDocuments:
    def test_committed_malformed_fixture_is_rejected_with_a_pointer(self) -> None:
        path = FIXTURES / "trial_malformed_calls.json"
        with pytest.raises(ForensicsError) as excinfo:
            forensics.load_trial(path)
        message = str(excinfo.value)
        # Actionable means: which trial, which JSON path, and both counts.
        assert "fixture_malformed_calls" in message
        assert "turns[1].execution.calls" in message
        assert "1 execution record(s)" in message
        assert "2 dispatched motion tool call(s)" in message

    def test_missing_top_level_key_names_the_key(self) -> None:
        with pytest.raises(ForensicsError, match=r"<root>\.turns: missing"):
            forensics.validate_document({"trial_id": "t", "config": _config()})

    def test_non_object_document_is_rejected(self) -> None:
        with pytest.raises(ForensicsError, match="must be a JSON object"):
            forensics.validate_document([])

    def test_execution_record_out_of_order_is_rejected(self) -> None:
        document = _document(
            [
                _turn(
                    1,
                    ["turn_to_heading", "move"],
                    [_exec("move", (1.0, 0.0)), _exec("turn_to_heading", (1.0, 0.0))],
                )
            ]
        )
        with pytest.raises(ForensicsError, match="execution records are out of order"):
            forensics.validate_document(document, require_final=False)

    def test_unexplained_partial_dispatch_is_rejected(self) -> None:
        turn = _turn(1, ["move", "look_around"], [_exec("move", (1.0, 0.0))])
        turn["model_output"]["dispatched"] = 1
        with pytest.raises(ForensicsError, match="without a 'declare_done' or a falling"):
            forensics.validate_document(_document([turn]), require_final=False)

    def test_historical_fall_explains_trailing_unrun_calls(self) -> None:
        execution = _exec("turn_to_heading", (1.0, 0.0))
        execution["fell"] = True
        turn = _turn(
            1,
            ["set_current_room", "turn_to_heading", "move", "get_observation"],
            [execution],
        )
        turn["model_output"]["dispatched"] = 2
        document = _document([turn])
        forensics.validate_document(document, require_final=False)
        calls = list(forensics.iter_tool_calls(document))
        assert [call.dispatched for call in calls] == [True, True, False, False]

    def test_positional_results_handle_interleaved_motion_rejection(self) -> None:
        turn = _turn(
            1,
            ["move", "turn_to_heading", "set_current_room"],
            [_exec("move", (1.0, 0.0))],
        )
        turn["model_output"]["dispatched"] = 2
        turn["tool_results"] = [
            _tool_result("move", {}),
            _tool_result("turn_to_heading", {"error": "not_executed"}),
            _tool_result("set_current_room", {"ok": True}),
        ]
        document = _document([turn])
        forensics.validate_document(document, require_final=False)
        calls = list(forensics.iter_tool_calls(document))
        assert [call.dispatched for call in calls] == [True, False, True]
        assert [call.call.name for call in forensics.iter_motion_calls(document)] == [
            "move"
        ]

    def test_orphan_correction_record_is_rejected(self) -> None:
        turn = _turn(1, ["look_around"], [])
        turn["memory_snapshot"]["corrections"] = [
            {
                "turn": 1,
                "stage": "find_kitchen",
                "old_xy": [0.5, 0.5],
                "new_xy": [1.0, 1.0],
                "reason": "phantom",
            }
        ]
        with pytest.raises(ForensicsError, match="only 0 dispatched position-correction"):
            forensics.correction_events(_document([turn]))

    def test_load_batch_on_an_empty_directory_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ForensicsError, match="no \\*.json trial documents"):
            forensics.load_batch(tmp_path)


# ---------------------------------------------------------------------------
# Correction ordering inside one turn
# ---------------------------------------------------------------------------


class TestCorrectionOrdering:
    """The reconstruction rule, isolated from the batch.

    The real batch happens to contain only corrections issued *before* any
    motion in their turn, so the after-motion branch has no live coverage. These
    synthetic turns cover both, which is what makes the pinned batch numbers
    trustworthy rather than accidentally right.
    """

    def test_correction_before_motion_uses_the_prior_turn_pose(self) -> None:
        document = _document(
            [
                _turn(1, ["move"], [_exec("move", (1.0, 0.0))], end=(1.0, 0.0)),
                _turn(
                    2,
                    ["correct_position", "move"],
                    [_exec("move", (2.0, 0.0))],
                    end=(2.0, 0.0),
                    corrections=[_correction(2, (1.2, 0.0), (0.0, 0.0))],
                ),
            ]
        )
        (event,) = forensics.correction_events(document)
        assert event.motion_calls_before == 0
        assert event.true_xy_source == "prior_turn"
        assert event.true_xy == (1.0, 0.0)
        (effect,) = forensics.correction_error_effects(document)
        assert effect.error_before_m == pytest.approx(0.2)
        assert effect.error_after_m == pytest.approx(1.0)
        assert effect.effect_m == pytest.approx(0.8)
        assert effect.worsened is True

    def test_correction_after_motion_uses_that_motion_call_pose(self) -> None:
        document = _document(
            [
                _turn(1, ["move"], [_exec("move", (1.0, 0.0))], end=(1.0, 0.0)),
                _turn(
                    2,
                    ["move", "correct_position"],
                    [_exec("move", (2.0, 0.0))],
                    end=(2.0, 0.0),
                    corrections=[_correction(2, (1.2, 0.0), (2.1, 0.0))],
                ),
            ]
        )
        (event,) = forensics.correction_events(document)
        assert event.motion_calls_before == 1
        assert event.true_xy_source == "motion_call[0]"
        assert event.true_xy == (2.0, 0.0)
        (effect,) = forensics.correction_error_effects(document)
        assert effect.error_before_m == pytest.approx(0.8)
        assert effect.error_after_m == pytest.approx(0.1)
        assert effect.effect_m == pytest.approx(-0.7)
        assert effect.improved is True

    def test_two_corrections_around_interleaved_motion(self) -> None:
        """move, correct, move, correct — each correction gets its own instant."""
        document = _document(
            [
                _turn(
                    1,
                    ["move", "correct_position", "move", "correct_position"],
                    [_exec("move", (1.0, 0.0)), _exec("move", (2.0, 0.0))],
                    end=(2.0, 0.0),
                    corrections=[
                        _correction(1, (1.5, 0.0), (1.1, 0.0)),
                        _correction(1, (2.4, 0.0), (2.05, 0.0)),
                    ],
                )
            ]
        )
        first, second = forensics.correction_events(document)
        assert (first.motion_calls_before, first.true_xy) == (1, (1.0, 0.0))
        assert (second.motion_calls_before, second.true_xy) == (2, (2.0, 0.0))
        assert first.true_xy_source == "motion_call[0]"
        assert second.true_xy_source == "motion_call[1]"
        before, after = forensics.correction_error_effects(document)
        assert before.effect_m == pytest.approx(0.1 - 0.5)
        assert after.effect_m == pytest.approx(0.05 - 0.4)

    def test_first_turn_correction_falls_back_to_spawn(self) -> None:
        document = _document(
            [
                _turn(
                    1,
                    ["correct_position"],
                    [],
                    end=None,
                    corrections=[_correction(1, (0.6, 0.5), (2.0, 2.0))],
                )
            ]
        )
        (event,) = forensics.correction_events(document)
        assert event.true_xy_source == "spawn"
        assert event.true_xy == (0.5, 0.5)
        (effect,) = forensics.correction_error_effects(document)
        assert effect.error_before_m == pytest.approx(0.1)
        assert effect.error_after_m == pytest.approx(math.dist((2.0, 2.0), (0.5, 0.5)))

    def test_anchor_correction_is_joined_to_its_memory_record(self) -> None:
        document = _document(
            [
                _turn(
                    1,
                    ["correct_to_anchor"],
                    [],
                    end=None,
                    corrections=[_correction(1, (0.6, 0.5), (0.5, 0.5))],
                )
            ]
        )
        (event,) = forensics.correction_events(document)
        assert event.accepted is True
        assert event.args == {}
        assert event.true_xy == (0.5, 0.5)

    def test_rejected_correction_has_no_effect_but_is_still_listed(self) -> None:
        document = _document([_turn(1, ["correct_position"], [], corrections=[])])
        (event,) = forensics.correction_events(document)
        assert event.accepted is False
        assert event.old_xy is None and event.new_xy is None
        (effect,) = forensics.correction_error_effects(document)
        assert effect.effect_m is None
        assert effect.worsened is False and effect.improved is False

    def test_stage_local_turn_numbers_do_not_cross_stages(self) -> None:
        """A find_kitchen turn 3 and a return_home turn 3 are different turns."""
        document = _document(
            [
                _turn(
                    3,
                    ["correct_position"],
                    [],
                    end=(1.0, 0.0),
                    corrections=[_correction(3, (1.1, 0.0), (1.4, 0.0))],
                ),
                _turn(
                    3,
                    ["correct_position"],
                    [],
                    end=(1.0, 0.0),
                    stage="return_home",
                    corrections=[
                        _correction(3, (1.1, 0.0), (1.4, 0.0)),
                        _correction(3, (1.4, 0.0), (1.0, 0.0), stage="return_home"),
                    ],
                ),
            ]
        )
        first, second = forensics.correction_events(document)
        assert first.stage == "find_kitchen" and first.new_xy == (1.4, 0.0)
        assert second.stage == "return_home" and second.new_xy == (1.0, 0.0)


# ---------------------------------------------------------------------------
# Synthetic document helpers
# ---------------------------------------------------------------------------


def _config() -> dict:
    return {
        "config_hash": "synthetic",
        "freeze_commit": "0" * 40,
        "seed": 101,
        "spawn": {"xy": [0.5, 0.5], "heading_deg": 90.0},
    }


def _document(turns: list[dict]) -> dict:
    return {"trial_id": "synthetic", "config": _config(), "turns": turns}


def _exec(tool: str, xy: tuple[float, float]) -> dict:
    return {
        "tool": tool,
        "policy_seconds_used": 1.0,
        "true_pose": [xy[0], xy[1], 0.0],
        "true_displacement_m": 0.0,
        "distance_moved_m": 0.0,
        "bumped": False,
        "counted_as_bump": False,
        "fell": False,
        "stop_reason": "reached",
    }


def _correction(
    turn: int,
    old_xy: tuple[float, float],
    new_xy: tuple[float, float],
    stage: str = "find_kitchen",
) -> dict:
    return {
        "turn": turn,
        "stage": stage,
        "old_xy": list(old_xy),
        "new_xy": list(new_xy),
        "reason": "synthetic",
    }


def _turn(
    turn_idx: int,
    tool_names: list[str],
    executions: list[dict],
    *,
    end: tuple[float, float] | None = None,
    stage: str = "find_kitchen",
    corrections: list[dict] | None = None,
) -> dict:
    turn: dict = {
        "stage": stage,
        "turn_idx": turn_idx,
        "global_turn_idx": turn_idx,
        "model_output": {
            "tool_calls": [{"name": name, "args": {}} for name in tool_names],
            "dispatched": len(tool_names),
        },
        "execution": {"calls": executions},
        "memory_snapshot": {"corrections": list(corrections or [])},
    }
    if end is not None:
        turn["true_pose"] = {"x": end[0], "y": end[1], "heading_deg": 0.0}
    return turn


def _tool_result(name: str, payload: dict) -> dict:
    return {
        "call_id": f"call_{name}",
        "name": name,
        "json_text": json.dumps(payload),
        "is_error": bool(payload.get("error")),
        "images": [],
    }
