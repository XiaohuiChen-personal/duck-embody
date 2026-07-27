"""Unit tests for ``duck_embody.charts``'s PURE helpers (PLAN T4.4).

Scope, deliberately narrow: path extraction, labels, trial picking, grouping
and the markdown table — the logic a figure's correctness rests on. No
image-diff tests (pixel output is matplotlib-version noise, not logic), and no
matplotlib import: the module contract (AGENTS.md rule 2 — the 0.5 s gate) is
that importing ``duck_embody.charts`` pulls no plotting stack, and the last
test asserts exactly that.

Fixtures: the golden trial (``tests/fixtures/trial_seed101_success.json``, the
writer's own shapes) wherever a real log shape matters; hand-built
:class:`TrialMetrics` only for the picking/grouping/table cases that need
controlled turn counts and NA cells.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from duck_embody import charts, scoring
from duck_embody.agent.memory import STAGE_FIND_KITCHEN, STAGE_RETURN_HOME
from duck_embody.scoring import (
    NA,
    Estimate,
    MapAccuracy,
    QAResult,
    StageMetrics,
    TrialMetrics,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_TRIAL = FIXTURES / "trial_seed101_success.json"


def golden_document() -> dict:
    return json.loads(GOLDEN_TRIAL.read_text(encoding="utf-8"))


def _stage(stage: str, *, turns_used: int, end_reason: str = "fall",
           success: bool = False, drift=NA) -> StageMetrics:
    return StageMetrics(
        stage=stage,
        outcome="fall" if end_reason == "fall" else "failure",
        end_reason=end_reason,
        success=success,
        success_preregistered=success,
        outcome_preregistered="fall" if end_reason == "fall" else "failure",
        d_nearest_counter_face_m=NA,
        d_initial_m=1.0,
        d_final_m=1.0,
        progress=0.0,
        oracle_path_m=1.0,
        true_path_m=1.0,
        spl=0.0,
        time_s=NA,
        turns_used=turns_used,
        drift_m=drift,
        corrections=0,
        correction_magnitudes_m=(),
    )


def _trial(model: str, seed: int, turns_used: int, *,
           end_reason: str = "fall", precision=NA) -> TrialMetrics:
    return TrialMetrics(
        trial_id=f"{charts.short_model_name(model)}_seed{seed}",
        model=model,
        seed=seed,
        stages={
            STAGE_FIND_KITCHEN: _stage(
                STAGE_FIND_KITCHEN, turns_used=turns_used, end_reason=end_reason
            ),
            STAGE_RETURN_HOME: _stage(
                STAGE_RETURN_HOME, turns_used=0, end_reason="not_run"
            ),
        },
        bumps=1,
        falls=1,
        map_accuracy=MapAccuracy(
            claimed=0,
            matched=0,
            true_rooms_visited=1,
            precision=precision,
            recall=0.0,
            matches=(),
            edges_claimed=0,
            edges_correct=0,
            edge_accuracy=NA,
        ),
        qa=QAResult(scores=(0.0,) * 5, answers=("",) * 5),
        visited_rooms=("living_room",),
    )


class TestModelNaming:
    """Figures label models by API id; colours key off the short roster name."""

    def test_short_name_accepts_both_conventions(self):
        assert charts.short_model_name("claude-fable-5") == "fable5"
        assert charts.short_model_name("fable5") == "fable5"
        assert charts.short_model_name("gpt-5.6-sol") == "gpt56sol"

    def test_unknown_model_passes_through(self):
        assert charts.short_model_name("mystery-9") == "mystery-9"
        assert charts.model_display_name("mystery-9") == "mystery-9"

    def test_display_name_is_the_api_id(self):
        assert charts.model_display_name("opus5") == "claude-opus-5"
        assert charts.model_display_name("claude-opus-5") == "claude-opus-5"

    def test_every_roster_model_has_a_distinct_colour(self):
        colours = [charts.model_color(m) for m in charts.MODEL_ORDER]
        assert len(set(colours)) == len(charts.MODEL_ORDER)
        assert charts.model_color("claude-fable-5") == charts.model_color("fable5")

    def test_unknown_model_gets_the_deterministic_fallback(self):
        assert charts.model_color("mystery-9") == charts.model_color("mystery-8")


class TestUnitInterval:
    """0–1 metrics get the fixed honest axis (doc 06 §10.2)."""

    def test_unit_interval_metrics(self):
        for key in (
            "find_kitchen.success_rate",
            "return_home.success_rate",
            "find_kitchen.progress",
            "find_kitchen.spl",
            "qa",
            "map_precision",
            "map_recall",
            "edge_accuracy",
        ):
            assert charts.is_unit_interval(key), key

    def test_unbounded_metrics(self):
        for key in ("bumps", "falls", "find_kitchen.drift_m",
                    "find_kitchen.turns_used", "find_kitchen.time_s"):
            assert not charts.is_unit_interval(key), key

    def test_headline_metrics_exist_in_the_scoring_columns(self):
        """The grid can only draw columns METRIC_COLUMNS actually produces —
        the one-table contract charts.py's docstring inherits."""
        for key, _title in charts.HEADLINE_METRICS:
            assert key in scoring.METRIC_COLUMNS, key


class TestBarAnnotation:
    """doc 06 §6: a missing CI is SAID, an NA cell prints NA, SR prints x/N."""

    def test_mean_with_ci(self):
        est = scoring.estimate([0.6, 0.5, 0.7, 0.5], resamples=200, seed=1)
        text = charts.bar_annotation(est)
        assert text.startswith(f"{est.mean:.2f}")
        assert "no CI" not in text and "n=" not in text

    def test_missing_ci_is_named_with_its_n(self):
        est = scoring.estimate([1.0, NA, NA, NA], resamples=200, seed=1)
        assert est.ci is None
        assert "(no CI, n=1)" in charts.bar_annotation(est)

    def test_partial_n_is_shown(self):
        est = scoring.estimate([0.5, 0.5, 0.5, NA], resamples=200, seed=1)
        assert est.ci is not None
        assert "(n=3/4)" in charts.bar_annotation(est)

    def test_all_na_prints_na(self):
        est = scoring.estimate([NA, NA, NA, NA], resamples=200, seed=1)
        assert charts.bar_annotation(est) == NA

    def test_success_ratio_prints_x_over_n(self):
        zeros = scoring.estimate([0.0] * 4, resamples=200, seed=1)
        assert charts.bar_annotation(zeros, success_ratio=True).startswith("0/4")
        three = scoring.estimate([1.0, 0.0, 1.0, 1.0], resamples=200, seed=1)
        assert charts.bar_annotation(three, success_ratio=True).startswith("3/4")


class TestAxisCeiling:
    def test_all_zero_or_na_gives_unit_ceiling(self):
        zeros = Estimate(values=(0.0, 0.0, 0.0), n_total=3, mean=0.0, ci=(0.0, 0.0))
        na = Estimate(values=(), n_total=4, mean=NA, ci=None)
        assert charts.axis_ceiling([zeros, na]) == 1.0

    def test_ceiling_covers_the_highest_ci(self):
        est = Estimate(values=(2.0, 4.0, 6.0), n_total=3, mean=4.0, ci=(2.0, 6.0))
        assert charts.axis_ceiling([est]) == pytest.approx(6.0 * 1.25)

    def test_ci_free_estimate_uses_its_mean(self):
        est = Estimate(values=(3.0,), n_total=4, mean=3.0, ci=None)
        assert charts.axis_ceiling([est]) == pytest.approx(3.0 * 1.25)


class TestBeliefPath:
    """The dead-reckoned trail is the per-turn obs.position_estimate column
    (loop.memory_snapshot: breadcrumbs are deliberately not logged because
    they duplicate exactly that column)."""

    def test_golden_trial_series(self):
        document = golden_document()
        path = charts.belief_path(document)
        expected = [
            (t["obs"]["position_estimate"]["x"], t["obs"]["position_estimate"]["y"])
            for t in document["turns"]
        ]
        assert path == expected
        assert path[0] == (0.5, 0.5)  # the spawn belief

    def test_position_estimate_end_is_appended_when_logged(self):
        document = golden_document()
        document["turns"][-1][scoring.POSITION_ESTIMATE_END] = {"x": 1.0, "y": 2.0}
        path = charts.belief_path(document)
        assert path[-1] == (1.0, 2.0)
        assert len(path) == len(document["turns"]) + 1

    def test_no_turns_is_an_empty_path(self):
        assert charts.belief_path({"turns": []}) == []

    def test_non_finite_estimate_raises(self):
        document = golden_document()
        document["turns"][0]["obs"]["position_estimate"]["x"] = float("nan")
        with pytest.raises(scoring.ScoringError):
            charts.belief_path(document)


class TestCorrectionSnaps:
    def test_golden_trial_has_the_one_logged_snap(self):
        snaps = charts.correction_snaps(golden_document())
        assert snaps == [((2.24, 1.01), (2.2, 1.05))]

    def test_no_turns_no_snaps(self):
        assert charts.correction_snaps({"turns": []}) == []


class TestClaimMarkers:
    """Claims are drawn at their FIRST logged claim position (a real
    room_evidence point), never at an invented one; evidence-free claims are
    reported for the caption instead."""

    def test_golden_trial_claims_are_drawn_at_first_evidence(self):
        document = golden_document()
        trial = scoring.score_trial(document)
        drawn, undrawn = charts.claim_markers(document, trial.map_accuracy.matches)
        assert undrawn == []
        evidence = scoring.room_evidence(document)
        names = [name for name, _xy, _match in drawn]
        assert names == scoring.claimed_rooms(document)
        for name, xy, matched in drawn:
            assert xy == evidence[name][0]
        matched_map = dict(trial.map_accuracy.matches)
        assert {n: m for n, _xy, m in drawn} == {
            n: matched_map.get(n) for n in names
        }

    def test_claim_without_evidence_is_undrawn(self):
        document = copy.deepcopy(golden_document())
        document["turns"][-1]["memory_snapshot"]["rooms"]["phantom"] = {
            "name": "phantom", "description": "", "landmarks": []
        }
        drawn, undrawn = charts.claim_markers(document, ())
        assert undrawn == ["phantom"]
        assert all(name != "phantom" for name, _xy, _m in drawn)


class TestPickTrajectoryTrials:
    """'Richest' = most stage-1 turns per model, deterministic tie-break; on
    the frozen batch this rule reproduces the three report trials."""

    def test_picks_the_most_turns_per_model(self):
        trials = [
            _trial("claude-fable-5", 101, 2),
            _trial("claude-fable-5", 102, 14),
            _trial("claude-opus-5", 102, 28),
            _trial("claude-opus-5", 103, 16),
            _trial("gpt-5.6-sol", 103, 27),
            _trial("gpt-5.6-sol", 104, 8),
        ]
        assert charts.pick_trajectory_trials(trials) == [
            "fable5_seed102", "opus5_seed102", "gpt56sol_seed103",
        ]

    def test_tie_breaks_on_trial_id(self):
        trials = [
            _trial("claude-fable-5", 104, 7),
            _trial("claude-fable-5", 101, 7),
        ]
        assert charts.pick_trajectory_trials(trials) == ["fable5_seed101"]

    def test_output_follows_model_order_with_unknowns_last(self):
        trials = [
            _trial("mystery-9", 101, 30),
            _trial("gpt-5.6-sol", 101, 3),
            _trial("claude-fable-5", 101, 3),
        ]
        assert charts.pick_trajectory_trials(trials) == [
            "fable5_seed101", "gpt56sol_seed101", "mystery-9_seed101",
        ]


class TestGroupByModel:
    def test_groups_are_api_id_keyed_in_roster_order(self):
        trials = [
            _trial("gpt-5.6-sol", 102, 3),
            _trial("claude-fable-5", 104, 3),
            _trial("claude-fable-5", 101, 3),
            _trial("claude-opus-5", 103, 3),
        ]
        groups = charts.group_by_model(trials)
        assert list(groups) == ["claude-fable-5", "claude-opus-5", "gpt-5.6-sol"]
        assert [t.seed for t in groups["claude-fable-5"]] == [101, 104]

    def test_unknown_models_follow_alphabetically(self):
        groups = charts.group_by_model(
            [_trial("zeta", 101, 1), _trial("alpha", 101, 1),
             _trial("claude-opus-5", 101, 1)]
        )
        assert list(groups) == ["claude-opus-5", "alpha", "zeta"]


class TestPerTrialTable:
    """doc 06 §6: the per-trial table ships with every aggregate, and an NA
    cell prints NA — never a coerced 0."""

    def test_one_row_per_trial_plus_header(self):
        trials = [_trial("claude-fable-5", s, 3) for s in (101, 102)]
        table = charts.per_trial_table(trials)
        lines = table.splitlines()
        assert len(lines) == 2 + len(trials)
        assert lines[0].count("|") == lines[2].count("|")

    def test_na_cells_print_na_not_zero(self):
        table = charts.per_trial_table([_trial("claude-opus-5", 101, 3)])
        row = table.splitlines()[-1]
        cells = [c.strip() for c in row.strip("|").split("|")]
        # precision, edge accuracy and drift were all NA in the fixture.
        assert cells.count(NA) >= 3

    def test_golden_trial_renders_a_success_row(self):
        trial = scoring.score_trial(golden_document())
        table = charts.per_trial_table([trial])
        row = table.splitlines()[-1]
        assert trial.trial_id in row
        assert "claude-fable-5" in row
        assert "| yes |" in row


class TestLazyPlotting:
    """AGENTS.md rule 2: importing charts must not pull the plotting stack."""

    def test_importing_charts_does_not_import_matplotlib(self):
        assert "duck_embody.charts" in sys.modules  # imported at module top
        if "matplotlib" in sys.modules:
            pytest.skip(
                "matplotlib already in sys.modules before this check — "
                "cannot attribute the import"
            )
        assert "matplotlib" not in sys.modules
