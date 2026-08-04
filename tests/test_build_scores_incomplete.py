"""Escape hatch: score a present subset without touching results/scores.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_allow_incomplete_scores_present_subset_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_scores as report

    src = REPO / "results" / "mini_v5d_r3"
    if not (src / "sonnet5_seed101.json").is_file():
        pytest.skip("mini_v5d_r3 fixtures not present")
    batch = tmp_path / "mini_like"
    batch.mkdir()
    for name in ("sonnet5_seed101.json", "sonnet5_seed102.json"):
        (batch / name).write_text((src / name).read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(report, "RAW", batch)
    monkeypatch.setattr(report, "ALLOW_INCOMPLETE", True)
    monkeypatch.setattr(report, "IS_DESCRIBED_BATCH", False)
    monkeypatch.setattr(report, "IS_V5D_R2_BATCH", False)
    monkeypatch.setattr(report, "MANIFEST_PATH", REPO / "results" / "manifests" / "v5d-r3-final-prod.json")

    frozen = (REPO / "results" / "scores.json").read_bytes()
    scores = report.build()
    table = report.write_table(scores, tmp_path / "summary.md")

    assert scores["matrix_completeness"]["present"] == 2
    assert scores["matrix_completeness"]["expected"] == 12
    assert scores["matrix_completeness"]["status"] == "incomplete"
    assert scores["disposition"]["status"] == "INCOMPLETE"
    assert len(scores["trials"]) == 2
    assert "INCOMPLETE 2/12" in table
    out = batch.parent / f"scores_{batch.name}.json"
    assert out.is_file()
    assert (REPO / "results" / "scores.json").read_bytes() == frozen


def test_incomplete_without_flag_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_scores as report

    src = REPO / "results" / "mini_v5d_r3"
    if not (src / "sonnet5_seed101.json").is_file():
        pytest.skip("mini_v5d_r3 fixtures not present")
    batch = tmp_path / "mini_like"
    batch.mkdir()
    (batch / "sonnet5_seed101.json").write_text(
        (src / "sonnet5_seed101.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(report, "RAW", batch)
    monkeypatch.setattr(report, "ALLOW_INCOMPLETE", False)
    monkeypatch.setattr(report, "IS_DESCRIBED_BATCH", False)
    monkeypatch.setattr(report, "IS_V5D_R2_BATCH", False)
    with pytest.raises(SystemExit, match="incomplete matrix"):
        report.build()
