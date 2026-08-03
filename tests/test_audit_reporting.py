"""TR.8 audit/report gates, including the frozen v5d_r2 regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from duck_embody import audit, forensics

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw_v5d_r2"
FREEZE = REPO / "results" / "freeze.json"
AUDITS = REPO / "results" / "audits_v5d_r2"


def test_legacy_missing_checks_are_incomplete_not_pass() -> None:
    result = audit.audit_trial(
        RAW / "sonnet5_seed101.json",
        batch_dir=RAW,
        manifest_path=FREEZE,
        repo_root=REPO,
    )
    assert result["status"] == audit.INCOMPLETE
    statuses = {row["name"]: row["status"] for row in result["checks"]}
    assert statuses["write-once manifest SHA"] == audit.INCOMPLETE
    assert statuses["trial manifest SHA match"] == audit.INCOMPLETE
    assert statuses["request reconstruction"] == audit.INCOMPLETE
    assert statuses["resolved model consistency"] == audit.INCOMPLETE
    assert statuses["provider usage completeness"] == audit.INCOMPLETE
    assert statuses["scorer replay"] == audit.PASS
    assert statuses["drift via canonical parser"] == audit.PASS


def test_missing_manifest_can_never_pass(tmp_path: Path) -> None:
    result = audit.audit_trial(
        RAW / "sonnet5_seed101.json",
        batch_dir=RAW,
        manifest_path=tmp_path / "missing.json",
        repo_root=REPO,
    )
    statuses = {row["name"]: row["status"] for row in result["checks"]}
    assert statuses["write-once manifest SHA"] == audit.INCOMPLETE
    assert statuses["trial manifest SHA match"] == audit.INCOMPLETE
    assert result["status"] != audit.PASS


def test_pending_visual_verdict_can_never_pass(tmp_path: Path) -> None:
    sheet = tmp_path / "trial.md"
    sheet.write_text(
        "\n".join(f"- {field}: _pending_" for field in audit.VERDICT_FIELDS),
        encoding="utf-8",
    )
    assert audit.visual_verdict_status(sheet)["status"] == audit.INCOMPLETE
    sheet.write_text(
        "\n".join(f"- {field}: reviewed" for field in audit.VERDICT_FIELDS),
        encoding="utf-8",
    )
    assert audit.visual_verdict_status(sheet) == {"status": audit.PASS, "missing": []}


def test_v5d_correction_regressions_use_forensics_parser() -> None:
    documents = {doc["trial_id"]: doc for doc in forensics.load_batch(RAW)}
    sonnet = forensics.correction_summary([documents["sonnet5_seed101"]])
    assert sonnet["calls"] == sonnet["accepted"] == 1
    assert sonnet["rejected"] == 0
    assert sonnet["ledger"][0]["effect_m"] == pytest.approx(1.480, abs=1e-3)
    opus = forensics.correction_summary([documents["opus5_seed104"]])
    assert opus["calls"] == opus["accepted"] == 3
    assert opus["rejected"] == 0


def test_v5d_report_is_provisional_and_paths_are_data_driven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.build_scores as report

    monkeypatch.setattr(report, "RAW", RAW)
    monkeypatch.setattr(report, "IS_DESCRIBED_BATCH", False)
    monkeypatch.setattr(report, "IS_V5D_R2_BATCH", True)
    scores = report.build()
    output = tmp_path / "nested" / "table.md"
    output.parent.mkdir()
    text = report.write_table(scores, output)
    assert "PROVISIONAL" in text
    assert "opus5_seed101" in text
    assert "was not offered `return_home`" in text
    assert "results/raw/" not in text
    assert "](videos/" not in text
    assert "Corr. A/R" in text
    assert "1/0" in text


def test_visual_publication_gate_requires_structured_verdicts(tmp_path: Path) -> None:
    result = audit.audit_batch(
        RAW,
        FREEZE,
        repo_root=REPO,
        audit_dir=AUDITS,
    )
    assert result["publication_gate"]["expected"] == 12
    assert result["publication_gate"]["written"] == 0
    assert len(result["publication_gate"]["pending"]) == 12
    assert result["status"] == audit.INCOMPLETE
