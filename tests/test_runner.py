"""Batch-runner unit tests: the freeze guard and the resume logic (doc 06 §7).

**No kit, no API, no money** — everything here drives the runner's pure
functions against FIXTURE directories, the same split ``tests/test_preflight.py``
uses for rule 1: a guard that only runs on launch night is a guard nobody has
ever seen fire until it matters.

The failures these tests exist to keep impossible, each measured or nearly
measured on this repo:

* **A silently pooled pre-freeze artifact.** ``results/raw/fable5_seed101.json``
  (the T3.5 sanity trial — "NOT benchmark data, never pooled") occupies exactly
  the batch's fable5/seed101 slot with a valid ``final``. Had its stale
  ``config_hash`` happened to match the live tree, a naive resume would have
  skipped it as a finished benchmark trial. The guard must HARD-REFUSE on a
  complete result under a different frozen configuration — never skip it,
  never rerun it (a rerun silently discards a paid result).

* **A smoke-capped run mistaken for a result.** ``run_trial.py --max-turns``
  records ``config.turn_cap_override`` precisely so this cannot happen; the
  resume gate must honour the marker (``gpt56sol_seed101.json`` on disk carries
  ``turn_cap_override: 5`` today).

* **A mid-batch edit to a frozen file that nothing notices.** doc 06 §2
  records the incident: an uncommitted ``memory.py`` edit left
  ``freeze_commit`` AND ``config_hash`` byte-identical across trials running
  different caps. The freeze guard hashes every FROZEN_FILES entry against
  ``results/freeze.json`` and must name exactly the drifted file.

* **The judge sweeping into the matrix.** ``configs/models/*.yaml`` also
  matches ``judge.yaml`` (the out-of-benchmark Sonnet 5 scene judge, doc 04
  §8); the manifest must be the ENUMERATED 15, and a benchmark-shaped foreign
  JSON in results/raw/ must be named at startup, not discovered in T4.4's
  aggregation.
"""

from __future__ import annotations

import html
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from duck_embody.agent.loop import FROZEN_FILES, config_hash
from duck_embody.runner import (
    FREEZE_SCHEMA,
    RERUN_LOG_HEADER,
    STATUS_COMPLETE,
    STATUS_HASH_DRIFT,
    STATUS_INCOMPLETE,
    STATUS_MANIFEST_DRIFT,
    STATUS_PENDING,
    STATUS_SMOKE_CAPPED,
    TrialPlan,
    append_rerun_log,
    batch_manifest_refusals,
    build_batch_manifest,
    build_parser,
    classify_trial,
    cmd_dry_run,
    cmd_freeze,
    cmd_run,
    commit_refusals,
    ensure_rerun_log,
    file_sha256,
    foreign_trials,
    format_dry_run,
    freeze_manifest,
    freeze_refusals,
    freeze_tree_refusals,
    load_matrix,
    load_calibration,
    manifest_sha256,
    midbatch_refusals,
    plan_batch,
    plan_refusals,
    provenance_disposition,
    read_freeze,
    record_resolved_model,
    retire_incomplete,
    run_one_trial,
    trial_matrix,
    verify_asset_checksums,
    write_manifest_once,
    write_freeze,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_06 = REPO_ROOT / "docs" / "designs" / "06-benchmark-evaluation.html"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_root(tmp_path: Path) -> Path:
    """A fixture repo root carrying byte-identical copies of the 15 frozen
    files (so every hash is real) and an empty results tree. No git — the
    commit-side guard is tested separately via :func:`git_freeze`."""
    root = tmp_path / "repo"
    for rel in FROZEN_FILES:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO_ROOT / rel).read_bytes())
    (root / "results" / "raw").mkdir(parents=True)
    return root


def git_freeze(root: Path) -> None:
    """Commit the fixture tree so ``freeze_commit`` returns a clean sha —
    the green path the real batch runs on."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *args], check=True, capture_output=True
    )
    run("init", "-q")
    run("add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "freeze")


def write_trial(
    path: Path,
    *,
    hash_value: str,
    final: bool = True,
    infra: bool = False,
    override: int | None = None,
) -> dict:
    document = {
        "trial_id": path.stem,
        "config": {
            "freeze_commit": "f" * 40,
            "config_hash": hash_value,
            "model": "claude-fable-5",
            "seed": 101,
            "spawn": {"xy": [0.0, 0.0], "heading_deg": 0.0},
        },
        "turns": [],
        "video_path": None,
    }
    if final:
        document["final"] = {"outcome": {"find_kitchen": "success"}}
    if infra:
        document["infra_failure"] = "Traceback (most recent call last):\nRuntimeError: boom"
    if override is not None:
        document["config"]["turn_cap_override"] = override
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return document


def doc_06_manifest_files() -> set[str]:
    """doc 06 §2's manifest list, extracted at test time — same extractor
    contract as ``tests/test_loop.py``: a hand-copied golden can be edited in
    the same commit as the code it polices, and nothing fails."""
    source = DOC_06.read_text(encoding="utf-8")
    start = source.index("THE HASHED MANIFEST")
    bullets = source[source.index("<ul>", start) : source.index("</ul>", start)]
    listed: set[str] = set()
    for item in bullets.split("<li>")[1:]:
        for raw in re.findall(r"<code>([^<]+)</code>", item):
            text = html.unescape(raw).strip()
            if not re.fullmatch(r"[\w./{},\-]+\.(py|yaml)", text):
                continue
            brace = re.search(r"\{([^}]*)\}", text)
            if brace:
                listed.update(
                    text[: brace.start()] + option.strip() + text[brace.end() :]
                    for option in brace.group(1).split(",")
                )
            else:
                listed.add(text)
    assert listed, "doc 06 §2's hashed-manifest list is no longer in the HTML"
    return listed


def run_args(root: Path) -> SimpleNamespace:
    """A cmd_run argument namespace pointed at a fixture root."""
    return SimpleNamespace(
        out_dir=str(root / "results" / "raw"),
        video_dir=str(root / "results" / "videos"),
        checkpoint=None,
        video_every_n=1,
        no_video=False,
        headed=False,
        infra_retries=1,
    )


def make_provenance_fixture(tmp_path: Path, monkeypatch):
    """A complete, tiny pre-Kit runtime for T6 refusal tests."""
    import hashlib

    root = make_root(tmp_path)
    for rel in ("duck_embody/runner.py", "pyproject.toml"):
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / rel).read_bytes())

    asset = root / "assets" / "payload.bin"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"asset-v1")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (root / "assets" / "checksums.txt").write_text(f"{digest}  ./payload.bin\n")

    parent = tmp_path / "parent"
    robot = parent / "mini_bdx/robots/open_duck_mini_v2/usd/open_duck_mini_v2.usd"
    robot.parent.mkdir(parents=True)
    robot.write_bytes(b"robot-usd-v1")
    subprocess.run(["git", "-C", str(parent), "init", "-q", "-b", "v2"], check=True)
    subprocess.run(["git", "-C", str(parent), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(parent), "-c", "user.email=t@t", "-c",
            "user.name=t", "commit", "-qm", "parent",
        ],
        check=True,
    )
    parent_commit = subprocess.run(
        ["git", "-C", str(parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        re.sub(
            r'parent_repo_commit = "[0-9a-f]+"',
            f'parent_repo_commit = "{parent_commit}"',
            pyproject.read_text(),
        )
    )
    monkeypatch.setenv("DUCK_EMBODY_PARENT_REPO", str(parent))

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    checkpoint_sha = file_sha256(checkpoint)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "id": "fixture-calibration",
                "checkpoint_sha256": checkpoint_sha,
                "values": {"k_velocity_realisation": 0.9617},
            }
        )
    )
    git_freeze(root)
    document = build_batch_manifest(
        batch_id="fixture",
        checkpoint=checkpoint,
        calibration_path=calibration,
        argv=["runner.py", "--batch-id", "fixture"],
        root=root,
    )
    assert batch_manifest_refusals(document, root) == []
    return root, parent, checkpoint, calibration, document


class TestImmutableBatchManifest:
    def test_manifest_is_self_hashed_and_write_once(self, tmp_path, monkeypatch):
        root, _, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        assert document["manifest_sha256"] == manifest_sha256(document)
        path = root / "results" / "manifests" / "fixture.json"
        write_manifest_once(path, document)
        original = path.read_bytes()
        with pytest.raises(FileExistsError, match="write-once"):
            write_manifest_once(path, document)
        assert path.read_bytes() == original

    def test_one_byte_checkpoint_mutation_refuses(self, tmp_path, monkeypatch):
        root, _, checkpoint, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
        assert any(
            "checkpoint SHA" in reason
            for reason in batch_manifest_refusals(document, root)
        )

    def test_parent_commit_mismatch_refuses_benchmark_but_warns_smoke(
        self, tmp_path, monkeypatch
    ):
        root, parent, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        (parent / "new.txt").write_text("new parent commit\n")
        subprocess.run(["git", "-C", str(parent), "add", "-A"], check=True)
        subprocess.run(
            [
                "git", "-C", str(parent), "-c", "user.email=t@t", "-c",
                "user.name=t", "commit", "-qm", "advance",
            ],
            check=True,
        )
        reasons = batch_manifest_refusals(document, root)
        assert any("parent commit differs" in reason for reason in reasons)
        hard, warnings = provenance_disposition(
            reasons,
            smoke=False,
            root=root,
            out_dir=root / "results" / "smoke-a",
            video_dir=root / "results" / "smoke-b",
        )
        assert hard and not warnings
        hard, warnings = provenance_disposition(
            reasons,
            smoke=True,
            root=root,
            out_dir=root / "results" / "smoke-a",
            video_dir=root / "results" / "smoke-b",
        )
        assert not hard and warnings == reasons

    def test_smoke_cannot_downgrade_inside_benchmark_dirs(self, tmp_path):
        hard, warnings = provenance_disposition(
            ["parent mismatch"],
            smoke=True,
            root=tmp_path,
            out_dir=tmp_path / "results" / "raw",
            video_dir=tmp_path / "results" / "smoke-video",
        )
        assert hard and not warnings

    def test_asset_mutation_refuses(self, tmp_path, monkeypatch):
        root, _, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        (root / "assets" / "payload.bin").write_bytes(b"mutated")
        assert verify_asset_checksums(root)["ok"] is False
        assert any(
            "asset checksum" in reason
            for reason in batch_manifest_refusals(document, root)
        )

    def test_midbatch_runner_edit_refuses(self, tmp_path, monkeypatch):
        root, _, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        with (root / "duck_embody" / "runner.py").open("ab") as handle:
            handle.write(b"\n# edited mid-batch\n")
        reasons = batch_manifest_refusals(document, root)
        assert any("runner.py" in reason for reason in reasons)

    def test_calibration_must_name_exact_checkpoint_sha(self, tmp_path):
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        calibration = tmp_path / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "id": "wrong",
                    "checkpoint_sha256": "0" * 64,
                    "values": {"k_velocity_realisation": 0.9617},
                }
            )
        )
        with pytest.raises(ValueError, match="not checkpoint"):
            load_calibration(calibration, file_sha256(checkpoint))

    def test_completed_trial_from_another_manifest_refuses(self, tmp_path):
        path = tmp_path / "sonnet5_seed101.json"
        document = write_trial(path, hash_value="a" * 64)
        document["config"]["batch_manifest_sha256"] = "b" * 64
        path.write_text(json.dumps(document))
        status, detail = classify_trial(path, "a" * 64, "c" * 64)
        assert status == STATUS_MANIFEST_DRIFT
        plan = TrialPlan("sonnet5", 101, path.stem, path, status, detail)
        assert any("another batch manifest" in reason for reason in plan_refusals([plan]))

    def test_trial_config_binds_manifest_policy_parent_and_resolved_model(self, tmp_path):
        provenance = {
            "batch_manifest_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "parent_commit": "c" * 40,
            "success_criterion": "criterion-v-test",
            "resolved_model": None,
        }
        class ExplodingSession:
            def reset(self, **kwargs):
                raise RuntimeError("stop after TrialLog construction")

        outcome = run_one_trial(
            ExplodingSession(),
            model_name="sonnet5",
            cfg=SimpleNamespace(model_id="configured-model"),
            provider=None,
            seed=101,
            out_dir=tmp_path,
            video_dir=tmp_path / "videos",
            no_video=True,
            provenance=provenance,
        )
        stored = json.loads(outcome.json_path.read_text())["config"]
        for key, value in provenance.items():
            assert stored[key] == value
        fake_log = SimpleNamespace(
            document={"config": {"resolved_model": None}},
            flush_calls=0,
        )
        fake_log.flush = lambda: setattr(
            fake_log, "flush_calls", fake_log.flush_calls + 1
        )
        record_resolved_model(
            fake_log,
            {"response_metadata": {"resolved_model_id": "provider-resolved-model"}},
        )
        record_resolved_model(
            fake_log,
            {"response_metadata": {"resolved_model_id": "later-model"}},
        )
        assert fake_log.document["config"]["resolved_model"] == "provider-resolved-model"
        assert fake_log.flush_calls == 1

    def test_model_document_outside_matrix_refuses(self, tmp_path, monkeypatch):
        root, _, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        document["models"]["judge"] = dict(next(iter(document["models"].values())))
        document["manifest_sha256"] = manifest_sha256(document)
        assert any(
            "outside the live matrix" in reason
            for reason in batch_manifest_refusals(document, root)
        )

    def test_manifest_contains_no_environment_values(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-value-must-not-appear")
        _, _, _, _, document = make_provenance_fixture(tmp_path, monkeypatch)
        encoded = json.dumps(document)
        assert "ANTHROPIC_API_KEY" in encoded
        assert "secret-value-must-not-appear" not in encoded

    def test_legacy_freeze_schema_remains_separate(self):
        assert FREEZE_SCHEMA == "duck-embody-freeze-v1"


# ===========================================================================
# 1. The freeze manifest (results/freeze.json)
# ===========================================================================


class TestFreezeManifest:
    def test_the_manifest_covers_exactly_doc_06_s2s_enumerated_files(self, tmp_path):
        """PLAN T4.2's own parenthetical was a 6-item SUBSET (and a glob);
        freeze.json generated from anything but FROZEN_FILES would re-open the
        gap doc 06 §2 records — an uncommitted memory.py edit invisible to the
        guard mid-batch."""
        root = make_root(tmp_path)
        manifest = freeze_manifest(root)
        assert set(manifest["files"]) == set(FROZEN_FILES) == doc_06_manifest_files()

    def test_judge_yaml_exists_and_is_deliberately_excluded(self):
        """The exclusion only guards something while the hazard is real: the
        out-of-benchmark judge config is on disk, matches the
        `configs/models/*.yaml` glob PLAN's parenthetical used, and must never
        enter the fairness contract."""
        assert (REPO_ROOT / "configs" / "models" / "judge.yaml").exists()
        assert "configs/models/judge.yaml" not in FROZEN_FILES

    def test_per_file_hash_is_sha256_of_raw_bytes(self, tmp_path):
        import hashlib

        root = make_root(tmp_path)
        manifest = freeze_manifest(root)
        rel = FROZEN_FILES[0]
        expected = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert manifest["files"][rel] == expected

    def test_combined_hash_is_loops_config_hash_never_reimplemented(self, tmp_path):
        """Two hash implementations is how the guard and the trial JSONs start
        disagreeing — freeze.json must carry the exact value TrialLog writes
        into every result."""
        root = make_root(tmp_path)
        assert freeze_manifest(root)["config_hash"] == config_hash(FROZEN_FILES, root)

    def test_a_missing_file_hashes_to_the_named_sentinel(self, tmp_path):
        assert file_sha256(tmp_path / "nope.py") == "<missing>"

    def test_write_freeze_round_trips_at_the_pinned_path(self, tmp_path):
        root = make_root(tmp_path)
        path = write_freeze(root)
        assert path == root / "results" / "freeze.json"
        stored = read_freeze(root)
        assert stored["schema"] == FREEZE_SCHEMA
        assert stored["files"] == freeze_manifest(root)["files"]
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", stored["frozen_at"]
        )

    def test_read_freeze_is_none_when_absent(self, tmp_path):
        assert read_freeze(make_root(tmp_path)) is None

    def test_the_matrix_is_read_from_benchmark_yaml_not_hardcoded(self, tmp_path):
        """benchmark.yaml is inside the hashed contract; a matrix hardcoded in
        the runner could diverge from the one every trial JSON claims."""
        root = make_root(tmp_path)
        config = root / "configs" / "benchmark.yaml"
        text = config.read_text()
        import re as _re
        text = text.replace("seeds: [101, 102, 103, 104]", "seeds: [7, 8]")
        # Regex, not a literal: the models line carries an amendment comment
        # since 2026-07-30 and a literal pinned to one revision no-ops silently
        # (which is exactly how this test broke once).
        text, n = _re.subn(r"^models: \[[^\]]*\][^\n]*", "models: [fable5]", text, count=1, flags=_re.M)
        assert n == 1, "did not find the models line to stub"
        config.write_text(text)
        assert freeze_manifest(root)["matrix"] == {"models": ["fable5"], "seeds": [7, 8]}
        models, seeds = load_matrix(root)
        assert trial_matrix(models, seeds) == [("fable5", 7, "fable5_seed7"), ("fable5", 8, "fable5_seed8")]

    def test_the_real_matrix_is_three_models_by_four_seeds(self):
        models, seeds = load_matrix(REPO_ROOT)
        assert len(models) == 3 and len(seeds) == 4
        assert len(trial_matrix(models, seeds)) == 12

    def test_trial_ids_are_the_resume_key_convention(self):
        """`{model_config}_seed{seed}` — the shipped convention (run_trial.py,
        the files on disk, and now doc 06 §4's example)."""
        models, seeds = load_matrix(REPO_ROOT)
        ids = [tid for _, _, tid in trial_matrix(models, seeds)]
        assert "sonnet5_seed101" in ids and "gpt56sol_seed104" in ids

    def test_the_order_is_model_major(self):
        models, seeds = load_matrix(REPO_ROOT)
        first_block = [m for m, _, _ in trial_matrix(models, seeds)][: len(seeds)]
        assert set(first_block) == {models[0]}


# ===========================================================================
# 2. The freeze guard
# ===========================================================================


class TestFreezeGuard:
    def test_an_untouched_tree_passes(self, tmp_path):
        root = make_root(tmp_path)
        write_freeze(root)
        assert freeze_refusals(read_freeze(root), root) == []

    def test_editing_every_frozen_file_is_refused_by_name(self, tmp_path):
        """doc 06 §7: "a message naming the changed file". A guard that fires
        without naming the file sends the operator diffing 15 files at 2 a.m.;
        one that misses a file re-opens the memory.py incident."""
        root = make_root(tmp_path)
        write_freeze(root)
        manifest = read_freeze(root)
        for rel in FROZEN_FILES:
            path = root / rel
            original = path.read_bytes()
            path.write_bytes(original + b"\n# mid-batch edit\n")
            refusals = freeze_refusals(manifest, root)
            assert refusals, f"editing {rel} did not trip the guard"
            assert any(rel in reason for reason in refusals), rel
            path.write_bytes(original)
        assert freeze_refusals(manifest, root) == []

    def test_a_deleted_frozen_file_is_refused_by_name(self, tmp_path):
        root = make_root(tmp_path)
        write_freeze(root)
        (root / FROZEN_FILES[0]).unlink()
        refusals = freeze_refusals(read_freeze(root), root)
        assert any(FROZEN_FILES[0] in r and "<missing>" in r for r in refusals)

    def test_a_non_frozen_edit_passes(self, tmp_path):
        """The guard must not be trigger-happy either: doc 06 §7 pins re-scoring
        and other non-frozen work as free, and a guard that refused on any edit
        would teach the operator to want the --force that must not exist."""
        root = make_root(tmp_path)
        write_freeze(root)
        (root / "duck_embody" / "unfrozen.py").write_text("x = 1\n")
        assert freeze_refusals(read_freeze(root), root) == []

    def test_manifest_set_drift_refuses_in_both_directions(self, tmp_path):
        root = make_root(tmp_path)
        write_freeze(root)
        manifest = read_freeze(root)
        removed = dict(manifest["files"])
        gone = FROZEN_FILES[2]
        removed.pop(gone)
        assert any(gone in r for r in freeze_refusals({**manifest, "files": removed}, root))
        extra = {**manifest["files"], "duck_embody/extra.py": "0" * 64}
        assert any("extra.py" in r for r in freeze_refusals({**manifest, "files": extra}, root))

    def test_an_unknown_schema_refuses(self, tmp_path):
        root = make_root(tmp_path)
        refusals = freeze_refusals({"schema": "v0-handmade"}, root)
        assert refusals and "schema" in refusals[0]

    def test_a_hand_edited_combined_hash_refuses(self, tmp_path):
        root = make_root(tmp_path)
        write_freeze(root)
        manifest = read_freeze(root)
        manifest["config_hash"] = "0" * 64
        assert any("hand" in r for r in freeze_refusals(manifest, root))

    def test_there_is_deliberately_no_force_flag(self):
        """doc 06 §7 forbids it by name; the parser is where it would sneak in."""
        assert "--force" not in build_parser()._option_string_actions

    def test_commit_refusal_when_no_git_history_exists(self, tmp_path):
        """`loop.freeze_commit` never raises by design — its docstring hands
        the hard refusal to the runner, and this is the runner honouring it."""
        refusals = commit_refusals(make_root(tmp_path))
        assert refusals and "unknown" in refusals[0]

    def test_a_clean_committed_tree_passes_the_commit_guard(self, tmp_path):
        root = make_root(tmp_path)
        git_freeze(root)
        assert commit_refusals(root) == []

    def test_an_uncommitted_frozen_edit_trips_the_commit_guard(self, tmp_path):
        """The doc 06 §2 incident: an uncommitted frozen-file edit mid-batch
        left freeze_commit byte-identical across incomparable trials."""
        root = make_root(tmp_path)
        git_freeze(root)
        (root / "duck_embody" / "agent" / "memory.py").write_bytes(b"# edited\n")
        refusals = commit_refusals(root)
        assert refusals and "UNCOMMITTED" in refusals[0]


# ===========================================================================
# 3. Resume classification (doc 06 §7, §9.1)
# ===========================================================================


class TestResumeClassification:
    LIVE = "a" * 64

    def test_a_missing_json_is_pending(self, tmp_path):
        assert classify_trial(tmp_path / "fable5_seed101.json", self.LIVE)[0] == STATUS_PENDING

    def test_a_complete_matching_result_is_skipped(self, tmp_path):
        path = tmp_path / "fable5_seed101.json"
        write_trial(path, hash_value=self.LIVE)
        assert classify_trial(path, self.LIVE)[0] == STATUS_COMPLETE

    def test_no_final_block_is_incomplete(self, tmp_path):
        path = tmp_path / "fable5_seed101.json"
        write_trial(path, hash_value=self.LIVE, final=False)
        status, detail = classify_trial(path, self.LIVE)
        assert status == STATUS_INCOMPLETE and "final" in detail

    def test_an_infra_failure_with_a_final_is_still_incomplete(self, tmp_path):
        """Both halves of scoring.is_complete: an `infra_failure` that somehow
        acquired a `final` is still not a result."""
        path = tmp_path / "fable5_seed101.json"
        write_trial(path, hash_value=self.LIVE, final=True, infra=True)
        status, detail = classify_trial(path, self.LIVE)
        assert status == STATUS_INCOMPLETE and "infra_failure" in detail

    def test_unparseable_json_is_incomplete_not_a_crash(self, tmp_path):
        path = tmp_path / "fable5_seed101.json"
        path.write_text('{"trial_id": "fable5_seed101", "turns": [')
        assert classify_trial(path, self.LIVE)[0] == STATUS_INCOMPLETE

    def test_a_smoke_capped_run_is_never_skipped_as_done(self, tmp_path):
        """run_trial.py records turn_cap_override so a capped smoke run "can
        never be mistaken for a result" — a resume gate that ignored the marker
        would pool a 5-turn smoke as a benchmark trial (the shape sitting in
        results/raw/gpt56sol_seed101.json today)."""
        path = tmp_path / "gpt56sol_seed101.json"
        write_trial(path, hash_value=self.LIVE, override=5)
        status, detail = classify_trial(path, self.LIVE)
        assert status == STATUS_SMOKE_CAPPED
        assert "turn_cap_override" in detail

    def test_a_crashed_smoke_run_is_smoke_capped_not_incomplete(self, tmp_path):
        """The override marker must win over completeness: a smoke run that
        infra-crashed (`turn_cap_override` present, NO `final`) is still the
        operator's parked file. Classifying it INCOMPLETE would silently
        retire it and spend a full PAID trial in a slot the operator earmarked
        for smoke — the refusal the marker exists to force never fires."""
        path = tmp_path / "gpt56sol_seed101.json"
        write_trial(path, hash_value=self.LIVE, final=False, override=5)
        status, detail = classify_trial(path, self.LIVE)
        assert status == STATUS_SMOKE_CAPPED
        assert "turn_cap_override" in detail

    def test_an_infra_failed_smoke_run_is_smoke_capped_too(self, tmp_path):
        path = tmp_path / "gpt56sol_seed101.json"
        write_trial(path, hash_value=self.LIVE, final=False, infra=True, override=5)
        assert classify_trial(path, self.LIVE)[0] == STATUS_SMOKE_CAPPED

    def test_a_complete_result_under_a_stale_hash_is_drift_not_rerun(self, tmp_path):
        """The T3.5 sanity hazard, measured: fable5_seed101.json holds a valid
        `final` under config_hash bb340a51… while the live tree hashes
        differently. Skipping pools sanity data; rerunning silently discards a
        paid result. The only honest status is a hard refusal."""
        path = tmp_path / "fable5_seed101.json"
        write_trial(path, hash_value="b" * 64)
        status, _ = classify_trial(path, self.LIVE)
        assert status == STATUS_HASH_DRIFT
        assert status != STATUS_INCOMPLETE  # never moved, never rerun

    def test_plan_refusals_name_the_offending_files_and_the_missing_force(self, tmp_path):
        raw = tmp_path / "raw"
        drift = raw / "fable5_seed101.json"
        smoke = raw / "gpt56sol_seed101.json"
        write_trial(drift, hash_value="b" * 64)
        write_trial(smoke, hash_value=self.LIVE, override=5)
        plans = plan_batch(("fable5", "gpt56sol"), (101,), raw, self.LIVE)
        refusals = plan_refusals(plans)
        assert any(str(drift) in r and "--force" in r for r in refusals)
        assert any(str(smoke) in r for r in refusals)

    def test_a_foreign_json_in_the_results_dir_is_named(self, tmp_path):
        """The judge hazard one directory later: a benchmark-shaped
        judge_seed101.json would be folded in by any glob-based aggregator."""
        raw = tmp_path / "raw"
        write_trial(raw / "judge_seed101.json", hash_value=self.LIVE)
        plans = plan_batch(("fable5",), (101,), raw, self.LIVE)
        foreign = foreign_trials(raw, plans)
        assert foreign == [raw / "judge_seed101.json"]
        assert any("judge_seed101" in r for r in plan_refusals(plans, foreign))

    def test_plan_batch_covers_the_whole_matrix_in_order(self, tmp_path):
        models, seeds = load_matrix(REPO_ROOT)
        plans = plan_batch(models, seeds, tmp_path, self.LIVE)
        assert [p.trial_id for p in plans] == [
            f"{m}_seed{s}" for m in models for s in seeds
        ]
        assert all(p.status == STATUS_PENDING for p in plans)


# ===========================================================================
# 4. results/incomplete/ + results/rerun_log.md
# ===========================================================================


class TestRetireIncomplete:
    def make_partial(self, tmp_path: Path, trial_id="fable5_seed101") -> tuple[Path, Path, Path]:
        raw = tmp_path / "results" / "raw"
        path = raw / f"{trial_id}.json"
        write_trial(path, hash_value="a" * 64, final=False)
        return path, tmp_path / "results" / "incomplete", tmp_path / "results" / "rerun_log.md"

    def test_the_move_keeps_the_bytes_and_stamps_the_name(self, tmp_path):
        path, incomplete, log = self.make_partial(tmp_path)
        original = path.read_bytes()
        dest = retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="no `final` block"
        )
        assert not path.exists(), "the slot must be free for the rerun"
        assert re.fullmatch(
            r"fable5_seed101\.\d{8}-\d{6}(-\d+)?\.json", dest.name
        ), dest.name
        assert dest.read_bytes() == original, "preserved, not rewritten"

    def test_every_move_is_logged_and_the_header_is_written_once(self, tmp_path):
        """doc 06 §7: the rerun log ships with the results — reruns are
        visible, not silent. A move without a row IS a silent rerun."""
        path, incomplete, log = self.make_partial(tmp_path)
        dest = retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="sim crash"
        )
        write_trial(path, hash_value="a" * 64, final=False)
        retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="second crash"
        )
        text = log.read_text()
        assert text.count("# Rerun log") == 1
        rows = [line for line in text.splitlines() if line.startswith("| fable5_seed101 ")]
        assert len(rows) == 2
        assert "sim crash" in rows[0] and dest.name in rows[0]

    def test_two_retirements_in_one_second_do_not_overwrite(self, tmp_path, monkeypatch):
        """An instant infra failure being retried can retire the same trial id
        twice inside one clock second; losing the first file loses the only
        evidence of the first failure."""
        import duck_embody.runner as runner_module

        monkeypatch.setattr(runner_module, "_stamp", lambda: "20260727-000000")
        path, incomplete, log = self.make_partial(tmp_path)
        first = retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="a"
        )
        write_trial(path, hash_value="a" * 64, final=False)
        second = retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="b"
        )
        assert first != second and first.exists() and second.exists()

    def test_the_frames_directory_moves_with_the_json(self, tmp_path):
        """The rerun's TrialLog WIPES frames/<trial_id>/ at construction (the
        T3.5 accumulation bug's fix), so frames left behind are destroyed
        seconds after the JSON referencing them was 'preserved'."""
        path, incomplete, log = self.make_partial(tmp_path)
        frames = path.parent / "frames" / "fable5_seed101"
        frames.mkdir(parents=True)
        (frames / "t001_0.jpg").write_bytes(b"jpegbytes")
        dest = retire_incomplete(
            path, incomplete_dir=incomplete, rerun_log_path=log, cause="crash"
        )
        assert not frames.exists()
        moved = incomplete / "frames" / dest.stem / "t001_0.jpg"
        assert moved.read_bytes() == b"jpegbytes"

    def test_a_pipe_in_the_cause_cannot_break_the_table(self, tmp_path):
        log = tmp_path / "rerun_log.md"
        append_rerun_log(log, "fable5_seed101", "a | b\nc", "evidence.json")
        row = log.read_text().splitlines()[-1]
        assert row.count(" | ") == 3, "extra cells would shift the evidence column"


# ===========================================================================
# 5. --dry-run and the run-mode guard
# ===========================================================================


def snapshot(root: Path) -> dict[str, bytes]:
    """Every tracked byte under root — except .git, whose index git itself
    refreshes on a read-only `git status` (the commit guard runs one)."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


class TestDryRunAndRunGuard:
    def seed_results(self, root: Path) -> None:
        # Named from the live matrix, not a hardcoded model: the matrix is an
        # amendable config (fable5 -> sonnet5, 2026-07-30) and a fixture pinned
        # to a retired model would silently stop exercising the skip/rerun path.
        live = config_hash(FROZEN_FILES, root)
        models, seeds = load_matrix(root)
        raw = root / "results" / "raw"
        write_trial(raw / f"{models[0]}_seed{seeds[0]}.json", hash_value=live)          # skip
        write_trial(raw / f"{models[0]}_seed{seeds[1]}.json", hash_value=live, final=False)  # rerun

    def test_dry_run_lists_every_trial_and_touches_nothing(self, tmp_path, capsys):
        root = make_root(tmp_path)
        git_freeze(root)
        write_freeze(root)
        self.seed_results(root)
        before = snapshot(root)
        code = cmd_dry_run(root)
        out = capsys.readouterr().out
        assert code == 0
        models, seeds = load_matrix(root)
        for _, _, trial_id in trial_matrix(models, seeds):
            assert trial_id in out
        assert STATUS_COMPLETE in out and STATUS_INCOMPLETE in out and STATUS_PENDING in out
        assert "would run 11, skip 1" in out
        assert snapshot(root) == before, "--dry-run must not move or write anything"

    def test_dry_run_refuses_on_a_frozen_mutation_naming_the_file(self, tmp_path, capsys):
        """PLAN T4.2's acceptance smoke, as a test: deliberately touch a frozen
        file -> the runner must refuse."""
        root = make_root(tmp_path)
        git_freeze(root)
        write_freeze(root)
        mutated = "duck_embody/agent/prompts.py"
        with (root / mutated).open("ab") as handle:
            handle.write(b"\n# tuned mid-batch\n")
        code = cmd_dry_run(root)
        out = capsys.readouterr().out
        assert code == 2
        assert mutated in out and "--force" in out

    def test_run_refuses_before_kit_without_a_freeze(self, tmp_path, capsys):
        """cmd_run must fail on the guard BEFORE any provider/sim import — a
        refusal after the multi-minute cold start wastes it, and one after a
        retirement has already moved someone's file."""
        root = make_root(tmp_path)
        git_freeze(root)
        code = cmd_run(run_args(root), root)
        out = capsys.readouterr().out
        assert code == 2 and "--freeze" in out

    def test_run_refuses_on_the_measured_t35_sanity_shape(self, tmp_path, capsys):
        """A complete pre-freeze artifact in a matrix slot (stale hash) must
        hard-refuse the batch — the alternative hashes happening to match would
        have silently pooled sanity data as a benchmark result."""
        root = make_root(tmp_path)
        git_freeze(root)
        write_freeze(root)
        write_trial(root / "results" / "raw" / "sonnet5_seed101.json", hash_value="b" * 64)
        code = cmd_run(run_args(root), root)
        out = capsys.readouterr().out
        assert code == 2 and "sonnet5_seed101.json" in out and "pre-freeze" in out

    def test_format_dry_run_flags_the_refusing_statuses_loudly(self, tmp_path):
        plans = [
            TrialPlan("fable5", 101, "fable5_seed101", tmp_path / "x.json",
                      STATUS_HASH_DRIFT, "stored != live"),
            TrialPlan("fable5", 102, "fable5_seed102", tmp_path / "y.json",
                      STATUS_SMOKE_CAPPED, "turn_cap_override=5"),
        ]
        text = format_dry_run(plans)
        assert "HASH-DRIFT" in text and "SMOKE-CAPPED" in text


# ===========================================================================
# 6. Shipped artifacts + the shared per-trial body
# ===========================================================================


class TestShippedArtifacts:
    def test_results_incomplete_exists_with_a_gitkeep(self):
        assert (REPO_ROOT / "results" / "incomplete" / ".gitkeep").exists()

    def test_the_rerun_log_ships_with_the_runner_header(self):
        """The committed log must BEGIN with ensure_rerun_log's header — same
        bytes — or a fresh batch dir and the shipped results tree would carry
        two different 'contracts' for the same table.

        PREFIX, not equality: pre-batch this file was the bare header and
        equality held, but doc 06 §7 says the rerun log SHIPS WITH the results,
        rows included — and the completed batch legitimately appended one (the
        opus5_seed101 attempt-1 Anthropic 529). Asserting emptiness after the
        batch would demand deleting evidence to green the suite. Any rows must
        still look like table rows.

        The row check stops at the first ``## `` heading: scoring.py's own
        header requires post-batch scoring changes to be "logged in
        results/rerun_log.md", and that record (criterion v2, 2026-07-27) is
        prose by nature. The runner only ever APPENDS ROWS to the table at the
        top, which is the region this contract governs.
        """
        shipped = (REPO_ROOT / "results" / "rerun_log.md").read_text(encoding="utf-8")
        assert shipped.startswith(RERUN_LOG_HEADER), (
            "the shipped log does not begin with the runner's header contract"
        )
        for line in shipped[len(RERUN_LOG_HEADER):].splitlines():
            if line.startswith("## "):
                break
            if line.strip():
                assert line.startswith("|") and line.count("|") >= 5, (
                    f"non-table content in the rerun log's runner table: {line!r}"
                )

    def test_run_trial_script_still_wires_after_the_refactor(self):
        """run_trial.py now delegates its per-trial body to
        runner.run_one_trial (one implementation, doc 06 §7); this pins that
        the script still imports and still reads the frozen matrix."""
        spec = importlib.util.spec_from_file_location(
            "run_trial_under_test", REPO_ROOT / "scripts" / "run_trial.py"
        )
        module = importlib.util.module_from_spec(spec)
        saved_argv = sys.argv
        try:
            sys.argv = ["run_trial.py"]
            spec.loader.exec_module(module)
        finally:
            sys.argv = saved_argv
        models, seeds = load_matrix(REPO_ROOT)
        assert module.frozen_matrix() == (models, seeds)
        parser = module.build_parser()
        assert "--force" not in parser._option_string_actions
        assert callable(module.main)

    def test_run_one_trial_and_announce_are_the_shared_surface(self):
        from duck_embody.runner import announce, run_one_trial

        assert callable(run_one_trial) and callable(announce)


# ===========================================================================
# 7. T4.2 second adversarial review (resume/freeze lens) — the six fixes
# ===========================================================================


class TestMidbatchGuard:
    """The freeze guard must re-run BETWEEN trials, not only at startup.

    The measured hazard shape (doc 06 §2's recorded incident + AGENTS.md §5's
    "this tree always carries uncommitted work"): a frozen file edited during
    trial 6 of an hours-long unattended batch. A startup-only guard runs
    trials 6-12 under different frozen bytes and fires only if the runner
    happens to be restarted; a batch that finishes in one go is never
    re-checked.
    """

    def make_frozen_root(self, tmp_path) -> Path:
        root = make_root(tmp_path)
        git_freeze(root)
        write_freeze(root)
        return root

    def test_a_clean_tree_has_no_midbatch_refusals(self, tmp_path):
        assert midbatch_refusals(self.make_frozen_root(tmp_path)) == []

    def test_a_mid_batch_frozen_edit_is_refused_by_name(self, tmp_path):
        root = self.make_frozen_root(tmp_path)
        mutated = "duck_embody/agent/prompts.py"
        with (root / mutated).open("ab") as handle:
            handle.write(b"\n# tuned during trial 6\n")
        refusals = midbatch_refusals(root)
        assert refusals and any(mutated in r for r in refusals)

    def test_a_vanished_freeze_json_refuses(self, tmp_path):
        root = self.make_frozen_root(tmp_path)
        (root / "results" / "freeze.json").unlink()
        refusals = midbatch_refusals(root)
        assert refusals and "freeze.json" in refusals[0]

    def test_a_corrupted_freeze_json_refuses(self, tmp_path):
        root = self.make_frozen_root(tmp_path)
        (root / "results" / "freeze.json").write_text("{ torn mid-write")
        refusals = midbatch_refusals(root)
        assert refusals and "unparseable" in refusals[0]

    def test_cmd_run_rechecks_before_the_retirement_and_before_each_retry(self):
        """Source-level, because the loop body needs kit — but the CALL SITES
        are the contract: the guard must fire (a) before retire_incomplete
        moves anyone's file for the rerun, and (b) inside the retry loop, so
        an infra retry cannot launch under bytes that drifted during the
        failed attempt."""
        source = (REPO_ROOT / "duck_embody" / "runner.py").read_text()
        body = source[source.index("def cmd_run(") : source.index("def main(")]
        code = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        guards = [n for n, text in enumerate(code) if "_abort_on_midbatch_drift(" in text]
        assert len(guards) >= 2, "guard must run before the launch AND before each retry"
        retire = min(n for n, text in enumerate(code) if "retire_incomplete(" in text)
        launch = min(n for n, text in enumerate(code) if "run_one_trial(" in text)
        loop = min(n for n, text in enumerate(code) if text.startswith("while True"))
        assert min(guards) < retire < launch, "a drifted tree must move nobody's file"
        assert any(loop < n < launch for n in guards), "retry attempts must re-check"


class TestAuditFreezeHashCheck:
    """T4.3's acceptance — '12/12 complete under ONE freeze hash' — must be a
    mechanical check, not a by-eye hash grep: audit_trial.py FAILs a
    results/raw/ trial whose config_hash differs from freeze.json."""

    @staticmethod
    def load_audit(tmp_root: Path):
        spec = importlib.util.spec_from_file_location(
            "audit_trial_under_test", REPO_ROOT / "scripts" / "audit_trial.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.REPO_ROOT = tmp_root  # point its results tree at the fixture
        return module

    def fixture(self, tmp_path: Path, *, trial_hash: str, frozen_hash: str, sub: str):
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "freeze.json").write_text(
            json.dumps({"schema": FREEZE_SCHEMA, "config_hash": frozen_hash})
        )
        trial = tmp_path / "results" / sub / "fable5_seed101.json"
        write_trial(trial, hash_value=trial_hash)
        return trial

    def test_a_drifted_hash_in_results_raw_fails_the_audit(self, tmp_path, capsys):
        module = self.load_audit(tmp_path)
        trial = self.fixture(
            tmp_path, trial_hash="b" * 64, frozen_hash="a" * 64, sub="raw"
        )
        code = module.audit(trial)
        out = capsys.readouterr().out
        assert code == 1
        assert "FAIL  config.config_hash matches results/freeze.json" in out

    def test_a_matching_hash_passes_the_freeze_check(self, tmp_path, capsys):
        module = self.load_audit(tmp_path)
        trial = self.fixture(
            tmp_path, trial_hash="a" * 64, frozen_hash="a" * 64, sub="raw"
        )
        module.audit(trial)
        out = capsys.readouterr().out
        assert "PASS  config.config_hash matches results/freeze.json" in out

    def test_pre_freeze_artifacts_outside_raw_are_not_failed_on_hash(self, tmp_path, capsys):
        """Scoped to results/raw/: the T3.5 sanity trials parked in
        results/logs/ legitimately carry a stale hash and must still audit."""
        module = self.load_audit(tmp_path)
        trial = self.fixture(
            tmp_path, trial_hash="b" * 64, frozen_hash="a" * 64, sub="logs"
        )
        module.audit(trial)
        out = capsys.readouterr().out
        assert "config.config_hash matches results/freeze.json" not in out
        assert "freeze-hash check skipped" in out


class TestOccupiedSlotGuard:
    """run_trial.py post-freeze: TrialLog OVERWRITES the JSON and WIPES
    frames/<trial_id>/ at construction, so a typo'd seed (or a deliberate
    'just re-check this one trial' — rule 3's forbidden selective retry)
    would silently destroy a paid COMPLETE result and its rule-11 evidence."""

    @staticmethod
    def load_run_trial():
        spec = importlib.util.spec_from_file_location(
            "run_trial_guard_tests", REPO_ROOT / "scripts" / "run_trial.py"
        )
        module = importlib.util.module_from_spec(spec)
        saved_argv = sys.argv
        try:
            sys.argv = ["run_trial.py"]
            spec.loader.exec_module(module)
        finally:
            sys.argv = saved_argv
        return module

    def test_pre_freeze_smoke_reruns_keep_working(self, tmp_path):
        """No freeze.json yet -> overwrite-your-own-smoke is the T3.5
        workflow and must stay legal."""
        module = self.load_run_trial()
        json_path = tmp_path / "raw" / "fable5_seed101.json"
        write_trial(json_path, hash_value="a" * 64)
        assert module.occupied_slot_refusal(json_path, tmp_path / "freeze.json") is None

    def test_post_freeze_a_complete_result_is_refused(self, tmp_path):
        module = self.load_run_trial()
        json_path = tmp_path / "raw" / "fable5_seed101.json"
        write_trial(json_path, hash_value="a" * 64)
        freeze = tmp_path / "freeze.json"
        freeze.write_text("{}")
        refusal = module.occupied_slot_refusal(json_path, freeze)
        assert refusal is not None
        assert "COMPLETE" in refusal and "runner" in refusal and "--force" in refusal

    def test_post_freeze_a_partial_artifact_is_refused_too(self, tmp_path):
        """Overwriting a post-freeze partial via run_trial.py is an UNLOGGED
        rerun — the batch runner retires it with a rerun_log row instead."""
        module = self.load_run_trial()
        json_path = tmp_path / "raw" / "fable5_seed101.json"
        write_trial(json_path, hash_value="a" * 64, final=False)
        freeze = tmp_path / "freeze.json"
        freeze.write_text("{}")
        assert module.occupied_slot_refusal(json_path, freeze) is not None

    def test_post_freeze_an_unparseable_json_is_still_refused(self, tmp_path):
        module = self.load_run_trial()
        json_path = tmp_path / "raw" / "fable5_seed101.json"
        json_path.parent.mkdir(parents=True)
        json_path.write_text('{"turns": [')
        freeze = tmp_path / "freeze.json"
        freeze.write_text("{}")
        assert module.occupied_slot_refusal(json_path, freeze) is not None

    def test_post_freeze_an_empty_slot_still_runs(self, tmp_path):
        module = self.load_run_trial()
        freeze = tmp_path / "freeze.json"
        freeze.write_text("{}")
        assert module.occupied_slot_refusal(tmp_path / "raw" / "x.json", freeze) is None

    def test_the_guard_fires_before_preflight_and_before_kit(self):
        """A refusal after the cold start wastes minutes; one after
        SimSession.launch would also need the session torn down again."""
        source = (REPO_ROOT / "scripts" / "run_trial.py").read_text()
        body = source[source.index("def main(") :]
        code = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        guard = min(n for n, t in enumerate(code) if "occupied_slot_refusal(" in t)
        preflight = min(n for n, t in enumerate(code) if "preflight_provider(" in t)
        launch = min(n for n, t in enumerate(code) if "SimSession.launch(" in t)
        assert guard < preflight < launch


class TestRerunLogCrashSafety:
    """The rerun log is written across an hours-long unattended batch — it is
    the one artifact whose whole job is surviving a crash."""

    def test_a_torn_append_cannot_fuse_two_rows(self, tmp_path):
        """Measured pre-fix: a power cut mid-append leaves no trailing
        newline, and the next row fused onto the same line — shifting every
        cell a reader (or T4.5's report) parses."""
        log = tmp_path / "rerun_log.md"
        ensure_rerun_log(log)
        with log.open("a", encoding="utf-8") as handle:
            handle.write("| fable5_seed102 | 2026-07-2")  # power cut mid-row
        append_rerun_log(log, "fable5_seed103", "later crash", "x.json")
        lines = log.read_text().splitlines()
        assert lines[-1].startswith("| fable5_seed103 |"), "new row on its own line"
        assert lines[-1].count(" | ") == 3
        assert lines[-2] == "| fable5_seed102 | 2026-07-2", "torn fragment left visible"

    def test_a_normal_append_adds_no_blank_lines(self, tmp_path):
        log = tmp_path / "rerun_log.md"
        append_rerun_log(log, "fable5_seed101", "crash", "x.json")
        append_rerun_log(log, "fable5_seed102", "crash", "y.json")
        lines = log.read_text().splitlines()
        # Consecutive rows on consecutive lines — the torn-tail newline guard
        # must not fire on a healthy file.
        assert lines[-2].startswith("| fable5_seed101 |")
        assert lines[-1].startswith("| fable5_seed102 |")

    def test_the_header_is_written_atomically_with_no_temp_left(self, tmp_path):
        """Header via temp + os.replace: a crash mid-write must not leave a
        half-header that exists() then treats as done forever."""
        log = tmp_path / "rerun_log.md"
        ensure_rerun_log(log)
        assert log.read_text(encoding="utf-8") == RERUN_LOG_HEADER
        assert list(tmp_path.glob("*.tmp")) == []

    def test_retirement_logs_before_it_moves(self, tmp_path, monkeypatch):
        """A crash between log-and-move must leave a LOGGED move that visibly
        failed (source file still in place), never an unlogged retirement —
        the 'silent rerun' doc 06 §7 forbids."""
        import duck_embody.runner as runner_module

        raw = tmp_path / "results" / "raw"
        path = raw / "fable5_seed101.json"
        write_trial(path, hash_value="a" * 64, final=False)
        log = tmp_path / "results" / "rerun_log.md"

        def crash_between(*args, **kwargs):
            raise OSError("power cut between log and move")

        monkeypatch.setattr(runner_module.shutil, "move", crash_between)
        with pytest.raises(OSError):
            retire_incomplete(
                path,
                incomplete_dir=tmp_path / "results" / "incomplete",
                rerun_log_path=log,
                cause="sim crash",
            )
        assert path.exists(), "the move failed — the file must still be in place"
        assert "| fable5_seed101 |" in log.read_text(), "…and the attempt logged"


class TestFreezeNightCleanTree:
    """--freeze refuses on ANY dirty tracked file, not only the frozen 15: a
    dirty runner.py/scoring.py at freeze time stamps freeze.json with a clean
    sha while the batch executors run code no commit contains."""

    def test_cmd_freeze_refuses_on_a_dirty_non_frozen_tracked_file(self, tmp_path, capsys):
        root = make_root(tmp_path)
        executor = root / "duck_embody" / "scoring.py"
        executor.write_text("# the scorer, committed at freeze\n")
        git_freeze(root)
        executor.write_text("# the scorer, edited on freeze night\n")
        code = cmd_freeze(root)
        out = capsys.readouterr().out
        assert code == 2
        assert "duck_embody/scoring.py" in out
        assert not (root / "results" / "freeze.json").exists()

    def test_cmd_freeze_ignores_untracked_scratch_files(self, tmp_path, capsys):
        """-uno by design: this tree carries scratch artifacts (AGENTS.md §5),
        and an untracked file cannot be code a commit claims to contain."""
        root = make_root(tmp_path)
        git_freeze(root)
        (root / "scratch_probe.py").write_text("x = 1\n")
        code = cmd_freeze(root)
        assert code == 0, capsys.readouterr().out
        assert (root / "results" / "freeze.json").exists()

    def test_freeze_tree_refusals_is_empty_without_git(self, tmp_path):
        """The no-git case belongs to commit_refusals (which hard-refuses it
        with the better message); this helper must not double-report."""
        assert freeze_tree_refusals(make_root(tmp_path)) == []

    def test_resume_keeps_the_narrow_frozen_file_scope(self, tmp_path):
        """T4.3 branch (a): a fix touching only NON-frozen code must still
        resume — widening the resume guard would forbid the sanctioned
        canary-fix path."""
        root = make_root(tmp_path)
        executor = root / "duck_embody" / "scoring.py"
        executor.write_text("# committed\n")
        git_freeze(root)
        executor.write_text("# the branch-(a) canary fix, uncommitted\n")
        assert commit_refusals(root) == []
        write_freeze(root)
        assert midbatch_refusals(root) == []


class TestSetupPhaseInfraBoundary:
    """A fault in session.reset / camera warmup / recorder attach must take
    the same retire+log+retry path as a mid-episode infra fault — not abort
    the whole unattended batch with a bare traceback and no rerun-log row."""

    class ExplodingSession:
        def __init__(self, exc: BaseException):
            self._exc = exc

        def reset(self, **kwargs):
            raise self._exc

    @staticmethod
    def cfg():
        return SimpleNamespace(model_id="claude-fable-5")

    def run(self, tmp_path, exc):
        return run_one_trial(
            self.ExplodingSession(exc),
            model_name="fable5",
            cfg=self.cfg(),
            provider=None,
            seed=101,
            out_dir=tmp_path / "raw",
            video_dir=tmp_path / "videos",
            no_video=True,
        )

    def test_a_reset_fault_is_an_infra_failure_with_a_resumable_json(self, tmp_path, capsys):
        outcome = self.run(tmp_path, RuntimeError("render fault during reset"))
        assert outcome.final is None and not outcome.interrupted
        assert "RuntimeError" in (outcome.infra_detail or "")
        document = json.loads(outcome.json_path.read_text(encoding="utf-8"))
        assert "infra_failure" in document and "final" not in document
        # The resume gate must see exactly a rerunnable incomplete trial.
        assert classify_trial(outcome.json_path, "a" * 64)[0] == STATUS_INCOMPLETE
        assert "INFRA FAILURE" in capsys.readouterr().out

    def test_an_interrupt_during_setup_is_an_abort_not_a_retry(self, tmp_path, capsys):
        """Ctrl-C in the setup phase is the operator's abort: cmd_run must see
        interrupted=True and stop the batch instead of 'retrying' it."""
        outcome = self.run(tmp_path, KeyboardInterrupt())
        assert outcome.interrupted and outcome.final is None

    def test_the_json_exists_before_any_sim_setup_runs(self):
        """Source pin: TrialLog is constructed BEFORE the try that wraps
        session.reset, so every setup fault has a JSON to record into and
        cmd_run's retirement path always finds a file to move."""
        source = (REPO_ROOT / "duck_embody" / "runner.py").read_text()
        body = source[source.index("def run_one_trial(") : source.index("def build_parser(")]
        code = [
            line.strip()
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # startswith, not `in` — the docstring PROSE also names session.reset
        # and would otherwise match (comment lines are filtered, prose is not).
        log_ctor = min(n for n, t in enumerate(code) if t.startswith("log = TrialLog("))
        reset = min(n for n, t in enumerate(code) if t.startswith("session.reset("))
        guard = min(n for n, t in enumerate(code) if t.startswith("except BaseException"))
        assert log_ctor < reset < guard


class TestEveryModelConfigIsAccountedFor:
    """No `configs/models/*.yaml` may sit unguarded and undeclared.

    `judge.yaml` already had an explicit exclusion test. `fable5.yaml` acquired
    the same status silently on 2026-07-30 when the matrix swapped and it left
    FROZEN_FILES — its bytes still certify the PUBLISHED v4 batch, so drift
    there would quietly invalidate results that are already in the repo, with
    nothing failing.
    """

    #: Not frozen, deliberately, each with the reason it is exempt.
    RETIRED_OR_EXCLUDED = {
        "judge.yaml": "out-of-benchmark scene judge (doc 04 §8)",
        "fable5.yaml": (
            "retired contestant (matrix swap 2026-07-30); bytes certify the "
            "published v4 batch via results/freeze_v4_baseline.json"
        ),
    }

    def test_each_model_config_is_frozen_or_explicitly_exempt(self):
        frozen = {
            Path(rel).name
            for rel in FROZEN_FILES
            if rel.startswith("configs/models/")
        }
        on_disk = {p.name for p in (REPO_ROOT / "configs" / "models").glob("*.yaml")}
        unaccounted = on_disk - frozen - set(self.RETIRED_OR_EXCLUDED)
        assert not unaccounted, (
            f"model configs neither frozen nor declared exempt: {sorted(unaccounted)} "
            "— add to FROZEN_FILES or to RETIRED_OR_EXCLUDED with a reason"
        )

    def test_every_live_matrix_entry_is_frozen(self):
        models, _ = load_matrix(REPO_ROOT)
        frozen = {
            Path(rel).name
            for rel in FROZEN_FILES
            if rel.startswith("configs/models/")
        }
        for model in models:
            assert f"{model}.yaml" in frozen, (
                f"{model} is in the live matrix but its config is not frozen — "
                "the batch would run against an unhashed contestant config"
            )
