"""Build results/scores.json + results/summary_table.md from the frozen batch.

Run with the kit python (needs duck_embody importable):

    ~/IsaacLab/_isaac_sim/python.sh scripts/build_scores.py [CLI_SCORES.json]

Inputs (all frozen, never modified):
  - results/raw/<model>_seed<seed>.json  (the 12 trial logs)
  - duck_embody/scoring.py               (every metric + the seeded bootstrap)
  - configs/benchmark.yaml scoring:      (bootstrap_resamples / bootstrap_seed)

Set ``DUCK_EMBODY_RAW_DIR`` and ``DUCK_EMBODY_MANIFEST`` together for any
non-default batch. The report records the supplied manifest's actual relative
path and provenance fields; it never guesses a manifest from a directory name.

The optional CLI_SCORES.json argument is the captured stdout of
``python -m duck_embody.scoring results/raw/*.json``; when given, this script
asserts its own ``score_trial`` output is byte-identical to it, so the shipped
scores.json is provably the scoring CLI's numbers.

Timestamps: every timestamp in the output is READ FROM THE TRIAL FILES (the
last turn's ``timestamp``); nothing here calls a clock.
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
import sys
from pathlib import Path

from duck_embody import forensics
from duck_embody.scoring import (
    NA,
    STAGES,
    SUCCESS_CRITERION,
    Estimate,
    estimate,
    historical_openai_cost_lower_bound,
    load_trial,
    score_trial,
    scoring_config,
    summarise,
)

REPO = Path(__file__).resolve().parents[1]
# RAW is overridable so a candidate batch living outside results/raw can be
# scored by THIS scorer rather than a reimplementation. Default unchanged, so
# every existing invocation and the frozen v4 record behave identically.
# (Found 2026-07-29: a v5d batch in results/raw_v5d would otherwise have been
# scored from results/raw, silently publishing v4 numbers as the v5d result.)
RAW = pathlib.Path(os.environ.get("DUCK_EMBODY_RAW_DIR") or (REPO / "results" / "raw"))

#: True only for the batch the hardcoded narrative below actually describes.
#: write_table()'s headline and Notes name specific trials, counts, and outcomes
#: from the 2026-07-27 fable5/opus5/gpt56sol batch. Emitted unconditionally they
#: would attach that story to ANY redirected batch — a report with correct
#: tables and a false narrative, naming a model not even in the file. Numbers
#: are matrix-driven; prose cannot be, so prose is gated.
IS_DESCRIBED_BATCH = RAW.resolve() == (REPO / "results" / "raw").resolve()
IS_V5D_R2_BATCH = RAW.resolve() == (REPO / "results" / "raw_v5d_r2").resolve()
MANIFEST_PATH = (
    Path(os.environ["DUCK_EMBODY_MANIFEST"]).expanduser()
    if os.environ.get("DUCK_EMBODY_MANIFEST")
    else None
)

# The matrix comes from the FROZEN benchmark config, not a hardcoded tuple
# (2026-07-30, when the owner swapped fable5 -> sonnet5): a scorer pinned to
# one matrix would either mis-score or silently skip a batch run under the
# other. Scoring a RAW dir that does not match the current config's matrix
# fails the completeness check below — which is the correct outcome, because
# the frozen artifacts for the OLD matrix (results/scores.json et al.) are
# committed and must not be regenerated under a different config.
import yaml as _yaml

MODELS = tuple(
    _yaml.safe_load((REPO / "configs" / "benchmark.yaml").read_text())["models"]
)
SEEDS = (101, 102, 103, 104)
STAGE1, STAGE2 = STAGES  # find_kitchen, return_home

#: Short config name -> the API model id the trial JSONs carry.
#: Superset across matrix revisions, so historical raw files stay resolvable.
_API_IDS = {
    "fable5": "claude-fable-5",
    "sonnet5": "claude-sonnet-5",
    "opus5": "claude-opus-5",
    "gpt56sol": "gpt-5.6-sol",
}


def cost_record(model: str, tokens: dict) -> dict:
    """Published cost disposition without mutating historical raw JSON."""
    original = float(tokens["cost_usd_estimate"])
    if IS_V5D_R2_BATCH and model == "gpt56sol" and "input_tokens_total" not in tokens:
        pricing = _yaml.safe_load(
            (REPO / "configs" / "models" / "gpt56sol.yaml").read_text()
        )
        lower = historical_openai_cost_lower_bound(
            tokens,
            input_per_mtok=float(pricing["price_in_per_mtok"]),
            cache_read_per_mtok=float(pricing["price_cache_read_per_mtok"]),
            output_per_mtok=float(pricing["price_out_per_mtok"]),
        )
        return {
            "cost_usd": lower,
            "cost_usd_original_reported": original,
            "cost_exact": False,
            "cost_basis": "lower_bound_missing_gpt56_cache_write_tokens",
        }
    return {
        "cost_usd": original,
        "cost_usd_original_reported": original,
        "cost_exact": True,
        "cost_basis": "normalized_usage" if "input_tokens_total" in tokens else "legacy_provider_usage",
    }


def median_or_na(values):
    usable = [v for v in values if not isinstance(v, str)]
    return round(statistics.median(usable), 4) if usable else NA


def manifest_metadata() -> dict:
    """Report provenance from an explicitly selected manifest, never a guess."""
    if MANIFEST_PATH is None:
        return {
            "path": None,
            "schema": None,
            "manifest_sha256": None,
            "checkpoint_sha256": None,
            "parent_commit": None,
            "status": "not_supplied",
        }
    path = MANIFEST_PATH.resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    relative = (
        str(path.relative_to(REPO))
        if path.is_relative_to(REPO)
        else str(path)
    )
    return {
        "path": relative,
        "schema": document.get("schema"),
        "manifest_sha256": document.get("manifest_sha256"),
        "checkpoint_sha256": (document.get("policy") or {}).get(
            "checkpoint_sha256"
        ),
        "parent_commit": (document.get("parent_repo") or {}).get("commit"),
        "status": (
            "complete"
            if document.get("manifest_sha256")
            else "legacy_no_write_once_sha"
        ),
    }


def build(cli_scores_path: Path | None = None):
    documents = {}
    metrics = {}
    for model in MODELS:
        for seed in SEEDS:
            trial_id = f"{model}_seed{seed}"
            document = load_trial(RAW / f"{trial_id}.json")
            documents[trial_id] = document
            metrics[trial_id] = score_trial(document)

    trial_dicts = [metrics[f"{m}_seed{s}"].as_dict() for m in MODELS for s in SEEDS]

    # Cross-check against the scoring CLI's captured stdout, if provided.
    if cli_scores_path is not None:
        cli = json.loads(cli_scores_path.read_text(encoding="utf-8"))
        if cli != trial_dicts:
            raise SystemExit("score_trial output != scoring CLI output — refusing to write")
        print("cross-check: score dicts identical to the scoring CLI output", file=sys.stderr)

    # Per-trial provenance extensions — each field is a verbatim copy of a
    # frozen trial-JSON field (or a length of a frozen array), never computed
    # from a clock or re-derived.
    for entry in trial_dicts:
        document = documents[entry["trial_id"]]
        turns = document["turns"]
        total_turns = len(turns)
        declared = sum(
            int(document["final"]["stages"][stage]["turns_used"]) for stage in STAGES
        )
        if total_turns != declared:
            raise SystemExit(
                f"{entry['trial_id']}: len(turns)={total_turns} != sum(turns_used)={declared}"
            )
        tokens = document["final"]["tokens"]
        entry.update(cost_record(next(k for k, v in _API_IDS.items() if v == entry["model"]), tokens))
        entry["tokens"] = {
            key: value
            for key, value in tokens.items()
            if key != "cost_usd_estimate"
        }
        entry["total_turns"] = total_turns
        entry["last_turn_timestamp"] = turns[-1]["timestamp"]
        entry["video"] = document["video_path"]
        correction = forensics.correction_summary([document])
        entry["correction_calls"] = {
            "calls": correction["calls"],
            "accepted": correction["accepted"],
            "rejected": correction["rejected"],
            "worsened": correction["worsened"],
            "improved": correction["improved"],
        }

    config = scoring_config()

    per_model = {}
    for model in MODELS:
        rows = [metrics[f"{model}_seed{s}"] for s in SEEDS]
        summary = summarise(model, rows)
        # Extras the task's table needs beyond summarise(): medians, cost, turns.
        summary[STAGE1]["progress"]["median"] = median_or_na(
            [t.stages[STAGE1].progress for t in rows]
        )
        summary[STAGE2]["progress"]["median"] = median_or_na(
            [t.stages[STAGE2].progress for t in rows]
        )
        costs = [
            next(
                entry["cost_usd"]
                for entry in trial_dicts
                if entry["trial_id"] == f"{model}_seed{seed}"
            )
            for seed in SEEDS
        ]
        summary["cost_usd"] = {**estimate(costs).as_dict(), "sum": round(sum(costs), 6)}
        totals = [float(len(documents[f"{model}_seed{s}"]["turns"])) for s in SEEDS]
        summary["total_turns"] = {**estimate(totals).as_dict(), "sum": int(sum(totals))}
        outcome_tally: dict[str, int] = {}
        for t in rows:
            reason = t.stages[STAGE1].end_reason
            outcome_tally[reason] = outcome_tally.get(reason, 0) + 1
        summary["find_kitchen_end_reasons"] = outcome_tally
        per_model[model] = summary

    first_doc = documents[f"{MODELS[0]}_seed{SEEDS[0]}"]
    batch_last = max(e["last_turn_timestamp"] for e in trial_dicts)
    scores = {
        "schema": "duck-embody-scores-v2",
        # The published stage-1 success predicate. v2 ("any counter face",
        # 2026-07-27, owner-directed, post-batch) is the union of the
        # pre-registered 0.35 m point disc and "within the same radius of any
        # kitchen counter footprint while inside the kitchen". The change is
        # logged in results/rerun_log.md; every trial also carries its
        # pre-registered verdict (success_preregistered / outcome_preregistered)
        # so the original reading stays reproducible from this same file.
        "scoring_criterion": {
            "name": SUCCESS_CRITERION,
            # v4 completed before the 2026-07-27 adoption. v5d_r2 ran after it;
            # calling that batch's criterion "changed post-batch" was false.
            "changed_post_batch": IS_DESCRIBED_BATCH,
            "rerun_log": "results/rerun_log.md",
            "preregistered_find_kitchen_successes": sum(
                1
                for e in trial_dicts
                if e["stages"][STAGE1]["success_preregistered"]
            ),
            "v2_find_kitchen_successes": sum(
                1 for e in trial_dicts if e["stages"][STAGE1]["success"]
            ),
        },
        "config_hash": first_doc["config"]["config_hash"],
        "freeze_commit": first_doc["config"]["freeze_commit"],
        "source_raw_dir": (
            str(RAW.resolve().relative_to(REPO))
            if RAW.resolve().is_relative_to(REPO)
            else str(RAW.resolve())
        ),
        "model_roster": list(MODELS),
        "batch_manifest": manifest_metadata(),
        # Read from the trial files (max over the 12 last-turn timestamps),
        # deliberately NOT a wall clock (doc 06 reproducibility stance).
        "batch_last_turn_timestamp": batch_last,
        "bootstrap": {
            "resamples": int(config["bootstrap_resamples"]),
            "seed": int(config["bootstrap_seed"]),
            "method": "percentile, mean over N=4 resamples, no CI when n_defined < 3",
        },
        "cost_accounting": {
            "schema": "normalized-provider-usage-v1",
            "historical_v5d_r2_gpt": (
                {
                    "exact": False,
                    "disposition": (
                        "Raw JSON is unchanged. Published GPT costs are corrected "
                        "lower bounds using recoverable cache reads; exact costs are "
                        "unrecoverable because legacy logs omitted cache-write usage."
                    ),
                    "pricing_version": "2026-08-02",
                    "pricing_source": "https://developers.openai.com/api/docs/pricing",
                }
                if IS_V5D_R2_BATCH
                else None
            ),
        },
        "trials": trial_dicts,
        "per_model": per_model,
    }
    default_raw = REPO / "results" / "raw"
    out = (
        REPO / "results" / "scores.json"
        if RAW.resolve() == default_raw.resolve()
        else RAW.parent / f"scores_{RAW.name}.json"
    )
    out.write_text(
        json.dumps(scores, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}", file=sys.stderr)
    return scores


# ---------------------------------------------------------------------------
# summary_table.md
# ---------------------------------------------------------------------------


def _num(value, digits=3):
    return NA if isinstance(value, str) else f"{value:.{digits}f}"


def _mean_ci(block, digits=3):
    """'mean [lo, hi]' with the n_defined/n_total shown when metrics were NA."""
    mean = block["mean"]
    if isinstance(mean, str):
        return NA
    text = f"{mean:.{digits}f}"
    if block.get("ci95"):
        lo, hi = block["ci95"]
        text += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    else:
        text += " (no CI, n<3)"
    if block["n_defined"] < block["n_total"]:
        text += f" (n={block['n_defined']}/{block['n_total']})"
    return text


def _sr(block):
    text = block["printed"]
    if isinstance(text, str) and text == NA:
        return NA
    if block.get("ci95"):
        lo, hi = block["ci95"]
        text += f" [{lo:.2f}, {hi:.2f}]"
    return text


def write_table(scores: dict, out_path: Path | None = None) -> str:
    per_model = scores["per_model"]
    trials = scores["trials"]
    boot = scores["bootstrap"]
    default_raw = REPO / "results" / "raw"
    out = out_path or (
        REPO / "results" / "summary_table.md"
        if RAW.resolve() == default_raw.resolve()
        else RAW.parent / f"summary_table_{RAW.name}.md"
    )

    def artifact_link(value: str) -> str:
        target = Path(value)
        if not target.is_absolute():
            target = REPO / target
        return os.path.relpath(target, out.parent).replace(os.sep, "/")

    def row(label, cell):
        return "| " + label + " | " + " | ".join(cell(per_model[m]) for m in MODELS) + " |"

    lines = [
        "# Duck Embody — 12-trial benchmark results"
        + (" (PROVISIONAL)" if IS_V5D_R2_BATCH else ""),
        "",
        f"Batch: {len(MODELS)} models x {len(SEEDS)} seeds ({SEEDS[0]}-{SEEDS[-1]}), config_hash `{scores['config_hash'][:12]}`, "
        f"freeze commit `{scores['freeze_commit'][:12]}`, last trial turn at "
        f"{scores['batch_last_turn_timestamp']} (read from the trial logs).",
        f"Manifest: `{scores['batch_manifest']['path']}` "
        f"({scores['batch_manifest']['status']}); manifest SHA "
        f"`{scores['batch_manifest']['manifest_sha256'] or 'unavailable'}`, "
        f"checkpoint SHA `{scores['batch_manifest']['checkpoint_sha256'] or 'unavailable'}`, "
        f"parent commit `{scores['batch_manifest']['parent_commit'] or 'unavailable'}`.",
        "",
        # PROSE GATE (2026-07-30). These sentences describe the 2026-07-27
        # batch specifically — trial names, fall counts, the criterion-widening
        # history. Correct there, false anywhere else, and the tables around
        # them are now matrix-driven so nothing else would look wrong.
        *([
        "**Headline: 1/12 find_kitchen successes under criterion v2 (any counter face); "
        "0/12 under the pre-registered point-disc criterion.** 10 trials ended in a fall "
        "(the audit-corrected decomposition, results/audit_notes.md: 5 hull-limit spin "
        "falls at |wz| = 0.5 exactly, 5 forward-step topples at |wz| 0.02-0.29); 2 "
        "ended by `declare_done`: gpt56sol_seed103 five cm from an "
        "east-wall counter face (a v2 success; `declared_elsewhere` as-run) and "
        "fable5_seed104 in the living room 1.40 m from any counter (a failure under both "
        "criteria). The scoring criterion was widened POST-BATCH (2026-07-27, "
        "owner-directed, all 12 trials re-scored together — see results/rerun_log.md): "
        "the objective text \"walk to the counter\" never disambiguates the two counter "
        "runs, so success is now the pre-registered 0.35 m disc UNION within 0.35 m of "
        "any kitchen-counter footprint while inside the kitchen. `return_home` never "
        "ran: the LIVE stage-2 gate used the pre-registered predicate, so the v2 success "
        "was never offered its return leg — its SR is 0/4 with the unrun stage counted "
        "a failure (doc 06 §3.2), and the conditional SR counts only offered legs "
        "(— , k=0). Differentiation between models lives in progress, map "
        "precision/recall, QA, bumps, and drift below.",
        "",
        f"Statistics: mean [95% bootstrap CI], percentile method, {boot['resamples']} "
        f"resamples, seed {boot['seed']} (configs/benchmark.yaml `scoring:`); \"—\" = "
        "undefined, excluded from means, never coerced to 0; no CI when n_defined < 3.",
        "",
        ] if IS_DESCRIBED_BATCH else [
            f"**Headline:** generated from `{RAW.name}` — {len(MODELS)} models "
            f"x {len(SEEDS)} seeds. The interpretive narrative in the default "
            "report describes the 2026-07-27 batch only and is omitted here; "
            "read the tables below plus the per-trial audits.",
            "",
            *(
                [
                    "**Publication status: PROVISIONAL.** This historical batch predates "
                    "write-once batch manifests and request journals, and its visual "
                    "publication gate is not complete. Missing evidence is classified "
                    "`INCOMPLETE`, never PASS.",
                    "",
                    "`opus5_seed101` satisfies the later published v2 counter-face "
                    "criterion but was not offered `return_home`: the live gate used "
                    "the point-disc verdict recorded during the run.",
                    "",
                ]
                if IS_V5D_R2_BATCH
                else []
            ),
        ]),
        "## Per-model aggregate (N=4 trials each)",
        "",
        "| Metric | " + " | ".join(f"{m} ({_API_IDS[m]})" for m in MODELS) + " |",
        "|---|" + "---|" * len(MODELS),
        row("find_kitchen SR (v2: any counter face)", lambda s: _sr(s[STAGE1]["success_rate"])),
        row(
            "find_kitchen SR (pre-registered point disc)",
            lambda s: "{}/{}".format(
                sum(
                    1
                    for e in trials
                    if e["model"] == _API_IDS[s["model"]]
                    and e["stages"][STAGE1]["success_preregistered"]
                ),
                s["n_trials"],
            ),
        ),
        row("return_home SR (unrun = failure)", lambda s: _sr(s[STAGE2]["success_rate"])),
        row(
            "return_home SR given stage-1 success (x/k)",
            lambda s: _sr(s[STAGE2]["success_rate_given_stage1"]),
        ),
        row("find_kitchen progress (mean)", lambda s: _mean_ci(s[STAGE1]["progress"])),
        row(
            "find_kitchen progress (median)",
            lambda s: _num(s[STAGE1]["progress"]["median"]),
        ),
        row("find_kitchen SPL", lambda s: _mean_ci(s[STAGE1]["spl"])),
        row("time-to-kitchen (s)", lambda s: _mean_ci(s[STAGE1]["time_s"], 1)),
        row("find_kitchen turns", lambda s: _mean_ci(s[STAGE1]["turns_used"], 2)),
        row("bumps / trial", lambda s: _mean_ci(s["bumps"], 2)),
        row("falls / trial", lambda s: _mean_ci(s["falls"], 2)),
        row("dead-reckoning drift (m, stage 1)", lambda s: _mean_ci(s[STAGE1]["drift_m"])),
        row("accepted position corrections (stage 1)", lambda s: _mean_ci(s[STAGE1]["corrections"], 2)),
        row("map precision", lambda s: _mean_ci(s["map_precision"])),
        row("map recall", lambda s: _mean_ci(s["map_recall"])),
        row("edge accuracy", lambda s: _mean_ci(s["edge_accuracy"])),
        row("QA score (0-1)", lambda s: _mean_ci(s["qa"])),
        row(
            "cost (USD / trial)"
            + ("; GPT lower bound" if IS_V5D_R2_BATCH else ""),
            lambda s: _mean_ci(s["cost_usd"], 3) + f", sum ${s['cost_usd']['sum']:.2f}",
        ),
        row(
            "total turns / trial",
            lambda s: _mean_ci(s["total_turns"], 2) + f", sum {s['total_turns']['sum']}",
        ),
        row(
            "stage-1 end reasons",
            lambda s: ", ".join(
                f"{k}: {v}" for k, v in sorted(s["find_kitchen_end_reasons"].items())
            ),
        ),
        "",
        *(
            [
                "**v5d_r2 GPT cost correction.** Raw trial JSON is unchanged. "
                "The GPT cost cells above are lower bounds computed from total "
                "input, recoverable cache reads, and output at the 2026-08-02 "
                "GPT-5.6 Sol rates. Legacy logs omitted `cache_write_tokens`, so "
                "the exact charge cannot be recovered; each hidden write would "
                "add the 25% write premium. Original reported → corrected lower "
                "bound: "
                + ", ".join(
                    f"`{entry['trial_id']}` ${entry['cost_usd_original_reported']:.6f}"
                    f" → ≥${entry['cost_usd']:.6f}"
                    for entry in trials
                    if entry["model"] == _API_IDS["gpt56sol"]
                )
                + ".",
                "",
            ]
            if IS_V5D_R2_BATCH
            else []
        ),
        # Batch-specific Notes: same gate as the headline. These name
        # gpt56sol_seed103 and the v4 batch's criterion history.
        *([
        "Notes: time-to-kitchen is defined only on the published (v2) success (doc 06 "
        "§5.4). SPL is 0.0 (not —) on failure by definition (§5.3); its stage-1 oracle "
        "`l` is the shortest path to the v2 SUCCESS REGION (disc ∪ counter band, "
        "ObjectNav convention), so `l` is shorter than the old point oracle for every "
        "spawn. progress / d_initial / d_final keep the pre-registered point reference "
        "for comparability (a success can therefore show progress < 1). return_home "
        "rows beyond SR are omitted: the stage never ran, so progress = 0.0 and "
        "drift = — for all 12 by convention (§3.2). Edge accuracy is — when a trial "
        "claimed no `leads_to:` edge. Of the two `declare_done` trials, "
        "gpt56sol_seed103 is the single v2 success (0.051 m from counter_5's face, in "
        "the kitchen; `declared_elsewhere` under the pre-registered criterion) and "
        "fable5_seed104 is a failure under both criteria — consistent with the videos "
        "(rule 11: video is authoritative; no metric-vs-video disagreement found).",
        "",
        ] if IS_DESCRIBED_BATCH else [
            "Notes: definitions per doc 06 §§5.3-5.4 (SPL is 0.0 on failure; "
            "time-to-kitchen defined only on success). Batch-specific "
            "commentary is omitted for a redirected results dir — see the "
            "per-trial audit files alongside the raw JSONs.",
            "",
        ]),
        "## Per-trial results",
        "",
    ]

    for model in MODELS:
        lines += [
            f"### {model}",
            "",
            "| Trial | Stage-1 outcome (v2) | Progress | SPL | Path (m) | Turns | Bumps | Falls "
            "| Drift (m) | Corr. A/R | Map P | Map R | Edge acc | QA | Cost ($) | Video |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in trials:
            if entry["model"] != _API_IDS[model]:
                continue
            s1 = entry["stages"][STAGE1]
            acc = entry["map_accuracy"]
            video = Path(entry["video"]).name
            video_link = artifact_link(entry["video"])
            correction = entry["correction_calls"]
            lines.append(
                "| {id} | {end} | {prog} | {spl} | {path} | {turns} | {bumps} | {falls} "
                "| {drift} | {corr} | {p} | {r} | {edge} | {qa} | {cost:.3f} | "
                "[{video}]({video_link}) |".format(
                    id=entry["trial_id"],
                    end=s1["outcome"],
                    prog=_num(s1["progress"]),
                    spl=_num(s1["spl"]),
                    path=_num(s1["true_path_m"], 2),
                    turns=s1["turns_used"],
                    bumps=entry["bumps"],
                    falls=entry["falls"],
                    drift=_num(s1["drift_m"]),
                    corr=f"{correction['accepted']}/{correction['rejected']}",
                    p=_num(acc["precision"], 2),
                    r=_num(acc["recall"], 2),
                    edge=_num(acc["edge_accuracy"], 2),
                    qa=_num(entry["qa"]["score"], 2),
                    cost=entry["cost_usd"],
                    video=video,
                    video_link=video_link,
                )
            )
        lines.append("")

    lines += [
        "Per-question QA scores, matched room names, visited rooms, token counts and "
        f"the return_home rows are in `{artifact_link(str((RAW.parent / ('scores_' + RAW.name + '.json')) if RAW.resolve() != default_raw.resolve() else (REPO / 'results' / 'scores.json')))}`; "
        f"raw evidence is under `{artifact_link(str(RAW))}/` and each video link above "
        "is derived from its trial JSON.",
        "",
        f"Generated by `scripts/build_scores.py` from "
        f"`{RAW.relative_to(REPO) if RAW.is_relative_to(REPO) else RAW}/*.json` via "
        "`duck_embody.scoring` (no frozen file touched).",
        "",
    ]

    text = "\n".join(lines)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return text


if __name__ == "__main__":
    write_table(build(Path(sys.argv[1]) if len(sys.argv) > 1 else None))
