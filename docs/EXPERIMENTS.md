# Experiments — the frozen 12-trial batch

One benchmark batch was run: **3 models × 4 seeds (101–104), 12 trials, one
freeze hash, zero selective reruns.** This file records exactly what ran, on
what, and the commands that verify or regenerate every artifact. Caveats and
claim limits: [README § Honest framing & caveats](../README.md#honest-framing--caveats)
and [doc 06 §1](designs/06-benchmark-evaluation.html); metric definitions:
[METRICS.md](METRICS.md).

## Batch identity

| Field | Value | Source |
|---|---|---|
| Config hash | `cf29ec164676a5da2d00fc1b92980db787484d988040c103b28c8525b45124c1` | [`results/freeze.json`](../results/freeze.json) |
| Freeze commit | `13f438d93e505462a60321005eeb84acdda641c4` — the commit that lands `results/freeze.json`; the manifest itself was written from the clean tree at `8eceaf26ec43` (its `freeze_commit` field), 2026-07-27T08:30:41Z | `results/freeze.json`, `git log` |
| Frozen files | 19 (agent loop/tools/memory/prompts, providers, sim wiring, env + scene, task, `configs/benchmark.yaml`, 3 model configs) — SHA-256 each | `results/freeze.json` `files` |
| Matrix | models `fable5, opus5, gpt56sol` × seeds `101, 102, 103, 104` | `results/freeze.json` `matrix` |
| Batch window | first turn 2026-07-27T08:32:58Z → last turn 09:31:34Z (~59 min wall-clock; timestamps read from the trial JSONs — the sim cold start precedes the first turn) | `results/raw/*.json` `turns[].timestamp` |
| Total cost | **$9.63** ($4.40 + $3.51 + $1.72 per model) | [`results/scores.json`](../results/scores.json) `per_model.*.cost_usd.sum` |
| Headline | 0/12 `find_kitchen` (10 falls, 2 `declare_done` outside the radius); `return_home` never ran | [`results/summary_table.md`](../results/summary_table.md) |

Every trial JSON records `config.freeze_commit` and `config.config_hash`; the
runner re-checks the freeze before every launch and there is deliberately no
`--force` flag.

## Environment

| Component | Version / identity |
|---|---|
| Machine | NVIDIA DGX Spark (aarch64), single **GB10** GPU, CUDA 13.0, headless |
| Isaac Sim | **5.1.0-rc.19** (`~/IsaacSim`) |
| Isaac Lab | **2.3.2** (commit `f4aa17f87e2`, `~/IsaacLab`) |
| Interpreter | the kit python only (`~/IsaacLab/isaaclab.sh -p` / `~/IsaacLab/_isaac_sim/python.sh`) — for the batch, the scoring, and the tests (AGENTS.md §4) |
| Locomotion policy | `v4_robust` `model_2999.pt`, vendored in [`policy/`](../policy/) with provenance + checksums ([`policy/README.md`](../policy/README.md)) |
| Robot / parent repo | Open Duck Mini Jetson, pinned commit `34f70fda182120369f954a4b1ccfa1edf58190ea` (asserted at import) |
| Fable 5 | `claude-fable-5`, Anthropic Messages API, thinking always on, explicit prompt caching ([`configs/models/fable5.yaml`](../configs/models/fable5.yaml)) |
| Opus 5 | `claude-opus-5`, Anthropic Messages API, thinking on by default, explicit prompt caching ([`configs/models/opus5.yaml`](../configs/models/opus5.yaml)) |
| GPT 5.6 sol | `gpt-5.6-sol`, OpenAI **Responses API** (`/v1/responses`; chat completions rejects function tools + reasoning for this model), automatic prompt caching at 0.1× ([`configs/models/gpt56sol.yaml`](../configs/models/gpt56sol.yaml)) |

No locked model supports deterministic decoding (Anthropic returns 400 on any
sampling parameter; OpenAI returns 400 on `temperature=0` — probes recorded in
the model configs). **Reproducibility rests on the fixed sim seeds alone**:
re-running the batch reproduces the protocol and the world, not the
transcripts.

## Reproduction commands

All commands run from the repo root with the kit python.

**0. Test gate** (must be green before any batch — AGENTS.md rule 2; 1549 tests):

```bash
bash scripts/run_tests.sh tests/ -q
```

**1. Freeze** (refuses from a dirty tree; writes `results/freeze.json`):

```bash
~/IsaacLab/isaaclab.sh -p -m duck_embody.runner --freeze
```

**2. Verify the freeze against what ran** — `audit_trial.py` FAILs any trial
whose `config.config_hash` differs from `results/freeze.json` (this, over all
12 files, was the batch acceptance check — 12/12 AUDIT PASS):

```bash
for f in results/raw/*.json; do
  ~/IsaacLab/_isaac_sim/python.sh scripts/audit_trial.py "$f"
done
```

**3. Run the batch** (dry-run first; the runner is resumable and skips trials
already complete under the matching hash):

```bash
~/IsaacLab/isaaclab.sh -p -m duck_embody.runner --dry-run   # must list the pending matrix
~/IsaacLab/isaaclab.sh -p -m duck_embody.runner             # ONE persistent sim process
```

**Per-cell semantics, stated honestly:** the runner's only unit is *the whole
remaining matrix*. There is no per-trial selection flag — doc 06 §7 and §3.2
forbid selective reruns (selection bias), so "reproduce one cell" means:
resume a batch in which that cell is not yet complete. A finished cell is never
re-run in place. Per-cell *verification*, by contrast, is always available
offline (steps 2, 4, 5).

**4. Re-score any trial from its frozen JSON** (scorer reads the trial JSON +
the layout ground truth, nothing else):

```bash
~/IsaacLab/isaaclab.sh -p -m duck_embody.scoring results/raw/<trial>.json
```

**5. Rebuild the published tables and figures** (nothing reads scores.json —
figures re-score the raw JSONs):

```bash
~/IsaacLab/_isaac_sim/python.sh scripts/build_scores.py   # -> results/scores.json + summary_table.md
bash scripts/make_figures.sh                              # -> results/figures/*.png
```

**6. The results GIF** (README embed; source video is a frozen artifact):

```bash
~/.local/bin/ffmpeg -i results/videos/gpt56sol_seed103.mp4 \
  -vf "setpts=0.5*PTS,fps=5,scale=480:-1:flags=lanczos,palettegen=max_colors=96:stats_mode=diff" palette.png
~/.local/bin/ffmpeg -i results/videos/gpt56sol_seed103.mp4 -i palette.png \
  -lavfi "setpts=0.5*PTS,fps=5,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=2:diff_mode=rectangle" \
  results/videos/gpt56sol_seed103.gif   # 4.1 MB, 480 px, 2x speed
```

## Per-trial record

Stage-1 end reasons from [`results/summary_table.md`](../results/summary_table.md)
(full metric rows there); every trial links its frozen log and its Rule-11
video. `declared_elsewhere` = the model called `declare_done` outside the
0.35 m goal radius.

| Trial | Model ID | Seed | Stage-1 end | Turns | Cost ($) | Log | Video |
|---|---|---|---|---|---|---|---|
| fable5_seed101 | claude-fable-5 | 101 | fall | 2 | 0.142 | [json](../results/raw/fable5_seed101.json) | [mp4](../results/videos/fable5_seed101.mp4) |
| fable5_seed102 | claude-fable-5 | 102 | fall | 14 | 2.676 | [json](../results/raw/fable5_seed102.json) | [mp4](../results/videos/fable5_seed102.mp4) |
| fable5_seed103 | claude-fable-5 | 103 | fall | 5 | 0.313 | [json](../results/raw/fable5_seed103.json) | [mp4](../results/videos/fable5_seed103.mp4) |
| fable5_seed104 | claude-fable-5 | 104 | declared_elsewhere (1.66 m) | 11 | 1.269 | [json](../results/raw/fable5_seed104.json) | [mp4](../results/videos/fable5_seed104.mp4) |
| opus5_seed101 | claude-opus-5 | 101 | fall | 2 | 0.069 | [json](../results/raw/opus5_seed101.json) | [mp4](../results/videos/opus5_seed101.mp4) |
| opus5_seed102 | claude-opus-5 | 102 | fall | 28 | 1.888 | [json](../results/raw/opus5_seed102.json) | [mp4](../results/videos/opus5_seed102.mp4) |
| opus5_seed103 | claude-opus-5 | 103 | fall | 16 | 1.482 | [json](../results/raw/opus5_seed103.json) | [mp4](../results/videos/opus5_seed103.mp4) |
| opus5_seed104 | claude-opus-5 | 104 | fall | 3 | 0.070 | [json](../results/raw/opus5_seed104.json) | [mp4](../results/videos/opus5_seed104.mp4) |
| gpt56sol_seed101 | gpt-5.6-sol | 101 | fall | 6 | 0.161 | [json](../results/raw/gpt56sol_seed101.json) | [mp4](../results/videos/gpt56sol_seed101.mp4) |
| gpt56sol_seed102 | gpt-5.6-sol | 102 | fall | 11 | 0.397 | [json](../results/raw/gpt56sol_seed102.json) | [mp4](../results/videos/gpt56sol_seed102.mp4) |
| gpt56sol_seed103 | gpt-5.6-sol | 103 | declared_elsewhere (0.83 m) | 27 | 0.897 | [json](../results/raw/gpt56sol_seed103.json) | [mp4](../results/videos/gpt56sol_seed103.mp4) |
| gpt56sol_seed104 | gpt-5.6-sol | 104 | fall | 8 | 0.262 | [json](../results/raw/gpt56sol_seed104.json) | [mp4](../results/videos/gpt56sol_seed104.mp4) |

`return_home` never ran in any trial (its gate is stage-1 success). Fall
diagnostics (`commanded [vx, vy, wz]`, tilt, seconds into the call) are in each
falling trial's log; the batch-level fall decomposition — 5 spin falls at
|wz| = 0.5 exactly, 5 forward-step topples at |wz| 0.02–0.29 — is the
audit-corrected wording recorded in
[`results/audit_notes.md`](../results/audit_notes.md).

## Audits

One trial per model received a frame-by-frame Rule-11 video audit against its
frozen log (fable5_seed102, opus5_seed102, gpt56sol_seed103 — all
**CONSISTENT**), and all 5 figures were independently recomputed from
`results/raw/*.json` (**CONSISTENT**). No metric-vs-video disagreement was
found anywhere, so Rule-11 resolution (video overrides metric) was never
needed. Verbatim audit outputs, including the one batch-headline wording
correction they forced: [`results/audit_notes.md`](../results/audit_notes.md).

## The rerun log — one infra failure, zero silent retries

The complete [`results/rerun_log.md`](../results/rerun_log.md) for the batch is
one line:

> `opus5_seed101` — 2026-07-27T08:51:25Z — infra failure (attempt 1):
> `anthropic.OverloadedError: Error code: 529 … 'overloaded_error'` — evidence
> `results/incomplete/opus5_seed101.20260727-085125.json`

Policy (doc 06 §7): **model failures — falls, cap-outs, wrong declares — are
final results and are never rerun**; the only legitimate rerun is a logged
infrastructure failure. The API returned 529 (provider overloaded) mid-trial;
the partial JSON was moved to `results/incomplete/` (preserved, committed), the
rerun was appended to the log, and the trial re-ran from scratch inside its
`--infra-retries 1` budget, completing normally. Neither of T4.3's restart
branches was ever taken: no code changed mid-batch, frozen or otherwise, and
all 12 trials completed under the one freeze hash.

## Outputs (all frozen; do not edit)

- `results/raw/*.json` — 12 trial logs (turns, tool calls, poses, tokens, QA)
- `results/videos/*.mp4` + `*_filmstrip.png` — Rule-11 evidence per trial
- `results/scores.json`, `results/summary_table.md` — derived scores (rebuildable, step 5)
- `results/figures/*.png` — per-metric bars, turns-survived, 3 trajectory-vs-belief figures, 3 audit filmstrips
- `results/audit_notes.md`, `results/rerun_log.md`, `results/freeze.json`, `results/incomplete/`
