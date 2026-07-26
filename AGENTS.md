# AGENTS.md — Rules & Context (single source of truth)

Every AI agent working in this repo (Claude Code, Cursor, or otherwise) reads this
file first. `CLAUDE.md` and `.cursor/rules/agents.mdc` are pointers to this file —
never duplicate rules there. **Update this file whenever a decision changes**; it is
the project's institutional memory and is deliberately context-rich so a fresh agent
can pick up work with no other briefing.

Last updated: 2026-07-26 (design docs drafted + adversarially reviewed; pending owner approval).

---

## 1. What this project is

**Duck Embody**: an LLM drives a 42 cm bipedal robot (Open Duck Mini v2) through an
**unknown multi-room apartment in Isaac Sim** using tool calls — velocity commands to
a pretrained RL locomotion policy — and must **find the kitchen and walk to the
counter** using only vision + a compass. There is **no classical SLAM anywhere**: the
LLM authors and maintains its own map as text (rooms / exits / landmarks), dead-reckons
its position, and corrects drift cognitively. We benchmark several frontier models on
identical tasks and publish results.

Inspired by Anthropic's ["How Claude Performs on Robotics Tasks"](https://www.anthropic.com/research/claude-plays-robotics)
(Frontier Red Team, Jul 9 2026). Their harness (`safety-research/embody`) is
**unreleased as of 2026-07-26** (404, verified); this is a from-scratch build, not a fork.
Positioning: *the embody navigation paradigm + the memory scaffolding their harness
lacked, on a biped instead of a Go2.* Prior-art sweep (2026-07-26, ~40 repos/papers
fetched) found **no existing system** where the LLM is sole holder of both map and
position estimate — closest are MapGPT (text map, but oracle discrete graph),
SG-Nav/SayNav (LLM reads a geometry-built graph), VLMnav (mapless, memoryless).

**Purpose**: portfolio piece for the owner's Anthropic Research Engineer (Agents)
application — the role asks for "agent harnesses (e.g. memory…)" and "quantitative
benchmarks for agentic tasks". Write READMEs/results for that reviewer. Keep the
honest AI-assisted-attribution section prominent. Also feeds Phase 5 (Cosmos VLM) of
the parent robot project.

## 2. Locked design decisions

| Decision | Choice | Why |
|---|---|---|
| Scope | Apartment task only (no arena tier) | Weekend deadline; smoke tests on the empty plane replace the arena |
| Scene | Custom apartment: primitive walls + NVIDIA SimReady/ArchVis furniture, **duck-scale (0.4×)** | No official apartment exists; 0.4× makes rooms readable from a 0.36 m camera |
| Memory | Single **max-scaffold** config (no ablation in MVP): LLM-authored room/exit/landmark graph re-injected every turn, breadcrumbs, dead-reckoned (x,y), `correct_position` loop closure | "Most robust LLM-as-SLAM"; ablation cells are stretch |
| Motion | Closed-loop macros `turn_to_heading` / `move` (+ raw `send_velocity`) | Motor precision is not the capability under test |
| Camera | Head-mounted egocentric, ~512 px, ~90–100° HFOV, frozen for all models | Matches planned hardware (IMX219 CSI in head); fairness requires one config |
| Models | **Fable 5, Opus 5, GPT 5.6 sol** (locked 2026-07-26; add more only as needed later) | Two Anthropic tiers (generational comparison, mirrors the paper) + one cross-lab point; dropping the open-weight VLM removes all local vLLM serving work — providers = Anthropic + OpenAI only |
| Protocol | Paused sim between LLM calls; 1 obs/turn (+ `look_around` panorama); context = first turn + last K + memory block | Paper's protocol; measures capability, not latency |
| Repo | This dedicated public repo; parent robot repo is a read-only pinned dependency | Portfolio readability |
| Trials | N=3–5 per model, fixed seed set, `find_kitchen` + `return_home` continuation | Comparison, not statistical paper |

The concrete implementation plan will live in `docs/PLAN.md` (placeholder until
written). The apartment layout dict (`duck_embody/env/apartment_layout.py`) is
**simultaneously the scene spec and the scoring ground truth** — never let scoring
depend on anything else.

**Low-level design docs** (`docs/designs/`, HTML — start at
[`index.html`](docs/designs/index.html); status: **APPROVED by owner 2026-07-26**).
**Read the relevant doc BEFORE implementing a component**; if implementation must
deviate, update the doc in the same commit:

| Component you're touching | Read |
|---|---|
| Overall architecture, process model, runtime, turn lifecycle | `docs/designs/01-system-architecture.html` |
| Policy loading, command injection, env cfg, motion macros, fall/bump logic | `docs/designs/02-policy-playback.html` |
| Apartment layout, assets, scene builder, scene validation | `docs/designs/03-scene-design.html` |
| Camera mount, capture pipeline, observation payload, look_around | `docs/designs/04-camera-observation.html` |
| Agent loop, tool schema, memory/map, prompts, providers, error policy | `docs/designs/05-agent-harness.html` |
| Trial protocol, log schema, metric formulas, runner, tests, reporting | `docs/designs/06-benchmark-evaluation.html` |

## 3. Hard rules

1. **ONE Isaac Sim / GPU job at a time** on this machine (DGX Spark). Never launch a
   second kit process; the second dies in its init banner. Check `nvidia-smi` before
   launching. Trials run sequentially in ONE persistent sim process (startup is
   minutes — never relaunch per trial).
2. **Scoring is unit-tested before any batch launches** (`tests/test_scoring.py`).
   Non-negotiable; a scoring bug discovered after the batch is a wasted batch.
3. **Evidence discipline** (inherited from the parent project): every number in any
   doc names its source (file, JSON field, command output). Report failures as
   failures; never hand-retry selected trials (selection bias) — caps and failures
   are logged and scored.
4. **No per-model prompt tuning.** One prompt template, one camera config, one tool
   set, frozen before the batch. Changing any of these invalidates the comparison.
5. **The harness stores and formats; the LLM perceives, estimates, and decides.**
   No geometric fact enters memory unless the model asserted it from its own
   observations. Exceptions (declared in docs): dead-reckoning integration of
   commanded velocity, compass heading, and closed-loop motion macros — these mirror
   the real robot's IMU + command interface and are sensor-realistic.
6. **Secrets**: API keys come from `.env` (gitignored; template `.env.example`),
   loaded via `python-dotenv` `load_dotenv()` at provider import — shell
   alternative `set -a && source .env && set +a`. Never write a key into a
   tracked file; never echo or log key values.
7. **Git**: do not commit or push without the owner asking. Fetched asset binaries
   (`assets/*.usd*`) stay out of git; checksums are committed. Results (JSON,
   figures, selected compressed videos) ARE committed — they are the portfolio.
8. **Parent repo is read-only.** Only `duck_embody/env/embody_env_cfg.py` may import
   from it. Record the pinned commit in `pyproject.toml`/README at first import.
9. Keep this file updated. When a decision changes or a gotcha is discovered,
   record it here in the same commit as the change.
10. **Per-task implementation loop** (mandatory for EVERY task in `docs/PLAN.md`,
    in this order — owner-mandated 2026-07-26):
    1. **Adversarial plan review first**: read the task's entry in `docs/PLAN.md`,
       the design doc sections it cites, AND the current repo state (what previous
       tasks actually built — do not trust the plan's assumptions about it). Try to
       find errors, staleness, or contradictions in the task plan. Fix `docs/PLAN.md`
       BEFORE implementing (same commit as the implementation).
    2. **Implement** per the (corrected) plan and the approved design docs.
    3. **Unit tests**: create/extend them wherever the deliverable has testable
       logic (pure functions, data transforms, schemas). Run them.
    4. **Adversarial implementation review**: review the new code with fresh eyes
       (a subagent or a deliberate reviewer pass) hunting for bugs, spec deviations,
       and silent failure modes. Fix what is confirmed.
    5. **Smoke test** per the task's Smoke section (see rule 11 for simulation
       smoke tests) and confirm behavior matches expectations.
    Fix every error encountered at ANY stage before marking the task complete;
    record completion in `docs/PLAN.md` (status + evidence: commands, outputs,
    artifact paths) in the same commit.
11. **Simulation smoke tests are video-verified.** The owner works over SSH from a
    MacBook Air — there is no live viewport, and aggregate numbers alone are NOT
    sufficient (parent-repo lesson: run 12 passed every metric while crawling; the
    video caught it). Any smoke test that runs simulation MUST: (a) capture an mp4
    of the run (robot-tracking or task-relevant camera); (b) extract a filmstrip
    (`ffmpeg -vf fps=1` — denser around critical moments) plus PNG stills; (c)
    analyze it frame by frame against the task's explicit expected-behavior
    checklist (locomotion baseline: trunk upright ~0.17 m, both feet alternate with
    real ground clearance, no drag/glide/crawl, heading straight, no action dither
    — extended per task: camera view correct, collisions bump-not-teleport, rooms
    recognizable); (d) store video + filmstrip under `results/` (or the task's log
    dir) and cite them in the task's completion evidence. **When metrics and video
    disagree, the video wins.** Serve artifacts to the owner via the existing
    tailscale serve when asked.

## 4. Runtime environment (this machine)

- **Interpreters are disjoint** (verified 2026-07-26): the kit python
  (`~/IsaacLab/isaaclab.sh -p`) has isaaclab/rsl_rl/torch-cuda/yaml/dotenv but NOT
  anthropic/openai until PLAN T0.0 installs them; system `python3` has the SDKs but
  NOT isaaclab. **All runtime and all pytest runs use the kit python.** `pxr` (USD)
  is importable from neither by default — use the pinned packman invocation in
  PLAN T0.2.
- DGX Spark, aarch64, single NVIDIA GB10, CUDA 13.0. Headless (no display).
- Isaac Sim **5.1.0-rc.19** at `~/IsaacSim` (build: `~/IsaacSim/_build/linux-aarch64/release`).
- Isaac Lab **2.3.2** at `~/IsaacLab` (commit f4aa17f87e2). Launch pattern:
  `~/IsaacLab/isaaclab.sh -p <script> --headless` (+ `--enable_cameras` for RGB).
  AppLauncher must be constructed BEFORE importing torch/isaaclab.
- Parent robot repo: `~/Projects/Open_Duck_Mini_Jetson` (branch `v2`). Its
  `isaac_lab_env` package registers the duck gym tasks; import it to trigger
  registration. **Pinned commit `34f70fda182120369f954a4b1ccfa1edf58190ea`**
  (recorded 2026-07-26 by T0.1 in `pyproject.toml` `[tool.duck-embody]`;
  `embody_env_cfg.py` asserts it at import and warns on mismatch).
- Policy checkpoint (**vendored into `policy/` by T0.1** — see `policy/README.md`
  for provenance, checksums, training config, and the eval record). Source:
  `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`
  (+ `params/{agent,env}.yaml`, `exported/policy.onnx`). Eval record: 5/5 gait gate,
  0.00% falls over 3,200 push-free episodes (parent repo `docs/jetson-mod/v4_comparison.md`).
- Asset catalog: anonymous public S3, verified reachable 2026-07-26:
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`
  - `…/Isaac/Environments/` — Simple_Room/Office/Hospital/warehouses (NO apartment;
    "Limited Use no-modification" license — do not remix).
  - `Assets/ArchVis/Residential/…` (bucket-root sibling of `Assets/Isaac/`, NOT under the
    5.1 tree; the earlier `…/NVIDIA/Assets/ArchVis/…` path 404s — corrected 2026-07-26,
    see design doc 03 §5/§10) — sofas/beds/fridge/oven/kitchen —
    **visual-only, no colliders** (verified by USD inspection); add bbox collider proxies.
  - `Assets/simready_content/common_assets/props/` — armchair, sofa, tables, chairs,
    desks, cabinets — **colliders + semantic labels included**; no bed/fridge/TV.
  - `…/Isaac/Props/Sektion_Cabinet/` — kitchen cabinet WITH dedicated collision USD.
  - Mirror every fetched asset into `assets/` immediately (`fetch_assets.sh`);
    the batch must never depend on the bucket mid-run.
  - Fallback if furnishing stalls: **MolmoSpaces** (allenai, USD exports of iTHOR/
    ProcTHOR household scenes, targets IsaacSim 5.1.0 + IsaacLab 2.3.1, CC BY 4.0).

## 5. Critical technical gotchas (all verified against local source, 2026-07-26)

Robot / policy (paths in parent repo or `~/IsaacLab`):
- **`heading_command=True` silently hijacks wz**: the inherited command term overwrites
  `vel_command_b[:,2]` from a heading P-controller EVERY step. Must set
  `term.cfg.heading_command = False` (pattern: parent `scripts/evaluate_policies.py:1412`)
  and `rel_standing_envs = 0.0`; pin ranges or the 10 s resampler clobbers written
  commands. #1 source of "the LLM's turn commands do nothing".
- **Auto-reset inside `env.step()`**: `time_out` truncates at `episode_length_s`
  (40 s in play cfgs) and teleports the duck to a RANDOM pose mid-episode. Set
  `terminations.time_out = None` (or raise the length) for agent episodes.
- **Fall detection is trunk-contact-only** (>1 N on `trunk_assembly`); no height/tilt
  term exists. In the apartment: replace with tilt/height-based fall; furniture/wall
  contact becomes a "bump" observation, NOT a termination. The policy is blind
  (59-dim proprioceptive) and cannot get up — a real fall ends the trial.
- **`gait_phase` obs (2 of 59 dims) is a side effect of the ImitationReward REWARD
  term** (module-global registry). Stripping/disabling the reward manager silently
  zeroes it and corrupts the policy input. Keep rewards computing.
- **59-dim actor obs order**: base_ang_vel(3), projected_gravity(3),
  velocity_commands(3), joint_pos(16), joint_vel(16), actions(16), gait_phase(2).
  NO base_lin_vel. Obs normalization is baked into the model — never pre-normalize.
- **Command envelope** (training hull): vx ∈ (−0.148, 0.222) m/s, vy ∈ (±0.111),
  wz ∈ (±0.5) rad/s. Clamp all commands. vy is near-useless for navigation (9 s/m) —
  prefer turn-then-drive. 50 Hz control (sim dt 0.005 × decimation 4);
  `duration_s → N = duration_s × 50` steps.
- **Net displacement was never measured** in the parent repo (velocity errors are
  instantaneous L2). `scripts/smoke_displacement.py` measures real achieved speed
  BEFORE episode caps/scoring are frozen.
- **Head camera frame trap**: `head_assembly`'s local frame is rotated (MJCF quat
  `0.707,0,-0.707,0`) — robot-forward = local **−Z**; an identity mount films the sky.
  Head height ≈ 0.36–0.41 m. The robot USD is instanceable — parenting a camera prim
  may fail (fallbacks: articulation root mount → local de-instanced USD copy →
  viewport render path). `scripts/smoke_camera.py` settles it with a PNG.
- **Exported ONNX is fixed batch [1,59] and parity-unverified** — use `model_2999.pt`
  via RSL-RL in-sim (deterministic inference = mean actions; use `torch.no_grad()`,
  NOT `inference_mode()`, if the env resets afterwards).
- **Pausing is safe**: physics advances ONLY inside `env.step()`. Not stepping = frozen.

Kit process / tooling (verified 2026-07-26 during PLAN T0.0):
- **`SimulationApp.close()` terminates the process.** Statements after it never
  execute — a script that closes the app and *then* prints its summary or writes
  its artifacts loses both, silently, with exit code 0. Write artifacts and print
  verdicts BEFORE closing. (Cost two wasted runs before it was identified.)
- **kit buffers stdout aggressively.** Run every sim script with
  `PYTHONUNBUFFERED=1`, or output produced during the run is discarded at exit.
- **`isaaclab_tasks` and `pxr` import only inside a running kit app.** Plain
  `isaaclab.sh -p -c "import isaaclab_tasks"` fails (`No module named 'pxr'`);
  after `AppLauncher(...)` both import fine (USD `(0, 24, 5)`). Offline USD
  inspection uses the pinned packman invocation in PLAN T0.2 instead.
- **numpy is pinned `<2`** in `pyproject.toml`: the kit python ships 1.26.0 and
  Isaac Sim 5.1.0's compiled extensions are built against the 1.x ABI.
- `pyproject.toml` needs explicit `[tool.setuptools.packages.find]` — flat-layout
  autodiscovery otherwise aborts on `assets/ policy/ results/ configs/`.

Scene / rendering:
- `UsdFileCfg(scale=0.4)` applies scale at spawn BEFORE PhysX parses → static
  colliders cook correctly. Runtime rescaling does NOT propagate. Uniform scale only.
- **Authored `contactOffset`/`restOffset` are absolute meters** and do not shrink
  with 0.4× scale (a 2 cm offset on a 20 cm table = invisible force field). Inspect
  offline with pxr (`scripts/inspect_assets.py`); override via `CollisionPropertiesCfg`.
- `UsdFileCfg(collision_props=…)` only MODIFIES existing colliders — it will not add
  one to a visual-only USD (renders fine, robot walks through). Verify per asset.
- Walls: `CuboidCfg` with `collision_props` creates real static colliders; ≥2–3 cm thick.
- One global `/World/Apartment`, `collision_group=-1`, `num_envs=1`.
- **First headless frames can be gray** (MDL compile/texture streaming) — warm up
  several render steps before capturing VLM frames.
- Adding an RTX camera makes `env.step()` render every `render_interval` (50 Hz) —
  raise `sim.render_interval` and render on demand (1 frame per LLM turn).
- Command-arrow debug markers (`debug_vis=True`) float above the robot and would
  leak the commanded velocity into frames — set `debug_vis=False`.

## 6. Prompt patterns to use (from verified prior art)

- **MapGPT map grammar** (proven zero-shot with GPT-4V): `Place N: <desc>` nodes,
  explicit adjacency lines, one-line trajectory history, separate
  seen-but-unexplored list. Render the LLM's graph this way every turn.
- **Plan carry-forward**: re-inject the model's previous multi-step plan each turn;
  ask update-or-keep. Fixes local-wandering loops.
- **CogNav-style state machine** in the prompt: broad search → contextual search →
  verify target, transitions decided by the model.
- **Text frontier scoring**: each turn, model rates unexplored exits for "likelihood
  of leading to the kitchen".
- Post-episode **layout QA** (embody's explore_report pattern): quiz the model about
  its own map; score against ground truth — map quality independent of task success.

## 7. Repo map

```
docs/            PLAN.md (placeholder) · EXPERIMENTS.md · METRICS.md
policy/          vendored v4_robust artifacts (checkpoint, yamls, ONNX) + provenance
assets/          fetch_assets.sh + checksums (USD binaries gitignored)
duck_embody/
  env/           embody_env_cfg (imports parent repo) · apartment_layout (ground truth)
                 · scene_builder · camera
  sim/           policy_wrapper (RSL-RL adapter, cmd injection) · session (persistent sim)
  agent/         loop · tools · memory · prompts · providers/{anthropic,openai}
  tasks/         find_kitchen (+ return_home continuation)
  runner.py      resumable sequential batch · scoring.py · charts.py
scripts/         inspect_assets → smoke_camera → smoke_displacement →
                 render_scene_survey → run_trial   (build order)
configs/         models/*.yaml · benchmark.yaml (frozen before batch)
tests/           test_scoring (gate) · test_memory · test_layout
results/         raw JSON · figures · videos (committed)
```

## 8. Current status

- [x] Design + feasibility research complete (2026-07-26; see §2 decisions, §5 gotchas)
- [x] Repo skeleton, rules, README draft
- [x] Low-level design docs (`docs/designs/01–06` + index) — adversarially reviewed,
      **APPROVED by owner 2026-07-26**
- [x] docs/PLAN.md — task-level implementation plan (every task follows hard
      rules 10–11)
- [ ] Vendor policy artifacts into `policy/`
- [ ] Smoke tests (camera PNG, net displacement, asset inspection)
- [ ] Apartment scene + survey renders
- [ ] Agent loop + tools + memory + providers
- [ ] Sanity LLM episode → freeze configs → batch (Fable 5, Opus 5, GPT 5.6 sol × N=3–5)
- [ ] Scoring, figures, README results

Owner deadline: Sunday night 2026-07-26. Cut order if behind: return_home stage →
GPT 5.6 sol (preserve the two-Claude comparison) → N→3 → panorama tool.
