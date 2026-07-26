# AGENTS.md — Rules & Context (single source of truth)

Every AI agent working in this repo (Claude Code, Cursor, or otherwise) reads this
file first. `CLAUDE.md` and `.cursor/rules/agents.mdc` are pointers to this file —
never duplicate rules there. **Update this file whenever a decision changes**; it is
the project's institutional memory and is deliberately context-rich so a fresh agent
can pick up work with no other briefing.

Last updated: 2026-07-26 (project start; skeleton commit).

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
6. **Secrets**: API keys come from environment variables / `.env` (gitignored).
   Never write a key into a tracked file.
7. **Git**: do not commit or push without the owner asking. Fetched asset binaries
   (`assets/*.usd*`) stay out of git; checksums are committed. Results (JSON,
   figures, selected compressed videos) ARE committed — they are the portfolio.
8. **Parent repo is read-only.** Only `duck_embody/env/embody_env_cfg.py` may import
   from it. Record the pinned commit in `pyproject.toml`/README at first import.
9. Keep this file updated. When a decision changes or a gotcha is discovered,
   record it here in the same commit as the change.

## 4. Runtime environment (this machine)

- DGX Spark, aarch64, single NVIDIA GB10, CUDA 13.0. Headless (no display).
- Isaac Sim **5.1.0-rc.19** at `~/IsaacSim` (build: `~/IsaacSim/_build/linux-aarch64/release`).
- Isaac Lab **2.3.2** at `~/IsaacLab` (commit f4aa17f87e2). Launch pattern:
  `~/IsaacLab/isaaclab.sh -p <script> --headless` (+ `--enable_cameras` for RGB).
  AppLauncher must be constructed BEFORE importing torch/isaaclab.
- Parent robot repo: `~/Projects/Open_Duck_Mini_Jetson` (branch `v2`). Its
  `isaac_lab_env` package registers the duck gym tasks; import it to trigger
  registration.
- Policy checkpoint (to be vendored into `policy/`):
  `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/model_2999.pt`
  (+ `params/{agent,env}.yaml`, `exported/policy.onnx`). Eval record: 5/5 gait gate,
  0.00% falls over 3,200 push-free episodes (parent repo `docs/jetson-mod/v4_comparison.md`).
- Asset catalog: anonymous public S3, verified reachable 2026-07-26:
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`
  - `…/Isaac/Environments/` — Simple_Room/Office/Hospital/warehouses (NO apartment;
    "Limited Use no-modification" license — do not remix).
  - `…/NVIDIA/Assets/ArchVis/Residential/…` — sofas/beds/fridge/oven/kitchen —
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
  agent/         loop · tools · memory · prompts · providers/{anthropic,openai,vllm}
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
- [ ] docs/PLAN.md (consolidated plan) — pending owner go
- [ ] Vendor policy artifacts into `policy/`
- [ ] Smoke tests (camera PNG, net displacement, asset inspection)
- [ ] Apartment scene + survey renders
- [ ] Agent loop + tools + memory + providers
- [ ] Sanity LLM episode → freeze configs → batch (Fable 5, Opus 5, GPT 5.6 sol × N=3–5)
- [ ] Scoring, figures, README results

Owner deadline: Sunday night 2026-07-26. Cut order if behind: return_home stage →
GPT 5.6 sol (preserve the two-Claude comparison) → N→3 → panorama tool.
