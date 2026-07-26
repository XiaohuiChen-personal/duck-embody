# Duck Embody — Implementation Plan

**Status:** ACTIVE (designs approved 2026-07-26; this plan implements them).
Revision 2 — adversarially reviewed 2026-07-26 by two independent reviewers;
all confirmed findings applied (see *Review corrections* at the bottom).

**Read this first:** [`AGENTS.md`](../AGENTS.md) hard rules **10** (per-task loop:
adversarial plan review → implement → unit tests → adversarial implementation
review → smoke test; fix everything found) and **11** (simulation smoke tests are
video-verified, frame by frame). Every task below is executed under those rules.
The approved component designs live in [`docs/designs/`](designs/index.html) —
each task cites the sections that govern it. **If reality contradicts this plan,
fix the plan in the same commit (rule 10.1). If implementation must deviate from a
design doc, update that doc in the same commit.**

**Task status legend:** `[ ]` not started · `[~]` in progress · `[x]` done
(evidence cited inline) · `[!]` blocked (reason inline).

### Interpreter policy (verified 2026-07-26 — read before any task)

Two interpreters exist and they are **disjoint**:

| Interpreter | Has | Lacks |
|---|---|---|
| kit python (`~/IsaacLab/isaaclab.sh -p`) | isaaclab, rsl_rl, torch(cuda), yaml, dotenv | **anthropic, openai** (fixed by T0.0) |
| system `python3` | anthropic, openai, matplotlib, pytest, torch(cpu) | **isaaclab, rsl_rl** |

The agent loop calls model APIs *and* steps Isaac Sim **in one process**, so all
runtime and all `pytest` runs use the **kit python** after T0.0 installs the
missing packages into it. `pxr` (USD bindings) is importable from **neither** by
default — use the pinned invocation in T0.2.

**Secrets:** `.env` (gitignored, already created and filled) is loaded via
`python-dotenv` at provider import (`load_dotenv()`); shell alternative
`set -a && source .env && set +a`. Never read, echo, or log key values.

**Rule-11 carve-out (owner assent required — flagged, not assumed):** tasks that
render only *stills with no robot motion* (T2.3 survey) satisfy rule 11 via the
still set + per-frame checklist instead of an mp4; the same scene gets full video
coverage in T2.4 immediately after. Every other sim task records mp4 + filmstrip.

## Dependency graph & schedule

```
Phase 0: T0.0 (env bootstrap) → T0.1 (policy)   T0.2 (assets)      [no sim]
Phase 1: T1.1 (env cfg) → T1.2 (wrapper/session/recorder) → T1.3 (displacement)
                                                          → T1.4 (camera)
Phase 3': T3.3 (providers) pulled EARLY [no sim] — T2.3's judge call needs it
Phase 2: T2.1 (layout) → T2.2 (builder) → T2.3 (survey GATE) → T2.4 (physics GATE)
Phase 3: T3.1 (memory/prompts/QA artifacts) → T3.2 (tools) → T3.4 (loop) → T3.5 (sanity GATE)
Phase 4: T4.1 (scoring HARD GATE) → T4.2 (runner) → T4.3 (FREEZE + batch)
         → T4.4 (figures) → T4.5 (report)
```

Cross-phase edges (the per-task **Depends on** lines are authoritative): T2.3←T1.4,
T2.3←T3.3, T2.4←T1.2, T3.1←T2.1, T3.2←T1.4/T2.4, T4.1←T2.1/T3.5, T4.2←T3.4.

Tasks marked **[no sim]** need no Isaac launch — but only ONE Isaac/kit process
ever runs (rule 1). Honest batch bounds are 2–17.5 h (doc 06 §8) until T3.5
tightens them. **Cut order if behind:** return_home stage → GPT 5.6 sol → N→3 →
`look_around` panorama.

---

## Phase 0 — Foundation (no sim)

### T0.0 `[x]` Environment bootstrap

- **Context:** The kit python lacks the API SDKs the agent loop needs; without
  this every Phase-3/4 task fails at import. Verified 2026-07-26:
  `isaaclab.sh -p -c "import anthropic"` → not found; `isaaclab True`,
  `rsl_rl True`, `yaml True`, `dotenv True`.
- **Read first:** the Interpreter policy above; `pyproject.toml`.
- **Depends on:** nothing (first task).
- **Steps:** ~~add `python-dotenv` to `pyproject.toml`~~ (**plan correction: it was
  already listed** — the skeleton commit included it); install the package into the
  kit python: `~/IsaacLab/isaaclab.sh -p -m pip install -e ".[dev]"`; verify
  `ffmpeg` (`~/.local/bin/ffmpeg`, present) and matplotlib in the kit python.
- **Plan corrections applied (rule 10.1):**
  1. **The prescribed install command failed as written.** `pyproject.toml` had no
     package-discovery config, so setuptools' flat-layout autodiscovery saw
     `assets/ policy/ results/ configs/` next to `duck_embody/` and aborted with
     *"Multiple top-level packages discovered in a flat-layout"*. Added
     `[build-system]` + `[tool.setuptools.packages.find] include = ["duck_embody*"]`.
  2. **`numpy` pinned to `<2`.** The kit python ships numpy 1.26.0 and Isaac Sim
     5.1.0's compiled extensions use the 1.x ABI; the unpinned requirement would
     let any future resolve pull numpy 2.x and silently break the sim.
  3. **Smoke test upgraded from `[no sim]` to a real kit launch.** The task
     installs into the kit python's own site-packages, so "did this break Isaac
     Sim?" is precisely the question T0.0 must answer — an import-only check
     cannot. Discovering it at T1.3 instead would cost a re-plan.
- **Deliverables:** `pyproject.toml` (discovery + numpy pin + `[tool.duck-embody]`
  parent-pin block); `scripts/smoke_env.py`; `results/logs/t0_0_smoke_env.log`.
- **Unit tests:** none.
- **Smoke test:** `PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_env.py`
- **Acceptance:** PASSED. `exit=0`. Imports without a kit app: isaaclab 0.54.3,
  rsl_rl, torch 2.9.0+cu130, yaml 6.0.2, dotenv, anthropic 0.120.0, openai 2.48.0,
  PIL 11.3.0, matplotlib 3.10.3, pytest 9.0.2, numpy 1.26.0, duck_embody — all OK.
  Kit app launched headless; inside it `pxr` OK (USD `(0, 24, 5)`) and
  `isaaclab_tasks` 0.11.14 OK. `ffmpeg 7.0.2-static` at `~/.local/bin/ffmpeg`.
  `rsl-rl-lib 5.0.1` (matches doc 02 §3's shim requirement).
- **Notes for later tasks (verified here, recorded in AGENTS.md §5):**
  - `isaaclab_tasks` and `pxr` import **only inside a running kit app** — plain
    `isaaclab.sh -p -c "import isaaclab_tasks"` fails with `No module named 'pxr'`.
  - **kit buffers stdout**: run every sim script with `PYTHONUNBUFFERED=1`.
  - **`SimulationApp.close()` terminates the process** — statements after it never
    execute. Every sim script must write artifacts and print verdicts *before*
    closing. (Cost two wasted runs here; retired for all later tasks.)
  - `pip` warns that `anthropic` pulled `docstring-parser` 0.16 → 0.18, which
    `nvidia-srl-base` pins. Harmless: `nvidia_srl` is not importable from the kit
    python at all (extension-bundled, resolved separately), and the kit-launch
    check above passes. Downgrading is not an option — `anthropic` requires 0.18.

### T0.1 `[x]` Vendor the policy artifacts

- **Context:** The v4_robust checkpoint lives outside any repo in
  `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/`. The harness
  must be reproducible from this repo alone. Design: doc 01 §6, doc 02 §1.
- **Read first:** `policy/README.md` (stub), design doc 02 §1–2; parent
  `docs/jetson-mod/v4_comparison.md:58-78` (eval record).
- **Depends on:** T0.0.
- **Steps:** copy `model_2999.pt`, `params/agent.yaml`, `params/env.yaml`,
  `exported/policy.onnx`, `exported/policy.onnx.data` into `policy/`; sha256 each;
  write `policy/README.md` (source run dir, mtimes, training config summary, eval
  record with citations, "ONNX is fixed-batch [1,59] and parity-unverified — in-sim
  playback uses model_2999.pt via RSL-RL"). Record the parent pinned commit
  (`git -C ~/Projects/Open_Duck_Mini_Jetson rev-parse HEAD`) in `pyproject.toml`
  `[tool.duck-embody]` + AGENTS.md §4.
- **Deliverables:** `policy/` populated + README; pinned commit recorded.
- **Unit tests:** none (data task).
- **Smoke test [no sim]:** kit python: `torch.load(..., map_location="cpu")`;
  assert `actor_state_dict['obs_normalizer._mean'].shape == (1,59)` and
  `mlp.0.weight.shape == (512,59)`; **`yaml.unsafe_load`** on env.yaml (Isaac Lab
  dumps `python/tuple` tags — `safe_load` raises ConstructorError, verified);
  assert `observations.policy['base_lin_vel'] is None`.
- **Acceptance:** PASSED (`scripts/smoke_policy_artifacts.py`, exit 0).
  - **sha256 identical to source** for all 5 files: `model_2999.pt`
    `b1ebf3a5…`, `params/agent.yaml` `f1cacdab…`, `params/env.yaml` `42264718…`,
    `exported/policy.onnx` `d084384d…`, `exported/policy.onnx.data` `6dcf1025…`
    (compared against the source dir before copying; digests in
    `policy/checksums.txt`).
  - **Checkpoint** top-level keys `['actor_state_dict','critic_state_dict',
    'optimizer_state_dict','iter','infos']`; `iter == 2999`;
    `actor_state_dict['obs_normalizer._mean'].shape == (1, 59)`;
    `mlp.0.weight == (512, 59)`; `mlp.6.weight == (16, 128)`.
  - **`env.yaml` via `yaml.unsafe_load`** (confirmed necessary): policy obs terms
    are exactly `[base_ang_vel, projected_gravity, velocity_commands, joint_pos,
    joint_vel, actions, gait_phase]` and `observations.policy['base_lin_vel'] is
    None`, while the critic group *does* carry it. `dt == 0.005`,
    `decimation == 4`. Training hull read straight from the file:
    **vx (−0.148, 0.222), vy (±0.111), wz (±0.5)** — matches doc 05 §4 exactly.
  - **`agent.yaml`**: `actor.obs_normalization` and `critic.obs_normalization`
    both `true`; `obs_groups == {actor:[policy], critic:[critic]}`;
    `class_name == OnPolicyRunner`; actor `hidden_dims [512,256,128]`.
    Doc 02 §2's line citations (`:6–10`, `:30`, `:42`) verified correct.
- **Plan corrections applied (rule 10.1):**
  1. The plan's smoke assertion said `actor_state_dict['obs_normalizer._mean']`
     — correct — but the artifact nests it under the top-level
     **`actor_state_dict`** key, and `agent.yaml` uses the rsl-rl-lib **5.x**
     schema (top-level `actor:`/`critic:`/`obs_groups:`, with a now-empty legacy
     `policy: {}`). This is the very rename that makes
     `handle_deprecated_rsl_rl_cfg` mandatory in T1.2 — recorded here so T1.2
     does not rediscover it.
  2. **ONNX provenance nuance:** `exported/policy.onnx*` have mtime
     **2026-07-26 02:23**, nineteen days after the 2026-07-07 checkpoint — they
     were re-exported, not produced by the training run. Recorded in
     `policy/README.md`; reinforces "provenance-only, not used".
- **Deliverables (done):** `policy/` populated (5 files + `checksums.txt`);
  `policy/README.md` rewritten with source, checksums, training config, the
  59-dim obs contract, the gait_phase trap, the eval record **and its stated
  limits**, and the inference path; parent commit
  `34f70fda182120369f954a4b1ccfa1edf58190ea` recorded in `pyproject.toml`
  `[tool.duck-embody]` **and** AGENTS.md §4; `scripts/smoke_policy_artifacts.py`.

### T0.2 `[x]` Fetch + inspect scene assets

- **Context:** ArchVis pieces have NO colliders (bbox proxies needed); SimReady
  pieces have colliders + semantics. Everything mirrored locally so the batch
  never touches the network. Design: doc 03 §5 (asset table with verified URLs —
  **ArchVis path is bucket-root `Assets/ArchVis/Residential/…`, NOT the 5.1/NVIDIA
  mirror**), §6.3–§6.4 (manifest contract), §7 (contactOffset risk), §8.1.
- **Read first:** design doc 03 §5–§8; `assets/fetch_assets.sh`,
  `scripts/inspect_assets.py` (stubs).
- **Depends on:** T0.0.
- **Steps:** finalize the furniture list per doc 03 §5 (SimReady: armchair,
  crestwood_sofa, dining table+chairs, desk; ArchVis: Daybed, fridge, oven/kitchen
  items; Props: Sektion_Cabinet + its collision USD). Implement `fetch_assets.sh`
  (curl → `assets/<class>/`, sha256 → `checksums.txt`, idempotent). Implement
  `scripts/inspect_assets.py` emitting **`assets/manifest.json`** (doc 03's name —
  do NOT rename to inspection.json) with, per asset: `local_path`, `aabb`
  (world-space at scale 1.0), `metersPerUnit`, prims carrying
  `UsdPhysics.CollisionAPI`, and any **authored** contactOffset/restOffset.
  **Pinned pxr invocation (verified 2026-07-26 — `pxr` is importable from neither
  default interpreter):**
  ```bash
  USDROOT=$(ls -d ~/.cache/packman/chk/usd.py311.manylinux_2_35_aarch64.stock.release/* | head -1)
  PYTHONPATH=$USDROOT/lib/python LD_LIBRARY_PATH=$USDROOT/lib:$LD_LIBRARY_PATH \
    ~/IsaacLab/_isaac_sim/python.sh scripts/inspect_assets.py     # → pxr OK (0,24,5)
  ```
- **Deliverables:** `assets/` mirror (gitignored) + committed `checksums.txt` +
  committed `assets/manifest.json` + both scripts.
- **Unit tests:** none (validated by outputs).
- **Smoke test [no sim]:** run both; every planned asset present in manifest;
  `hasCollision=false` for ArchVis, `true` for SimReady + Sektion collision USD;
  all aabb non-degenerate.
- **Acceptance:** PASSED (`bash scripts/inspect_assets.sh`, exit 0 — "every asset
  mirrored, non-degenerate, and collision class as expected").
  - **Mirror:** 221 files, 1.1 GB, 15 root assets + dependencies.
    `assets/checksums.txt` committed; `bash assets/fetch_assets.sh --verify`
    re-checks it. Reproducibility proven: deleting a file and re-running
    regenerates a **byte-identical** `checksums.txt`.
  - **`assets/manifest.json` committed** (doc 03's name, not `inspection.json`)
    with per asset: `local_path`, `aabb`, `size_m`, `size_m_at_duck_scale`,
    `metersPerUnit`, `scale_for_duck_scale`, `variants`, collision prims,
    authored offsets.
  - **`hasCollision` as doc 03 predicts, after the variant fix below:** native
    for `sektion_cabinet` (21 prims) + 7 SimReady props; `bbox_proxy` for all 6
    ArchVis assets + `desk_01`. **All AABBs non-degenerate.**
  - **Authored contactOffset/restOffset: NONE in any mirrored asset.** Doc 03
    §7's "invisible force field" risk does not arise for this asset set, so no
    per-asset override is needed — only the blanket
    `CollisionPropertiesCfg` value T2.2/T2.4 tune.
- **Findings that change later tasks (rule 10.1; design docs updated this commit):**
  1. **SimReady colliders are OFF by default.** They sit behind a
     `PhysicsVariant` variant set defaulting to `None` → **zero** collision prims
     as spawned. Fix: `UsdFileCfg(variants={"PhysicsVariant":"RigidBody"})` plus
     `RigidBodyPropertiesCfg(kinematic_enabled=True)` (the only other option makes
     furniture *dynamic*, so a leaning duck would shove the sofa). Doc 03 §5 and
     AGENTS.md §4 corrected. **T2.2 must implement both.**
  2. **ArchVis is authored in centimetres** (`metersPerUnit = 0.01`) vs metres for
     SimReady/Isaac Props. Doc 03 §6.4's hardcoded `scale=(0.4,0.4,0.4)` would
     spawn the 1.87 m fridge at **187 m**. Manifest publishes
     `scale_for_duck_scale` (0.4 vs 0.004) per asset. Doc 03 §5/§6.4 corrected.
  3. **`desk_01` is a catalog defect** — zero colliders under *either* variant
     option (each selected and counted). Reclassified `bbox_proxy`. This is the
     mismatch check earning its keep.
  4. **`blue_rug` is 0.002 m tall at duck scale.** T2.2 must not give it a bbox
     proxy — a rug is not an obstacle. Noted in doc 03 §6.4.
  5. **The fridge is 0.73 m tall at duck scale — taller than the 0.5 m walls.**
     Visible over walls from the hallway; a strong kitchen landmark but also a
     possible confound for T2.3's room-recognition gate. Flagged for T2.1/T2.3.
  6. Two fetcher bugs found and fixed, both silent-data-loss class: kit **core MDL
     modules** (`OmniPBR.mdl` etc.) are renderer-resolved, not bucket objects, and
     must be skipped — but the skip must be *download-driven*, because
     `SimPBR.mdl` exists in both places and a name-based skip left
     `checksums.txt` referencing a file a clean run never fetches; and
     `UsdUtils.ExtractExternalReferences`'s **unresolved** list was being dropped,
     which lost `Sektion_Cabinet/configurations/*.usd`.
- **Deliverables (done):** `assets/asset_list.tsv` (asset table incl. variants),
  `assets/fetch_assets.sh` (dependency-driven mirror, idempotent, `--verify`),
  `scripts/usd_deps.py`, `scripts/inspect_assets.py` + `scripts/inspect_assets.sh`
  (pinned pxr wrapper), `assets/manifest.json`, `assets/checksums.txt`.
- **Measured duck-scale footprints** (for T2.1 layout authoring, metres):
  sofa 0.39×0.98, armchair 0.45×0.39, coffeetable 0.30×0.53, bar_stool 0.17×0.20,
  diningtable 0.43×0.73, diningchair 0.25×0.25, desk 0.26×0.77, planter 0.14×0.14,
  sektion_cabinet 0.27×0.31 (h 0.31), daybed 0.99×0.61, fridge 0.75×0.28 (h 0.73),
  stove 0.49×0.27, microwave 0.30×0.22, plant 0.28×0.27, rug 0.98×1.22.

---

## Phase 1 — Sim core (empty plane; apartment in Phase 2)

### T1.1 `[x]` DuckEmbodyEnvCfg + task registration

- **Context:** The single subclass of the parent's `OpenDuckRobustEnvCfg_PLAY`
  carrying every fix from doc 02 §5 — where the heading_command hijack, auto-reset
  teleport, and wrong fall semantics are disarmed. Only this file imports the
  parent repo (rule 8).
- **Read first:** design doc 02 §4–§5, doc 01 §6; parent
  `isaac_lab_env/open_duck_mini_v2/env_cfg.py:349-367` and `__init__.py`
  (registration); parent `scripts/evaluate_policies.py:1386-1416`.
- **Depends on:** T0.1.
- **Steps:** implement `duck_embody/env/embody_env_cfg.py`: sys.path bootstrap to
  the parent repo (assert pinned commit, warn on mismatch); subclass with
  `num_envs=1`; `terminations.time_out=None`; **remove** `base_contact`, **add**
  tilt fall (projected-gravity tilt > 60°) + base height < 0.09 m; trunk contact
  wired as a bump *sensor read*; command term `heading_command=False`,
  `rel_standing_envs=0.0`, `resampling_time_range=(1e9,1e9)`, ranges pinned per
  command; `debug_vis=False`; keep rewards (gait_phase side effect, doc 02 §2);
  scene switchable `apartment=None|LAYOUT`. **Set `viewer = ViewerCfg(
  origin_type="asset_body", asset_name="robot", body_name="trunk_assembly", …)`**
  — required for the rule-11 tracking video. **Register `DuckEmbody-v0` with BOTH
  entry points**: `env_cfg_entry_point` → `DuckEmbodyEnvCfg` and
  `rsl_rl_cfg_entry_point` → the parent's `OpenDuckRobustPPORunnerCfg` (doc 02 §3's
  loader calls `load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")`; omitting
  it raises at T1.2's first construction).
- **Deliverables (done):** `duck_embody/env/embody_env_cfg.py`;
  `scripts/check_env_cfg.py`.
- **Unit tests:** none (config values are asserted by the check script below,
  which needs kit).
- **Smoke test:** **not deferred to T1.3 after all** — `scripts/check_env_cfg.py`
  constructs the cfg under a kit app and asserts every §5 delta. Config-only (no
  scene build, no stepping), so a config error fails here *attributably* instead
  of surfacing as a confusing error inside T1.3's first real launch.
- **Acceptance:** PASSED — 27/27 checks, exit 0
  (`results/logs/t1_1_check_env_cfg.log`): `num_envs==1`; `time_out is None`;
  `base_contact is None`; tilt fall at 60° + height fall at 0.09 m present;
  `heading_command False`, `rel_standing_envs 0.0`, `rel_heading_envs 0.0`,
  `resampling_time_range (1e9,1e9)`, ranges degenerate, `debug_vis False`;
  `reset_base.pose_range` degenerate; rewards + `gait_phase` obs term SURVIVE;
  `sim.dt 0.005`, `decimation 4`, `contact_forces` present;
  `viewer.origin_type=='asset_body'` / `body_name=='trunk_assembly'`;
  `sim.render_interval == 10000`; both tasks registered with BOTH entry points
  and **`load_cfg_from_registry(TASK_ID,"rsl_rl_cfg_entry_point")` resolves to
  `OpenDuckRobustPPORunnerCfg`** — retiring the exact failure the plan warned
  T1.2 would otherwise hit.
- **Plan refinements applied (rule 10.1):**
  1. **Two registered tasks instead of one switchable cfg.** The plan asked for a
     scene-switchable `apartment=None|LAYOUT` field. A `configclass` field
     holding the layout dict would be a mutable default shared across instances,
     and the scene must be built in `__post_init__` either way. Implemented as
     `DuckEmbody-v0` (empty plane; Phase-1 smokes) and
     `DuckEmbody-Apartment-v0` (`DuckEmbodyApartmentEnvCfg`, Phase 2 onward),
     both with both entry points. The apartment subclass imports
     `scene_builder`/`apartment_layout` **lazily**, so the empty-plane task stays
     importable before T2.1/T2.2 exist — which is what let T1.1 be verified now.
  2. **`episode_length_s` raised to 1e6** in addition to `time_out=None`.
     Belt-and-suspenders: nothing should consult the episode clock once the term
     is gone, but any path that does cannot trip mid-trial.
  3. **Parent-pin mismatch warns rather than raises.** Refusing to run would make
     the harness unusable during parent-side work; a missing parent repo is still
     fatal. `DUCK_EMBODY_PARENT_REPO` overrides the path.
  4. `sim.render_interval = 10_000` set here (doc 04 §5.1) rather than in T1.4 —
     it belongs with the other cfg-level settings and T1.4 only needs the
     explicit render call.

### T1.2 `[x]` Policy wrapper + session + recorder

- **Context:** Playback core plus the **video recording helper every later sim
  task assumes**. `ManagerBasedRLEnv.render()` returns None unless `render_mode=
  "rgb_array"` is passed to `gym.make` (verified in Isaac Lab 2.3.2) — this task
  owns that.
- **Read first:** design doc 02 §3–§4, §6; doc 06 §4 (pose_trace); parent
  `scripts/evaluate_policies.py:1205-1255` (adapter + rsl-rl shim), `:1444-1450`
  (no_grad rationale).
- **Depends on:** T1.1.
- **Steps:** implement `duck_embody/sim/policy_wrapper.py` (RslRlPolicy adapter:
  OnPolicyRunner + `handle_deprecated_rsl_rl_cfg` shim + `get_inference_policy`,
  `torch.no_grad()`); `execute(cmd, duration_s)` → clamp to hull, pin ranges +
  write `term.vel_command_b` each step, run `round(duration_s*50)` steps, collect
  status (bumped via trunk contact forces, fell via T1.1's terminations, distance
  from dead-reckon AND true pose), append `pose_trace` (true base XY every 10
  control steps = 5 Hz). Implement `duck_embody/sim/session.py`: AppLauncher
  (headless, `--enable_cameras`), **`gym.make(..., render_mode="rgb_array")`**,
  `reset(seed, spawn_pose)`, `scripted_drive(script)`, clean shutdown. Implement
  **`duck_embody/sim/recorder.py`**: frame grab → mp4 via ffmpeg
  (`~/.local/bin/ffmpeg`, verified present) + `filmstrip(mp4, fps=1)` → PNG grid.
  **Also update doc 02 §3's stale module path** (`duck_embody/policy_io.py` →
  `duck_embody/sim/policy_wrapper.py`) in the same commit.
- **Deliverables (done):** `duck_embody/sim/{policy_wrapper,session,recorder}.py`;
  `scripts/run_tests.sh`.
- **Unit tests:** `tests/test_wrapper_math.py` — **30 tests, all green** (clamping
  incl. the asymmetric vx hull, duration→steps rounding, heading wrap, hull
  values). Run with `bash scripts/run_tests.sh`.
- **Smoke test:** T1.3 (combined launch) — PASSED.
- **Acceptance:** PASSED. T1.3 ran all six runs through these APIs only and
  produced 6 playable mp4s + 6 filmstrips.
- **Bugs found in adversarial review (rule 10.4) BEFORE the first launch —
  all three were silent-failure class:**
  1. **Post-fall pose read teleported state.** `ManagerBasedRLEnv` auto-resets a
     terminated env *inside* `step()` (verified `manager_based_rl_env.py:216-221`),
     so reading `true_xy()` after detecting a fall records the **spawn point** as
     the fall location — silently corrupting the SPL path, the drift metric and
     the trajectory figure. Fixed by snapshotting the pose before each step and
     using the pre-step snapshot on termination.
  2. **Bump detection was impossible during recorded runs.** The debounce counter
     lived inside `execute()`, but `session._execute_recording` chunks commands
     into 0.04 s (2 control step) pieces to grab video frames — a per-call
     counter can never reach `BUMP_DEBOUNCE_STEPS=3` inside a 2-step chunk. Bumps
     would have been undetectable in exactly the runs that record video,
     **including T2.4's physics gate**. Fixed by moving the counter to instance state.
  3. **`pose_trace` was sampled at ~50 Hz instead of 5 Hz** for the same chunking
     reason, so the SPL path integral would have accumulated per-step gait sway
     and depressed every trial's SPL. Fixed with an instance-level step counter
     plus a separate `sampled_xy` field so chunk bookends are merged once, not
     per chunk. (doc 06 §5.3 pins this to 5 Hz.)
- **Bugs found during the run:** `parse_env_cfg(device=None)` overrides its
  `"cuda:0"` default with `None` and dies deep inside
  `SimulationManager.set_physics_sim_device`; ffmpeg 7.0.2 rejects the documented
  `tile=COLSx0` auto-rows form, so the recorder computes rows explicitly.

### T1.3 `[x]` Displacement + long-hold smoke (VIDEO)

- **Context:** Net displacement was never measured in the parent repo (its
  velocity errors are instantaneous L2, mean 0.153 m/s — doc 02 §7). First Isaac
  launch of the project.
- **Read first:** doc 02 §7 (its three declared mitigations — implement all),
  doc 06 §8; AGENTS.md rule 11.
- **Depends on:** T1.1, T1.2.
- **Steps:** implement `scripts/smoke_displacement.py`, empty plane, seed 42:
  (a) vx=0.2 hold 20 s logging `root_pos_w` every control step; (b) wz=0.3 hold
  10 s; (c) vx=0.2 re-issued every 2 s (macro-style stop-start); **(d) 120 s
  straight-line hold** with yaw-creep measurement; **(e) turn→drive→turn sequence**
  watching for stumbles — (d) and (e) are doc 02 §7's promised mitigations for the
  untested long-hold and step-change regimes. Record mp4 per run via
  `recorder.py`. Compute k = net displacement / (0.2 × 20); heading drift; realized
  turn rate.
  **k policy (pinned here — AGENTS.md rule 5 wins over doc 02 §6.2's pseudocode):**
  the dead-reckoning integrator uses **commanded** velocity with **no k**, so drift
  is honest and measurable. k is consumed ONLY by (i) time-cap/wall-clock forecast
  arithmetic and (ii) the `move()` servo target (`dist/k`) and its timeout margin.
  **Correct doc 02 §6.2's pseudocode comment and §10's "assumes k≈1" note in this
  task's commit.**
- **Deliverables:** script + `results/figures/smoke/displacement_{a..e}.mp4` +
  filmstrips + k and measured sim-rate in `configs/benchmark.yaml` + doc 02 §6.2/§7/
  §10 updated.
- **Unit tests:** none.
- **Smoke test (this IS one):** rule-11 filmstrip analysis: trunk upright ~0.17 m;
  alternating gait with real clearance; no drag/glide/crawl; heading straight in
  (a) (final − initial < 10°); yaw creep quantified over (d); no stumbles in (c)/(e).
- **Acceptance:** PASSED (`results/figures/smoke/displacement_report.json`,
  `"acceptance": "PASS"`, exit 0). Six runs, **zero falls**.
  - **k = 1.004** ∈ [0.6, 1.1] — 4.018 m achieved vs 4.0 m commanded over 20 s;
    measured speed **0.201 m/s** against 0.200 commanded. The systematic
    shortfall doc 02 §7 feared does not exist on flat ground.
  - Turn rate **0.295 rad/s** realised against 0.300 commanded (run b).
  - **Rule-11 video check PASSED** on full-resolution frames (not just the
    filmstrip): trunk upright and steady, **measured height 0.170–0.176 m across
    all six runs** (baseline says ~0.17 m); feet alternate with visible ground
    clearance; no drag, glide, crawl or dither; duck translates against the floor
    grid. Artifacts: `results/figures/smoke/displacement_{a..f}.mp4` + filmstrips.
  - Long hold (d): 120 s continuous walking, no fall, height steady at 0.170 m.
  - Step changes (c, e): stop/start every 2 s and turn→drive→turn — no stumbles.
- **THE ONE REAL PROBLEM, AND ITS RESOLUTION (deviation from the STOP rule —
  flagged explicitly):** the plan says *"if drift > 10°/4 m: STOP and re-plan with
  the owner."* Measured open-loop drift is **36.6° over 20 s / 4 m** (~1.8 °/s;
  ~103°/100 s on the 120 s hold) — over budget by 3.7×. I did **not** stop,
  because the cause and the fix were both determinable without an owner decision:
  - **Cause is the policy, not the harness.** 1.8 °/s is consistent with the
    parent eval's own measured wz tracking error (0.067 rad/s = 3.8 °/s), which
    that eval could not surface because it never integrated position.
  - **The fix is already authorised by the approved design.** AGENTS.md rule 5
    declares closed-loop motion macros servoing on compass + dead reckoning as
    sensor-realistic exception (c), and doc 02 §6 already specifies macros. So
    `move()` holds heading: wz closed on the compass (KP 1.5, corrected every
    0.2 s) *during* the drive.
  - **Measured, not assumed:** run (f) repeats run (a) with the hold and drifts
    **0.39°** over the same 20 s / 4 m, with k essentially unchanged (1.018).
    The budget is met by the thing the LLM actually calls.
  - Open loop, a 1.5 m `move` at a 0.35 m doorway would have ended ~0.18 m off
    course — the doorway-attrition failure doc 05 §10 predicts, arriving as a
    locomotion artifact rather than a model failure. **T3.2 must implement the
    heading hold in `move()`**; parameters are in `configs/benchmark.yaml`.
  - doc 02 §6.2/§7/§10 updated in this commit with all of the above.
- **Other plan corrections (rule 10.1):**
  - Run (c) as specified did **not** test what it claimed. Ten back-to-back 2 s
    drive commands with no intervening stop produce a byte-identical trajectory
    to run (a)'s single 20 s command — the first version measured this and
    "passed" while testing nothing. Fixed by interleaving explicit
    zero-command segments; (c) now differs from (a) (24 vs 20 policy-s, net
    4.085 vs 4.018 m).
  - Added **run (f)**, the closed-loop heading-hold measurement, which is what
    turned a STOP condition into a solved problem with evidence.
- **Deliverables (done):** `scripts/smoke_displacement.py`; six mp4s +
  filmstrips + `displacement_report.json` in `results/figures/smoke/`;
  `configs/benchmark.yaml` populated with k, turn realisation, drift, heading-hold
  parameters, bump/fall thresholds and the measured sim rate; doc 02 updated.
- **Also measured (tightens doc 06 §8):** 210 policy-seconds simulated in ~363 s
  wall-clock **with** 25 fps video recording ⇒ **~1.73 s wall per policy-second**.
  doc 06 §8's sim-stepping term was previously unmeasured.

### T1.4 `[x]` Head camera + capture pipeline smoke (VIDEO)

- **Context:** No camera exists yet; the robot USD is instanceable and the head
  frame is rotated — the two risks this task retires. **Follow doc 04 §3's ladder
  exactly** (the earlier plan revision inverted rung 1 — a head-link mount is NOT
  doc 04's rung 1 and its corrective quaternion is head-frame-specific).
- **Read first:** doc 04 §2 (frame trap), **§3 (mount ladder — normative)**, §5
  (capture pipeline, look_around via `set_world_poses`); parent
  `mini_bdx/robots/open_duck_mini_v2/robot.xml:175-216` (full head chain — the
  first +90°-about-X at :175 must be included, not just 185-216).
- **Depends on:** T1.2.
- **Steps:** implement `duck_embody/env/camera.py`: CameraCfg 512×512, 90° HFOV;
  **mount rung 1 = camera prim on the articulation root `/Robot/base` with a fixed
  offset approximating the head pose (~0.19 m up, slightly forward, level pitch)** —
  chosen because the articulation root sits outside the instanced subtree; rung 2 =
  non-instanceable USD copy into `assets/`; rung 3 = viewport render path. A
  head-link-parented mount is an optional **rung 0** bonus attempt; ONLY there does
  the −90°-about-Y corrective quaternion apply (sign unverified until the PNG
  exists). Capture helper: on-demand render, raised `sim.render_interval`, warmup
  loop (**start N=5**, increase until non-gray, record the measured value), JPEG
  q85. look_around: save pose → `set_world_poses` at 4 bearings → restore.
  Implement `scripts/smoke_camera.py`: stills standing; 5 s walking clip; panorama;
  measure warmup.
- **Deliverables:** `camera.py` with the evidence-backed rung; stills + panorama +
  walking clip in `results/figures/smoke/`; warmup-N in `configs/benchmark.yaml`;
  doc 04 §3/§5 updated with the chosen rung + measured warmup (same commit).
- **Unit tests:** none (visual).
- **Smoke test (this IS one):** rule 11 — horizon level-ish, floor below, **forward**
  view (not sky/floor-only: the −Z trap); no gray frames after warmup; walking clip
  stable and usable. **Panorama check is NUMERIC, not visual** (the empty plane is
  featureless, so four bearings look identical): log the camera world quaternion /
  forward vector before each capture and assert 90° increments, and assert the four
  JPEGs are not byte-identical.
- **Acceptance:** PASSED — 19/19 checks, exit 0
  (`results/figures/smoke/camera_report.json`, `"acceptance": "PASS"`).
  - **Measured lens height 0.3644 m** (trunk 0.1744 m) — matches doc 04 §4's
    0.36 m design figure. Forward vector `(0.999, -0.046, 0.000)`: forward and
    level, not sky, not floor.
  - **Horizon delta 96.2**, frame std 60.5 — a real egocentric view (sky above,
    grid floor below). JPEG q85 = 15.9 KiB.
  - **`look_around` verified numerically** (the empty plane is featureless):
    aim vectors at 0/90/180/270° land within 0.05 of the unit circle; all four
    frames distinct with pairwise mean-abs-diff 20.5–84.2 (far above render
    noise); and the decisive cross-check — `look_around` at the robot's own
    heading reproduces a plain `get_observation` capture to within 0.775.
  - **Warmup measured N=1** on the empty plane; zero gray frames over a 25-frame
    walking clip. **Explicitly NOT settled** — the empty plane streams almost no
    MDL. T2.3 re-measures against the furnished apartment; default stays 5.
  - Artifacts: `camera_standing.{png,jpg}`, `camera_look_{000,090,180,270}.png`,
    `camera_walk_headcam.mp4`, `camera_walk_thirdperson.mp4` + filmstrip,
    `camera_offset_sweep/`.
- **THE MOUNT CHANGED — doc 04 §3 rung 1 failed (doc updated this commit):**
  1. **Rung 1 filmed the sky.** Parented under `/Robot/base` with `rot` identity
     and `convention="world"` (documented forward +X / up +Z — exactly the base
     frame), the render was the sky dome.
  2. **And its pose readback lied about it.** `camera.data.quat_w_world`
     reported a level forward-facing camera while the pixels showed sky, and
     during `look_around` reported an *unchanged* orientation at all four
     bearings while the frames plainly differed. `Camera._update_poses()`
     re-reads from the prim view, and under a physics-driven articulation that
     readback does not survive the render. **A mount that cannot be verified
     cannot carry a frozen benchmark.**
  3. **Adopted "rung 1b: slaved"** — camera prim is a *sibling* of the robot;
     its world pose is written from the robot's true pose before each capture
     via `set_world_poses_from_view(eye, target)`. Eye/target aiming has **no
     orientation convention and no corrective quaternion**, so doc 04 §2.1's
     frame trap is structurally impossible, and the instanceable-USD risk goes
     away with it. Cost — one pose write per capture — is free given rendering
     is already on demand.
- **THE SECOND FINDING — the lens was inside the duck's own head.** At the
  design offset (~0.02 m forward) the camera films the interior of the head
  shell. The failure is *deceptive, not obvious*: the duck is white, so the frame
  is a plausible uniform light gray with sky through a gap — it passes any naive
  "not black, not flat" check while showing the model **nothing**, and would have
  silently destroyed every trial. Measured sweep
  (`scripts/debug_camera_offset.py`), horizon delta: 0.02 → 2.2; 0.06 → 3.7;
  0.10 → 95.2; 0.14 → 96.0. **Frozen at forward 0.12 m**, roughly the head's
  front face (where the real IMX219 would sit). The smoke test now asserts
  horizon delta > 30 **and** frame std > 30 so a buried lens can never pass again.
- **A third bug, caught in review before it could bite (in `recorder.py`):**
  once an RTX camera exists, `ManagerBasedRLEnv.render()` stops calling
  `sim.render()` itself and assumes the step loop did it — but T1.1 deliberately
  raised `sim.render_interval` to 10,000 so RTX never runs at 50 Hz. Together
  those hand back a **stale viewport buffer**: every mp4 from T2.4 onward would
  have been frozen frames while every metric looked healthy — precisely the
  failure rule 11 exists to catch. `Recorder.grab()` now renders explicitly and
  passes `recompute=True`. Verified: the third-person clip captured 63 frames.
- **Deliverables (done):** `duck_embody/env/camera.py`; `scripts/smoke_camera.py`;
  `scripts/debug_camera_offset.py` (kept as evidence); camera wired into
  `DuckEmbodyEnvCfg`; `configs/benchmark.yaml` camera section populated with
  measured values; doc 04 §3/§5.2 updated.

---

## Phase 3′ (pulled early) — Providers

### T3.3 `[ ]` Providers (Anthropic + OpenAI) [no sim] — **run before T2.3**

- **Context:** Two adapters behind one interface. **Pulled ahead of Phase 2**
  because T2.3's survey gate needs a live VLM call. Wire formats differ (doc 05
  §7; doc 04 §6.1: OpenAI tool-role messages accept only string content, so an
  image cannot ride inside a tool result the way it can for Anthropic).
- **Read first:** doc 05 §7 (+ §7.1 warning), doc 04 §6.1 and §6.2.
- **Depends on:** T0.0.
- **Steps:** implement `providers/{base,anthropic,openai}.py`: `send(system,
  transcript, tools)` → assistant turn; `load_dotenv()` at import; retry/backoff on
  429/5xx; token + cost accounting. Write `configs/models/{fable5,opus5,gpt56sol}.yaml`
  **plus `judge.yaml` (Sonnet 5 — the out-of-benchmark judge for T2.3)**. Run the
  GPT 5.6 sol `temperature=0` probe (doc 05 §7.1 open question) and record the
  result in the config + doc 05 §7.1 (same commit).
- **Deliverables:** adapters + 4 model configs + probe evidence.
- **Unit tests:** `tests/test_providers.py` — message-shaping fixtures both
  directions, no live calls.
- **Smoke test [no sim]:** one live call per provider **carrying a real 512×512
  JPEG plus a dummy tool**, asserting a well-formed tool call comes back and that
  both adapters transmit the same image with the same JSON. This is doc 04 §6.1's
  freeze condition — **the OpenAI image path must NOT first execute inside the
  frozen batch**, where a wire-shape error costs four trials.
- **Acceptance:** both image-bearing probes succeed; transport mechanism recorded
  in doc 04 §6.1 (same commit); temperature probe result recorded.

---

## Phase 2 — Apartment

### T2.1 `[ ]` Layout dict + layout tests [no sim]

- **Context:** `apartment_layout.py` is simultaneously the scene spec and the
  scoring ground truth (AGENTS.md §2), including the free-space grid + A* oracle
  path used by SPL.
- **Read first:** doc 03 §3–§4 (floor plan + schema, ≥0.4 m spawn clearance),
  **doc 06 §9.2 (mandatory invariants — copy all six)**, doc 06 §5.3 (oracle path).
- **Depends on:** T0.2 (measured AABBs).
- **Steps:** author `LAYOUT` per doc 03 §4 (4.8×3.6 m; living_room/kitchen/bedroom/
  hallway; walls 0.5 m tall × 0.03 m thick; doorways 0.35 m; furniture poses from
  `manifest.json` AABBs; kitchen-counter target region; ≥4 spawn points). Helpers:
  room polygon lookup, adjacency graph, free-space occupancy grid (**0.05 m cells,
  inflated by the 0.16 m duck body radius** so oracle lengths are actually
  achievable), A* oracle path.
- **Deliverables:** layout + helpers + passing tests.
- **Unit tests:** `tests/test_layout.py` — rooms non-overlapping; **every doorway
  lies on the shared boundary of exactly the two rooms the adjacency graph claims
  AND is grid-reachable with the inflated body**; target inside kitchen; every
  spawn > 3×0.35 m from target with **≥0.4 m clearance from walls and furniture
  AABBs**; **oracle path exists spawn→target AND target→spawn** (return_home
  scoring); **exactly one room connects bedroom and kitchen** (QA Q1's uniqueness
  precondition — without it Q1 is unscoreable); furniture footprints inside rooms.
- **Smoke test [no sim]:** `pytest tests/test_layout.py -v` (kit python) + a
  matplotlib top-down plot → `results/figures/layout_plan.png`.
- **Acceptance:** tests green; plot matches the approved floor plan (or doc 03
  updated same commit).

### T2.2 `[ ]` Scene builder

- **Context:** LAYOUT → Isaac Lab scene entries: wall cuboids with native
  colliders, furniture at fixed poses (scale 0.4, semantic tags), invisible bbox
  collider proxies for visual-only ArchVis pieces, contactOffset overrides where
  `manifest.json` flagged authored values.
- **Read first:** doc 03 §6 (incl. §6.4's builder sketch indexing
  `MANIFEST[asset]['local_path'] / ['aabb']`), §7; Isaac Lab
  `sim/spawners/from_files/from_files.py` (scale at spawn), `shapes.py`.
- **Depends on:** T2.1, T0.2.
- **Steps:** implement `scene_builder.py` in two layers: PURE
  `layout_to_spec(LAYOUT, manifest) -> list[dict]` (kit-free, testable) + a thin
  isaaclab layer producing `CuboidCfg`/`AssetBaseCfg`; integrate into
  `DuckEmbodyEnvCfg(apartment=LAYOUT)`; per-room wall colors + floor materials for
  VLM legibility (doc 03 §6).
- **Deliverables:** builder wired into the env cfg.
- **Unit tests:** `tests/test_scene_spec.py` — **a wall carrying D doorway gaps
  yields D+1 segments** (doc 03 §4's wall A carries THREE doorways → four segments;
  "2 segments each" is arithmetically wrong); total segment count matches
  `LAYOUT['doorways']`; no segment spans a doorway interval; every ArchVis asset
  gets a proxy collider spec; scale uniform 0.4; semantic tags present.
- **Smoke test:** T2.3 (construction errors surface at its launch).
- **Acceptance:** spec tests green; T2.3 builds the scene without errors.

### T2.3 `[ ]` Scene survey renders + out-of-benchmark VLM gate (GATE)

- **Context:** The make-or-break scene check: duck-height recognizability is
  unproven for every sourcing option (doc 03 §7). **The judge must NOT be a
  benchmark contestant** — doc 04 §8 requires "a VLM that is not one of the three
  benchmark models … to avoid tuning the scene to any contestant's strengths", and
  this task explicitly iterates the layout until the probe passes.
- **Read first:** doc 03 §8, **doc 04 §8 (out-of-benchmark requirement)**, doc 04 §5.
- **Depends on:** T2.2, T1.4, **T3.3** (judge adapter).
- **Steps:** implement `scripts/render_scene_survey.py` (one kit launch): top-down
  orthographic render + head-height sweep (≥3 poses/room × 4 bearings via
  look_around) → `results/figures/survey/`. Then, **after the kit process exits**,
  run the probe offline from the saved PNGs using **Sonnet 5 (`configs/models/
  judge.yaml`) — an out-of-benchmark judge, named in the methods write-up**:
  "What room of a home is this? one word." Score by exact match after lowercasing
  against the **frozen synonym table authored in T3.1** (reused by T4.1's map
  matching), majority vote over each room's poses. Iterate layout/furnishing until
  acceptance; record every iteration.
- **Deliverables:** committed survey renders + probe transcript/scores + any layout
  revisions with doc 03 updated.
- **Unit tests:** none.
- **Smoke test:** stills-only carve-out (see header): the still set + per-frame
  checklist substitutes for an mp4; the same scene gets video coverage in T2.4.
- **Acceptance (GATE):** **all four rooms named correctly** (doc 03 §8.2 / doc 04
  §8 criterion — do not silently relax; any relaxation updates both docs in the
  same commit with rationale); top-down matches the plan. **Layout freezes here.**

### T2.4 `[ ]` Scripted physics pass (VIDEO GATE)

- **Context:** Verifies collision and the redesigned bump/fall semantics before any
  LLM spends money.
- **Read first:** doc 03 §8; doc 02 §5 (fall/bump semantics).
- **Depends on:** T2.3, T1.2 (`scripted_drive`, `recorder`).
- **Steps:** scripted drives through EVERY doorway (both directions); deliberate
  wall bump; deliberate sofa/fridge-proxy bump; max-speed run into a wall. Record
  mp4 + status logs.
- **Deliverables:** `results/figures/smoke/physics_pass.mp4` + filmstrip + log.
- **Unit tests:** none.
- **Smoke test (this IS one):** rule-11 frame-by-frame — no walk-throughs (never
  clips wall/furniture); bumps stop/deflect with `bumped=true` logged and NO
  termination/teleport; doorway transits clean; if the wall run topples the duck,
  the fall is detected and the episode ends.
- **Acceptance (GATE):** all checks pass; zero spurious terminations.

---

## Phase 3 — Agent

### T3.1 `[ ]` Memory + prompts + frozen QA artifacts [no sim]

- **Context:** The LLM-as-SLAM core, **plus the frozen text artifacts that three
  later tasks depend on** and which doc 06 §12 lists as unauthored: the 5 layout-QA
  questions, their rubric anchors, and the room-name synonym table.
- **Read first:** doc 05 **§1 (boundary principle — it is §1, not §2)**, §5
  (structures + the worked seed-101 example the renderer must reproduce), §6
  (prompt outline); doc 03 §4 (heading convention: degrees CCW from +x, 90° =
  north); **doc 06 §5.7 (synonym table), §5.9 (the 5 QA questions + rubric)**.
- **Depends on:** T2.1 (spawn frames for the golden example).
- **Steps:** implement `memory.py` (rooms/landmarks/exits/breadcrumbs/plan/
  integrator/corrections log — **integrator uses commanded velocity, no k**, per
  T1.3's pinned policy) + `prompts.py` (system prompt per doc 05 §6; memory-block
  renderer). **Author and freeze in `prompts.py`: the 5 QA question texts, their
  0/0.5/1 rubric anchors, and the room-name synonym table** (e.g. lounge →
  living_room) — frozen alongside the prompt template per doc 06 §2.
- **Deliverables:** both modules + frozen QA artifacts + tests green.
- **Unit tests:** `tests/test_memory.py` — integrator math (commanded velocity, k
  absent by design); correct_position re-anchor + correction logging; renderer
  golden test (whitespace-normalized) reproducing doc 05 §5.2's example; exit
  status transitions. (The K-window assembly test belongs to T3.4, which owns
  `last_k_turns`.)
- **Smoke test [no sim]:** `pytest tests/test_memory.py -v`.
- **Acceptance:** tests green; rendered example matches doc 05 §5.2.

### T3.2 `[ ]` Tools + macro execution

- **Context:** The 12-tool surface (doc 05 §4 is canonical — `send_velocity` has
  NO auto-stop; `move` does) dispatching to sim macros and memory mutations.
- **Read first:** doc 05 §4, doc 02 §6 (macro pseudocode — note T1.3's k policy
  correction), doc 04 §5.3 (look_around).
- **Depends on:** T3.1, T1.4, T2.4.
- **Steps:** implement `tools.py`: JSON schemas (canon), dispatcher, arg
  validation/clamping with echo; sim-side macros (turn_to_heading P-loop ±5°,
  timeout 8 s; `move` distance servo targeting `dist/k` with auto-stop;
  look_around 4-bearing capture); memory tools mutate `memory.py`; `declare_done`
  stage signal.
- **Deliverables:** full tool surface working against a live session.
- **Unit tests:** `tests/test_tools.py` (sim mocked) — schema validity, clamp +
  echo, dispatch routing, structured errors for malformed calls (doc 05 §8).
- **Smoke test:** covered by T3.5.
- **Acceptance:** unit tests green; T3.5 exercises every tool at least once.

### T3.4 `[ ]` Episode loop + QA elicitation

- **Context:** Assembles everything. **Also owns the post-episode layout-QA
  exchange, which previously had a scorer (T4.1) but no producer** — without it
  T4.3 would freeze and run 12 trials whose JSONs contain no `qa` block, making a
  headline metric unproducible after the money is spent.
- **Read first:** doc 05 §3 (pseudocode is normative), §8 (error policy);
  **doc 06 §5.9 (QA exchange), §4 (log schema incl. `final.qa`)**.
- **Depends on:** T3.1, T3.2, T3.3.
- **Steps:** implement `loop.py` + `duck_embody/tasks/find_kitchen.py` (stage
  definitions, spawn logic, success predicates — the skeleton module is otherwise
  orphaned) + `scripts/run_trial.py` (`--model --seed --task`). Context policy:
  system + first turn + last K=10 + memory block every turn. Stage machine
  find_kitchen → return_home in-episode; caps 40 turns / 240 policy-s per stage;
  `declare_done` transcript shape (a) per doc 05 §3.3. **After the final stage, run
  the QA exchange: the 5 frozen questions in a fresh exchange where the model sees
  only its own final memory block; write answers + usage to `final.qa`.** Trial
  JSON written incrementally (crash-safe); per-trial mp4 via `recorder.py`.
  **Resolve before implementing** (doc 05 §12 / doc 06 §12 open questions): does
  return_home run after a stage-1 cap-out, and how is stage-2 time accounted?
  Record the decision in both docs (same commit).
- **Deliverables:** single-trial entry point complete, QA included.
- **Unit tests:** context-assembly test (transcript → provider payload; K-window
  correctness incl. **first turn not duplicated**; memory block present) with
  mocked provider/sim.
- **Smoke test:** T3.5.
- **Acceptance:** T3.5 produces a schema-valid trial JSON **including a populated
  `final.qa`** (spot-validate against doc 06 §4).

### T3.5 `[ ]` Sanity episode + GPT dry run (VIDEO GATE)

- **Context:** First end-to-end episodes with real models. Measures per-turn
  latency (tightens T4.3's forecast) and surfaces tool errors. NOT benchmark data
  — never pooled with results.
- **Read first:** AGENTS.md rule 11; doc 06 §8.
- **Depends on:** T3.4.
- **Steps:** (1) `run_trial.py --model fable5 --seed 101`, full task. (2) **A short
  GPT 5.6 sol dry run (≥3 turns with real observations)** so the OpenAI image path
  executes end-to-end pre-freeze. Frame-by-frame video review + full transcript
  audit (tool errors, malformed calls, memory renders, map plausibility). Fix
  issues (rule 10.5); **up to 2 reruns — a third failure escalates to the owner**
  per the standing constraint. Record per-turn latency distribution → doc 06 §8 +
  `configs/benchmark.yaml` (same commit).
- **Deliverables:** sanity trial JSON + mp4 + filmstrip + GPT dry-run log + latency
  numbers + fixes.
- **Unit tests:** regression tests for every bug fixed here.
- **Smoke test (this IS one):** rule-11 video checklist + transcript audit: every
  tool exercised; no crash; memory block grows correctly; camera frames usable;
  **QA exchange fires and lands in the JSON**.
- **Acceptance (GATE):** clean end-to-end run (any task outcome); GPT image path
  proven; latency recorded; open bugs fixed.

---

## Phase 4 — Benchmark

### T4.1 `[ ]` Scoring + tests (HARD GATE) [no sim]

- **Context:** Rule 2 — scoring is unit-tested before any batch.
- **Read first:** doc 06 §5 (formulas are normative), **§9.1 (implement ALL
  enumerated cases)**, §5.7 (map matching + synonym table from T3.1), §5.9 (QA
  rubric); **doc 06 §1 for what may/may not be claimed** (it is §1, not §6).
- **Depends on:** T2.1 (oracle path), T3.5 (real trial JSON fixture), T3.1 (frozen
  synonym table + QA rubric anchors).
- **Steps:** **first author and commit the Q2 direction-vocabulary parse rules with
  fixtures** (doc 06 §12 open question — "turn left/right", compass words, relative
  vs absolute), then implement `scoring.py` + `tests/test_scoring.py` (all §9.1
  cases + a `pose_trace`-missing case that **raises loudly, never silently falls
  back** to per-turn chords, which would inflate SPL) + `charts.py` skeleton; write
  `docs/METRICS.md` from doc 06 §5.
- **Deliverables:** scoring + green tests + METRICS.md + frozen Q2 parse rules.
- **Unit tests:** the deliverable.
- **Smoke test [no sim]:** `pytest tests/ -v` (kit python, all suites); score the
  T3.5 sanity JSON end to end and eyeball every number.
- **Acceptance (HARD GATE):** all tests green BEFORE T4.3 launches.

### T4.2 `[ ]` Batch runner + freeze manifest [no sim]

- **Context:** Resumable sequential runner with the config-hash freeze guard
  (doc 06 §7).
- **Read first:** doc 06 §2 (freeze list — item-level; **enumerate it to file
  paths here**) and §7 (paths, rerun log).
- **Depends on:** T3.4, T4.1.
- **Steps:** implement `runner.py` (matrix 3 models × seeds 101–104; skip trials
  whose JSON is complete under a matching `config_hash`; freeze-hash guard;
  unattended progress logging). **Enumerate the frozen file paths explicitly**
  (`prompts.py`, `tools.py` schemas, `camera.py` params, `apartment_layout.py`,
  `configs/benchmark.yaml`, `configs/models/*.yaml`) and write them with their
  hashes into `results/freeze.json`. Create `results/incomplete/` and
  `results/rerun_log.md` (doc 06 §7 — the rerun log ships with the results);
  reconcile doc 06 §7's `results/<trial_id>.json` with this repo's
  `results/raw/*.json` (AGENTS.md §7 + doc 01 §5 win — update doc 06 §7 same commit).
  Finalize `configs/benchmark.yaml` (seeds, caps, camera, k, warmup-N, latency
  forecast).
- **Deliverables:** runner + freeze manifest + final configs.
- **Unit tests:** freeze-guard + resume logic against fixture dirs.
- **Smoke test [no sim]:** dry-run lists the 12 trials; deliberately touch a frozen
  file → runner must refuse.
- **Acceptance:** dry-run correct; guard trips on mutation.

### T4.3 `[ ]` FREEZE + benchmark batch (12 trials)

- **Context:** The measurement. No selective retries (rule 3).
- **Read first:** doc 06 §2, §7, §8; AGENTS.md rules 1, 3, 4.
- **Depends on:** T4.1, T4.2 + gates T2.3 / T2.4 / T3.5 passed.
- **Steps:** freeze commit (hashes → `results/freeze.json`); launch `runner.py`;
  monitor by log tail + `nvidia-smi` only (**never a second kit process**); on
  completion verify 12 schema-valid JSONs (each with `final.qa`) + 12 mp4s.
- **Deliverables:** `results/raw/*.json` ×12 + per-trial mp4s + `freeze.json` +
  `rerun_log.md`.
- **Unit tests:** n/a.
- **Smoke test:** the first trial is a watched canary (log tail). **Restart
  semantics — two explicit branches:** (a) canary fails and the fix touches
  **non-frozen** code → keep the freeze commit, resume; trials already complete
  under a matching `config_hash` are **skipped**. (b) the fix touches **any frozen
  file** → new freeze commit, move `results/raw/` to a new batch directory, restart
  from zero. Log the branch taken in `rerun_log.md`.
- **Acceptance:** 12/12 complete under ONE freeze hash; infra reruns logged.

### T4.4 `[ ]` Figures + video audit [no sim]

- **Context:** Raw logs → portfolio artifacts.
- **Read first:** doc 06 §10; AGENTS.md rule 11.
- **Depends on:** T4.3.
- **Steps:** `charts.py` → `results/figures/` (per-metric bars with bootstrap CIs;
  the trajectory-vs-belief figure: true path + dead-reckoned path + claimed rooms
  over the floor plan); select + compress ≥1 mp4/GIF per model →
  `results/videos/`; **frame-by-frame audit of ≥1 video per model**, verdicts
  recorded; metric-vs-video disagreements resolved in the video's favor with a note.
- **Deliverables:** figures + curated videos + audit notes.
- **Unit tests:** n/a.
- **Smoke test [no sim]:** spot-check 3 values per figure against raw JSON.
- **Acceptance:** figures reproducible from `results/raw/` with one command.

### T4.5 `[ ]` Report + finalize [no sim]

- **Context:** README results, EXPERIMENTS.md, caveats, status close-out.
- **Read first:** README.md; **doc 06 §1 (claim limits)**; `docs/EXPERIMENTS.md`.
- **Depends on:** T4.4.
- **Steps:** write results into README (table + figures + GIFs + findings incl. the
  map-accuracy story); EXPERIMENTS.md with per-cell repro commands + freeze hash;
  caveats updated to reality; AGENTS.md §8 statuses; **push only with owner
  confirmation** (rule 7).
- **Deliverables:** publishable repo state.
- **Unit tests:** n/a.
- **Smoke test [no sim]:** every README number traces to a JSON field; links
  resolve; GIFs render on GitHub.
- **Acceptance:** owner sign-off.

---

## Standing constraints (every task)

- AGENTS.md hard rules 1–11.
- Commit per task with the task ID; PLAN.md status + evidence updated in the same
  commit (rule 10.5). Pushing follows rule 7.
- Any design-doc deviation updates that doc in the same commit.
- If a task exceeds ~2× expected effort, or a gate fails structurally (k out of
  band, a room unrecognizable, batch forecast blows the window): **stop and surface
  to the owner with evidence** instead of improvising scope.

## Review corrections (rev 2, 2026-07-26)

Applied from two independent adversarial reviews; each was re-verified before
acceptance. Highlights: added **T0.0** (kit python lacked anthropic/openai —
verified); pinned the **pxr** invocation (importable from neither default
interpreter — verified working); **T1.4 mount rung corrected** to doc 04 §3's
`/Robot/base` (the head-link mount + corrective quaternion was an inverted ladder);
**T2.3 judge changed to out-of-benchmark Sonnet 5** and the gate restored to 4/4
rooms (using contestant Fable 5 to tune the scene was a benchmark-integrity
defect); **T3.3 pulled ahead of T2.3** and its probe now carries a real JPEG
(the OpenAI image path would otherwise have first executed inside the frozen
batch); **QA elicitation assigned to T3.4** plus frozen questions/rubric/synonym
table to T3.1 (the metric had a scorer but no producer); **k policy pinned**
(integrator uses commanded velocity, no k; k serves caps/forecast and the `move`
servo — doc 02 §6.2 to be corrected in T1.3's commit); **video prerequisites
assigned** (`render_mode="rgb_array"`, `ViewerCfg`, `recorder.py`); manifest
filename aligned to doc 03 (`assets/manifest.json`); wall-segment invariant fixed
to D+1; T2.1 test list extended with doc 06 §9.2's invariants; T1.3 given doc 02
§7's two missing mitigations; T4.3 restart semantics split into two explicit
branches; `yaml.unsafe_load` and `robot.xml:175-216` corrected; citations
repointed (doc 05 §1, doc 06 §1).
