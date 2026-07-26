# Vendored locomotion policy — `v4_robust`

These artifacts are **copied, not referenced**. Duck Embody must be reproducible
from this repo alone; the batch can never depend on a training log directory that
may be moved, pruned, or overwritten (AGENTS.md rule 8, design doc 01 §6).

Vendored 2026-07-26 by PLAN task **T0.1**.

## Source

| | |
|---|---|
| Source run dir | `~/IsaacLab/logs/rsl_rl/open_duck_ppo_robust/2026-07-07_00-15-43/` |
| Parent repo | `Open_Duck_Mini_Jetson`, branch `v2`, commit `34f70fda182120369f954a4b1ccfa1edf58190ea` |
| Trained with | RSL-RL PPO (`rsl-rl-lib` 5.0.1) in Isaac Lab 2.3.2 on a DGX Spark |
| Training iterations | 3000 (`model_2999.pt` = iteration 2999, the final checkpoint) |

## Files

| File | Bytes | mtime | sha256 (first 16) |
|---|---|---|---|
| `model_2999.pt` | 4,748,341 | 2026-07-07 02:14 | `b1ebf3a5d7d866ef` |
| `params/agent.yaml` | 1,176 | 2026-07-07 00:19 | `f1cacdab6648495d` |
| `params/env.yaml` | 27,967 | 2026-07-07 00:19 | `4226471874ca94ef` |
| `exported/policy.onnx` | 14,558 | 2026-07-26 02:23 | `d084384de0485f18` |
| `exported/policy.onnx.data` | 787,968 | 2026-07-26 02:23 | `6dcf1025725ae6b4` |

Full digests: [`checksums.txt`](checksums.txt). Verify with
`cd policy && sha256sum -c checksums.txt`.

> **Note the ONNX mtimes.** `policy.onnx*` were re-exported on **2026-07-26**,
> nineteen days after the checkpoint was trained — they are not original training
> outputs. That is one more reason they are provenance-only (see below).

## Training configuration (from `params/`)

- **Algorithm** PPO, 5 learning epochs, 4 mini-batches, lr 1e-3 (adaptive
  schedule, desired KL 0.01), γ 0.97, λ 0.95, entropy coef 0.005, seed 42.
- **Networks** actor and critic are both MLPs, hidden dims `[512, 256, 128]`,
  ELU. Actor output: 16 joint targets.
- **Asymmetric actor/critic** `obs_groups: {actor: [policy], critic: [critic]}`
  — the actor sees 59 proprioceptive dims; the critic additionally sees true base
  linear velocity (62 dims). The real robot's BNO055 IMU cannot measure linear
  velocity, so the deployed network must not depend on it.
- **Observation normalization is baked into the checkpoint**
  (`actor.obs_normalization: true`, `agent.yaml:30`). RSL-RL's
  `get_inference_policy()` applies the learned running mean/std internally.
  **Never pre-normalize observations** — doing so double-normalizes and yields
  actions that look almost plausible while being wrong (doc 02 §2).
- **Control rate** `sim.dt = 0.005 s` × `decimation = 4` → **50 Hz**, so
  `duration_s → N = round(50 × duration_s)` control steps.
- **Command hull (training ranges, verified in `params/env.yaml`)**
  `vx ∈ (−0.148, 0.222)` m/s, `vy ∈ (−0.111, 0.111)` m/s,
  `wz ∈ (−0.5, 0.5)` rad/s. Every command the LLM issues is clamped to this hull.

### The 59-dim actor observation, in order

`base_ang_vel(3)`, `projected_gravity(3)`, `velocity_commands(3)`,
`joint_pos(16)`, `joint_vel(16)`, `actions(16)`, `gait_phase(2)` = **59**.
No `base_lin_vel`. Confirmed against the vendored `env.yaml` by
`scripts/smoke_policy_artifacts.py`.

> **`gait_phase` trap.** Dims 58–59 are produced by a function that reads the
> `ImitationReward` *reward* term through a module-global registry and returns
> **zeros** if no instance is registered. Disabling the reward manager therefore
> silently zeroes 2 of 59 input dims and degrades gait with no error. The embody
> env config keeps the full reward manager computing every step; the reward
> values are discarded (doc 02 §2).

## Evaluation record

Under the parent project's evaluation protocol — 3,200 episodes
(5 command conditions × 10 windows × 64 envs, 30 s each, deterministic, seed 42):

| Metric | v4_robust |
|---|---|
| Gait-validity gate | **5/5** |
| Fall rate | **0.00 %** (0.0 in every one of the 5 conditions) |
| Reference RMS | 4.49° |
| v_xy error | 0.153 m/s |
| wz error | 0.067 rad/s |

Source: parent repo `docs/jetson-mod/v4_comparison.md`, `v4_robust` row (line 62)
and the per-condition fall table (line 70).

> **What that record does not cover.** Those windows are push-free, flat-ground,
> and hold a *fixed* command for 30 s, and the velocity metric is an
> *instantaneous* L2 error — position was never integrated, so **net displacement
> was never measured**. Duck Embody's macros switch commands every 0.2 s and hold
> for up to 240 policy-seconds in a furnished apartment. PLAN T1.3
> (`scripts/smoke_displacement.py`) measures real achieved displacement, long-hold
> yaw creep, and step-change stability before any cap or scoring constant is
> frozen (doc 02 §7).

## Inference path

In-sim playback loads **`model_2999.pt`** through RSL-RL's `OnPolicyRunner` +
`get_inference_policy()` (deterministic mean actions), stepping under
`torch.no_grad()` — never `torch.inference_mode()`, which poisons lazily-created
sim tensors and makes a later `env.reset()` crash (doc 02 §3).

**`exported/policy.onnx` is vendored for provenance only and is NOT used.**
It is fixed-batch `[1, 59]`, parity against the checkpoint is unverified, and it
was re-exported after training (see mtimes above).

## Verification

`scripts/smoke_policy_artifacts.py` (T0.1 smoke test) re-checks every claim above
that is machine-checkable — all checksums, the `(1, 59)` normalizer and
`(512, 59)` / `(16, 128)` actor shapes, iteration 2999, absence of
`base_lin_vel` from the policy obs group and its presence in the critic group,
dt/decimation, and the normalization flags. Run it with the kit python:

```bash
PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_policy_artifacts.py
```
