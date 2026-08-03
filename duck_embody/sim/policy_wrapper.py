"""Locomotion-policy playback and velocity-command injection.

Wraps the frozen ``v4_robust`` PPO checkpoint so the rest of the harness can say
"walk at 0.2 m/s for 3 seconds" and get back what actually happened. The loading
path is byte-for-byte the pattern the parent repo already validated over 3,200
evaluation episodes (``scripts/evaluate_policies.py:1205-1255``); novelty here
would buy only risk.

Two subtleties that are not obvious from the code alone:

* **``torch.no_grad()``, never ``torch.inference_mode()``.** Stepping the env
  inside ``inference_mode`` marks lazily-created sim-state tensors as inference
  tensors, and the next out-of-scope ``env.reset()`` — which we do between
  trials — dies with *"Inplace update to inference tensor outside InferenceMode"*.

* **Observations are fed raw.** Normalisation is baked into the checkpoint and
  applied inside ``get_inference_policy()``. Normalising upstream would
  double-normalise and produce actions that look almost plausible.

The pure math here (clamping, duration→steps, heading wrap) lives in module-level
functions with no Isaac dependency so ``tests/test_wrapper_math.py`` can exercise
it without a kit process.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# --- Constants -------------------------------------------------------------

#: Training command hull, read from policy/params/env.yaml (T0.1 verified).
#: Commands outside it are not just unwise, they are outside the reference-motion
#: library the gait was trained against.
VX_RANGE = (-0.148, 0.222)
VY_RANGE = (-0.111, 0.111)
WZ_RANGE = (-0.5, 0.5)

#: 50 Hz control = sim dt 0.005 s x decimation 4 (parent env_cfg.py:189-190).
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

#: True base XY is sampled into `pose_trace` every this many control steps.
#: 10 steps = 5 Hz. Scoring integrates this for SPL path length; sampling only
#: once per turn would miss within-turn curvature and inflate SPL (doc 06 §5.3).
POSE_TRACE_EVERY = 10

#: Contact force (N) on any NON-FOOT body that counts as a bump, and how many
#: consecutive control steps must exceed it. Debouncing matters: a single
#: grazing spike while squeezing through a 0.35 m doorway must not read as a
#: collision (doc 02 §6.2).
#:
#: TUNED IN T2.4 against the real apartment, as this constant always promised.
#: The bodies are the change that mattered — see PolicyPlayback.__init__; the
#: 1.0 N threshold survived measurement. Real contacts land at 28-499 N, two
#: orders of magnitude above it, while free walking leaves every non-foot body
#: under 1 N (3 runs x 60+ steps, scripts/debug_bump_bodies.py). There is no
#: near-threshold regime to tune into: the gap is the whole point.
BUMP_FORCE_N = 1.0
BUMP_DEBOUNCE_STEPS = 3
#: Contact must remain continuously confirmed for 0.4 s before a locomotion
#: macro treats it as blocking. Shorter contacts are grazes: they are reported,
#: but neither a turn nor a drive is aborted. Release uses the same duration so
#: gait-cycle force troughs cannot split one physical contact into many events.
CONTACT_SUSTAINED_S = 0.4
CONTACT_SUSTAINED_STEPS = round(CONTACT_SUSTAINED_S * CONTROL_HZ)
CONTACT_STATES = (
    "free",
    "candidate_contact",
    "sustained_contact",
    "candidate_release",
)

#: Fall thresholds, MIRRORED from DuckEmbodyEnvCfg (doc 02 §5). Duplicated
#: rather than imported because embody_env_cfg pulls in the parent repo and
#: needs a running kit app, while this module's pure half must stay importable
#: for the unit tests. `tests/test_wrapper_math.py` asserts the two agree, so
#: the copy cannot drift — the failure it would otherwise cause is a fall
#: report whose stated threshold is not the one that actually fired.
FALL_MIN_HEIGHT_M = 0.09
FALL_TILT_LIMIT_DEG = 60.0

# --- Motion macros (doc 02 §6) ---------------------------------------------
# These live in the playback layer, not the tool layer: doc 02 owns the macros
# and `tools.py` only wires tool schemas to them. Putting them here means the
# T2.4 physics pass and the LLM drive the *same* code.

#: Commanded forward speed for `move` (doc 02 §6.2).
MOVE_SPEED_MPS = 0.2
#: Reverse walking is deliberately slower than forward walking and remains
#: inside the asymmetric trained hull (whose reverse edge is -0.148 m/s).
REVERSE_MOVE_SPEED_MPS = 0.10
#: Per-call distance cap, so one tool call cannot cross the apartment blind.
MOVE_MAX_DISTANCE_M = 1.5
#: Servo/correction interval. 0.2 s = 10 control steps.
MACRO_CHUNK_S = 0.2
#: Extra time allowed before a macro gives up, as a multiple of the ideal.
MACRO_TIME_MARGIN = 1.6

#: P gain on heading error (radians) -> wz. Saturates the +/-0.5 rad/s hull at
#: ~19 deg of error. Mirrors Isaac Lab's own heading controller structure.
KP_HEADING = 1.5
TURN_TOLERANCE_DEG = 5.0
TURN_TIMEOUT_S = 8.0

# --- Simulated leg odometry (2026-07-30 redesign) ---------------------------
#: The dead-reckoning integrator no longer consumes COMMANDED velocity. It
#: consumes simulated LEG ODOMETRY: the robot's true per-call displacement with
#: a seeded error model. Rationale, measured the hard way:
#:
#:   * Commanded-velocity reckoning credited motion to a wedged robot. Batch
#:     trial fable5_seed101: 49 send_velocity calls credited 27.09 m against
#:     1.99 m of true displacement — the belief left the building (26.65 m off
#:     in a 4.8 x 3.6 m flat) and ~95% of it was this accounting, not drift.
#:   * The contact-time discount that tried to fix it is POLICY-DEPENDENT and
#:     was measured ineffective: v4 bounces off furniture (contact force above
#:     1 N on only 6.9% of wedged steps — an impulse train, nothing to latch),
#:     while v5d leans in (100% duty once pressed). A crediting rule that works
#:     for one gait fails for the other.
#:
#: Leg odometry is what the REAL robot can honestly compute: 14 joint encoders
#: + foot contact switches -> stance-leg forward kinematics. Published legged
#: odometry lands at a few percent of distance travelled; the constants below
#: sit in that band. The sim models the OUTPUT of that stack (true displacement
#: x seeded noise) rather than re-deriving FK, which is a recorded deviation,
#: not a secret: the estimate the model sees still drifts a few percent of
#: distance — enough that loop closure genuinely matters against the 0.35 m
#: success radius — but a robot that does not move no longer believes it moved.
#: Heading remains the absolute compass (BNO055), unchanged.
ODOM_SCALE_STD = 0.04          #: per-trial systematic scale error (std dev)
ODOM_SCALE_CLIP = (0.90, 1.10)
ODOM_NOISE_FRAC = 0.03         #: white noise at 1 m travelled (per-axis std dev)
#: Standing/slipping odometry error per SECOND, not per call. It MUST be a rate:
#: recording used to slice every command into 0.04 s pieces whenever video was
#: on — which was every batch trial — and merge_exec_results SUMS per-piece
#: contributions. A per-CALL floor therefore accrued ~25x/s: a wedged 3 s
#: command reported 0.094 m recorded vs 0.0013 m unrecorded, i.e. 4.6 m of
#: phantom distance over the 49-call wedge episode this redesign exists to fix,
#: and it made the model-facing distance depend on whether video was being
#: captured. TR.3 removed the slicing itself (recording is now observational),
#: but the rate form is kept and is now the honest one: the noise is generated
#: per CONTROL STEP, so a rate is the only dimensionally correct spelling.
ODOM_NOISE_FLOOR_RATE_MPS = 0.001

#: TR.3 (2026-08-02): the odometry noise process is defined by its VARIANCE
#: rates, not by a per-call sigma. This is the whole fix for forensics F-03.
#:
#: The old model drew ONE Gaussian per `execute()` call with
#: `sigma = ODOM_NOISE_FRAC * call_distance + ODOM_NOISE_FLOOR_RATE_MPS * call_seconds`.
#: Sigma is additive there, and sigma does not add — variance does. Splitting a
#: command into N pieces gave each piece ~sigma/N, and vector-summing N
#: independent draws yields `sqrt(N) * sigma/N = sigma/sqrt(N)`: the noise
#: process shrank by ~sqrt(N) purely because video was attached (N ~ 75 for a
#: 3 s command → an 8.7x quieter sensor on the paid path than on the path the
#: unit tests and the first odometry smoke exercised).
#:
#: Now noise is drawn once per CONTROL STEP with additive variance
#:
#:     variance_per_axis = ODOM_VAR_PER_M * step_distance
#:                       + ODOM_VAR_PER_S * CONTROL_DT
#:
#: Independent per-step draws sum to a total variance of
#: `ODOM_VAR_PER_M * total_distance + ODOM_VAR_PER_S * total_time`, which
#: depends only on what the robot DID — never on how many external calls the
#: steps were partitioned into. That is the invariance
#: `tests/test_wrapper_math.py::TestOdometryIsChunkInvariant` pins at
#: 1 / 5 / 75 calls over one fixed step sequence.
#:
#: Calibration: the rates are the squares of the legacy constants, so at 1 m
#: travelled the per-axis sigma is unchanged at ODOM_NOISE_FRAC (3 cm). Beyond
#: 1 m the error now grows as sqrt(distance) rather than linearly — which is
#: what an accumulation of independent per-step measurement errors actually
#: does (a random walk), and is the published shape for legged odometry.
ODOM_VAR_PER_M = ODOM_NOISE_FRAC ** 2              #: m^2 of variance per metre
ODOM_VAR_PER_S = ODOM_NOISE_FLOOR_RATE_MPS ** 2    #: m^2 of variance per second

#: RECORDED DEVIATION, carried forward unchanged (AGENTS.md rule 5). There is
#: **no slip term**: a robot wedged against furniture measures ~0 m of motion
#: and therefore *knows* it did not move. Real leg odometry with slipping feet
#: would integrate some phantom forward motion, so the simulated duck's
#: certainty while blocked is optimistic. Left deliberately — the owner's
#: acceptance criterion for the 2026-07-30 redesign was that serious drift
#: during collision no longer exist, and a slip term reintroduces exactly that
#: error class. TR.3 does not revisit it: this task changes only WHERE the
#: noise is generated (per control step, additive variance), not the physical
#: content of the model. Revisit only if the benchmark ever claims to
#: characterise odometry quality itself.

#: MEASURED velocity realisation factor (T1.3): net displacement / commanded.
#: Used only for timeout forecasting. Target completion is decided from
#: accumulated leg odometry, never this calibration factor.
#:
#: THIS IS A PROPERTY OF THE POLICY, NOT OF THE HARNESS. Re-measure it with
#: `scripts/smoke_displacement.py --checkpoint <policy>` and re-derive with
#: `scripts/derive_calibration.py` whenever the shipped policy changes; the
#: v4-vs-v5d gap is 4.3 %, which is ~16 cm over a 4 m path against a 0.35 m
#: success radius. Note that configs/benchmark.yaml's `locomotion:` block is
#: DOCUMENTATION ONLY — no Python reads it (verified 2026-07-29: zero hits for
#: the yaml key outside the config, this file's docstring and the derivation
#: script), so editing the yaml alone changes nothing at runtime. This line is
#: the one that matters, and tests/test_tools.py pins the two behaviours that
#: depend on it.
#:
#: 2026-07-29: 1.004 (v4_robust) -> 0.9617 (v5d_contact_wrench), measured
#: 3.8470 m achieved / 4.0 m commanded, with the v4 control reproducing the old
#: value to 1.0044 in the same session.
K_VELOCITY_REALISATION = 0.9617


def move_servo_plan(distance_m: float) -> tuple[float, float, int]:
    """Forward timeout plan: (clamped request, k-adjusted forecast, chunks).

    Extracted from ``PolicyPlayback.move`` on 2026-07-29 because the arithmetic
    existed in TWO places — here and re-implemented inside the test suite's
    ``FakePlayback.move`` — with nothing keeping them in step. That was proved
    the hard way: mutating the real ``move`` (``ceil`` -> ``floor``, and then
    deleting the ``/ k`` entirely) left all 547 ``tests/test_tools.py`` tests
    GREEN, because those tests drive the fake. The real servo was unguarded.

    Being a pure function, this can be asserted on directly, so a mutation to
    the arithmetic now fails a test instead of shipping. ``FakePlayback`` is
    pinned against it by an agreement test.

    ``k`` is consumed only to forecast a generous timeout. The second item
    retains the historical ``distance / k`` contract used by wall-clock
    planning and test doubles, but it must never decide whether the target was
    reached: :meth:`PolicyPlayback.move` closes that loop exclusively on
    accumulated leg odometry.
    """
    distance = max(0.0, min(distance_m, MOVE_MAX_DISTANCE_M))
    target_dist = distance / K_VELOCITY_REALISATION
    ideal_s = target_dist / MOVE_SPEED_MPS if MOVE_SPEED_MPS else 0.0
    n_chunks = max(1, int(math.ceil(ideal_s * MACRO_TIME_MARGIN / MACRO_CHUNK_S)))
    return distance, target_dist, n_chunks


def clamp_command(
    vx: float, vy: float, wz: float
) -> tuple[tuple[float, float, float], list[str]]:
    """Clamp to the training hull. Returns the clamped triple and any notes.

    The notes are echoed back to the model so clamping is visible rather than
    silent — a model that asks for 0.5 m/s should learn it did not get it.
    """
    notes: list[str] = []
    out = []
    for name, value, (lo, hi) in (
        ("vx", vx, VX_RANGE),
        ("vy", vy, VY_RANGE),
        ("wz", wz, WZ_RANGE),
    ):
        clamped = min(max(value, lo), hi)
        if clamped != value:
            notes.append(f"{name} {value:+.3f} clamped to {clamped:+.3f} (hull [{lo}, {hi}])")
        out.append(clamped)
    return (out[0], out[1], out[2]), notes


def duration_to_steps(duration_s: float) -> int:
    """``duration_s`` -> control steps at 50 Hz, minimum 1."""
    return max(1, round(duration_s * CONTROL_HZ))


def wrap_deg(angle_deg: float) -> float:
    """Wrap to [0, 360)."""
    return angle_deg % 360.0


def shortest_angle_diff_deg(target_deg: float, current_deg: float) -> float:
    """Signed smallest rotation from ``current`` to ``target``, in **[-180, 180)**.

    Note the half-open interval: an exact 180° error returns ``-180.0``, not
    ``+180.0``. Either direction is equally short at half a turn, so the choice
    is arbitrary — but it is deterministic, which is what the ``turn_to_heading``
    P-loop needs (an implementation that flipped sign at the boundary could
    dither there forever).
    """
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class StepObservation:
    """One control step, as seen by a passive observer (TR.3, forensics F-03).

    The point of this type is that **observing must not change execution**.
    Before TR.3 the only way to interleave anything with stepping — a viewport
    grab, for video — was ``attach_recorder`` replacing ``playback.execute``
    with a wrapper that re-entered it in 0.04 s pieces. That moved the command
    boundary, and the command boundary is load-bearing: it carried the bump
    debounce window, the pose-trace phase, the clamp-note list, the fall
    diagnostics stamp and the odometry noise draw. Every one of those has
    already been the subject of a separate bug fix (see this module's history
    and ``merge_exec_results``), which is the tell that the seam was in the
    wrong place. A recorded run and an unrecorded run were different
    experiments, and the paid batch only ever ran the recorded one.

    So consumers now get told what happened and cannot influence it: an
    observer is called once per non-teleported control step and its return
    value is ignored.

    ``true_pose`` is **AUDIT ONLY** — the same side of doc 05 §1's boundary as
    ``ExecResult.true_pose``. It exists so a recorder/auditor can annotate a
    frame with ground truth; anything on a model-facing path that reads it is a
    benchmark-invalidating leak.
    """

    #: Trial-scoped control-step index (``PolicyPlayback._step_counter``), so
    #: an observer's sampling grid is independent of call boundaries — which is
    #: the whole point.
    step_index: int
    #: True on the step whose ``env.step()`` terminated the episode. Isaac Lab
    #: auto-resets a terminated env INSIDE ``step()``, so live scene state is
    #: already the teleported spawn pose: a recorder MUST NOT grab a frame for
    #: this step (it would end every fall video on a healthy duck at spawn) and
    #: ``true_pose`` below is the PRE-step snapshot.
    terminated: bool
    #: (x, y, heading_deg). AUDIT ONLY.
    true_pose: tuple[float, float, float]
    #: Peak contact force over the non-foot bodies at this step, newtons.
    contact_force_n: float
    #: Body regions above threshold at this step, first-seen order. Empty
    #: unless the debounce had already confirmed contact — this mirrors what
    #: ``execute()`` itself sampled, so an observer sees no extra GPU reads.
    contact_groups: tuple[str, ...]
    #: Latched contact state (post-debounce, pre-release-hysteresis) — the same
    #: flag that charges ``ExecResult.contact_steps``. T4's contact state
    #: machine extends this without touching ``execute()``'s loop.
    in_contact: bool


@dataclass
class ExecResult:
    """What one command execution actually did."""

    commanded: tuple[float, float, float]
    duration_s: float
    steps: int
    policy_seconds: float
    bumped: bool
    fell: bool
    #: Control steps spent in CONFIRMED sustained contact (past the
    #: BUMP_DEBOUNCE_STEPS debounce). Proprioception, not ground truth — a real
    #: duck with foot switches and a torso bump sensor knows this too, so it is
    #: on the same side of the observability boundary as `contact_groups`.
    #:
    #: Exists because dead reckoning was crediting commanded motion to a robot
    #: that was wedged: measured on results/raw_v5d/fable5_seed101.json, 49
    #: `send_velocity` calls reported 27.09 m travelled against 1.99 m of true
    #: displacement, and the worst single calls credited 0.60 m for 0.01 m of
    #: real motion while `bumped=True` for their whole 3 s. That single tool
    #: accounted for 25.10 m of a 26.65 m position error — i.e. ~95% of what
    #: looked like "drift" was this accounting bug, not odometry.
    contact_steps: int = 0
    #: Simulated leg-odometry displacement for THIS call, world frame, noise
    #: applied. What the dead-reckoning integrator consumes. (0, 0) while
    #: wedged, because odometry measures motion, not intent.
    odom_dxy: tuple[float, float] = (0.0, 0.0)
    #: Path length of the same measurement (for `distance_moved_m` reporting).
    odom_distance_m: float = 0.0
    #: True base XY sampled at 5 Hz during the motion, bracketed by the exact
    #: start and end poses. SCORING ONLY — never shown to the model (doc 06 §4).
    pose_trace: list[tuple[float, float]] = field(default_factory=list)
    #: Just the periodic 5 Hz samples, without the start/end bookends. Callers
    #: that stitch several executions together (session._execute_recording)
    #: must merge THIS and add bookends once, or every chunk boundary
    #: contributes two extra near-duplicate points at the full 50 Hz step rate.
    sampled_xy: list[tuple[float, float]] = field(default_factory=list)
    #: True pose at the end, scoring only.
    true_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: Straight-line true displacement over this execution, scoring only.
    true_displacement_m: float = 0.0
    clamp_notes: list[str] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str = ""
    #: Which body regions were in contact when `bumped` went true — one or more
    #: of head / torso / left_leg / right_leg. Proprioception, shown to the
    #: model: it refines the `bumped` boolean without revealing WHAT was hit.
    contact_groups: list[str] = field(default_factory=list)
    #: Why the trial ended, captured at the terminating step (height, tilt,
    #: which term fired, and the command in flight). None unless this call
    #: terminated. SCORING/AUDIT ONLY — never shown to the model.
    fall_diagnostics: dict | None = None
    #: Distance the DEAD-RECKONING integrator believes was covered. This is what
    #: the model is told; `true_displacement_m` above is scoring-only and never
    #: shown. Set by `move`; the gap between them is the drift being measured.
    dead_reckoned_distance_m: float = 0.0
    #: Signed distance requested from ``move`` after the per-call clamp. Zero
    #: for non-move commands.
    requested_distance_m: float = 0.0
    #: Accumulated leg-odometry progress used by the move servo. This excludes
    #: the trailing settle chunk, which cannot make a target become reached.
    measured_distance_m: float = 0.0
    #: True only when the relevant closed-loop target was actually reached.
    #: Contact/fall/budget stops win even if the final chunk crossed the target.
    target_reached: bool = False
    #: Persistent contact-machine state at the end of this result.
    contact_state: str = "free"
    #: Monotonic trial-scoped ID of the latest sustained contact event.
    contact_event_id: int | None = None
    contact_onset_step: int | None = None
    contact_release_step: int | None = None
    contact_event_regions: list[str] = field(default_factory=list)
    #: Ordered summaries for compound macros such as ``turn_and_move``.
    phase_results: list[dict] = field(default_factory=list)


def merge_exec_results(total: "ExecResult | None", part: "ExecResult") -> "ExecResult":
    """Fold one more chunk's ``ExecResult`` into a running total. THE merge.

    This is the single merge implementation for BOTH stitching layers — the
    macros' 0.2 s servo chunks (:meth:`PolicyPlayback._merge` delegates here)
    and the recorder's 0.04 s video chunks (``recorder.chunked_execute``).
    They used to be two hand-mirrored copies, and the copies drifted: the
    recorder's dropped ``contact_groups`` and ``fall_diagnostics`` entirely, so
    every video-recorded run — the default and the rule-11-mandatory batch path
    — reported ``bumped: true, contact: []`` to the model and ``fell: true``
    with no diagnostics to the audit log whenever the confirming chunk was not
    the first (the COMMON case: BUMP_DEBOUNCE_STEPS=3 exceeds a 2-step chunk).
    One function, so the two paths structurally cannot disagree again.

    Field rules that are policy, not plumbing:

    * ``contact_groups`` is the UNION over chunks, preserving first-seen order
      — never last-wins. A drive that catches the head on a shelf and then
      scrapes the torso felt both, and the earlier region is exactly the one a
      last-wins merge silently destroyed. Matches ``execute()``'s own
      within-call accumulation, so all three layers agree.
    * ``fall_diagnostics`` belongs to the chunk that terminated; a later None
      never overwrites a captured one. Its ``policy_seconds_into_call`` is
      re-stamped with the ACCUMULATED seconds into this call: ``execute()`` can
      only know its own chunk, which bounded every recorded fall at 0.04 s
      regardless of how deep into the command it happened.
    * ``clamp_notes`` extend WITHOUT duplicates. Every 0.04 s recording chunk
      of one out-of-hull command carries the identical note; extending blindly
      would echo it ~75 times per 3 s command, turning the model-facing
      ``notes`` key from a signal into noise. (Macro chunks pre-clamp and
      carry no notes, so this changes nothing for them.)
    * ``duration_s`` and ``commanded`` are deliberately NOT merged and stay the
      first chunk's values — stale by design; never read them off a merged
      result (AGENTS.md §5).
    """
    if total is None:
        return part
    total.steps += part.steps
    total.policy_seconds += part.policy_seconds
    total.bumped = total.bumped or part.bumped
    # Contact steps SUM across chunks, like steps/policy_seconds and unlike the
    # boolean. A drive that scrapes through eight 0.2 s chunks accumulated eight
    # chunks' worth of contact, and the reported distance is computed from it.
    total.contact_steps += part.contact_steps
    total.odom_dxy = (
        total.odom_dxy[0] + part.odom_dxy[0],
        total.odom_dxy[1] + part.odom_dxy[1],
    )
    # NET displacement of the accumulated vector — NOT a sum of per-part
    # magnitudes. Magnitude-sum is not chunking-invariant: the recorder slices
    # a command into 0.04 s pieces (every batch trial), and a duck vibrating
    # against furniture has a small true displacement in a RANDOM direction
    # each slice, so the magnitudes accumulate as path length. Measured on real
    # physics: a wedged 3 s command reported 0.72 m of "travel" for 0.09 m of
    # net motion, purely from summed jitter. The vector sum cancels that
    # exactly, at any slicing. Semantics the model needs from
    # `distance_moved_m` is "how far did I actually get", which is net
    # displacement; a there-and-back excursion honestly reports ~0.
    total.odom_distance_m = math.hypot(*total.odom_dxy)
    for group in part.contact_groups:
        if group not in total.contact_groups:
            total.contact_groups.append(group)
    if part.fall_diagnostics is not None:
        total.fall_diagnostics = part.fall_diagnostics
        # AFTER the policy_seconds accumulation above, so the stamp covers
        # every chunk of this call including the terminating one (G9): the
        # chunk-local figure said every recorded fall happened <= 0.04 s in.
        total.fall_diagnostics["policy_seconds_into_call"] = round(
            total.policy_seconds, 3
        )
    total.fell = total.fell or part.fell
    total.sampled_xy.extend(part.sampled_xy)
    total.true_pose = part.true_pose
    total.stopped_early = part.stopped_early
    total.stop_reason = part.stop_reason or total.stop_reason
    total.contact_state = part.contact_state
    if part.contact_event_id is not None:
        total.contact_event_id = part.contact_event_id
        total.contact_onset_step = part.contact_onset_step
        total.contact_release_step = part.contact_release_step
        total.contact_event_regions = list(part.contact_event_regions)
    total.phase_results.extend(part.phase_results)
    for note in part.clamp_notes:
        if note not in total.clamp_notes:
            total.clamp_notes.append(note)
    return total


class PolicyPlayback:
    """Loads ``model_2999.pt`` and drives the env under injected commands."""

    #: Passive per-control-step hook, ``Callable[[StepObservation], None] | None``
    #: (TR.3). Declared at CLASS level, not only in ``__init__``, so the many
    #: ``PolicyPlayback.__new__(PolicyPlayback)`` test doubles in the suite
    #: inherit a working default instead of raising ``AttributeError`` from
    #: inside the step loop. Register through :meth:`register_step_observer`,
    #: which refuses to silently displace an existing one.
    step_observer = None

    def __init__(self, gym_env, task_id: str, checkpoint_path: str, device: str | None = None):
        # Imported here, not at module scope: these require a running kit app,
        # while the pure functions above must stay importable for unit tests.
        import importlib.metadata as metadata

        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rsl_rl.runners import OnPolicyRunner

        self._torch = torch
        self.gym_env = gym_env
        self.base_env = gym_env.unwrapped

        agent_cfg = load_cfg_from_registry(task_id.split(":")[-1], "rsl_rl_cfg_entry_point")
        # REQUIRED with rsl-rl-lib 5.x: the config keys were renamed, and
        # OnPolicyRunner otherwise dies with KeyError: 'class_name'. T0.1
        # confirmed the vendored agent.yaml uses the new schema.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        if device is not None:
            agent_cfg.device = device

        self.env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        self.runner = OnPolicyRunner(
            self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        self.runner.load(checkpoint_path)
        # Deterministic mean actions, with the baked-in normaliser applied.
        self.policy = self.runner.get_inference_policy(device=self.base_env.device)

        self._obs = None
        self._fell = False
        self.step_observer = None
        # Bump debounce state lives on the INSTANCE, not inside execute().
        # Recording used to chunk a long command into 0.04 s (2 control step)
        # pieces so it could grab a video frame between them; a per-call
        # counter could never reach BUMP_DEBOUNCE_STEPS=3 inside a 2-step
        # chunk, so bumps would have been undetectable in exactly the runs that
        # record video — including T2.4's physics gate. TR.3 removed the
        # chunking (recording observes steps now), so this state is no longer
        # load-bearing for correctness across a recorded/unrecorded split — but
        # it stays instance-scoped: debounce and release hysteresis are
        # properties of the CONTACT, not of the tool call that noticed it, and
        # the macros still stitch 0.2 s servo chunks.
        # Leg-odometry error model: seeded per trial in reset(seed=...), so a
        # trial's odometry is reproducible from its seed alone.
        self._odom_rng = random.Random(0)
        self._odom_scale = 1.0
        self._last_seed = None
        self._bump_run = 0
        self._contact_state = "free"
        self._contact_candidate_onset_step: int | None = None
        self._contact_clear_run = 0
        self._contact_candidate_regions: list[str] = []
        self._contact_event_id_counter = 0
        self._contact_event_id: int | None = None
        self._contact_event_onset_step: int | None = None
        self._contact_event_release_step: int | None = None
        self._contact_event_regions: list[str] = []
        self._last_contact_event: dict | None = None
        self._fall_diagnostics: dict | None = None
        # Likewise a persistent control-step counter for pose_trace sampling.
        # doc 06 §5.3 pins that trace to 5 Hz; a per-call index would restart at
        # 0 in every 2-step recording chunk, fire `step % 10 == 0` on the first
        # step of each, and sample at ~50 Hz instead. The extra points are pure
        # per-step gait sway, which inflates the SPL path integral and would
        # have quietly depressed every recorded trial's SPL.
        self._step_counter = 0

        self.command_term = self.base_env.command_manager.get_term("base_velocity")
        self._defuse_command_term()

        self._contact_sensor = self.base_env.scene.sensors["contact_forces"]
        # Bump = contact on any body that is NOT a foot. The feet are excluded
        # because they carry the robot: they read 80-200 N continuously against
        # the floor, so including them would report a permanent bump.
        #
        # NOT trunk-only, which is what doc 02 §6.2 originally specified and
        # what T2.4 MEASURED to be wrong. scripts/debug_bump_bodies.py logged
        # per-body forces while driving at three obstacle classes:
        #   sofa (0.42 m seat) -> trunk_assembly, 499 N, step 75
        #   fridge proxy       -> head_assembly,   40 N, step 62   <- trunk never
        #   wall A  (0.7 m)    -> head_assembly,  115 N, step 249  <- trunk never
        # The duck's head leads at its own height, so a trunk-only test is blind
        # to walls — the most common obstacle in the apartment. The failure mode
        # that produced was silent: the model drove into a wall, was told
        # `bumped=false`, kept pushing, and eventually toppled, ending the trial
        # with no collision ever reported.
        all_ids = list(range(len(self._contact_sensor.body_names)))
        foot_ids, foot_names = self._contact_sensor.find_bodies(".*foot.*")
        self._bump_body_ids = [i for i in all_ids if i not in set(foot_ids)]
        self._bump_body_names = [
            self._contact_sensor.body_names[i] for i in self._bump_body_ids
        ]
        self._foot_body_names = foot_names

        # Contact grouped by kinematic region, so a bump can say WHERE it was
        # felt rather than merely that it happened.
        #
        # Grouped by position in the articulation tree, NOT by name suffix: the
        # names lie. `knee_and_ankle_assembly_2` sits under
        # `left_roll_to_pitch_assembly` and is a LEFT-leg body, while `_3`/`_4`
        # sit under `right_roll_to_pitch_assembly`. A suffix heuristic gets that
        # exactly backwards, which is how a "left/right" report would have been
        # confidently wrong.
        #
        # The tree, from the T2.4 sensor dump:
        #   0-2   base / trunk_assembly / trunk          -> torso
        #   3-8   hip_roll_assembly .. left_foot         -> left leg
        #   9-15  neck_* / head_assembly / head / antennae -> head
        #   16-21 hip_roll_assembly_2 .. right_foot      -> right leg
        self._contact_groups: dict[str, list[int]] = {"head": [], "torso": [],
                                                      "left_leg": [], "right_leg": []}
        seen_right_hip = False
        for idx, body in enumerate(self._contact_sensor.body_names):
            low = body.lower()
            if "hip_roll_assembly_2" in low or "right_roll_to_pitch" in low:
                seen_right_hip = True
            if any(k in low for k in ("head", "neck", "antenna")):
                group = "head"
            elif any(k in low for k in ("hip_roll", "roll_to_pitch", "knee", "ankle", "foot")):
                group = "right_leg" if seen_right_hip else "left_leg"
            else:
                group = "torso"
            if idx in set(self._bump_body_ids):
                self._contact_groups[group].append(idx)
        if not self._bump_body_ids:
            raise RuntimeError(
                "Every contact-sensor body matched the foot pattern; bump "
                f"detection would silently never fire. Bodies: "
                f"{self._contact_sensor.body_names}"
            )

        self._robot = self.base_env.scene["robot"]

    # -- passive step observation (TR.3) -------------------------------------

    def register_step_observer(self, observer) -> "callable":
        """Register the per-control-step hook. Returns an unregister callable.

        Refuses to displace an existing observer rather than overwriting it. A
        second registration means two consumers (say, a nested recorder from a
        missing ``detach()``) each believe they are receiving every step, and
        the loser silently receives none — which is how ``runner.py`` describes
        the nested-recorder hazard for the patch this replaces. Loud beats a
        video that is quietly empty (rule 11).
        """
        if self.step_observer is not None:
            raise RuntimeError(
                "a step observer is already registered; unregister it first "
                "(a second registration would silently starve one consumer)"
            )
        self.step_observer = observer

        def unregister() -> None:
            if self.step_observer is observer:
                self.step_observer = None

        return unregister

    def _emit_step(
        self,
        terminated: bool,
        true_pose: tuple[float, float, float],
        contact_force_n: float,
        contact_groups,
        in_contact: bool,
    ) -> None:
        observer = self.step_observer
        if observer is None:
            return
        # Deliberately NOT wrapped in try/except: a recorder that cannot grab a
        # frame must fail the run loudly, not leave a green trial with no
        # rule-11 evidence.
        observer(
            StepObservation(
                step_index=self._step_counter,
                terminated=terminated,
                true_pose=true_pose,
                contact_force_n=contact_force_n,
                contact_groups=tuple(contact_groups),
                in_contact=in_contact,
            )
        )

    # -- simulated leg odometry (per control step, TR.3) ---------------------

    def _odometer_step(self, dx: float, dy: float) -> tuple[float, float]:
        """One control step of simulated leg odometry from a true XY delta.

        THE odometer. It is driven from the fixed 50 Hz control step, not from
        the command boundary, so the noise a trial accumulates is a function of
        the steps it took and nothing else — see ODOM_VAR_PER_M/PER_S for why
        the per-call spelling made the sensor depend on the recorder.

        Consumes the per-trial RNG stream seeded in :meth:`reset`, so a trial
        replays identically from its seed, at any external call partitioning.
        """
        sigma = math.sqrt(
            ODOM_VAR_PER_M * math.hypot(dx, dy) + ODOM_VAR_PER_S * CONTROL_DT
        )
        return (
            dx * self._odom_scale + self._odom_rng.gauss(0.0, sigma),
            dy * self._odom_scale + self._odom_rng.gauss(0.0, sigma),
        )

    # -- command channel ----------------------------------------------------

    def _defuse_command_term(self) -> None:
        """Re-assert the cfg-level defusal on the LIVE term.

        ``DuckEmbodyEnvCfg`` already sets these, but the term reads its cfg
        dynamically on every resample, so re-asserting on the constructed term
        costs nothing and closes the gap if the env was ever built from a
        different cfg (doc 02 §4).
        """
        cfg = self.command_term.cfg
        cfg.heading_command = False
        cfg.rel_standing_envs = 0.0
        cfg.resampling_time_range = (1.0e9, 1.0e9)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Pin the ranges (belt) and write the buffer directly (suspenders).

        The direct write takes effect on the very next control step instead of
        waiting for a resample; the degenerate ranges mean that if anything ever
        *does* resample — notably ``env.reset()`` — it redraws the same value.
        """
        cfg = self.command_term.cfg
        cfg.ranges.lin_vel_x = (vx, vx)
        cfg.ranges.lin_vel_y = (vy, vy)
        cfg.ranges.ang_vel_z = (wz, wz)
        self.command_term.vel_command_b[:, 0] = vx
        self.command_term.vel_command_b[:, 1] = vy
        self.command_term.vel_command_b[:, 2] = wz

    # -- true state (SCORING ONLY — never shown to the model) ---------------

    def true_xy(self) -> tuple[float, float]:
        pos = self._robot.data.root_pos_w[0]
        return (float(pos[0]), float(pos[1]))

    def true_height(self) -> float:
        return float(self._robot.data.root_pos_w[0, 2])

    def compass_deg(self) -> float:
        """Absolute heading, degrees CCW from +x (doc 03 §3 convention).

        This IS given to the model — declared sensor-realistic exception (a):
        the physical duck's BNO055 IMU provides absolute yaw.
        """
        quat = self._robot.data.root_quat_w[0]
        w, x, y, z = (float(v) for v in quat)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return wrap_deg(math.degrees(yaw))

    @property
    def fall_diagnostics(self) -> dict | None:
        """The last fall's diagnostics, or None if this run has not fallen.

        Instance state, and safe to read as a fallback: a fall ends the TRIAL
        (doc 01 §8), so within one run there is at most one, and `reset()`
        clears it between runs.
        """
        return self._fall_diagnostics

    def tilt_deg(self) -> float:
        """Trunk tilt from vertical, in degrees. SCORING/DIAGNOSTIC ONLY.

        Derived from projected gravity, the same signal `mdp.bad_orientation`
        terminates on, so a logged tilt and the term that fired cannot disagree.
        """
        gz = float(self._robot.data.projected_gravity_b[0][2])
        return math.degrees(math.acos(max(-1.0, min(1.0, -gz))))

    def bump_contact_force(self) -> float:
        """Peak contact force (N) over every non-foot body. See __init__."""
        forces = self._contact_sensor.data.net_forces_w[0, self._bump_body_ids]
        return float(forces.norm(dim=-1).max())

    def contact_groups(self) -> list[str]:
        """Which parts of the body are in contact right now, coarsely.

        One of `head`, `torso`, `left_leg`, `right_leg`. This is proprioception,
        not ground truth: it says what the robot FELT, never what it hit or
        where that thing is, so it sits on the sensor side of doc 05 §1's
        boundary — the same side as the compass and the `bumped` flag it
        refines.
        """
        forces = self._contact_sensor.data.net_forces_w[0]
        norms = forces.norm(dim=-1)
        out = []
        for group, ids in self._contact_groups.items():
            if ids and float(norms[ids].max()) > BUMP_FORCE_N:
                out.append(group)
        return out

    def contact_report(self) -> dict[str, float]:
        """Per-body force for the non-foot bodies above threshold. Debug only."""
        forces = self._contact_sensor.data.net_forces_w[0, self._bump_body_ids]
        return {
            name: round(float(f), 2)
            for name, f in zip(self._bump_body_names, forces.norm(dim=-1).tolist())
            if f > BUMP_FORCE_N
        }

    # -- persistent contact state ------------------------------------------

    def _ensure_contact_state(self) -> None:
        """Initialise contact fields for lightweight ``__new__`` test doubles."""
        defaults = {
            "_bump_run": 0,
            "_contact_state": "free",
            "_contact_candidate_onset_step": None,
            "_contact_clear_run": 0,
            "_contact_candidate_regions": [],
            "_contact_event_id_counter": 0,
            "_contact_event_id": None,
            "_contact_event_onset_step": None,
            "_contact_event_release_step": None,
            "_contact_event_regions": [],
            "_last_contact_event": None,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value.copy() if isinstance(value, list) else value)

    def _remember_contact_regions(self, groups) -> None:
        for group in groups:
            if group not in self._contact_candidate_regions:
                self._contact_candidate_regions.append(group)
            if (
                self._contact_state in ("sustained_contact", "candidate_release")
                and group not in self._contact_event_regions
            ):
                self._contact_event_regions.append(group)

    def _update_last_contact_event(self) -> None:
        if self._contact_event_id is None:
            return
        self._last_contact_event = {
            "contact_event_id": self._contact_event_id,
            "onset_step": self._contact_event_onset_step,
            "release_step": self._contact_event_release_step,
            "regions": list(self._contact_event_regions),
            "state": self._contact_state,
        }

    def _update_contact_state(self, raw_contact: bool, groups) -> None:
        """Advance the trial-scoped contact machine by one control step."""
        self._ensure_contact_state()

        if raw_contact:
            self._bump_run += 1
            self._contact_clear_run = 0
            self._remember_contact_regions(groups)

            if self._contact_state == "free":
                if self._bump_run >= BUMP_DEBOUNCE_STEPS:
                    self._contact_state = "candidate_contact"
                    self._contact_candidate_onset_step = (
                        self._step_counter - self._bump_run + 1
                    )
            elif self._contact_state == "candidate_release":
                # A short force trough belongs to the same physical event.
                self._contact_state = "sustained_contact"
                self._contact_event_release_step = None

            if self._contact_state == "candidate_contact":
                onset = self._contact_candidate_onset_step
                continuous_steps = (
                    self._step_counter - onset + 1 if onset is not None else 0
                )
                if continuous_steps >= CONTACT_SUSTAINED_STEPS:
                    self._contact_state = "sustained_contact"
                    self._contact_event_id_counter += 1
                    self._contact_event_id = self._contact_event_id_counter
                    self._contact_event_onset_step = onset
                    self._contact_event_release_step = None
                    self._contact_event_regions = list(
                        self._contact_candidate_regions
                    )
            self._update_last_contact_event()
            return

        self._bump_run = 0
        if self._contact_state == "candidate_contact":
            # It cleared before the sustained threshold: a reportable graze,
            # not a blocking event.
            self._contact_state = "free"
            self._contact_candidate_onset_step = None
            self._contact_candidate_regions = []
        elif self._contact_state == "sustained_contact":
            self._contact_state = "candidate_release"
            self._contact_clear_run = 1
        elif self._contact_state == "candidate_release":
            self._contact_clear_run += 1
            if self._contact_clear_run >= CONTACT_SUSTAINED_STEPS:
                self._contact_state = "free"
                self._contact_event_release_step = self._step_counter
                self._contact_candidate_onset_step = None
                self._contact_candidate_regions = []
        self._update_last_contact_event()

    @property
    def contact_state(self) -> str:
        self._ensure_contact_state()
        return self._contact_state

    @property
    def last_contact_event(self) -> dict | None:
        """Copy of the latest sustained event, safe for the tools layer."""
        self._ensure_contact_state()
        if self._last_contact_event is None:
            return None
        event = dict(self._last_contact_event)
        event["regions"] = list(event["regions"])
        return event

    # -- execution ----------------------------------------------------------

    def reset(self, seed: int | None = None):
        self._obs, _ = self.env.reset()
        self._fell = False
        # Per-trial odometry error: one systematic scale draw plus a fresh
        # white-noise stream. Seeded so a trial replays identically.
        # Remembered so the lazy `self.reset()` inside execute() (the _obs-is-None
        # path) cannot silently replace a trial's seeded error model with the
        # seed-0 one mid-trial — a re-seed that would be invisible in every
        # artifact.
        self._last_seed = seed if seed is not None else getattr(self, "_last_seed", None)
        self._odom_rng = random.Random(0 if self._last_seed is None else int(self._last_seed))
        lo, hi = ODOM_SCALE_CLIP
        self._odom_scale = min(hi, max(lo, self._odom_rng.gauss(1.0, ODOM_SCALE_STD)))
        # Cleared with `_fell`, or a fallback read could serve the PREVIOUS
        # trial's fall to this one.
        self._fall_diagnostics = None
        self._bump_run = 0
        self._contact_state = "free"
        self._contact_candidate_onset_step = None
        self._contact_clear_run = 0
        self._contact_candidate_regions = []
        self._contact_event_id_counter = 0
        self._contact_event_id = None
        self._contact_event_onset_step = None
        self._contact_event_release_step = None
        self._contact_event_regions = []
        self._last_contact_event = None
        self._step_counter = 0
        return self._obs

    @property
    def fell(self) -> bool:
        return self._fell

    def execute(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration_s: float,
        stop_on_bump: bool = False,
        stop_predicate=None,
    ) -> ExecResult:
        """Run one velocity command for ``duration_s`` and report what happened.

        ``stop_on_bump`` is what separates ``move`` (auto-stops on collision)
        from ``send_velocity`` (runs its full duration; doc 05 §4.2).
        ``stop_predicate(step_idx)`` lets the distance servo in ``tools.move``
        end the command as soon as it has covered the requested distance.
        """
        torch = self._torch
        (cvx, cvy, cwz), notes = clamp_command(vx, vy, wz)
        n_steps = duration_to_steps(duration_s)

        if self._obs is None:
            self.reset()

        start_xy = self.true_xy()
        sampled_xy: list[tuple[float, float]] = []
        bumped = False
        contact_groups: list[str] = []
        contact_steps = 0
        in_contact = False
        stopped_early = False
        stop_reason = ""
        steps_done = 0
        # Odometry accumulates per CONTROL STEP (TR.3). The call-scoped sum is
        # just bookkeeping; the process itself is the fixed-step odometer, which
        # is why splitting a motion across 1, 5 or 75 calls gives the identical
        # total from the same seed and step sequence.
        odom_dx = 0.0
        odom_dy = 0.0
        self._ensure_contact_state()

        # Last pose observed while the episode was still live. On a fall this is
        # the only trustworthy final pose — see the termination branch below.
        last_live_xy = start_xy
        last_live_heading = self.compass_deg()
        terminated_this_call = False

        for step in range(n_steps):
            # Re-write every step: cheap, and it makes the command immune to
            # anything that might touch the buffer between steps.
            self.set_command(cvx, cvy, cwz)

            # Snapshot BEFORE stepping. If this step turns out to be the one
            # that terminated the episode, the post-step scene has already been
            # teleported (see below) and this snapshot — one control step, 20 ms,
            # earlier — is the closest true pose we can honestly report.
            pre_step_xy = self.true_xy()
            pre_step_heading = self.compass_deg()
            pre_step_height = self.true_height()
            pre_step_tilt = self.tilt_deg()

            with torch.no_grad():
                actions = self.policy(self._obs)
                self._obs, _, _, _ = self.env.step(actions)

            steps_done = step + 1
            self._step_counter += 1

            # A fall is a real termination (tilt/height, per T1.1).
            # CRITICAL: Isaac Lab auto-resets a terminated env INSIDE step()
            # (manager_based_rl_env.py:216-221) and returns the post-reset
            # observation — so by the time we get here the robot has already
            # been teleported back to spawn. Reading true_xy() now would record
            # the spawn point as the fall location, quietly corrupting the SPL
            # path, the drift metric and the trajectory figure. Use the
            # pre-step snapshot and stop touching live state.
            if bool(self.base_env.termination_manager.terminated[0]):
                self._fell = True
                stopped_early = True
                stop_reason = "fell"
                terminated_this_call = True
                last_live_xy = pre_step_xy
                last_live_heading = pre_step_heading
                # WHY it ended, captured here or not at all. A fall ends the
                # whole trial (doc 01 §8), making it the single most
                # consequential event in a run — and T3.5 recorded one that
                # could not be audited afterwards: the JSON said `fell` with no
                # height, no tilt and no term (the recorder-merge drop, since
                # fixed — see merge_exec_results), and the audit video's last
                # frame was grabbed AFTER the auto-reset, i.e. a healthy duck
                # at spawn (also fixed: recorder.chunked_execute skips the
                # grab on a falling piece), so neither artifact could say
                # whether the fall was genuine.
                #
                # These are the PRE-STEP values, for the same reason the pose is:
                # the env has already auto-reset, so live state now describes a
                # healthy duck standing at spawn.
                self._fall_diagnostics = {
                    "height_m": round(pre_step_height, 4),
                    "tilt_deg": round(pre_step_tilt, 2),
                    "terms": {
                        name: bool(
                            self.base_env.termination_manager.get_term(name)[0]
                        )
                        for name in self.base_env.termination_manager.active_terms
                    },
                    "height_threshold_m": FALL_MIN_HEIGHT_M,
                    "tilt_threshold_deg": FALL_TILT_LIMIT_DEG,
                    "commanded": (cvx, cvy, cwz),
                    "policy_seconds_into_call": round(steps_done * CONTROL_DT, 3),
                    # Self-description, because the numbers can otherwise look
                    # wrong on their own: pre-step values are one control step
                    # (20 ms) BEFORE the thresholds fired, so a recorded tilt
                    # of 59.4 deg can sit beside a 60.0 threshold in the same
                    # dict with fell_over true — honest, but inexplicable to a
                    # reader who was not told the sampling instant.
                    "values_pre_step": True,
                }
                # Observers still hear the terminating step — that is how a
                # recorder knows NOT to grab (the scene is already teleported)
                # and how an auditor learns when the trial ended. No odometry
                # for this step: the post-step pose is the spawn point, so any
                # delta measured across it is fiction, and drawing noise for it
                # would make the RNG stream depend on the fall.
                self._emit_step(
                    terminated=True,
                    true_pose=(pre_step_xy[0], pre_step_xy[1], pre_step_heading),
                    contact_force_n=0.0,
                    contact_groups=(),
                    in_contact=in_contact,
                )
                break

            last_live_xy = self.true_xy()
            last_live_heading = self.compass_deg()

            # Leg odometry for THIS step, from the true pre/post delta. Measured
            # step by step rather than once per call so the noise process is
            # invariant to how the caller sliced the motion up (forensics F-03).
            _sdx = last_live_xy[0] - pre_step_xy[0]
            _sdy = last_live_xy[1] - pre_step_xy[1]
            _odx, _ody = self._odometer_step(_sdx, _sdy)
            odom_dx += _odx
            odom_dy += _ody

            step_force = self.bump_contact_force()
            raw_contact = step_force > BUMP_FORCE_N
            raw_groups = tuple(self.contact_groups()) if raw_contact else ()
            self._update_contact_state(raw_contact, raw_groups)
            in_contact = self._contact_state != "free"
            step_groups: tuple[str, ...] = ()
            if self._contact_state != "free":
                bumped = True
                step_groups = tuple(
                    self._contact_candidate_regions
                    if self._contact_state == "candidate_contact"
                    else self._contact_event_regions
                )
                for group in step_groups:
                    if group not in contact_groups:
                        contact_groups.append(group)

            # Charged for every latched step, not just the confirming one.
            if in_contact:
                contact_steps += 1

            if self._step_counter % POSE_TRACE_EVERY == 0:
                sampled_xy.append(last_live_xy)

            self._emit_step(
                terminated=False,
                true_pose=(last_live_xy[0], last_live_xy[1], last_live_heading),
                contact_force_n=step_force,
                contact_groups=step_groups,
                in_contact=in_contact,
            )

            if stop_on_bump and self._contact_state == "sustained_contact":
                stopped_early = True
                stop_reason = "sustained_contact"
                break

            if stop_predicate is not None and stop_predicate(step):
                stopped_early = True
                stop_reason = "target_reached"
                break

        # Never leave a command armed: the sim pauses between LLM turns, and a
        # live command must not be waiting when the next macro starts.
        # (Safe even after a termination — it only writes command buffers.)
        self.set_command(0.0, 0.0, 0.0)

        end_xy = last_live_xy
        end_heading = last_live_heading
        if not terminated_this_call:
            end_xy = self.true_xy()
            end_heading = self.compass_deg()
        pose_trace = [start_xy, *sampled_xy, end_xy]

        # Simulated leg odometry for this call is the SUM of the per-step
        # odometer above — no call-level draw. A wedged call's steps each
        # measure ~zero true motion, so odometry reports ~zero: motion is
        # measured, not assumed from the command.
        #
        # The retired call-level version drew one Gaussian with
        # `sigma = FRAC*call_distance + RATE*call_seconds`. Sigma is not
        # additive, so slicing a command into N pieces (which recording used to
        # do, N ~ 75) shrank the aggregate noise by ~sqrt(N): the sensor's
        # accuracy depended on whether video was attached. See ODOM_VAR_PER_M.
        _ox = odom_dx
        _oy = odom_dy

        return ExecResult(
            commanded=(cvx, cvy, cwz),
            duration_s=duration_s,
            steps=steps_done,
            policy_seconds=steps_done * CONTROL_DT,
            bumped=bumped,
            contact_groups=contact_groups,
            fell=self._fell,
            fall_diagnostics=self._fall_diagnostics if terminated_this_call else None,
            pose_trace=pose_trace,
            sampled_xy=sampled_xy,
            true_pose=(end_xy[0], end_xy[1], end_heading),
            true_displacement_m=math.dist(start_xy, end_xy),
            clamp_notes=notes,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            contact_steps=contact_steps,
            odom_dxy=(_ox, _oy),
            odom_distance_m=math.hypot(_ox, _oy),
            contact_state=self._contact_state,
            contact_event_id=self._contact_event_id,
            contact_onset_step=self._contact_event_onset_step,
            contact_release_step=self._contact_event_release_step,
            contact_event_regions=list(self._contact_event_regions),
        )

    def settle(self, duration_s: float = 0.4) -> None:
        """Step with a zero command so the gait comes to rest before a capture."""
        self.execute(0.0, 0.0, 0.0, duration_s)

    # -- motion macros (doc 02 §6) ------------------------------------------

    def _merge(self, total: ExecResult | None, part: ExecResult) -> ExecResult:
        # Pure delegation to the ONE shared merge. The recorder's chunked path
        # merges through the same function, so the two layers cannot drift —
        # the hand-mirrored copy this used to be beside dropped
        # contact_groups/fall_diagnostics on every recorded run (see
        # merge_exec_results).
        return merge_exec_results(total, part)

    def turn_to_heading(
        self,
        heading_deg: float,
        tol_deg: float = TURN_TOLERANCE_DEG,
        timeout_s: float = TURN_TIMEOUT_S,
        on_chunk=None,
    ) -> ExecResult:
        """Rotate in place to an absolute compass heading, closed-loop.

        P-control on the compass, clamped to the training hull, with a timeout
        instead of spinning forever. Reports the residual error so the model can
        decide whether to retry (doc 05 §4.2).
        """
        target = wrap_deg(heading_deg)
        start_xy = self.true_xy()
        merged: ExecResult | None = None
        n_chunks = max(1, int(timeout_s / MACRO_CHUNK_S))

        # Same post-fall rule as move(): never re-read live state after a
        # termination, because the env has already teleported.
        last_pose = (start_xy[0], start_xy[1], self.compass_deg())
        reason = "timeout"

        for _ in range(n_chunks):
            err = shortest_angle_diff_deg(target, self.compass_deg())
            if abs(err) <= tol_deg:
                break
            wz = max(-WZ_RANGE[1], min(WZ_RANGE[1], KP_HEADING * math.radians(err)))
            part = self.execute(
                0.0, 0.0, wz, MACRO_CHUNK_S, stop_on_bump=True
            )
            merged = self._merge(merged, part)
            last_pose = part.true_pose
            if on_chunk is not None:
                on_chunk()
            if part.fell:
                reason = "fall"
                break
            if part.stop_reason == "budget":
                reason = "budget"
                break
            if (
                part.stop_reason == "sustained_contact"
                or part.contact_state == "sustained_contact"
            ):
                reason = "sustained_contact"
                break

        if reason not in ("fall", "budget"):
            # Settle so the next capture shows a still robot rather than a turn
            # in progress, and so no command is left armed across the LLM think.
            settle = self.execute(
                0.0, 0.0, 0.0, MACRO_CHUNK_S, stop_on_bump=True
            )
            merged = self._merge(merged, settle)
            last_pose = settle.true_pose
            if on_chunk is not None:
                on_chunk()
            # A topple DURING the settle is still a fall. The stale local flag
            # used to win here, so the very command that ended the trial
            # reported stop_reason "reached"/"timeout" — and tools.py derives
            # the model-facing `timed_out` from that exact string, telling the
            # model its turn timed out on a trial that was already over.
            if settle.fell:
                reason = "fall"
            elif settle.stop_reason == "budget":
                reason = "budget"
            elif (
                settle.stop_reason == "sustained_contact"
                or settle.contact_state == "sustained_contact"
            ):
                reason = "sustained_contact"

        residual = shortest_angle_diff_deg(target, last_pose[2])
        if reason not in ("fall", "budget", "sustained_contact"):
            reason = "reached" if abs(residual) <= tol_deg else "timeout"
        merged.stop_reason = reason
        merged.stopped_early = reason in ("fall", "budget", "sustained_contact")
        merged.target_reached = reason == "reached"
        merged.true_pose = last_pose
        merged.pose_trace = [start_xy, *merged.sampled_xy, (last_pose[0], last_pose[1])]
        merged.true_displacement_m = math.dist(start_xy, (last_pose[0], last_pose[1]))
        return merged

    def move(
        self,
        distance_m: float,
        hold_heading: bool = True,
        stop_on_bump: bool = True,
        on_chunk=None,
    ) -> ExecResult:
        """Walk forward, servoing on dead-reckoned distance AND heading.

        **Heading hold is not optional decoration.** T1.3 measured the bare
        policy yawing ~1.8 deg/s when commanded straight — 36.6 deg over 4 m.
        Open loop, a 1.5 m move aimed at a 0.35 m doorway ends ~0.18 m off
        course, which would show up as "the model cannot navigate" when it is
        really the gait. Closing wz on the compass during the drive cuts that to
        0.39 deg over the same distance. AGENTS.md rule 5 declares closed-loop
        macros servoing on compass + dead reckoning a sensor-realistic exception,
        so this is in scope by design, not a workaround.

        Auto-stops on collision (this is the tool that does; `send_velocity`
        deliberately does not — doc 05 §4.2).
        """
        signed_distance = max(
            -MOVE_MAX_DISTANCE_M, min(distance_m, MOVE_MAX_DISTANCE_M)
        )
        requested_distance, timeout_forecast, n_chunks = move_servo_plan(
            abs(signed_distance)
        )
        speed = MOVE_SPEED_MPS if signed_distance >= 0.0 else -REVERSE_MOVE_SPEED_MPS
        if signed_distance < 0.0:
            # The public helper forecasts the historical forward macro. Reverse
            # uses the same k forecast with its deliberately lower speed.
            ideal_s = timeout_forecast / REVERSE_MOVE_SPEED_MPS
            n_chunks = max(
                1,
                int(math.ceil(ideal_s * MACRO_TIME_MARGIN / MACRO_CHUNK_S)),
            )

        held_heading = self.compass_deg()
        start_xy = self.true_xy()
        measured_distance = 0.0
        merged: ExecResult | None = None
        reason = "reached" if requested_distance == 0.0 else "timeout"
        # The last pose observed while the episode was LIVE. Re-reading
        # self.true_xy() after the loop would report the TELEPORTED pose on a
        # fall, because Isaac auto-resets a terminated env inside step() — the
        # same trap execute() already guards against, reintroduced here. It made
        # a duck that walked 1.1 m into a wall and toppled report 0.02 m.
        last_pose = (start_xy[0], start_xy[1], held_heading)

        for _ in range(0 if requested_distance == 0.0 else n_chunks):
            wz = 0.0
            if hold_heading:
                err = shortest_angle_diff_deg(held_heading, self.compass_deg())
                wz = max(-WZ_RANGE[1], min(WZ_RANGE[1], KP_HEADING * math.radians(err)))

            part = self.execute(
                speed, 0.0, wz, MACRO_CHUNK_S, stop_on_bump=stop_on_bump
            )
            merged = self._merge(merged, part)
            if on_chunk is not None:
                on_chunk()

            # The move servo closes exclusively on measured leg odometry. The
            # policy realisation factor only sized n_chunks (the timeout).
            measured_distance += part.odom_distance_m
            last_pose = part.true_pose

            if part.fell:
                reason = "fall"
                break
            if part.stop_reason == "budget":
                reason = "budget"
                break
            if stop_on_bump and (
                part.stop_reason == "sustained_contact"
                or part.contact_state == "sustained_contact"
            ):
                reason = "sustained_contact"
                break
            if measured_distance >= requested_distance:
                reason = "reached"
                break

        if reason not in ("fall", "budget"):
            stop = self.execute(
                0.0, 0.0, 0.0, MACRO_CHUNK_S, stop_on_bump=stop_on_bump
            )
            merged = self._merge(merged, stop)
            last_pose = stop.true_pose
            if on_chunk is not None:
                on_chunk()
            # A topple DURING the settle is still a fall. `reason` was decided
            # before the settle ran, so without this the audit record carried
            # the contradiction `fell: true, stop_reason: "reached",
            # stopped_early: false` on the command that ended the trial.
            if stop.fell:
                reason = "fall"
            elif stop.stop_reason == "budget":
                reason = "budget"
            elif stop_on_bump and (
                stop.stop_reason == "sustained_contact"
                or stop.contact_state == "sustained_contact"
            ):
                reason = "sustained_contact"

        end_xy = (last_pose[0], last_pose[1])
        merged.stop_reason = reason
        merged.stopped_early = reason in ("sustained_contact", "fall", "budget")
        merged.true_pose = last_pose
        merged.pose_trace = [start_xy, *merged.sampled_xy, end_xy]
        merged.true_displacement_m = math.dist(start_xy, end_xy)
        merged.requested_distance_m = signed_distance
        merged.measured_distance_m = measured_distance
        merged.target_reached = (
            reason == "reached" and measured_distance >= requested_distance
        )
        merged.dead_reckoned_distance_m = measured_distance
        return merged

    def move_backward(
        self,
        distance_m: float,
        hold_heading: bool = True,
        stop_on_bump: bool = True,
        on_chunk=None,
    ) -> ExecResult:
        """Conservative reverse move; accepts a positive distance magnitude."""
        return self.move(
            -abs(distance_m),
            hold_heading=hold_heading,
            stop_on_bump=stop_on_bump,
            on_chunk=on_chunk,
        )

    @staticmethod
    def _phase_summary(name: str, result: ExecResult) -> dict:
        return {
            "phase": name,
            "stop_reason": result.stop_reason,
            "target_reached": result.target_reached,
            "steps": result.steps,
            "policy_seconds": result.policy_seconds,
            "requested_distance_m": result.requested_distance_m,
            "measured_distance_m": result.measured_distance_m,
        }

    def turn_and_move(
        self,
        heading_deg: float,
        distance_m: float,
        hold_heading: bool = True,
        stop_on_bump: bool = True,
        on_chunk=None,
    ) -> ExecResult:
        """Turn first, then move only after a successful heading lock."""
        turn = self.turn_to_heading(heading_deg, on_chunk=on_chunk)
        phase_results = [self._phase_summary("turn", turn)]
        if not turn.target_reached:
            turn.phase_results = phase_results
            return turn

        move = self.move(
            distance_m,
            hold_heading=hold_heading,
            stop_on_bump=stop_on_bump,
            on_chunk=on_chunk,
        )
        phase_results.append(self._phase_summary("move", move))
        merged = self._merge(turn, move)
        merged.phase_results = phase_results
        merged.stop_reason = move.stop_reason
        merged.stopped_early = move.stopped_early
        merged.target_reached = move.target_reached
        merged.requested_distance_m = move.requested_distance_m
        merged.measured_distance_m = move.measured_distance_m
        merged.dead_reckoned_distance_m = move.dead_reckoned_distance_m
        return merged
