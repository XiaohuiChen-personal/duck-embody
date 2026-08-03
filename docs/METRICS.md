# Metrics

How every published number is computed, in plain language, with its formula, its
units, and the edge case that decides it. The normative source is
[design doc 06 §5](designs/06-benchmark-evaluation.html); the implementation is
`duck_embody/scoring.py` and the gate in front of it is `tests/test_scoring.py`,
which must be green **before any batch trial launches** (doc 06 §9, AGENTS.md
rule 2).

Run it with:

```bash
bash scripts/run_tests.sh tests/test_scoring.py tests/test_layout.py -q
```

Score a trial and print every number:

```bash
$HOME/IsaacLab/isaaclab.sh -p -m duck_embody.scoring results/raw/<trial>.json
```

---

## 0. What this benchmark claims — and does not claim

Quoted from doc 06 §1, because these limits bind every sentence written about
the numbers below.

> What the benchmark **does** claim, if the data supports it: relative ranking of
> the three models on this task suite, under this one scaffold, in this one
> simulated apartment, with honest per-trial evidence published for every number.

> What it does **not** claim:
>
> - **No statistical significance claims.** N=4 trials per model is far too
>   small; we report ranking hypotheses with uncertainty intervals (§6).
> - **No generality claims beyond this embodiment** (one biped, one apartment,
>   simulation only, one prompt, one camera config).
> - **No claims about the unreleased original harness.** Anthropic's
>   `safety-research/embody` is unreleased (404 verified 2026-07-26); this is a
>   from-scratch build of the paradigm, not a reproduction of their code.
> - **No memory-scaffold causality claims** — without the ablation, we cannot
>   attribute success or failure to the scaffold itself.

Three writing rules follow from doc 06 §6's honesty clause and apply to the
README, the report and every figure caption:

1. the per-trial table is published alongside **any** aggregate;
2. the phrase "statistically significant" appears **nowhere**;
3. overlapping confidence intervals are reported as **"indistinguishable at this
   N"**, never as a difference.

---

## 1. What the scorer reads

Doc 06 §5's preamble says "the trial JSON plus the layout ground truth — nothing
else". Stated honestly, that is three frozen inputs:

| Input | Why it is allowed |
|---|---|
| the per-trial JSON (doc 06 §4) | the only record of the run; the scorer never touches the simulator |
| `duck_embody/env/apartment_layout.py` | scene spec **and** answer key in one dict (AGENTS.md §2), read through its own helpers so the world and the key cannot drift |
| `duck_embody/tasks/find_kitchen.py::score_stage` and `duck_embody/agent/prompts.py`'s `ROOM_SYNONYMS` / `LAYOUT_QA_QUESTIONS` | doc 06 §9.1(iii) and §5.7/§5.9 **require** reusing these rather than re-authoring them, so the scorer and the live gate cannot disagree |

`configs/benchmark.yaml`'s `scoring:` block supplies §6's bootstrap constants —
the only place the RNG seed exists.

The scorer reads ground truth (`true_pose`, `pose_trace`, oracle paths, room
polygons) on purpose. The separation that matters runs the other way: nothing
under `duck_embody/agent/` may import `duck_embody.scoring`, and
`tests/test_scoring.py::TestPackageSeparation` asserts it.

### "—" is a value, never a number

A metric that is genuinely undefined for a trial prints an em dash. It is
excluded from means and confidence intervals and is **never** coerced to `0.0` —
a `0.0` in a time column would read as "arrived instantly". The cases are named
per metric below.

### A corrupt log raises; it never produces a number

Every failure mode below was measured to produce a *plausible and flattering*
number before it was guarded, which is the only kind that survives review. Each
now raises `ScoringError` naming the trial:

| Guard | What it caught, measured |
|---|---|
| **chord floor** (§2.3) | blanking every `pose_trace` to `[]` — the module's own documented value for "this turn stepped no physics" — gave `p = 0` and **`SPL = 1.0` on both stages**, indistinguishable from a perfect run. Keeping only each trace's last sample gave `p = 1.4922` against a true 2.2985 (−35 %) and still `SPL = 1.0` |
| **`turns_used` cross-check** | `turns_used = 1` against four logged stage-1 turns was republished verbatim, while the analogous `final.bumps` disagreement already raised by design |
| **unknown `turns[].stage`** | relabelling one stage-1 turn made it vanish from *both* path integrals: `p` fell 2.2985 → 1.8174 (−21 %) and SPL rose |
| **`declare_done` with no turns** | a `turns: []` log with a `final` claiming success scored `SPL 1.0, progress 0.985, time_s 0.0` — a teleport scoring perfectly |
| **spawn cross-check** | a spawn that disagrees with `spawn_pose(seed)` corrupts stage-1 `d_initial`, the return-home goal *and* radius, the SPL oracle and Q4's gold bearing at once |
| **non-finite values** | `json.loads('{"x": NaN}')` succeeds, so a PhysX blow-up round-trips through the log. `progress(2.0, nan)` returned **1.0** (because `min(1.0, nan)` is `1.0`), and one NaN turned a model's whole published mean *and* interval into NaN while still reporting `n_defined = 3`. Serialisation uses `allow_nan=False`; bare `NaN` is not valid JSON |
| **`pose_trace: null`, `args: null`** | raw `TypeError`/`AttributeError` mid-pass, with no trial named |

The rule they share: **losing motion data always pushes SPL up**, because
`max(p, l)` caps the ratio at 1.0. So under-measurement can never be treated as
conservative.

---

## 2. The metrics

### 2.1 Success rate (SR) — doc 06 §5.1, criterion v2

`SR = successes / N`, per model **per stage**, `N = 4`.

**The `find_kitchen` success predicate is criterion v2 ("any counter face").**
Adopted POST-BATCH (2026-07-27, owner-directed;
[`results/rerun_log.md`](../results/rerun_log.md)) and **unified with the live
stage machine by TR.2** (2026-08-02). Until TR.2 it was published-only: v2 lived
in `scoring.py` while the running task still gated on the point disc, so the
benchmark reported one task and the robot played another. `opus5_seed101`
declared 0.3607 m from the point (a live failure, `declared_elsewhere`) and
0.0577 m from a counter face inside the kitchen (a published success), and the
`return_home` leg it had earned was never offered. There is now ONE
implementation — `duck_embody.tasks.find_kitchen.position_success` — consulted
by the live loop and by the scorer, with the counter geometry in
`env/apartment_layout.py` so neither side owns a private copy.

A stage-1 declare succeeds
if the true base position is within 0.35 m of the pre-registered target point
**or** within the same 0.35 m of any of the five kitchen counters' footprint
rectangles *while standing inside the kitchen room polygon*. Rationale: the
frozen objective ("walk to the counter") never disambiguates the kitchen's two
counter runs, and the batch produced a declare 5 cm from the non-target run.
The union (not the counter band alone) because the regions are **not nested**:
the pinned target point is 0.397 m from the nearest counter footprint, so a
pure any-counter test would fail a robot standing exactly on the old goal.
The in-kitchen condition is load-bearing: `counter_4/5` back onto the bedroom
partition, and a bedroom pose 4 cm through that wall is within 0.35 m of their
rectangles in Euclidean distance.

A stage still succeeds only if **both** hold: the position condition **and**
the model called `declare_done` there. Arriving without declaring is a timeout —
"the model must *know* it arrived; stumbling through the goal region does not
demonstrate localization". v2 widened only WHERE arrival counts, never HOW.

- radii: **0.35 m** (point disc and counter band alike), **0.5 m** back to the
  spawn (`apartment_layout.LAYOUT`, mirrored in `configs/benchmark.yaml`);
  `return_home`'s predicate is unchanged by v2
- the boundary is **inclusive**: exactly 0.35 m is a success; the counter-band
  distance is point-to-rectangle (a corner approach is credited up to 0.35 m
  off a footprint corner — the natural rectangle generalisation of the disc)
- the distance is measured from **ground truth**, not the model's estimate. A
  model whose belief drifted onto the counter while the robot stood in the
  hallway fails. That asymmetry is the benchmark.
- the log carries two different flags and confusing them inflates SR:
  `stages.*.score.success` is the bare position test, `stages.*.success` is
  position **and** `declare_done`. **Which predicate wrote them depends on the
  trial**, and the trial says so: `config.success_criterion` is
  `"v2_any_counter"` on any log written after TR.2, and absent on a legacy log
  whose live gate was the point disc. The scorer branches on that field and
  never on the freeze commit:
  - **stamped `v2_any_counter`** — recompute under v2 and validate the logged
    live outcome against it. Live and published agree by construction; a
    disagreement is a corrupt log and raises.
  - **unstamped (legacy: v4, `raw_v5d`, `raw_v5d_r2`)** — validate the as-run
    point-disc verdict, then publish the v2 sensitivity result beside it
    exactly as before: `success_preregistered` / `outcome_preregistered` per
    stage in `scores.json`. An old `stages.*.success` is **never** re-read as
    though v2 had decided it.

  On the frozen v4 batch: v2 1/12, pre-registered 0/12; the single flip is
  `gpt56sol_seed103`.

`return_home` is reported twice, per doc 06 §3.2: `x/4` with an unrun stage
counted a failure and the denominator printed literally, **plus** a conditional
`x/k` over the stage-1 successes **whose return leg actually ran** — on a legacy
batch the live gate consulted the pre-registered predicate, so a v2-only success
was never offered stage 2, and counting it in `k` would report a failure for a
leg the model never attempted. The exclusion is published as
`stage1_successes_never_offered_return`; `x/k` prints `—` when `k = 0`, and no
confidence interval when `k < 3`.

On a TR.2-stamped batch that exclusion must be **empty**: the live gate is v2,
so every v2 stage-1 success is offered its return leg. A nonzero
`stage1_successes_never_offered_return` on a stamped batch is a defect, not a
disclosure — the two predicates have drifted apart again — and it is asserted as
zero in `tests/test_scoring.py`.

**SR ships with an interval like every other column.** Doc 06 §10's README table
asks for "SR (both stages), progress, SPL, … each as mean ± 95 % CI", so every
`success_rate` block carries **both** the honest `x/N` ratio *and* §4's bootstrap
over the binary per-trial indicator (1.0 for a success, 0.0 otherwise). The
`k < 3` rule needs no special case — it falls out of the same minimum every other
column uses. Print the ratio; the interval is what the figures draw.

### 2.2 Progress — doc 06 §5.2

```
progress = clip(1 − d_final / d_initial, 0, 1)        [dimensionless, 0–1]
```

Straight-line distances, in metres, **per stage**: `find_kitchen` runs from the
seed's spawn to the counter, `return_home` from the true pose at the stage
boundary back to the spawn.

- **No success override.** A success reports the same formula value as a
  failure, so every published number is reproducible from the formula alone.
- **The point reference survives criterion v2 deliberately.** `d_initial` /
  `d_final` / `progress` still measure to the pre-registered target point:
  they are continuous distance metrics whose cross-batch comparability matters
  more than folding a discontinuous region distance (through a wall, the
  nearest counter is metres of walking away at centimetres of Euclidean
  distance) into a gradient. Consequence: a v2 success can show
  `progress < 1` — `gpt56sol_seed103` succeeds at progress 0.739.
- Clipped, so wandering away cannot go negative, and `d_final = 0` on a
  **failure** legitimately scores 1.0.
- `d_initial = 0` scores 0.0 rather than dividing by zero. It cannot happen for
  stage 1 (`tests/test_layout.py` pins every spawn beyond `3 × 0.35 m` from the
  target); it is representable for an unrun stage 2, and 0.0 is what doc 06 §3.2
  requires there anyway.

### 2.3 SPL — Success weighted by Path Length — doc 06 §5.3

```
SPL_i = S_i · l_i / max(p_i, l_i)                     [dimensionless, 0–1]
SPL   = (1/N) Σ_i SPL_i
```

Anderson et al. 2018, *On Evaluation of Embodied Navigation Agents*
([arXiv:1807.06757](https://arxiv.org/abs/1807.06757)), Eq. (1); the ObjectNav
convention since Batra et al. 2020.

- `S` — the binary success indicator of §2.1 above (criterion v2 for stage 1).
- `l` — the **oracle shortest path**, in metres, on the layout's free-space
  grid (5 cm cells, obstacles inflated by the 0.08 m body radius). For
  `find_kitchen` under criterion v2, `l` runs from the stage's start to the
  nearest point of the **success region** (disc ∪ counter band — the ObjectNav
  convention: path to the nearest success viewpoint), computed by uniform-cost
  search over the same grid with the same no-corner-cutting rule; it is
  therefore shorter than the old point oracle for every spawn (measured:
  2.05–3.14 m vs 2.18–4.17 m across seeds 101–104). `return_home` keeps the
  point-to-point oracle (its criterion did not change).
- `p` — the **integrated true path length**, in metres:
  `Σ ‖pose(t+1) − pose(t)‖` over the 5 Hz `execution.pose_trace` samples,
  segmented per stage.

Two implementation facts that are load-bearing:

- **`p` never comes from the once-per-turn `true_pose` entries.** Chord-summing
  those under-measures every within-turn curve (`send_velocity` arcs,
  deflections during a bumped `move`), which shrinks `p` and **inflates** SPL —
  i.e. it would make every model look better than it was, in the headline metric.
  A turn whose `execution` has no `pose_trace` key **raises**; an *empty* trace
  is the legitimate "this turn stepped no physics" case, and doc 06 §4 exists to
  keep those two distinguishable.
- **`S = 0` short-circuits before the division.** After a stage-1 failure the end
  pose is arbitrary, so `l` can be ~0 and `max(p, l)` can be 0/0.

`max(p, l)` caps the ratio at 1.0 when `p < l`, which happens through drift,
rounding, or corner-cutting relative to the grid's 5 cm resolution. So
`SPL ≤ SR` always.

**The chord floor.** An *empty* `pose_trace` is legal, so a recorder that wrote
`[]` on every turn would be silent — and would publish `SPL = 1.0`. `p` is
therefore cross-checked against a lower bound computed from a *different* field
of the same log: `Σ ‖true_pose(n) − true_pose(n−1)‖` over the stage's turns,
anchored at the stage's start. Physics advances only inside `env.step()` and
every step is traced, so by the triangle inequality this can never exceed `p`;
`p` falling below it means samples were lost, and the scorer raises. The bound is
tight — measured margin on the committed fixtures is `0.000000 m`
(`find_kitchen`) and `−1e-16 m` (`return_home`), straight-line traces being
exactly their own chords — so the tolerance is deliberately small:
`1e-3 m + 1.5e-4 m per chord`, the second term covering `true_pose`'s 4-decimal
rounding (≤ 7.08e-5 m per endpoint, ≤ 1.42e-4 m per chord).

### 2.4 Time-to-kitchen / time-to-home — doc 06 §5.4

Cumulative **policy-seconds** (simulated seconds of commanded motion) from the
stage's start to its successful `declare_done`.

- **`—` on a failure.** Not a number, not zero.
- Policy-seconds, not wall-clock: the sim pauses while the model thinks, so
  wall-clock would measure API latency, not navigation efficiency.
- A value **above the 240 s cap is not a bug**: caps are checked after a whole
  turn, so one chained turn can legitimately end at 251 s.

### 2.5 Turns used — doc 06 §5.5

Model turns consumed in the stage, cap 40, counted **per stage** (the budget
resets at the boundary). A proxy for deliberation cost, complementary to §2.4's
motion cost.

### 2.6 Bumps and falls — doc 06 §5.6

`bumps` counts collision events over the **whole trial** (both stages) from
exactly two sources: `move` auto-stops, and `bumped = true` reports for
`send_velocity` commands (one per command).

`turn_to_heading` is **deliberately excluded**. `PolicyPlayback._bump_run` is
instance state that is not reset between calls, so after a bump-stopped `move`
the debounce counter already sits at its threshold and the first control step of
the recovery turn re-flags `bumped`. Bump-then-turn-away is the canonical
recovery, so counting rotations would score a model that turns away *worse* than
one that reverses, for the same number of real collisions — behaviour-dependent
inflation, not a stricter measurement.

The scorer reads `final.bumps` and **cross-checks** it against the count of
per-call `counted_as_bump` flags; a disagreement means the log is internally
inconsistent and raises rather than publishing either number.

`falls ∈ {0, 1}` — a fall ends the trial.

### 2.7 Map accuracy — doc 06 §5.7

Trial-scoped (memory carries across the stage boundary), in two parts.

**Room-node precision / recall.** The model's claimed rooms — its `update_room`
entries, under whatever names it coined — are matched one-to-one to true rooms. A
claim matches iff **both**:

1. **name** — it normalises to that room through the *frozen* synonym table in
   `prompts.py::ROOM_SYNONYMS` (case- and punctuation-insensitive exact match; no
   fuzzy or embedding matching). `"lounge"` → `living_room`; `"kitchenette"` is
   deliberately **absent** from the table and matches nothing.
2. **place** — a **majority** of its evidence points fall inside that room's
   polygon. Evidence = the true pose at each turn on which the model named the
   room via `update_room`, `set_current_room` or `add_landmark`.

Evidence counts only calls the harness actually **ran**. `loop._run_turn` stops
at `declare_done` and answers every later tool block with `not_executed`;
`model_output.dispatched` records how many ran, and the scorer reads only those.
Otherwise a claim the harness had rejected could tip the majority rule. One
residual: a call with a `parse_error` *is* counted in `dispatched` (dispatch
answers it with an error and never touches memory) and `parse_errors` records
names, not indices, so it cannot be excluded unambiguously — such a call can only
add evidence for a name some other, successful call already put in the snapshot.

Greedy, deterministic order: **evidence count** (descending), then
**name similarity** — an exact canonical name outranks a synonym-mediated match,
the only two grades a fixed table can produce — then **earliest claim time**.

```
precision = matched / claimed          recall = matched / true_rooms_visited
```

- `claimed = 0` ⇒ precision is **`—`**, excluded from the aggregate mean, never
  coerced to a number; recall is then 0 whenever at least one room was visited.
- `true_rooms_visited` counts only rooms the robot actually entered, from the
  true trace — the model cannot be penalised for not mapping rooms it never saw.

**Adjacency edge accuracy.** Exits whose status is `leads_to:<room>` are the
model's adjacency assertions; `unexplored` exits are excluded (claiming "there's
an unexplored exit north" asserts nothing about adjacency).

```
edge_accuracy = correct claimed edges / all claimed edges
```

An edge is correct iff **both** endpoints resolve to true rooms through the
matching above **and** the pair is a real doorway in the layout graph. An edge
naming a room that matched nothing is **wrong, not excluded** — excluding it
would make an unmatched claim free after precision already counted it. With no
`leads_to:` edge at all, edge accuracy is `—`.

Doc 06 §5.7's wording ("exits … define claimed edges **between matched rooms**")
also reads the other way, as "only matched pairs are claimed edges at all", and
the two conventions publish different numbers. The choice above is deliberate but
it has a real cost, so `edges_unresolved` is **published alongside**
`edges_claimed` and `edges_correct`: a reader can recompute the other convention
as `edges_correct / (edges_claimed − edges_unresolved)` without re-running the
scorer. The cost, stated plainly: the system prompt tells the model to coin its
own room names, so a model that calls its rooms "Place 1" and "Place 3" and
asserts a **perfectly correct** adjacency between them scores `edge_accuracy
0.00`, while a model that asserted nothing scores `—` and is excluded from the
mean. Asserting a correct edge in your own vocabulary is worse than asserting
none. This is the same defect as §5 limitation 1, in a third place.

### 2.8 Dead-reckoning drift — doc 06 §5.8

```
drift = ‖position_estimate − true_pose‖               [metres]
```

measured **per stage** at the stage's last logged turn — which is the
`declare_done` turn when the stage ended that way. Reported alongside it: the
count of accepted corrections in that stage and the magnitude
`‖old_xy − new_xy‖` of each.

**Two correction tools since TR.1 (2026-08-02), one record.** A correction now
comes from either `correct_to_anchor(name, reason)` — snapping to a point the
model recorded with `record_anchor` while standing on it — or coordinate-only
`correct_position(x, y, reason)`. Both append the same `Correction`, so the
count and the magnitudes mean what they meant before; the record additionally
carries `anchor` (the name, or `null`), so the two arms can be separated. Report
them separately whenever both occur: they are different cognitive acts, and on
the batch that motivated the split they had very different consequences. The
pre-TR.1 `correct_position(place=…)` mode resolved a room's or a doorway's
*first-sighting* position — where the robot happened to stand when it described
an area, not a point it could return to — and across `raw_v5d_r2`, **14 of 15
accepted corrections made the true error worse, for a net +3.72 m**
(`results/forensics_v5d_r2/batch_summary.json`; worst single case
`sonnet5_seed101` t21, 0.024 m → 1.504 m). A drift table from that batch is
therefore not comparable to one from a post-TR.1 batch on the corrections axis:
the tool the model was offered is different.

Dead reckoning integrates **simulated leg odometry** (the call's true
displacement through a seeded error model), so it drifts because measurement
error compounds. Final drift measures how honest the estimate ended up; the
corrections series shows whether the model actively did cognitive loop closure
or just rode the drift.

> **Not comparable across the 2026-07-30 redesign boundary.** Before that date
> the integrator consumed COMMANDED velocity, so drift also absorbed every
> metre credited to a robot that was blocked — in the v4 batch that accounting
> artifact dominated the number (one trial: 25.10 m of a 26.65 m error). Drift
> figures from the v4 batch and from later batches measure different things and
> must not be plotted on one axis or differenced. Compare within a batch.

**The two halves must be sampled at the same instant.** They are not stored that
way. Inside one turn record, `obs.position_estimate` is captured *before* the
tool calls are dispatched and `true_pose` *after* them, and doc 05 §3.3
explicitly allows `declare_done` to follow a `move` in the same turn. Pairing
them across that gap charges the model **the whole length of its last move** as
"drift".

That is not a rounding concern; it was the dominant term. Measured on a trial
whose belief equals ground truth at the instant `obs` was captured on every turn
— a mathematically perfect dead-reckoner — the old pairing reported
**1.3583 m**, exactly `dist((1.2, 0.9), (2.55, 0.75))`, the final move; the same
two moves with `declare_done` split into its own turn reported **0.0000 m**. Same
robot, same belief, 1.36 m apart. On the golden fixture it published 0.4810 m
where the model had just `correct_position`-ed *onto* the true pose. Bundling
`move` + `declare_done` is a per-model phrasing habit, so the headline metric of
the whole memory-scaffolding claim was partly measuring turn packing.

Four conventions this implementation pins:

- **Which pair — preferred.** `turns[].position_estimate_end`, the integrator
  *after* dispatch, against that same turn's `true_pose`. This is §5.8's "at the
  moment of `declare_done`" verbatim. A `correct_position` on the turn is already
  folded in, because it re-anchors the integrator. Writing this key is a
  `loop.py` change T4.1 **reports rather than makes** (file ownership), so the
  key is optional; every value it needs already exists in memory
  (`tools._record_motion` breadcrumbs `integrator.xy` after every motion) and is
  simply not written out. Each drift record publishes `paired_at` so a disputed
  number says which convention produced it.
- **Which pair — fallback**, when that key is absent: the last turn's
  `obs.position_estimate` against the true pose **at the instant that `obs` was
  captured**, which is the *previous* turn's logged `true_pose` (physics advances
  only inside `env.step()` and the sim is paused while the request is assembled),
  or the stage's start pose for the stage's first turn. Its cost, stated plainly:
  it measures the belief one turn before the declaration, so drift accrued during
  a final bundled move is not counted. Under-measuring a fraction of one move is
  the honest error; the old pairing *added* a whole move. A `correct_position` on
  that turn still supersedes the estimate — it is the model's corrected belief
  about the pose it was looking at — which assumes the correction preceded that
  turn's motion. `Correction.turn` records no intra-turn position, so that
  assumption is unverifiable from the log; the preferred pairing above removes
  the ambiguity entirely, which is the argument for landing it.
- **Which turn.** The stage's last turn, so a capped or fallen stage still gets a
  number. §5.8's rationale applies just as much there, and dropping it would
  delete the metric for exactly the trials most worth explaining. Only an unrun
  stage is `—`.
- **Corrections after the declaration are ignored.** Filtering is by the
  correction's stamped `stage` first (`Correction.turn` is stage-local, so two
  stages share turn numbers), then by `turn ≤ the declaring turn`. When several
  land on the declaring turn, the **last** wins: that is what the model believed
  when it declared.

**Precision.** `obs.position_estimate` is logged to 2 decimals, so a
fallback-paired drift carries ±0.7 cm of quantisation while being reported to 4.
`position_estimate_end` would be logged to 4.

### 2.9 Layout QA — doc 06 §5.9

After the episode ends — **including** after a cap-out or a fall — the model is
asked five fixed questions in a genuinely fresh exchange: no system prompt, no
tools, no camera, nothing but its own final memory block. Each answer scores
**0 / 0.5 / 1**; the QA score is the **mean of the five**.

The question texts and all fifteen rubric anchors are frozen in
`prompts.py::LAYOUT_QA_QUESTIONS`; `tests/test_memory.py` asserts they still
match doc 06 §5.9 verbatim. An answer the loop could not parse is logged as `""`
and **scores 0** — the harness never invents text, and `final.qa_raw` keeps the
unsplit reply so any disputed score can be re-derived.

| # | Question | 1 | 0.5 | 0 |
|---|---|---|---|---|
| 1 | Which room connects the bedroom to the kitchen? | the unique connector (`hallway`) | a room adjacent to exactly one of the two | anything else |
| 2 | Starting at the front of the sofa, directions to the fridge | oracle room sequence **and** initial direction **and** ends at the fridge | exactly one of those three wrong | route would not reach the kitchen |
| 3 | How many rooms did you visit? Name them | names and count match the true visited set | names right but count off by one, or one room missing/extra | otherwise |
| 4 | Which compass direction is the kitchen from your spawn? | the true bearing bucketed 8-way | an adjacent bucket | otherwise |
| 5 | Name one landmark in each room you visited | a true layout landmark for every visited room | correct for all but one | otherwise |

Parse rules, per question:

- **Q1** — the question names two rooms, and models restate them ("the room that
  connects the *bedroom* to the *kitchen* is the hallway"), so the scorer takes
  the first mentioned room that is **not** `bedroom` or `kitchen`, falling back
  to the first mention if the answer names nothing else.
- **Q2** — see §3 below.
- **Q3** — room names come from the frozen synonym table; a stated count is read
  only when it sits next to the word "room(s)", so "Room 1 was…" is not a count.
  A count the answer never states is taken as implied by the naming. Rooms the
  answer says it did **not** visit are not counted (see *negation* below).
- **Q4** — bucketing uses `apartment_layout.compass_8` itself, never a second
  bucketer. Compass words and **uppercase-only** abbreviations are accepted
  (lowercase would read the "e" of "i.e." as east), and an abbreviation must
  **stand alone**: one followed by `/`, or by `.` plus an alphanumeric, is
  punctuation rather than a compass claim. Both traps were live — `score_q4("N/A")`
  returned **0.5**, half a point for declining to answer, and `"E.g. somewhere to
  the southwest"` was scored on the abbreviation of *exempli gratia* rather than
  on the model's actual answer. A trailing full stop at a sentence end is fine
  ("the kitchen is NE."). The **first surviving** token is scored — the answer's
  leading claim; scoring the last would read "Northeast. The bedroom is to the
  south." as an answer of *south*.
  Gold answers, computed from the committed layout:
  **101 = NE, 102 = SW, 103 = SE, 104 = SE.** Seed 101's bearing is 22.521°, i.e.
  **0.021° past the E/NE boundary** — a model answering "E" is 0.021° from
  correct and scores 0.5. That is the rubric working as intended, and the report
  must say so.
- **Q5** — the answer is segmented by room mention, so a landmark is credited to
  the room it was attached to ("Living room: the fridge." earns nothing). Each
  mention owns text on **both sides** of itself: forward to the next mention, and
  backward as far as the previous mention but **never across a sentence
  boundary**. Forward-only was the original rule and it scored every
  landmark-before-room phrasing 0.0 while the answer named a true landmark for
  every visited room — measured: "The sofa is in the living room and the fridge
  is in the kitchen." → 0.0, "A sofa in the living room, a fridge in the
  kitchen." → 0.0, "Sofa - living room. Fridge - kitchen." → 0.0, "I saw a blue
  rug (living room) and a fridge (kitchen)." → 0.0. Answer order is a per-model
  habit, so that shifted a published comparison for a reason unrelated to map
  quality. The sentence clamp is what keeps the swap case scoring 0.
  A landmark's head noun ("table" for "coffee table") is accepted because every
  head noun in this layout is unique across rooms; `tests/test_scoring.py`
  asserts that stays true. The denominator is **every visited room**, named or
  not, and 0.5 ("correct for all but one") requires at least one room correct —
  otherwise a blank answer against a single-room trial scored half a point.

**Negation (Q1, Q3, Q4).** A room or compass word inside the scope of a negation
is not a claim, it is the opposite of one. Enumerating what you did *not* visit
is ordinary LLM answer style, and the mention-set reading scored "I visited two
rooms, the living room and the kitchen. I did not see the bedroom or the
hallway." — a fully correct answer — **0.0**. Scope runs from the cue (`not`,
`never`, `nor`, `neither`, `without`, `except`, `didn't`…) to the end of its
**clause**, where a clause ends at a sentence boundary (including an em dash), a
comma, or a contrastive conjunction (`but`, `however`, `though`, `yet`…). Bare
"no" is deliberately **not** a cue: "No, I visited the living room and the
kitchen." is an affirmative answer that starts with it. Q5 is exempt — there a
mention is a structural anchor for segmentation, not a claim about visiting.

Two residuals, stated rather than discovered later: a comma-separated negated
list ("I did not see the bedroom, the hallway") only negates up to the comma
(that direction errs towards today's behaviour and costs half a point, not a
full one), and a negation that *follows* its target ("The living room does not
connect them — the hallway does.", Q1) is not detected.

### 2.10 The true trace

`true_rooms_visited` (§2.7) and Q3/Q5's gold visited set both come from the
**union** of the 5 Hz `pose_trace` samples, the per-turn `true_pose` entries and
the spawn. Doc 06 says "the true trace" without saying which series; the union is
the honest reading — the 5 Hz samples are the finest-grained record of where the
robot was, and the per-turn poses cover turns that stepped no physics and
therefore produced no samples. Room polygons tile the whole apartment floor, so a
sample inside a room means the robot's centre was in that room; there is no
corner-clipping case where the union over-reports a visit.

It is **trial-scoped, not stage-scoped** — deliberately, because memory is: the
map carries across the stage boundary, so a room first entered on the way home
still belongs in §2.7's recall denominator and in Q3/Q5's gold set. A
stage-1-only reading would shrink both for every trial that takes a different
route home, which is the common case.

**There is no dwell threshold, and that is a decision with a cost.** `room_at`'s
bounds are half-open, so a doorway centre belongs to exactly one room —
`room_at(2.55, 2.7)` is `hallway`. A robot that pokes into the hallway for a
single 0.2 s sample and retreats therefore *has visited* the hallway: it raises
§2.7's recall denominator (an otherwise perfect map scores 2/3), it makes Q3's
gold count 3 (so "I visited two rooms…" drops from 1.0 to 0.5) and it makes Q5
require a hallway landmark the model barely saw. Four centimetres of trajectory
can decide those. The alternative — requiring N ≥ 2 consecutive samples — trades
that for the opposite error, dropping a room genuinely entered at the very end of
a trace, and would silently under-report. The union rule is kept, is fixtured
(`test_a_single_doorway_sample_counts_as_a_visit`) so it cannot change by
accident, and **the write-up must state it** wherever recall or Q3/Q5 is
discussed.

---

## 3. Q2's direction-vocabulary parse rules (doc 06 §12, resolved)

Doc 06 §12 left Q2's operationalization open and PLAN T4.1 requires the rules to
be **authored and committed with fixtures before the batch**, so the vocabulary
cannot be tuned after model answers are visible.

**Where they live.** Module-level constants in `duck_embody/scoring.py` —
deliberately **not** a config file. They are post-hoc scorer logic: they never
touch what a model sees, so they are not a doc 06 §2 fairness item, and §7's
config-hash guard does not hash `scoring.py` (re-scoring is free; re-running a
paid batch is not). The honesty mechanism is the **commit**, not the hash. **Any
post-batch change to these rules re-scores all models together and is logged in
`results/rerun_log.md`.**

**Gold facts, computed from the committed layout** (never transcribed from doc 06
§11, whose row was produced against that section's own representative layout):

| Fact | Value |
|---|---|
| start room (`room_at(sofa)`) | `living_room` |
| goal room (`room_at(fridge)`) | `kitchen` |
| oracle route | `living_room → kitchen` (direct doorway at (1.8, 1.2)) |
| start point — "the front of the sofa" | **(0.4955, 1.60)**, the midpoint of the sofa's east face |
| gold initial bearing (direct doorway) | **342.953°** (`compass_8` → E) |
| gold initial bearing (hallway route) | **69.810°** |
| `oracle_length(sofa, fridge)` | **3.1521 m** direct vs **3.6107 m** via the hallway (**+14.55 %**) |

**Scored predicates.** The rubric asks for "initial direction correct", so only
the **first** direction is scored; the route is never simulated, because the
answers carry no reliable distances and simulating would invent geometry.

1. **Room sequence** — the frozen `ROOM_SYNONYMS`, matched whole-word,
   longest-phrase-first, non-overlapping, in reading order, with immediate
   repeats collapsed. Whole-word matching is why `"kitchenette"` still matches
   nothing.

   Three entries — **`entry`, `entryway`, `landing`** — are skipped when scanning
   free prose, because in English they are ordinary words for a *doorway*. The
   frozen table maps them to `hallway`, so before this they inserted a phantom
   room: measured, "walk east through the **doorway** into the kitchen" scored
   **1.0** and the byte-identical sentence with "**entry**" scored **0.0**,
   because the sequence became `living_room → hallway → kitchen`, which costs the
   exact-route defect *and* recomputes the gold bearing to the hallway doorway
   (69.810°), turning the correct "east" into a second defect. One word choice,
   unrelated to map quality, moved 20 % of the QA metric by a full point.
   `extract_room_mentions("the landing gear")` was `["hallway"]`.
   The skip is **scoring-local and prose-only**: `ROOM_SYNONYMS` itself is a doc
   06 §2 frozen fairness item shared with T2.3's already-passed scene gate and is
   not touched, and room-*name* normalisation (§2.7) keeps the full table, so a
   model that names a room "Entry" is still matched. Residual risk: `hall`,
   `passage`, `corridor` and `den` stay in the prose vocabulary because here they
   are genuine room words far more often than not. The list is fixtured on both
   sides.
2. **Initial direction** — the first **cue-anchored** direction token, resolved
   to an absolute bearing in the prompt's own frame (degrees counter-clockwise
   from east). Correct iff `|Δbearing| ≤ 45°` against the **continuous** gold
   bearing.
3. **Ends at the fridge** — the answer contains "fridge" or "refrigerator".

**Cue anchoring** is what stops a spatial *description* being read as an
instruction: a direction token counts only if a motion cue
(`turn/head/walk/go/face/toward/…`) occurs within 4 tokens before it, or the
token's own first word is a cue (`bear left`, `turn around`). There is no
charitable fallback to un-cued tokens — an answer that only describes the layout
has a "missing turn direction", which the frozen 0.5 anchor already covers. So
"The sofa is against the **west** wall. Walk **east**…" scores on *east*.

**Why a 45° wedge and not a `compass_8` bucket.** The oracle path's first metre
bears **336.80°**, which is **0.70°** from the SE/E boundary — bucketing would
make Q2's headline bit turn on 0.7°. The wedge accepts "east" (17.0° off) and
"southeast" (28.0° off — the leg that actually clears the coffee table, whose
inflated south-west corner bears 330.8° from the start) and rejects "northeast"
(62.1°) and "left"/"north" (72.9°). The cost, stated plainly: a model guessing
uniformly among the eight compass words has a **25 % chance** of scoring the
direction clause.

**Relative → absolute** resolves against `INITIAL_FACING_DEG = 0` (east — the
sofa's long axis runs north–south against the west wall with the coffee table and
rug directly east of it): `left +90`, `right −90`, `straight/forward/ahead 0`,
`bear|veer|slight ±45`, `sharp|hard ±135`, `turn around / u turn / reverse 180`,
`turn N degrees left|right ±N`. Absolute: the eight compass words and their
`-ward(s)` forms, **uppercase-only** `E/NE/N/NW/W/SW/S/SE`, and explicit headings
(`heading 345`, `345 degrees`, `345°`). Explicit degrees are matched **first** so
"turn 90 degrees left" is never also read as the absolute heading 90.

Two English traps, both fixed narrowly and fixtured:

- `"go right through the door"` — `right` as an intensifier, blocked before a
  short list of adverbial heads (`through/past/away/up/down/…`). `at` and `into`
  are deliberately **not** blocked: "turn right at the doorway" is a real turn.
- bare `"back"` / `"behind"` are **not** in the vocabulary (`"the back wall"`,
  `"walk behind the counter"`). Only unambiguous forms remain: `go/head/walk/turn
  back`, `double back`, `backward(s)`, `turn around`, `u turn`, `about face`,
  `reverse`.

**Route tolerance — doc 06 §5.9's "decide before freeze".** The hallway route
scores **0.5 — not 1, and not 0**. It is the only reading consistent with the
frozen anchors: the 1 anchor requires the sequence to match the *oracle* route,
and the 0 anchor is "route would not reach the kitchen", which the hallway route
plainly does (3.611 m, +14.55 %, a route the robot plausibly walked). It is
scored as the 0.5 anchor's "one wrong room name". When the answer offers the
hallway route the gold initial bearing is **recomputed for that route** (69.810°),
so a hallway answer is penalised once for the extra room rather than twice for a
single decision.

**Normalizations**, applied in order to the mention list:

- **N1** collapse runs of the same room;
- **N2** drop a goal-room-only preamble ("The fridge is in the kitchen. From the
  living room, …"), only if the remainder still reaches the goal;
- **N3** prepend the implied start room, only if the answer names no start room
  **and** gave a cue-anchored direction (without that condition, "The fridge is
  in the kitchen. I do not remember the way." would be normalized into the oracle
  route);
- **N4** reverse-order salvage ("walk into the kitchen from the living room").
  This **always** counts as a defect, so a backwards-phrased answer can never
  score 1.0 — the parser cannot tell a phrasing quirk from a reversed route.

**Scoring ladder.** Floor first: if the sequence is not a graph walk ending in
the kitchen (with at most one extra room), the answer scores **0**, whatever else
it got right. Otherwise count defects — not the oracle route, direction wrong or
missing, fridge not named — and map `0 → 1.0`, `1 → 0.5`, `≥2 → 0.0`. The
multi-defect case is the **one extension** beyond the frozen anchors, which
enumerate no such case.

**Fixtures.** 35 cases in `tests/fixtures/qa_q2_answers.json`, run against
`scoring.score_q2` in `tests/test_scoring.py::TestQ2ParseRules`. Every frozen
constant is pinned from **both** sides, because a constant pinned from one side
is a constant that can drift:

| Constant | Pinned below by | Pinned above by |
|---|---|---|
| `DIRECTION_TOL_DEG = 45` | `heading 27.9` → 1.0 | `heading 28.1` → 0.5 |
| `CUE_WINDOW = 4` | `q2_1_cue_at_the_window_edge` (cue exactly 4 tokens out) | `q2_05_uncued_direction_just_out_of_reach` (uncued at 5) |
| `MAX_EXTRA_ROOMS = 1` | `q2_05_hallway_detour` | `q2_0_four_room_walk` |
| `PROSE_AMBIGUOUS_SYNONYMS` | `q2_1_through_the_entry` | `test_the_prose_ambiguous_list_is_pinned_on_BOTH_sides` |

The tolerance pair also pins the start point: they flip if either constant moves.

---

## 4. Statistics — doc 06 §6

Every metric is reported as **mean ± bootstrap 95 % confidence interval** over
the N=4 trials.

The bootstrap, fully specified so the numbers are reproducible:

- resample the **defined** per-trial values with replacement — a `—` cell is
  excluded from the resample, not zero-filled;
- **10,000** resamples (`configs/benchmark.yaml: scoring.bootstrap_resamples`);
- take the mean of each resample;
- report the **2.5th and 97.5th percentiles**, **percentile method, not BCa** —
  at N=4 the sophistication of the interval method is noise, reproducibility is
  not;
- percentiles use **linear interpolation** (the numpy default: position
  `(n−1)·q`), pinned because at N=4 nearest-rank would give a different interval;
- each interval is drawn from **its own** `random.Random(seed)`, with
  `seed = 20260726` (`configs/benchmark.yaml: scoring.bootstrap_seed`) — the only
  place the seed exists. Per interval rather than one RNG threaded through the
  run, so an interval is a pure function of (values, resamples, seed) and can be
  re-derived in isolation from the committed config. (This paragraph previously
  claimed one RNG for the whole run, which described a threading the code never
  did; corrected here in the same commit, AGENTS.md rule 5.) The frozen config is
  handed out **read-only** — `lru_cache` caches the object, not the file read, so
  one accidental write would silently re-seed every later bootstrap in the run.

The defaults are exercised, not just declared: `bootstrap_ci(values)` with no
arguments is asserted equal to `bootstrap_ci(values, resamples=…, seed=…)` read
from the YAML, and unequal under a different seed and a different resample count.
Without that, the code path the real batch runs through had no coverage at all —
every other statistics test passes both explicitly.

**No interval is reported when fewer than 3 values are defined.** Doc 06 §3.2
states this for the conditional return-home SR ("a bootstrap over one value is
theatre"); it generalises to every metric, because a metric with two defined
values out of four is the same situation. A missing interval is drawn and printed
as *missing*, with its `n` — never as a zero-width whisker, which would claim a
precision the data does not have.

[measured 2026-07-26] At N=4 and 10,000 resamples the reported endpoints are
identical across seeds tried (1, 7, 20260726): the locked seed buys exact
reproducibility, and the interval width is a property of the four values rather
than of the RNG.

---

## 5. Known limitations, stated rather than discovered later

These are real and affect published numbers. None is a bug; each is a
consequence of a frozen decision.

1. **Model-coined room names are penalised twice.** The system prompt tells the
   model "you choose your own names for the rooms you find". An answer like "From
   Place 1 walk east into Place 3" contains no matchable room name, so it scores
   0 on Q2 *and* costs map precision (§2.7) — the same defect counted in two
   metrics. Q2 is fixtured for this case (`q2_0_own_room_names`). Make that
   **three** metrics: an adjacency asserted between two coined names counts as a
   wrong edge, so a correct adjacency in the model's own vocabulary scores
   `edge_accuracy 0.00` while asserting none scores `—`. `edges_unresolved` is
   published so the other convention stays recomputable (§2.7).
2. **Q2 is partly determined by the spawn, not the model.** Seed 102 spawns in
   the bedroom, whose oracle route to the kitchen never enters the living room. A
   seed-102 trial can therefore succeed at `find_kitchen` without ever seeing the
   sofa, and is then near-certain to score 0 on Q2 — 20 % of that trial's QA
   score — for a reason unrelated to map quality. Seeds 101/103/104 all have the
   living room on their route. **Report Q2 per seed alongside the mean.**
3. **Q4 has a 0.021° margin on seed 101.** See §2.9.
4. **Q2's initial facing is pinned, not detected.** "Starting at the front of the
   sofa" does not say which way you face; east is pinned. "Left" and "right" are
   wrong under *both* readings, so the constant only matters for
   straight/forward/ahead and the ±45 tokens — but an answer that re-frames
   itself ("with your back to the sofa, turn right") is still scored against the
   pinned facing.
5. **Q2 normalization N2 can promote a 0.5 to a 1** for an answer that describes
   a wrong route and then separately restates the endpoints in oracle order. It
   can never rescue a 0 (the trimmed sequence must still be a graph walk ending
   in the kitchen) and needs unusual phrasing. `final.qa_raw` keeps the original
   reply, so any disputed score is re-scorable.
6. **Map evidence is the turn's post-execution pose.** A room claimed *before*
   that turn's `move` is placed where the turn ended, up to one turn's motion
   away. The majority rule absorbs it; the alternative — reconstructing intra-turn
   ordering from `model_output.tool_calls` against `execution.calls` — mis-aligns
   whenever a motion call errored and produced no execution record.
7. **Q1's answer is the first mention that is not one of the question's two
   rooms**, and a negation that *follows* it is not detected: "The living room
   does not connect them — the hallway does." scores 0.5, not 1.0. The backward
   negation scope (§2.9) does not reach it, and a forward rule would need to know
   which room the sentence's verb is about.
8. **Q4 is scored on the first surviving compass token**, so an un-negated
   self-correction ("the kitchen is east… more precisely northeast") is scored on
   *east* and gets 0.5. The alternative (score the last token) breaks the equally
   common "Northeast. The bedroom is to the south."; both orderings are fixtured
   so the choice is at least visible.
9. **Drift's fallback pairing measures one turn early.** Until `loop.py` logs
   `position_estimate_end` (§2.8), a stage whose `declare_done` is bundled with a
   final `move` has that move's accrued drift excluded. This under-reports by a
   fraction of one move; the pairing it replaced *over*-reported by a whole move,
   and did so as a function of each model's turn-packing habit.
10. **The Q2 fixture corpus is hand-authored.** T3.5's sanity trial has not run,
   so no fixture is a real model answer. Hand-written English systematically
   under-represents bullet lists, markdown and step numbering. **At least one
   genuine answer per model should join the corpus before the freeze**, and PLAN
   T4.1's smoke step ("score the T3.5 sanity JSON end to end and eyeball every
   number") must be re-run against the real JSON when it exists.
11. **The success criterion was widened after the batch (v2), which is a
   post-hoc choice and is disclosed as one.** §2.1's any-counter criterion was
   adopted 2026-07-27 with all 12 results visible, motivated by one specific
   trial (`gpt56sol_seed103`). The protections: it was applied to all trials
   of all models together, it is a strict superset of the pre-registered
   region (no success was revoked; by construction it can only add), both
   verdicts ship per trial in `scores.json`, the sensitivity analysis that
   preceded adoption is committed (`scripts/rescore_any_counter.py`), and the
   change is logged in `results/rerun_log.md` with the geometry adversarially
   verified. What it cannot fix: the live stage-2 gate ran under the
   pre-registered predicate, so the v2 success has no `return_home` data, and
   the conditional return rate excludes it (published as
   `stage1_successes_never_offered_return`).

---

## 6. Provider usage and cost accounting

`Usage` stores one normalized partition for every response:

- `input_tokens_total = input_tokens_uncached + cache_read_tokens +
  cache_write_tokens`;
- `output_tokens_total`;
- nullable `reasoning_tokens` and `provider_reported_total_tokens`;
- `cost_usd_estimate`, `pricing_version`, and `pricing_source`.

Anthropic's Messages API reports `input_tokens` as the uncached remainder, with
`cache_read_input_tokens` and `cache_creation_input_tokens` outside it. Therefore
the adapter adds all three to obtain total input. OpenAI's Responses API reports
`usage.input_tokens` as total input; `input_tokens_details.cached_tokens` and
`cache_write_tokens` are subsets. Therefore the adapter subtracts both to obtain
uncached input before billing. The provider's complete raw usage object is also
archived under response metadata; it contains no API key, prompt, or reasoning
content.

Prices are explicit per bucket in each model YAML. For GPT-5.6 Sol standard
short-context processing on 2026-08-02: uncached input $5.00/MTok, cache reads
$0.50/MTok, cache writes $6.25/MTok, and output $30.00/MTok. Source:
https://developers.openai.com/api/docs/pricing and
https://developers.openai.com/api/docs/guides/prompt-caching. Reasoning tokens
are already included in OpenAI `output_tokens` and are not billed a second time.

A controlled GPT-5.6 Sol probe on 2026-08-02 confirmed the documented
partition. With one explicit 7,203-token breakpoint, call 1 reported 7,210 total
= 7 uncached + 7,203 writes; the identical call and a changed-suffix call each
reported 7,210 = 7 uncached + 7,203 reads. Estimated costs were $0.04520375 for
the write and $0.00378650 for each read. The complete raw usage objects and
response IDs (no prompt, cache key, output, or credential) are archived at
`results/probes/gpt56_cache_usage_20260802.json`.

### Historical `v5d_r2` disposition

Raw trial JSON is immutable and remains unchanged. Its GPT records captured
total input and cache reads but not GPT-5.6 cache writes. Exact GPT cost is
therefore unrecoverable. `scripts/build_scores.py` publishes both the original
reported value and a corrected lower bound:

`(input_total - cache_reads) × ordinary_input_rate + cache_reads × read_rate +
output × output_rate`.

This treats every unknown non-read token as ordinary input. Any hidden cache
write can only increase the charge by the 25% write premium, so the result is a
genuine lower bound rather than an invented point estimate. For seeds 101–104,
original → lower bound is respectively $0.869054 → ≥$0.576264,
$1.373387 → ≥$0.876917, $1.582117 → ≥$1.085647, and
$1.450030 → ≥$0.940830. Source fields: each immutable
`results/raw_v5d_r2/gpt56sol_seed*.json final.tokens`; formula:
`duck_embody.scoring.historical_openai_cost_lower_bound`.

**Audit/report amendment (TR.8, 2026-08-02).** The table is PROVISIONAL, not a
publication-ready benchmark report. This batch predates write-once manifest
SHAs, request reconstruction journals, per-request frame hashes, resolved-model
metadata, and normalized usage fields. The strict auditor reports those checks
as `INCOMPLETE`; it never treats “not recorded” as PASS. Correction columns are
now accepted/rejected counts from `duck_embody.forensics` (16 calls = 15/1 for
the batch), and drift is replayed through `scoring.stage_drift`, not read from
nonexistent `final.stages.*.drift_m` fields. `opus5_seed101` is disclosed as a
published-v2 success that the historical live point-disc gate did not offer a
return leg.
