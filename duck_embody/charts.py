"""Figures for README/results — doc 06 §10.2/§10.4 + the reporting extras.

doc 06 §10 enumerates four reporting deliverables; the figures land here:

* **§10.2 per-metric bar charts** with CI whiskers — :func:`bar_with_ci` (one
  metric per file) and :func:`per_metric_bars` (the 8-panel headline grid).
* **§10.4 trajectory-vs-belief map figures — "the money figure"** — one per
  trial: the floor plan with (a) the true path, (b) the dead-reckoned belief,
  and (c) the model's claimed rooms. :func:`trajectory_vs_belief`.
* :func:`turns_survived` — per-trial turns with end-reason colouring (the
  "how did every trial end" figure an 11/12-failure batch needs).
* :func:`per_trial_table` — doc 06 §6's per-trial markdown table.

Every number a figure draws comes from :mod:`duck_embody.scoring` —
:func:`~duck_embody.scoring.metric_estimates` for the bars (the same call
``summarise`` is built from, so a figure can never disagree with the table
beside it), :func:`~duck_embody.scoring.score_trial` output for the per-trial
figures. Nothing here re-derives a metric.

Honesty conventions (doc 06 §6, kept everywhere):

1. **A missing CI is drawn as a missing CI.** ``Estimate.ci is None`` means
   fewer than three trials had a defined value (§3.2's ``k < 3`` rule); the bar
   is drawn with no whisker and annotated with its ``n``, never with a
   zero-width whisker.
2. **"—" is never plotted as 0.** An excluded cell is annotated, not drawn.
3. **Axes are honest**: 0–1 metrics get a fixed 0–1 axis; count/metre axes
   start at 0. A zero success rate is drawn (zero-height bars labelled
   ``0/4``), never dropped from the grid — under criterion v2 the batch pools
   1/12, with two of three models at 0/4.

**Palette** (validated with the dataviz six-checks validator, light surface
``#fcfcfb``): the three model hues pass the all-pairs CVD gate (worst pair
ΔE 9.2 deutan, normal-vision 24.0); the two end-reason hues pass adjacent
(ΔE 41.0). ``#1baf7a`` and ``#eda100`` sit below 3:1 contrast on the light
surface, so every bar carries a visible direct label (the validator's relief
rule), and ``declare_done`` bars additionally carry a hatch so end-reason is
never colour-alone. One colour per model, fixed in :data:`MODEL_ORDER` order
across every figure, so a reader can follow a model through the write-up.

**Dead-reckoned path source**: the per-turn ``obs.position_estimate`` column.
``loop.memory_snapshot`` deliberately does not log the breadcrumb list because
"the series is exactly the per-turn ``obs.position_estimate`` column"
(loop.py); :func:`belief_path` reads that column, plus the last turn's optional
``position_estimate_end`` when a future log carries it.

**Claimed-room placement**: each claim is drawn at its FIRST logged claim
position — the turn's ``true_pose`` at the first ``update_room`` /
``set_current_room`` / ``add_landmark`` call naming it (the same
:func:`~duck_embody.scoring.room_evidence` points §5.7's matching uses) — so
nothing is drawn at an invented position. Matched claims are annotated with the
true room they matched; a claim with no logged evidence point is listed in the
caption instead of drawn. (The T4.1 skeleton sketched matched-centroid
placement; claim positions are what the task's figure needs and are equally
non-invented — recorded here rather than silently diverging.)

``matplotlib`` is imported lazily inside the figure functions: importing this
module must stay free for the test suite (AGENTS.md rule 2 — the gate runs in
0.5 s and must not pull a plotting stack). The pure helpers (path extraction,
labels, trial picking, the markdown table) never touch matplotlib and are the
unit-tested surface.

CLI (what ``scripts/make_figures.sh`` runs — reproducible from raw logs alone):

    python -m duck_embody.charts results/raw --out results/figures
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from duck_embody.env.apartment_layout import (
    LAYOUT,
    room_bounds,
    room_centroid,
    wall_rects,
)
from duck_embody.scoring import (
    NA,
    POSITION_ESTIMATE_END,
    STAGE_FIND_KITCHEN,
    Estimate,
    TrialMetrics,
    _finite,
    claimed_rooms,
    kitchen_counter_rects,
    room_evidence,
    true_trace,
)

#: One colour per model, fixed so a reader can follow a model across every
#: figure in the write-up (short keys; ``config.model`` holds the API ids).
MODEL_ORDER: tuple[str, ...] = ("fable5", "opus5", "gpt56sol")

#: Short key -> the API id the batch logged in ``config.model``. Figures label
#: models by API id (the honest, unambiguous name).
MODEL_API_IDS: dict[str, str] = {
    "fable5": "claude-fable-5",
    "opus5": "claude-opus-5",
    "gpt56sol": "gpt-5.6-sol",
}
_API_TO_SHORT: dict[str, str] = {api: short for short, api in MODEL_API_IDS.items()}

#: Categorical slots 1–3 of the validated reference palette — the only three
#: slots that pass the all-pairs CVD gate on the light surface.
MODEL_COLORS: dict[str, str] = {
    "fable5": "#2a78d6",  # blue
    "opus5": "#eb6834",  # orange
    "gpt56sol": "#1baf7a",  # aqua (sub-3:1 on light surface -> direct labels)
}
_FALLBACK_COLOR = "#4a3aa7"

#: Stage-1 end reasons -> colour (+ hatch as the colour-independent channel).
END_REASON_COLORS: dict[str, str] = {
    "fall": "#4a3aa7",  # violet
    "declare_done": "#eda100",  # yellow (sub-3:1 -> direct labels + hatch)
}
END_REASON_HATCH: dict[str, str] = {"fall": "", "declare_done": "///"}
END_REASON_LABELS: dict[str, str] = {
    "fall": "fall (tilt-60 termination)",
    # Under criterion v2 a declare_done can be either verdict (the batch has
    # one of each), so the legend names the reason and leaves the verdict to
    # the per-trial outcome column in summary_table.md.
    "declare_done": "declare_done (v2 verdict varies)",
    "turn_cap": "turn cap reached",
    "policy_seconds_cap": "policy-seconds cap reached",
}

# Chart chrome (reference palette "chrome & ink", light mode — PNGs commit to
# one mode deliberately; there is no viewer theme for a file on disk).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: The §10.2 headline grid: (metric key, panel title). Keys are
#: :data:`duck_embody.scoring.METRIC_COLUMNS` keys.
HEADLINE_METRICS: tuple[tuple[str, str], ...] = (
    ("find_kitchen.success_rate", "find_kitchen success rate"),
    ("find_kitchen.progress", "progress toward kitchen (0–1)"),
    ("find_kitchen.spl", "SPL (0–1)"),
    ("bumps", "bumps / trial"),
    ("find_kitchen.drift_m", "dead-reckoning drift (m)"),
    ("qa", "layout QA score (0–1)"),
    ("map_precision", "map precision"),
    ("map_recall", "map recall"),
)

#: Metric keys whose value space is the unit interval — their axis is a fixed,
#: honest 0–1 (doc 06 §10.2's "honest axes").
_UNIT_INTERVAL_BARE = {"map_precision", "map_recall", "edge_accuracy", "qa"}
_UNIT_INTERVAL_SUFFIXES = (".success_rate", ".progress", ".spl")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; no matplotlib)
# ---------------------------------------------------------------------------


def short_model_name(model: str) -> str:
    """``"claude-fable-5"`` or ``"fable5"`` -> ``"fable5"`` (identity for
    unknown names, so an unexpected model still gets a deterministic slot)."""
    return _API_TO_SHORT.get(model, model)


def model_display_name(model: str) -> str:
    """The API id to print on a figure, from either naming convention."""
    short = short_model_name(model)
    return MODEL_API_IDS.get(short, model)


def model_color(model: str) -> str:
    return MODEL_COLORS.get(short_model_name(model), _FALLBACK_COLOR)


def is_unit_interval(metric: str) -> bool:
    """Does this metric key live in [0, 1]? Drives the fixed 0–1 axis."""
    return metric in _UNIT_INTERVAL_BARE or metric.endswith(_UNIT_INTERVAL_SUFFIXES)


def bar_annotation(est: Estimate, *, success_ratio: bool = False) -> str:
    """The visible direct label a bar carries (the validator's relief rule).

    ``success_ratio`` prints doc 06 §3.2's honest ``x/N`` instead of a mean.
    A missing CI is *said* (``no CI, n=k``) — never drawn as a zero whisker —
    and a partially defined metric names its ``n``.
    """
    if isinstance(est.mean, str):
        return NA
    if success_ratio:
        text = f"{int(round(sum(est.values)))}/{est.n_total}"
    else:
        text = f"{est.mean:.2f}"
    n_defined = len(est.values)
    if est.ci is None:
        text += f" (no CI, n={n_defined})"
    elif n_defined < est.n_total:
        text += f" (n={n_defined}/{est.n_total})"
    return text


def axis_ceiling(estimates: Sequence[Estimate]) -> float:
    """Top of an unbounded (count/metre) axis: the largest mean-or-CI-high
    with 25 % headroom for labels; 1.0 when everything is zero/undefined."""
    tops = []
    for est in estimates:
        if isinstance(est.mean, str):
            continue
        tops.append(est.mean if est.ci is None else max(est.mean, est.ci[1]))
    peak = max(tops, default=0.0)
    return peak * 1.25 if peak > 0 else 1.0


def belief_path(document: dict) -> list[tuple[float, float]]:
    """The dead-reckoned belief trail: ``obs.position_estimate`` per turn.

    This IS the breadcrumb series — ``loop.memory_snapshot`` documents that the
    snapshot omits breadcrumbs because they duplicate exactly this column.
    The last turn's optional ``position_estimate_end`` (the post-dispatch
    belief, when a log carries it) is appended so the trail ends at the same
    vintage §5.8's preferred drift pairing uses.
    """
    points: list[tuple[float, float]] = []
    turns = document.get("turns", [])
    for turn in turns:
        shown = turn["obs"]["position_estimate"]
        points.append(
            (_finite(shown["x"], "obs.position_estimate.x"),
             _finite(shown["y"], "obs.position_estimate.y"))
        )
    if turns:
        end = turns[-1].get(POSITION_ESTIMATE_END)
        if isinstance(end, dict):
            points.append(
                (_finite(end["x"], f"{POSITION_ESTIMATE_END}.x"),
                 _finite(end["y"], f"{POSITION_ESTIMATE_END}.y"))
            )
    return points


def correction_snaps(
    document: dict,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Every ``correct_position`` snap, oldest first: ``(old_xy, new_xy)``.

    Read from the LAST turn's ``memory_snapshot.corrections``, which carries
    the whole series (the same source §5.8's scorer reads).
    """
    turns = document.get("turns", [])
    if not turns:
        return []
    snaps = []
    for c in turns[-1].get("memory_snapshot", {}).get("corrections", []):
        snaps.append(
            (
                (_finite(c["old_xy"][0], "correction.old_xy"),
                 _finite(c["old_xy"][1], "correction.old_xy")),
                (_finite(c["new_xy"][0], "correction.new_xy"),
                 _finite(c["new_xy"][1], "correction.new_xy")),
            )
        )
    return snaps


def claim_markers(
    document: dict, matches: Sequence[tuple[str, str]]
) -> tuple[list[tuple[str, tuple[float, float], str | None]], list[str]]:
    """Where to draw each claimed room, and which claims cannot be drawn.

    Returns ``(drawn, undrawn)``: ``drawn`` is
    ``(claimed name, first claim position, matched true room | None)`` per
    claim with at least one logged evidence point (the true pose at its first
    claiming call — a real logged position, never invented); ``undrawn`` lists
    claims with no evidence point, for the caption.
    """
    matched = dict(matches)
    evidence = room_evidence(document)
    drawn: list[tuple[str, tuple[float, float], str | None]] = []
    undrawn: list[str] = []
    for name in claimed_rooms(document):
        points = evidence.get(name)
        if points:
            drawn.append((name, points[0], matched.get(name)))
        else:
            undrawn.append(name)
    return drawn, undrawn


def pick_trajectory_trials(trials: Sequence[TrialMetrics]) -> list[str]:
    """One trial id per model for the §10.4 figures: the RICHEST run, i.e. the
    one with the most stage-1 turns (ties broken by trial id, deterministic).

    On the frozen batch this selects fable5_seed102 (14 turns),
    opus5_seed102 (28) and gpt56sol_seed103 (27).
    """
    best: dict[str, TrialMetrics] = {}
    for trial in trials:
        key = short_model_name(trial.model)
        turns = trial.stages[STAGE_FIND_KITCHEN].turns_used
        current = best.get(key)
        if (
            current is None
            or turns > current.stages[STAGE_FIND_KITCHEN].turns_used
            or (
                turns == current.stages[STAGE_FIND_KITCHEN].turns_used
                and trial.trial_id < current.trial_id
            )
        ):
            best[key] = trial
    ordered = [m for m in MODEL_ORDER if m in best]
    ordered += sorted(k for k in best if k not in MODEL_ORDER)
    return [best[key].trial_id for key in ordered]


def group_by_model(trials: Sequence[TrialMetrics]) -> dict[str, list[TrialMetrics]]:
    """Trials per model (API-id keys), in :data:`MODEL_ORDER`, seeds ascending;
    models outside the roster follow, alphabetically."""
    buckets: dict[str, list[TrialMetrics]] = {}
    for trial in trials:
        buckets.setdefault(short_model_name(trial.model), []).append(trial)
    ordered = [m for m in MODEL_ORDER if m in buckets]
    ordered += sorted(k for k in buckets if k not in MODEL_ORDER)
    return {
        MODEL_API_IDS.get(key, key): sorted(buckets[key], key=lambda t: (t.seed, t.trial_id))
        for key in ordered
    }


def _cell(value: float | str, digits: int = 3) -> str:
    return value if isinstance(value, str) else f"{value:.{digits}f}"


def per_trial_table(trials: Sequence[TrialMetrics]) -> str:
    """doc 06 §6's honesty clause: the per-trial table published with EVERY
    aggregate. Markdown, so it drops straight into the README and the report.

    "—" cells are printed as "—" (:data:`~duck_embody.scoring.NA`), never as 0.
    """
    lines = [
        "| Trial | Model | Stage-1 end | Success | Progress | SPL | Path (m) "
        "| Turns | Bumps | Falls | Drift (m) | Corr. | Map P | Map R "
        "| Edge acc | QA |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for trial in trials:
        s1 = trial.stages[STAGE_FIND_KITCHEN]
        acc = trial.map_accuracy
        lines.append(
            "| {id} | {model} | {end} | {ok} | {prog} | {spl} | {path} | {turns} "
            "| {bumps} | {falls} | {drift} | {corr} | {p} | {r} | {edge} | {qa} |".format(
                id=trial.trial_id,
                model=model_display_name(trial.model),
                end=s1.end_reason,
                ok="yes" if s1.success else "no",
                prog=_cell(s1.progress),
                spl=_cell(s1.spl),
                path=_cell(s1.true_path_m, 2),
                turns=s1.turns_used,
                bumps=trial.bumps,
                falls=trial.falls,
                drift=_cell(s1.drift_m),
                corr=s1.corrections,
                p=_cell(acc.precision, 2),
                r=_cell(acc.recall, 2),
                edge=_cell(acc.edge_accuracy, 2),
                qa=_cell(trial.qa.score, 2),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib plumbing (lazy — AGENTS.md rule 2)
# ---------------------------------------------------------------------------


def _plt():
    """Import matplotlib headless (this machine has no display) and apply the
    chart chrome once. Returns ``pyplot``."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "hatch.linewidth": 0.7,
        }
    )
    return plt


def _draw_metric_panel(
    ax,
    estimates: dict[str, Estimate],
    *,
    unit_interval: bool,
    success_ratio: bool = False,
    ylabel: str | None = None,
) -> None:
    """One metric, one bar per model, CI whiskers — shared by
    :func:`bar_with_ci` and :func:`per_metric_bars`."""
    models = list(estimates)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    if unit_interval:
        ax.set_ylim(0.0, 1.06)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.set_ylim(0.0, axis_ceiling(list(estimates.values())))

    for index, model in enumerate(models):
        est = estimates[model]
        if isinstance(est.mean, str):
            # "—" is never plotted as 0: annotate the excluded cell instead.
            ax.annotate(
                f"{NA} (n=0/{est.n_total})",
                xy=(index, 0.0),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK_MUTED,
            )
            continue
        ax.bar(
            index,
            est.mean,
            width=0.55,
            color=model_color(model),
            zorder=3,
        )
        top = est.mean
        if est.ci is not None:
            low, high = est.ci
            ax.errorbar(
                index,
                est.mean,
                yerr=[[max(0.0, est.mean - low)], [max(0.0, high - est.mean)]],
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=1.1,
                capsize=3,
                capthick=1.1,
                zorder=4,
            )
            top = max(top, high)
        # Visible direct label: the relief for the sub-3:1 hues, and the n /
        # missing-CI annotation doc 06 §6 requires.
        ax.annotate(
            bar_annotation(est, success_ratio=success_ratio),
            xy=(index, top),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_SECONDARY,
            annotation_clip=False,
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [model_display_name(m) for m in models], fontsize=7.5, rotation=12
    )
    ax.set_xlim(-0.6, len(models) - 0.4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def bar_with_ci(
    metric: str,
    estimates: dict[str, Estimate],
    out_path: Path | str,
    *,
    ylabel: str | None = None,
) -> Path:
    """doc 06 §10.2: one headline metric, one bar per model, CI whiskers.

    ``estimates`` maps model name → :class:`~duck_embody.scoring.Estimate` for
    ONE metric, i.e. ``{model: metric_estimates(trials_of(model))[metric]}``.
    An estimate whose ``ci`` is ``None`` is drawn WITHOUT a whisker and
    labelled with its ``n_defined`` (doc 06 §3.2/§6): a zero-width whisker
    would claim a precision the data does not have.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.6, 3.0), layout="constrained")
    _draw_metric_panel(
        ax,
        estimates,
        unit_interval=is_unit_interval(metric),
        success_ratio=metric.endswith(".success_rate"),
        ylabel=ylabel,
    )
    ax.set_title(metric)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def per_metric_bars(
    estimates_by_model: dict[str, dict[str, Estimate]],
    out_path: Path | str,
    *,
    metrics: Sequence[tuple[str, str]] = HEADLINE_METRICS,
) -> Path:
    """The §10.2 grid: every headline metric, three bars each, CI whiskers.

    ``estimates_by_model`` maps model name →
    :func:`~duck_embody.scoring.metric_estimates` output for that model's
    trials. The success-rate panel prints ``x/N`` per bar and the figure
    subtitle states the pooled rate outright, under both criteria (1/12 v2,
    0/12 pre-registered on the frozen batch) — the headline is in the figure,
    not a footnote.
    """
    plt = _plt()
    from matplotlib.patches import Patch

    models = list(estimates_by_model)
    columns = 4
    rows = math.ceil(len(metrics) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 2.9 * rows + 0.9),
        layout="constrained",
    )
    grid = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for slot, (metric, title) in enumerate(metrics):
        ax = grid[slot]
        _draw_metric_panel(
            ax,
            {model: estimates_by_model[model][metric] for model in models},
            unit_interval=is_unit_interval(metric),
            success_ratio=metric.endswith(".success_rate"),
        )
        ax.set_title(title)
    for slot in range(len(metrics), len(grid)):
        grid[slot].set_visible(False)

    # The pooled success null, computed from the same estimates the bars use.
    successes = total = 0
    for model in models:
        est = estimates_by_model[model]["find_kitchen.success_rate"]
        successes += int(round(sum(est.values)))
        total += est.n_total
    fig.suptitle(
        "Duck Embody 12-trial benchmark — per-metric comparison "
        "(N=4 seeds/model; mean, 95% percentile-bootstrap CI)\n"
        f"find_kitchen: {successes}/{total} successes under criterion v2 "
        "(any counter face; 0/12 pre-registered) — SPL is 0 for every failure "
        "by definition; return_home never ran (the live gate used the "
        f"pre-registered criterion). {NA} = undefined, excluded from means, "
        "never plotted as 0.",
        fontsize=10.5,
    )
    # The null belongs IN the figure, not only above it: the success-rate
    # panel's empty plot area states it outright.
    if metrics and metrics[0][0] == "find_kitchen.success_rate":
        grid[0].text(
            0.5, 0.55,
            f"{successes}/{total} successes\n(pooled over all models)",
            transform=grid[0].transAxes, ha="center", va="center",
            fontsize=10, color=INK,
        )
    handles = [
        Patch(facecolor=model_color(m), label=model_display_name(m))
        for m in models
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), fontsize=8.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# §10.4 trajectory vs belief — the money figure
# ---------------------------------------------------------------------------


def _draw_floor_plan(ax) -> None:
    """Walls + furniture + room names + target from the frozen layout."""
    from matplotlib.patches import Circle, Rectangle

    for x0, y0, x1, y1 in wall_rects():
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=INK_SECONDARY,
                      edgecolor="none", zorder=2)
        )
    for item in LAYOUT["furniture"]:
        cx, cy = item["pos"]
        w, h = item["footprint"]
        solid = item["collision"] != "none"
        ax.add_patch(
            Rectangle(
                (cx - w / 2.0, cy - h / 2.0), w, h,
                facecolor=GRIDLINE if solid else "#f0efec",
                edgecolor=BASELINE if solid else "none",
                linewidth=0.6,
                zorder=1,
            )
        )
    for room in LAYOUT["rooms"]:
        cx, cy = room_centroid(room)
        ax.text(
            cx, cy, room.replace("_", " ").upper(),
            ha="center", va="center", fontsize=8, color=INK_MUTED,
            alpha=0.85, zorder=1.5,
        )
    # Criterion v2's success region is ONE thing with two kinds of lobe — the
    # pre-registered target disc and the five counter bands (same radius) — so
    # every lobe is drawn in the SAME style under ONE legend entry. Drawing the
    # disc differently from the bands (the first rendering did) reads as "old
    # criterion still shown", when the disc is a live lobe of v2. The star
    # keeps its own entry: the point is still what progress/d_initial/d_final
    # measure to (docs/METRICS.md §2.2).
    target = LAYOUT["target"]
    tx, ty = target["point"]
    radius = target["radius"]
    region_style = dict(
        fill=False, linestyle=":", edgecolor=INK_MUTED, linewidth=0.9, zorder=2.4
    )
    ax.add_patch(
        Circle(
            (tx, ty), radius,
            label=f"success region v2 (target disc ∪ counter bands, {radius} m)",
            **region_style,
        )
    )
    ax.plot(
        [tx], [ty], marker="*", markersize=11, color=INK, linestyle="none",
        zorder=5, label="pre-registered goal point (progress reference)",
    )
    # The counter lobes: the same radius around each kitchen counter footprint
    # (Minkowski sum = rounded rectangle), CLIPPED to the kitchen — through-wall
    # proximity is not success and must not be drawn as if it were. Geometry
    # comes from the scorer, not re-derived here.
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.path import Path as MplPath

    kx0, ky0, kx1, ky1 = room_bounds("kitchen")
    kitchen_clip = MplPath(
        [(kx0, ky0), (kx1, ky0), (kx1, ky1), (kx0, ky1), (kx0, ky0)]
    )
    for _name, (x0, y0, x1, y1) in kitchen_counter_rects():
        band = FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad={radius}",
            **region_style,
        )
        ax.add_patch(band)
        # Clip AFTER add_patch (adding resets the artist's clip box) and in
        # explicit data coordinates: an un-added Rectangle as clip_path
        # silently fails to clip, drawing the through-wall zone as success.
        band.set_clip_path(kitchen_clip, ax.transData)


def _draw_scale_bar(ax, y: float) -> None:
    x0 = 0.15
    ax.plot([x0, x0 + 1.0], [y, y], color=INK, linewidth=1.6,
            solid_capstyle="butt", zorder=5)
    for x in (x0, x0 + 1.0):
        ax.plot([x, x], [y - 0.03, y + 0.03], color=INK, linewidth=1.2, zorder=5)
    ax.text(x0 + 0.5, y - 0.07, "1 m", ha="center", va="top", fontsize=8,
            color=INK_SECONDARY, zorder=5)


def trajectory_vs_belief(
    trial: TrialMetrics,
    document: dict,
    out_path: Path | str,
) -> Path:
    """doc 06 §10.4's money figure: true path vs dead-reckoned belief.

    Three layers over the floor plan drawn from
    ``duck_embody.env.apartment_layout``:

    (a) the true path — spawn + the 5 Hz ``pose_trace`` samples + per-turn
        ``true_pose`` (:func:`scoring.true_trace`), so within-turn curvature is
        visible rather than chorded away;
    (b) the dead-reckoned path — :func:`belief_path`'s
        ``obs.position_estimate`` series (one point per turn), with each
        ``memory_snapshot.corrections`` entry drawn as a snap arrow from
        ``old_xy`` to ``new_xy``;
    (c) the model's claimed rooms at their first logged claim positions
        (:func:`claim_markers`), matched claims annotated with the true room
        (``TrialMetrics.map_accuracy.matches``); claims with no logged claim
        position are listed in the caption rather than drawn somewhere
        invented.

    The gap between (a) and (b) IS the drift story; the published §5.8 drift
    number is quoted in the caption rather than drawn as a segment, because
    its pairing (see ``scoring.stage_drift``) compares instants one turn
    apart — a segment between the two trail ends would redraw the discredited
    pairing.
    """
    plt = _plt()
    color = model_color(trial.model)
    fig, ax = plt.subplots(figsize=(9.4, 8.6), layout="constrained")
    ax.set_aspect("equal")
    _draw_floor_plan(ax)

    # (a) True path.
    truth = true_trace(document)
    if truth:
        xs, ys = zip(*truth)
        ax.plot(xs, ys, color=INK, linewidth=1.7, alpha=0.9, zorder=4,
                solid_capstyle="round", label="true path (ground truth, 5 Hz)")
        ax.plot([xs[-1]], [ys[-1]], marker="x", markersize=9, markeredgewidth=2,
                color=INK, linestyle="none", zorder=6,
                label=f"true end ({trial.stages[STAGE_FIND_KITCHEN].end_reason})")

    # (b) Dead-reckoned belief.
    belief = belief_path(document)
    if belief:
        bx, by = zip(*belief)
        ax.plot(bx, by, color=color, linewidth=1.9, linestyle="--", marker="o",
                markersize=3.5, zorder=5, dashes=(4, 2.4),
                label="dead-reckoned belief (position_estimate / turn)")
        ax.plot([bx[-1]], [by[-1]], marker="s", markersize=8, fillstyle="none",
                markeredgewidth=1.8, color=color, linestyle="none", zorder=6,
                label="belief at end")
    snaps = correction_snaps(document)
    for (ox, oy), (nx, ny) in snaps:
        ax.annotate(
            "", xy=(nx, ny), xytext=(ox, oy),
            arrowprops={"arrowstyle": "->", "color": INK_SECONDARY,
                        "linewidth": 1.3, "shrinkA": 0, "shrinkB": 0},
            zorder=6,
        )
    if snaps:
        ax.plot([], [], color=INK_SECONDARY, linewidth=1.3,
                label=f"correct_position snap ({len(snaps)})")

    # Spawn marker (the trial's logged spawn — scoring cross-checks it).
    sx, sy = document["config"]["spawn"]["xy"]
    ax.plot([sx], [sy], marker="^", markersize=10, color=INK,
            markerfacecolor=SURFACE, markeredgewidth=1.6, linestyle="none",
            zorder=6, label="spawn")

    # (c) Claimed rooms.
    extent_x, _extent_y = LAYOUT["extents"]
    drawn, undrawn = claim_markers(document, trial.map_accuracy.matches)
    for name, (cx, cy), matched in drawn:
        ax.plot([cx], [cy], marker="D", markersize=7, color=INK_SECONDARY,
                markerfacecolor="none", markeredgewidth=1.5, linestyle="none",
                zorder=6)
        note = f"= {matched}" if matched else "no match"
        # Labels near the east wall flip to the left so they never run off
        # the figure edge.
        eastern = cx > 0.68 * extent_x
        ax.annotate(
            f"“{name}” ({note})",
            xy=(cx, cy),
            xytext=(-6 if eastern else 6, 6),
            textcoords="offset points",
            ha="right" if eastern else "left",
            fontsize=8, color=INK_SECONDARY, zorder=7,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": SURFACE,
                  "edgecolor": BASELINE, "linewidth": 0.5, "alpha": 0.85},
        )
    if drawn:
        ax.plot([], [], marker="D", markersize=7, color=INK_SECONDARY,
                markerfacecolor="none", linestyle="none",
                label="claimed room (at first claim position)")

    extent_x, extent_y = LAYOUT["extents"]
    ax.set_xlim(-0.25, extent_x + 0.25)
    ax.set_ylim(-0.55, extent_y + 0.25)
    _draw_scale_bar(ax, y=-0.32)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    s1 = trial.stages[STAGE_FIND_KITCHEN]
    outcome = s1.outcome
    if s1.outcome != s1.outcome_preregistered:
        outcome = f"{s1.outcome} (v2; {s1.outcome_preregistered} pre-registered)"
    caption = (
        f"stage-1 outcome: {outcome} · progress {_cell(s1.progress)} · "
        f"dead-reckoning drift {_cell(s1.drift_m)} m · bumps {trial.bumps} · "
        f"turns {s1.turns_used}"
    )
    if undrawn:
        caption += " · claims without a logged position (not drawn): " + ", ".join(
            f"“{name}”" for name in undrawn
        )
    fig.suptitle(
        f"{trial.trial_id} — {model_display_name(trial.model)}: "
        "true path vs dead-reckoned belief",
        fontsize=11.5,
    )
    ax.set_title(caption, fontsize=8.5, color=INK_SECONDARY, pad=8)
    fig.legend(loc="outside lower center", ncol=3, fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Turns survived
# ---------------------------------------------------------------------------


def turns_survived(
    trials: Sequence[TrialMetrics],
    out_path: Path | str,
    *,
    turn_cap: int | None = None,
) -> Path:
    """Per-trial stage-1 turns, coloured (and hatched) by end reason.

    ``turn_cap`` is the stage turn budget read from the trial logs
    (``turns[0].budget.stage_turn_cap``) — passed in rather than hard-coded so
    the reference line always states what the batch actually enforced.
    """
    plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    groups = group_by_model(trials)
    fig, ax = plt.subplots(figsize=(9.0, 4.4), layout="constrained")
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)

    positions: list[float] = []
    labels: list[str] = []
    reasons_seen: list[str] = []
    base = 0.0
    label_transform = blended_transform_factory(ax.transData, ax.transAxes)
    for model, rows in groups.items():
        for offset, trial in enumerate(rows):
            s1 = trial.stages[STAGE_FIND_KITCHEN]
            x = base + offset
            reason = s1.end_reason
            if reason not in reasons_seen:
                reasons_seen.append(reason)
            ax.bar(
                x,
                s1.turns_used,
                width=0.62,
                color=END_REASON_COLORS.get(reason, INK_MUTED),
                hatch=END_REASON_HATCH.get(reason, "xx"),
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(
                str(s1.turns_used),
                xy=(x, s1.turns_used),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )
            positions.append(x)
            labels.append(str(trial.seed))
        center = base + (len(rows) - 1) / 2.0
        ax.text(center, -0.13, model, transform=label_transform,
                ha="center", va="top", fontsize=9, color=INK)
        base += len(rows) + 1.2

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)  # seeds; the model names group them
    ax.set_ylabel("turns survived (find_kitchen)")

    top = max((t.stages[STAGE_FIND_KITCHEN].turns_used for t in trials), default=1)
    if turn_cap is not None:
        ax.axhline(turn_cap, color=BASELINE, linewidth=1.0, linestyle=(0, (5, 4)),
                   zorder=2)
        ax.annotate(
            f"stage turn cap = {turn_cap}",
            xy=(1.0, turn_cap), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color=INK_MUTED,
        )
        top = max(top, turn_cap)
    ax.set_ylim(0, top * 1.12)

    failed = sum(1 for t in trials if not t.stages[STAGE_FIND_KITCHEN].success)
    ax.set_title(
        f"Turns survived per trial — find_kitchen ({failed}/{len(trials)} "
        "trials failed)"
    )
    handles = [
        Patch(
            facecolor=END_REASON_COLORS.get(reason, INK_MUTED),
            hatch=END_REASON_HATCH.get(reason, "xx"),
            edgecolor=SURFACE,
            label=END_REASON_LABELS.get(reason, reason),
        )
        for reason in reasons_seen
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI — the scoring sweep + every figure, from results/raw/ alone
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Score every trial JSON in RAW_DIR and write the report figures.

    Reproducible from the raw logs alone: nothing is read from scores.json or
    any other derived artefact. ``scripts/make_figures.sh`` is the one-command
    wrapper.
    """
    import argparse
    import sys

    from duck_embody.scoring import IncompleteTrialError, load_trial, metric_estimates, score_trial

    parser = argparse.ArgumentParser(
        prog="python -m duck_embody.charts",
        description="Scoring sweep over raw trial JSONs + report figures.",
    )
    parser.add_argument("raw_dir", help="directory of trial JSONs (results/raw)")
    parser.add_argument("--out", default="results/figures", help="output directory")
    parser.add_argument(
        "--trajectories",
        default="auto",
        help="comma-separated trial ids for trajectory-vs-belief figures, "
        "'auto' (richest per model = most stage-1 turns) or 'none'",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)
    documents: dict[str, dict] = {}
    metrics: dict[str, TrialMetrics] = {}
    for path in sorted(raw_dir.glob("*.json")):
        try:
            document = load_trial(path)
        except IncompleteTrialError as error:
            print(f"skipping {path.name}: {error}", file=sys.stderr)
            continue
        trial = score_trial(document)
        documents[trial.trial_id] = document
        metrics[trial.trial_id] = trial
    if not metrics:
        print(f"no complete trial JSONs under {raw_dir}", file=sys.stderr)
        return 2

    trials = list(metrics.values())
    groups = group_by_model(trials)
    estimates_by_model = {
        model: metric_estimates(rows) for model, rows in groups.items()
    }

    written = [per_metric_bars(estimates_by_model, out_dir / "per_metric_bars.png")]

    caps = {
        document["turns"][0]["budget"]["stage_turn_cap"]
        for document in documents.values()
        if document.get("turns")
    }
    turn_cap = caps.pop() if len(caps) == 1 else None
    written.append(
        turns_survived(trials, out_dir / "turns_survived.png", turn_cap=turn_cap)
    )

    if args.trajectories != "none":
        if args.trajectories == "auto":
            chosen = pick_trajectory_trials(trials)
        else:
            chosen = [t.strip() for t in args.trajectories.split(",") if t.strip()]
        for trial_id in chosen:
            if trial_id not in metrics:
                print(f"unknown trial id {trial_id!r}; have {sorted(metrics)}",
                      file=sys.stderr)
                return 2
            written.append(
                trajectory_vs_belief(
                    metrics[trial_id],
                    documents[trial_id],
                    out_dir / f"trajectory_vs_belief_{trial_id}.png",
                )
            )

    for path in written:
        print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())
