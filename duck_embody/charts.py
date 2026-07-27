"""Figures for README/results — SKELETON. T4.4 fills these in.

doc 06 §10 enumerates four reporting deliverables; two of them are figures and
land here:

* **§10.2 per-metric bar charts** with CI whiskers — one chart per headline
  metric, three bars (one per model). :func:`bar_with_ci`.
* **§10.4 trajectory-vs-belief map figures — "the money figure"** — one per
  trial: the floor plan with (a) the true path from ``true_pose``, (b) the
  dead-reckoned path from ``obs.position_estimate`` with ``correct_position``
  snap points marked, and (c) the model's claimed rooms overlaid at their matched
  positions. :func:`trajectory_vs_belief`.

Why the stubs exist now rather than in T4.4: PLAN T4.1's deliverable is a
"``charts.py`` skeleton", and the signatures are what pin the contract between
the scorer and the plotting task. Every input below is already produced by
:mod:`duck_embody.scoring` — :class:`~duck_embody.scoring.Estimate` carries the
mean, the interval and ``n_defined``; :func:`~duck_embody.scoring.score_trial`
carries the matched rooms. Nothing here re-derives a metric, so a figure can
never disagree with the table beside it.

**Where the estimates come from.** ``scoring.summarise`` renders every
``Estimate`` down to a plain dict for the results JSON, so a figure fed from it
would have to rebuild the columns itself — exactly the duplication this stub
exists to prevent. :func:`~duck_embody.scoring.metric_estimates` is the
accessor: it returns ``flat metric key -> Estimate`` for one model's trials, and
``summarise`` is built from the same call, so the figure and the table are fed by
one function. Keys are ``"<stage>.<metric>"`` for per-stage columns
(``"find_kitchen.spl"``, ``"return_home.drift_m"``, and ``"…success_rate"``,
which carries the bootstrap over the binary per-trial indicator that doc 06 §10's
"SR … as mean ± 95 % CI" asks for) and a bare name for trial-scoped ones
(``"qa"``, ``"bumps"``, ``"map_precision"``, ``"edge_accuracy"``).

Two conventions T4.4 must keep, both from doc 06 §6's honesty clause:

1. **A missing CI is drawn as a missing CI.** ``Estimate.ci is None`` means fewer
   than three trials had a defined value (§3.2's ``k < 3`` rule); the bar is
   drawn with no whisker and annotated with its ``n``, never with a zero-width
   whisker, which would read as a precise estimate.
2. **"—" is never plotted as 0.** An excluded cell is annotated, not drawn.

``matplotlib`` is imported lazily inside the functions: importing this module
must stay free for the test suite (AGENTS.md rule 2 — the gate runs in 0.5 s and
must not pull a plotting stack).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from duck_embody.scoring import Estimate, TrialMetrics

#: One colour per model, fixed so a reader can follow a model across every
#: figure in the write-up. Filled in by T4.4 alongside the palette decision.
MODEL_ORDER: tuple[str, ...] = ("fable5", "opus5", "gpt56sol")


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
    An estimate whose ``ci`` is ``None`` must be drawn WITHOUT a whisker and
    labelled with its ``n_defined`` (doc 06 §3.2/§6): a zero-width whisker would
    claim a precision the data does not have.
    """
    raise NotImplementedError(
        "T4.4 implements doc 06 §10.2's bar charts; T4.1 fixes the signature"
    )


def trajectory_vs_belief(
    trial: TrialMetrics,
    document: dict,
    out_path: Path | str,
) -> Path:
    """doc 06 §10.4's money figure: true path vs dead-reckoned belief.

    Three layers over the floor plan drawn from
    ``duck_embody.env.apartment_layout``:

    (a) the true path — ``turns[].execution.pose_trace``, the 5 Hz samples, so
        within-turn curvature is visible rather than chorded away;
    (b) the dead-reckoned path — the ``obs.position_estimate`` series, with each
        ``memory_snapshot.corrections`` entry marked as a snap from ``old_xy`` to
        ``new_xy``;
    (c) the model's claimed rooms, placed at the centroid of the true room each
        was matched to (``TrialMetrics.map_accuracy.matches``), with unmatched
        claims listed in the caption rather than drawn somewhere invented.
    """
    raise NotImplementedError(
        "T4.4 implements doc 06 §10.4's trajectory-vs-belief figure; "
        "T4.1 fixes the signature"
    )


def per_trial_table(trials: Sequence[TrialMetrics]) -> str:
    """doc 06 §6's honesty clause: the per-trial table published with EVERY
    aggregate. Markdown, so it drops straight into the README and the report."""
    raise NotImplementedError(
        "T4.4 implements doc 06 §10.1's per-trial table; T4.1 fixes the signature"
    )
