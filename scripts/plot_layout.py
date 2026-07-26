"""T2.1 smoke: render the layout so a human can compare it to doc 03's plan.

Two panels, because the layout has two jobs and they fail differently:

* **Left — the world.** Rooms, walls with real thickness, doorway gaps,
  furniture at measured footprints, the scored target disc, and the four spawn
  poses. This is what the scene builder will construct.
* **Right — what the robot can actually reach.** The free-space grid after
  inflating every obstacle by the body radius, with the oracle paths A* finds
  from each spawn to the target drawn on top. A layout can look perfect on the
  left and still have an unreachable doorway or a target walled in behind a
  counter; only this panel shows it.

Colour follows the dataviz method: room fills are low-chroma tints carrying no
information (every room is directly labelled), and the three saturated slots are
spent on the marks that mean something — doorways, target, spawns. Those three
were validated all-pairs for colour-vision deficiency before use; the aqua slot
sits under 3:1 contrast, which the direct seed labels relieve.

Run:  ~/IsaacLab/isaaclab.sh -p scripts/plot_layout.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from duck_embody.env.apartment_layout import (  # noqa: E402
    BODY_RADIUS_M,
    LAYOUT,
    grid,
    oracle_length,
    oracle_path,
    room_bounds,
    spawn_pose,
    target_point,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "figures" / "layout_plan.png"

# --- palette (validated: see module docstring) ------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
DOORWAY = "#2a78d6"  # slot 1
TARGET = "#eb6834"  # slot 2
SPAWN = "#1baf7a"  # slot 3
# Room tints are decorative only — identity comes from the printed room name.
ROOM_TINT = {
    "living_room": "#eef2f7",
    "kitchen": "#f7f5ec",
    "bedroom": "#f3eff6",
    "hallway": "#f2f2f1",
}
#: Label anchors chosen to clear the furniture in each room. Presentation only —
#: deliberately NOT in LAYOUT, which is the scene spec and the answer key.
ROOM_LABEL_AT = {
    "living_room": (0.90, 2.45),
    "kitchen": (2.45, 2.45),
    "bedroom": (4.05, 2.45),
    "hallway": (2.20, 3.15),
}


def draw_world(ax) -> None:
    w, h = LAYOUT["extents"]

    for name in LAYOUT["rooms"]:
        x0, y0, x1, y1 = room_bounds(name)
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, facecolor=ROOM_TINT[name], edgecolor="none", zorder=0
            )
        )
        lx, ly = ROOM_LABEL_AT[name]
        ax.text(lx, ly, name, ha="center", va="center", fontsize=9, color=INK_2, zorder=6)

    # Walls, drawn at true thickness so the doorway gaps are literal.
    half = LAYOUT["wall_thickness"] / 2.0
    for seg in LAYOUT["walls"]:
        (x0, y0), (x1, y1) = seg["start"], seg["end"]
        ax.add_patch(
            mpatches.Rectangle(
                (min(x0, x1) - half, min(y0, y1) - half),
                abs(x1 - x0) + 2 * half,
                abs(y1 - y0) + 2 * half,
                facecolor=INK,
                edgecolor="none",
                zorder=3,
            )
        )

    for item in LAYOUT["furniture"]:
        cx, cy = item["pos"]
        fw, fd = item["footprint"]
        solid = item["collision"] != "none"
        ax.add_patch(
            mpatches.Rectangle(
                (cx - fw / 2, cy - fd / 2),
                fw,
                fd,
                facecolor="#dcdbd6" if solid else "none",
                edgecolor=MUTED,
                linestyle="-" if solid else ":",
                linewidth=0.8,
                zorder=2,
            )
        )
        # The rug sits under the coffee table, so label it at its own top edge
        # rather than its centre — otherwise the two captions collide.
        label_y = cy + fd / 2 - 0.06 if item["collision"] == "none" else cy
        ax.text(cx, label_y, item["name"].replace("_", " "), ha="center", va="center",
                fontsize=5.2, color=INK_2, zorder=5)

    for door in LAYOUT["doorways"]:
        cx, cy = door["center"]
        half_w = door["width"] / 2
        horizontal = abs(cy - round(cy, 1)) < 1e-9 and cy in (2.7,)
        if horizontal:
            ax.plot([cx - half_w, cx + half_w], [cy, cy], color=DOORWAY, lw=2.5, zorder=4)
        else:
            ax.plot([cx, cx], [cy - half_w, cy + half_w], color=DOORWAY, lw=2.5, zorder=4)

    tx, ty = target_point()
    r = LAYOUT["target"]["radius"]
    ax.add_patch(
        mpatches.Circle((tx, ty), r, facecolor=TARGET, alpha=0.16, edgecolor=TARGET,
                        linestyle="--", linewidth=1.4, zorder=4)
    )
    ax.plot([tx], [ty], marker="o", ms=5, color=TARGET, zorder=6)
    ax.text(tx, ty - r - 0.12, f"target r={r}", ha="center", fontsize=6.5, color=TARGET, zorder=6)

    for seed in sorted(LAYOUT["spawn_points"]):
        (sx, sy), heading = spawn_pose(seed)
        ax.plot([sx], [sy], marker="o", ms=7, color=SPAWN, zorder=6)
        ax.arrow(
            sx, sy,
            0.22 * math.cos(math.radians(heading)), 0.22 * math.sin(math.radians(heading)),
            head_width=0.07, head_length=0.06, fc=SPAWN, ec=SPAWN, zorder=6, length_includes_head=True,
        )
        # Direct label: relieves the aqua slot's sub-3:1 contrast.
        ax.text(sx, sy - 0.20, str(seed), ha="center", fontsize=7, color=INK, zorder=7)

    ax.set_title("Apartment (duck scale, 0.4x)", fontsize=10, color=INK, pad=8)
    ax.set_xlim(-0.15, w + 0.15)
    ax.set_ylim(-0.15, h + 0.15)


def draw_reachability(ax) -> None:
    g = grid()
    w, h = LAYOUT["extents"]

    for j in range(g.ny):
        for i in range(g.nx):
            if g.free[j][i]:
                cx, cy = g.center(i, j)
                ax.add_patch(
                    mpatches.Rectangle(
                        (cx - g.cell / 2, cy - g.cell / 2), g.cell, g.cell,
                        facecolor="#e7ebf0", edgecolor="none", zorder=0,
                    )
                )

    tx, ty = target_point()
    for seed in sorted(LAYOUT["spawn_points"]):
        (sx, sy), _ = spawn_pose(seed)
        path = oracle_path((sx, sy), (tx, ty))
        if path:
            ax.plot([p[0] for p in path], [p[1] for p in path],
                    color=DOORWAY, lw=1.4, alpha=0.85, zorder=3)
            length = oracle_length((sx, sy), (tx, ty))
            mid = path[len(path) // 2]
            ax.text(mid[0], mid[1] + 0.08, f"{seed}: {length:.2f} m",
                    fontsize=6, color=DOORWAY, ha="center", zorder=5)
        ax.plot([sx], [sy], marker="o", ms=6, color=SPAWN, zorder=4)

    ax.plot([tx], [ty], marker="o", ms=6, color=TARGET, zorder=4)
    ax.set_title(
        f"Reachable free space (body radius {BODY_RADIUS_M} m) + oracle paths",
        fontsize=10, color=INK, pad=8,
    )
    ax.set_xlim(-0.15, w + 0.15)
    ax.set_ylim(-0.15, h + 0.15)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_aspect("equal")
        # Recessive axes: the geometry is the message.
        for spine in ax.spines.values():
            spine.set_color("#e2e1dc")
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("x (m, east) — heading 0 deg", fontsize=7.5, color=MUTED)
        ax.set_ylabel("y (m, north) — heading 90 deg", fontsize=7.5, color=MUTED)

    draw_world(axes[0])
    draw_reachability(axes[1])

    legend = [
        mpatches.Patch(facecolor="#dcdbd6", edgecolor=MUTED, label="furniture (solid)"),
        mpatches.Patch(facecolor="none", edgecolor=MUTED, linestyle=":", label="visual only (rug)"),
        mpatches.Patch(color=DOORWAY, label="doorway 0.35 m / oracle path"),
        mpatches.Patch(color=TARGET, label="kitchen-counter target"),
        mpatches.Patch(color=SPAWN, label="spawn + heading"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Duck Embody apartment — scene spec AND scoring ground truth "
        "(duck_embody/env/apartment_layout.py)",
        fontsize=11, color=INK, y=0.99,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")

    print("\noracle path lengths (spawn -> target -> spawn):")
    tx, ty = target_point()
    for seed in sorted(LAYOUT["spawn_points"]):
        (sx, sy), _ = spawn_pose(seed)
        out = oracle_length((sx, sy), (tx, ty))
        back = oracle_length((tx, ty), (sx, sy))
        print(f"  seed {seed}: out {out:.3f} m ({out / 0.2:5.1f} policy-s)  "
              f"back {back:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
