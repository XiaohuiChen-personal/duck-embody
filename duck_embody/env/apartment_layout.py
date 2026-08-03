"""THE apartment layout — scene specification AND scoring ground truth.

One dict, two jobs. ``scene_builder.py`` reads it to place walls and furniture;
``scoring.py`` reads it for room polygons, adjacency, the target region, spawn
points, and oracle shortest paths — and reads **nothing else**. If the world and
the answer key were two files they would drift, and the map-accuracy and SPL
numbers would quietly stop describing the world the robot walked through
(AGENTS.md §2, doc 01 §5).

Conventions, fixed by doc 03 §3 and shared with the compass the model sees:
**x** runs east (0 at the west wall), **y** runs north (0 at the south wall),
and headings are **degrees counter-clockwise from +x**, so 0° faces east and
90° faces north.

All lengths are metres at **duck scale** (0.4x human) — the decision that makes
rooms legible from a 0.36 m camera instead of a toddler's-eye view of table
undersides. Divide by 0.4 for human-equivalent numbers.

Furniture footprints here are **measured, not estimated**: they come from
``assets/manifest.json``, which ``scripts/inspect_assets.py`` produced by
reading the actual USD bounding boxes. ``tests/test_layout.py`` asserts they
still match, so a re-fetched or swapped asset cannot silently invalidate the
clearances this file guarantees.
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Half the duck's ~0.16 m body width (AGENTS.md §5, doc 03 §3.1).
#:
#: PLAN CORRECTION (T2.1): the plan said to inflate the free-space grid "by the
#: 0.16 m duck body radius". 0.16 m is the body *width*; used as a radius it
#: leaves a 0.35 m doorway with 0.35 - 2(0.16) = 0.03 m of free width — less
#: than one grid cell — so every doorway would be impassable and every
#: oracle-path invariant would fail. The radius is 0.08 m, leaving 0.19 m.
BODY_RADIUS_M = 0.08

#: Free-space grid resolution for the oracle path (doc 06 §5.3).
GRID_CELL_M = 0.05

LAYOUT = {
    "units": "m",
    "world_scale": 0.4,
    "extents": (4.8, 3.6),
    # RAISED 0.5 -> 0.7 by T2.3 iteration 1. doc 03 §7 named this exact
    # contingency ("if the VLM mislabels rooms due to over-wall leakage, raise
    # walls to 0.7 m"), and the gate produced exactly that evidence: at 0.5 m
    # the judge read the hallway as an "outdoor courtyard" and named it from
    # the living-room sofa and the bed visible OVER the walls.
    "wall_height": 0.7,
    "wall_thickness": 0.03,
    "body_radius": BODY_RADIUS_M,
    "grid_cell": GRID_CELL_M,
    # -----------------------------------------------------------------------
    # Walls. Door gaps are encoded by SPLITTING a wall into segments, so the
    # builder only ever adds boxes and never subtracts geometry. A wall
    # carrying D doorways yields D+1 segments.
    # -----------------------------------------------------------------------
    "walls": [
        # Outer shell.
        {"name": "outer_S", "start": (0.0, 0.0), "end": (4.8, 0.0)},
        {"name": "outer_N", "start": (0.0, 3.6), "end": (4.8, 3.6)},
        {"name": "outer_W", "start": (0.0, 0.0), "end": (0.0, 3.6)},
        {"name": "outer_E", "start": (4.8, 0.0), "end": (4.8, 3.6)},
        # Wall A: y = 2.7, separates the three rooms from the hallway.
        # Carries THREE doorways -> four segments.
        {"name": "A1", "start": (0.0, 2.7), "end": (0.725, 2.7)},
        {"name": "A2", "start": (1.075, 2.7), "end": (2.375, 2.7)},
        {"name": "A3", "start": (2.725, 2.7), "end": (3.875, 2.7)},
        {"name": "A4", "start": (4.225, 2.7), "end": (4.8, 2.7)},
        # Wall B: x = 1.8, living_room | kitchen. ONE doorway -> two segments.
        {"name": "B1", "start": (1.8, 0.0), "end": (1.8, 1.025)},
        {"name": "B2", "start": (1.8, 1.375), "end": (1.8, 2.7)},
        # Wall C: x = 3.3, kitchen | bedroom. No doorway -> one segment, which
        # is what forces bedroom access through the hallway and gives QA
        # question 1 ("which room connects bedroom to kitchen?") a unique answer.
        {"name": "C", "start": (3.3, 0.0), "end": (3.3, 2.7)},
    ],
    # -----------------------------------------------------------------------
    # Rooms: axis-aligned polygons. SCORING GROUND TRUTH for room
    # precision/recall, "which room is the robot in", and the layout QA.
    # Per-room wall and floor colours exist for VLM legibility (doc 03 §6):
    # at duck height the palette is often the clearest room cue in frame.
    # -----------------------------------------------------------------------
    "rooms": {
        "living_room": {
            "poly": [(0.0, 0.0), (1.8, 0.0), (1.8, 2.7), (0.0, 2.7)],
            "wall_color": (0.72, 0.76, 0.82),
            "floor_color": (0.45, 0.33, 0.22),
            "landmarks": ["sofa", "armchair", "coffee table", "blue rug"],
        },
        "kitchen": {
            "poly": [(1.8, 0.0), (3.3, 0.0), (3.3, 2.7), (1.8, 2.7)],
            "wall_color": (0.90, 0.88, 0.78),
            "floor_color": (0.82, 0.82, 0.80),
            "landmarks": ["counter", "fridge", "stove", "bar stool"],
        },
        "bedroom": {
            "poly": [(3.3, 0.0), (4.8, 0.0), (4.8, 2.7), (3.3, 2.7)],
            "wall_color": (0.78, 0.72, 0.82),
            "floor_color": (0.50, 0.36, 0.28),
            "landmarks": ["bed", "desk"],
        },
        "hallway": {
            "poly": [(0.0, 2.7), (4.8, 2.7), (4.8, 3.6), (0.0, 3.6)],
            "wall_color": (0.86, 0.86, 0.86),
            "floor_color": (0.38, 0.30, 0.24),
            "landmarks": ["potted plant"],
        },
    },
    # -----------------------------------------------------------------------
    # Doorways: ground-truth adjacency edges + gap geometry.
    # -----------------------------------------------------------------------
    "doorways": [
        {"between": ("hallway", "living_room"), "center": (0.9, 2.7), "width": 0.35},
        {"between": ("hallway", "kitchen"), "center": (2.55, 2.7), "width": 0.35},
        {"between": ("hallway", "bedroom"), "center": (4.05, 2.7), "width": 0.35},
        {"between": ("living_room", "kitchen"), "center": (1.8, 1.2), "width": 0.35},
    ],
    # -----------------------------------------------------------------------
    # Furniture: SCENE SPEC ONLY — scoring never reads this.
    #
    # `footprint` is the placed (post-yaw) extent in world x, y, taken from the
    # MEASURED AABBs in assets/manifest.json. `collision`: "native" = the USD
    # ships colliders (SimReady assets need their PhysicsVariant selected, doc
    # 03 §5); "bbox_proxy" = an invisible cuboid provides the physics;
    # "none" = no collider at all.
    # -----------------------------------------------------------------------
    "furniture": [
        # --- living_room --------------------------------------------------
        {"name": "sofa", "room": "living_room", "asset": "crestwood_sofa",
         "pos": (0.30, 1.60), "yaw_deg": 0, "footprint": (0.391, 0.975),
         "collision": "native"},
        {"name": "coffee_table", "room": "living_room", "asset": "appleseed_coffeetable",
         "pos": (0.88, 1.60), "yaw_deg": 0, "footprint": (0.300, 0.527),
         "collision": "native"},
        # MOVED 0.95 -> 0.72 by the T2.4 physics pass. At y=0.95 the armchair's
        # body-inflated footprint reached y=1.2225, leaving only ~7 cm of free
        # centre-line inside the living-room/kitchen doorway (corridor
        # y 1.105-1.295). A* still found a route, so `test_every_room_reachable`
        # passed and the defect was invisible — but the ORACLE path then ran
        # through a 7 cm slot, and every model that sensibly detoured via the
        # hallway would have been scored against a threading the robot cannot
        # reliably achieve. That is a scoring-fairness bug, not a driving one.
        # At y=0.72 the corridor clears by 0.113 m and the chair is off the rug.
        {"name": "armchair", "room": "living_room", "asset": "armchair",
         "pos": (1.40, 0.72), "yaw_deg": 0, "footprint": (0.447, 0.385),
         "collision": "native"},
        # A rug is not an obstacle: 0.002 m tall, so NO collider. A bbox proxy
        # would put an invisible 2 mm lip across the living-room floor.
        {"name": "blue_rug", "room": "living_room", "asset": "blue_rug",
         "pos": (0.95, 1.60), "yaw_deg": 0, "footprint": (0.975, 1.219),
         "collision": "none"},
        # --- kitchen -------------------------------------------------------
        # The counter run: three cabinets along the south wall. The scored
        # target sits in front of it (see "target" below).
        {"name": "counter_1", "room": "kitchen", "asset": "sektion_cabinet",
         "pos": (2.45, 0.20), "yaw_deg": 0, "footprint": (0.267, 0.306),
         "collision": "native"},
        {"name": "counter_2", "room": "kitchen", "asset": "sektion_cabinet",
         "pos": (2.72, 0.20), "yaw_deg": 0, "footprint": (0.267, 0.306),
         "collision": "native"},
        {"name": "counter_3", "room": "kitchen", "asset": "sektion_cabinet",
         "pos": (2.99, 0.20), "yaw_deg": 0, "footprint": (0.267, 0.306),
         "collision": "native"},
        {"name": "stove", "room": "kitchen", "asset": "stove",
         "pos": (2.06, 0.20), "yaw_deg": 0, "footprint": (0.487, 0.274),
         "collision": "bbox_proxy"},
        # The fridge stands 0.734 m tall — ABOVE the 0.5 m walls, so its top is
        # visible from the hallway. Deliberate: a strong kitchen landmark.
        # Flagged for T2.3, which must confirm it does not let the judge name
        # the kitchen from outside the kitchen.
        {"name": "fridge", "room": "kitchen", "asset": "fridge",
         "pos": (3.10, 2.30), "yaw_deg": 90, "footprint": (0.280, 0.746),
         "collision": "bbox_proxy"},
        {"name": "bar_stool", "room": "kitchen", "asset": "bar_stool",
         "pos": (2.15, 1.65), "yaw_deg": 0, "footprint": (0.167, 0.195),
         "collision": "native"},
        # --- kitchen, T2.3 iteration 3 ------------------------------------
        # The gate failed the kitchen twice: the judge saw "a mostly empty white
        # room with a chair and a cabinet" and called it a living room. All the
        # kitchen-ness sat in one low run along the south wall, so from the
        # middle or north of the room there was nothing to see. These add
        # kitchen signal where the earlier poses were looking.
        #
        # A second cabinet run along the east wall.
        {"name": "counter_4", "room": "kitchen", "asset": "sektion_cabinet",
         "pos": (3.13, 1.15), "yaw_deg": 90, "footprint": (0.306, 0.267),
         "collision": "native"},
        {"name": "counter_5", "room": "kitchen", "asset": "sektion_cabinet",
         "pos": (3.13, 1.45), "yaw_deg": 90, "footprint": (0.306, 0.267),
         "collision": "native"},
        # A microwave ON the counter (z = counter height). Visual only: it sits
        # on a solid cabinet, so it needs no collider of its own and must not
        # get one — a proxy would float an invisible box above the counter.
        {"name": "microwave", "room": "kitchen", "asset": "microwave",
         "pos": (2.72, 0.20), "z": 0.314, "yaw_deg": 0, "footprint": (0.298, 0.218),
         "collision": "none"},
        # --- bedroom -------------------------------------------------------
        {"name": "bed", "room": "bedroom", "asset": "daybed",
         "pos": (4.20, 0.95), "yaw_deg": 90, "footprint": (0.605, 0.988),
         "collision": "bbox_proxy"},
        # desk_01 is the SimReady catalog defect (zero colliders under EITHER
        # PhysicsVariant option), so it needs a proxy — doc 03 §5.
        {"name": "desk", "room": "bedroom", "asset": "desk_01",
         "pos": (3.55, 1.90), "yaw_deg": 0, "footprint": (0.256, 0.765),
         "collision": "bbox_proxy"},
        # --- hallway -------------------------------------------------------
        # Plant at the EAST end. The west end was rejected in design review for
        # sitting ~0.17 m from spawn 103 (doc 03 §4).
        {"name": "plant", "room": "hallway", "asset": "plant_01",
         "pos": (4.62, 3.40), "yaw_deg": 0, "footprint": (0.275, 0.273),
         "collision": "bbox_proxy"},
        # T2.3 iteration 4. The hallway was the least stable room in the gate —
        # across three judge runs on identical frames it scored 3/3, 2/3 and a
        # 1/1/1 tie, because a corridor with one plant at the far end has little
        # of its own to show and the judge kept naming whichever room it could
        # see through a doorway. Two planters give it furniture of its own along
        # its length. They sit against the north side, leaving ~0.66 m of free
        # corridor — well clear for a 0.16 m body.
        #
        # NOTE they are NOT at the west end: doc 03's design review already moved
        # a plant away from there for sitting ~0.17 m from spawn 103, and that
        # spawn is still there.
        {"name": "planter_w", "room": "hallway", "asset": "gardenplanter_medium",
         "pos": (1.20, 3.45), "yaw_deg": 0, "footprint": (0.142, 0.142),
         "collision": "native"},
        {"name": "planter_e", "room": "hallway", "asset": "gardenplanter_medium",
         "pos": (3.30, 3.45), "yaw_deg": 0, "footprint": (0.142, 0.142),
         "collision": "native"},
    ],
    # -----------------------------------------------------------------------
    # Target. SEMANTICS PINNED: this disc IS "the kitchen-counter target
    # region". The scorer measures Euclidean distance to `point`, NEVER to the
    # cabinet footprint — a robot pressed against the counter face but off to
    # one side is a FAILURE, and that exact case is a required fixture in
    # tests/test_scoring.py (doc 03 §4).
    # -----------------------------------------------------------------------
    "target": {
        "name": "kitchen_counter",
        "room": "kitchen",
        "point": (2.55, 0.75),
        "radius": 0.35,
    },
    # -----------------------------------------------------------------------
    # Spawns: seed -> pose. Identical across models; the seed IS the trial id.
    # -----------------------------------------------------------------------
    "spawn_points": {
        101: {"pos": (0.5, 0.5), "heading_deg": 90, "room": "living_room"},
        102: {"pos": (4.3, 2.2), "heading_deg": 270, "room": "bedroom"},
        # 103 and 104 were nudged 2-3 cm off doc 03 §3.1's coordinates. That
        # doc said they "sit exactly at the 0.4 m wall boundary", measuring to
        # the wall CENTRELINE; walls are 0.03 m thick, so the real clearance to
        # the surface the robot actually collides with was 0.385 m. Measuring
        # to the face is the honest test, so the spawns moved to satisfy it.
        103: {"pos": (0.43, 3.15), "heading_deg": 0, "room": "hallway"},
        104: {"pos": (1.37, 2.27), "heading_deg": 180, "room": "living_room"},
    },
    #: return_home succeeds within this radius of the spawn (doc 06 §3.1).
    "return_home_radius": 0.5,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def room_bounds(room: str) -> tuple[float, float, float, float]:
    """(x_min, y_min, x_max, y_max) of a room's axis-aligned polygon."""
    poly = LAYOUT["rooms"][room]["poly"]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def room_at(x: float, y: float) -> str | None:
    """Which room contains (x, y)? ``None`` if outside every room.

    Boundaries are half-open on the max side so a point on a shared edge
    belongs to exactly one room. Otherwise "which room is the robot in" would
    be ambiguous precisely at the doorways, where it matters most.
    """
    for name in LAYOUT["rooms"]:
        x0, y0, x1, y1 = room_bounds(name)
        if x0 <= x < x1 and y0 <= y < y1:
            return name
    # Points exactly on the outer north/east edge still belong to a room.
    for name in LAYOUT["rooms"]:
        x0, y0, x1, y1 = room_bounds(name)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


def room_centroid(room: str) -> tuple[float, float]:
    x0, y0, x1, y1 = room_bounds(room)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def adjacency() -> dict[str, set[str]]:
    """Room graph implied by the doorways. The map-accuracy answer key."""
    graph: dict[str, set[str]] = {name: set() for name in LAYOUT["rooms"]}
    for door in LAYOUT["doorways"]:
        a, b = door["between"]
        graph[a].add(b)
        graph[b].add(a)
    return graph


def connecting_rooms(a: str, b: str) -> list[str]:
    """Rooms adjacent to BOTH ``a`` and ``b`` — QA question 1's answer."""
    graph = adjacency()
    return sorted(graph[a] & graph[b])


def room_path(start: str, goal: str) -> list[str] | None:
    """Shortest room sequence through the adjacency graph (BFS). QA question 2."""
    if start == goal:
        return [start]
    graph = adjacency()
    seen = {start}
    queue: list[list[str]] = [[start]]
    while queue:
        path = queue.pop(0)
        for nxt in sorted(graph[path[-1]]):
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            queue.append(path + [nxt])
    return None


def bearing_deg(origin: tuple[float, float], target: tuple[float, float]) -> float:
    """Heading from origin to target, degrees CCW from +x, in [0, 360)."""
    return math.degrees(math.atan2(target[1] - origin[1], target[0] - origin[0])) % 360.0


COMPASS_8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def compass_8(bearing: float) -> str:
    """Bucket a bearing to 8-way compass. QA question 4's answer key."""
    return COMPASS_8[int(((bearing % 360.0) + 22.5) // 45.0) % 8]


# ---------------------------------------------------------------------------
# Obstacle model and the free-space grid
# ---------------------------------------------------------------------------


def wall_rects() -> list[tuple[float, float, float, float]]:
    """Wall segments as (x_min, y_min, x_max, y_max), including thickness."""
    half = LAYOUT["wall_thickness"] / 2.0
    rects = []
    for seg in LAYOUT["walls"]:
        (x0, y0), (x1, y1) = seg["start"], seg["end"]
        rects.append(
            (
                min(x0, x1) - half,
                min(y0, y1) - half,
                max(x0, x1) + half,
                max(y0, y1) + half,
            )
        )
    return rects


def furniture_rects(
    include_non_colliding: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Furniture footprints as rectangles.

    Items with ``collision: "none"`` (the rug) are excluded by default: they are
    visible but not solid, so treating them as obstacles would make the oracle
    path detour around something the robot simply walks over — inflating `l`
    and flattering every model's SPL.
    """
    rects = []
    for item in LAYOUT["furniture"]:
        if item["collision"] == "none" and not include_non_colliding:
            continue
        cx, cy = item["pos"]
        w, d = item["footprint"]
        rects.append((cx - w / 2.0, cy - d / 2.0, cx + w / 2.0, cy + d / 2.0))
    return rects


def _dist_point_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    """Euclidean distance from a point to an axis-aligned rectangle (0 inside)."""
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# The kitchen counter run — success-criterion ground truth (criterion v2)
# ---------------------------------------------------------------------------
#
# These live HERE, not in `scoring.py`, because criterion v2 has two consumers:
# the live stage gate in `tasks/find_kitchen.py` and the post-hoc scorer. That
# is exactly the split F-02 recorded — the published criterion widened while the
# live gate stayed on the point disc, so `opus5_seed101` was a published success
# that was never offered its return leg. One geometry source, imported by both,
# is the structural fix; two copies of "which rectangles are the counters" is
# the same defect waiting to recur.
#
# NOTE this reads `LAYOUT["furniture"]`, whose header says "SCENE SPEC ONLY —
# scoring never reads this". That header is AMENDED by criterion v2 (adopted
# 2026-07-27, results/rerun_log.md; unified live-and-published by TR.2): the
# counter footprints ARE scoring ground truth now. Amended in the open rather
# than quietly ignored.

#: The asset every kitchen counter is an instance of. Counters are selected
#: structurally (kitchen + this asset), never by name, so a renamed counter
#: cannot silently fall out of the success region.
KITCHEN_COUNTER_ASSET = "sektion_cabinet"

#: How many counters the frozen layout is known to contain. A different count
#: means the criterion no longer matches the scene — raise, never guess.
KITCHEN_COUNTER_COUNT = 5


class LayoutError(ValueError):
    """The layout no longer supports something that reads it. Never swallowed."""


@lru_cache(maxsize=1)
def kitchen_counter_rects() -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    """The five kitchen counters' footprint AABBs, in layout order.

    Footprints are world-axis full extents (the same reading
    :func:`furniture_rects` uses), so the rectangles are axis-aligned by
    construction.
    """
    rects: list[tuple[str, tuple[float, float, float, float]]] = []
    for item in LAYOUT["furniture"]:
        if item["room"] == "kitchen" and item["asset"] == KITCHEN_COUNTER_ASSET:
            cx, cy = item["pos"]
            w, d = item["footprint"]
            rects.append(
                (item["name"], (cx - w / 2.0, cy - d / 2.0, cx + w / 2.0, cy + d / 2.0))
            )
    if len(rects) != KITCHEN_COUNTER_COUNT:
        raise LayoutError(
            f"expected {KITCHEN_COUNTER_COUNT} kitchen {KITCHEN_COUNTER_ASSET} "
            f"counters in the frozen layout, found {len(rects)} — the success "
            "criterion no longer matches the scene"
        )
    return tuple(rects)


def nearest_counter_face(xy: tuple[float, float]) -> tuple[str, float]:
    """``(counter name, Euclidean distance to its footprint rectangle)``.

    Distance is to the rectangle (0 inside), through :func:`_dist_point_rect`
    so the criterion and the free-space grid share one geometry. A corner
    approach is credited up to the radius off a footprint corner — the natural
    rectangle generalisation of the primary disc's semantics, same inclusive
    boundary, same radius.
    """
    name, dist = min(
        (
            (name, _dist_point_rect(xy[0], xy[1], rect))
            for name, rect in kitchen_counter_rects()
        ),
        key=lambda pair: pair[1],
    )
    return name, dist


def clearance(px: float, py: float, include_furniture: bool = True) -> float:
    """Distance from a point to the nearest wall or furniture footprint."""
    rects = wall_rects()
    if include_furniture:
        rects = rects + furniture_rects()
    return min(_dist_point_rect(px, py, r) for r in rects)


class FreeSpaceGrid:
    """Occupancy grid of poses the duck's *body* can actually occupy.

    Obstacles are inflated by the body radius so a path through this grid is
    achievable rather than merely collision-free for a point robot. Without the
    inflation the oracle would hug walls and thread doorway edges no 0.16 m-wide
    robot could take, and SPL would be measured against a route nothing can walk
    — flattering every model equally, but meaninglessly.
    """

    def __init__(self, cell: float = GRID_CELL_M, inflate: float = BODY_RADIUS_M):
        self.cell = cell
        self.inflate = inflate
        self.w, self.h = LAYOUT["extents"]
        self.nx = int(round(self.w / cell))
        self.ny = int(round(self.h / cell))
        rects = wall_rects() + furniture_rects()
        self.free = [
            [
                min(_dist_point_rect(*self.center(i, j), r) for r in rects) > inflate
                for i in range(self.nx)
            ]
            for j in range(self.ny)
        ]

    def center(self, i: int, j: int) -> tuple[float, float]:
        return ((i + 0.5) * self.cell, (j + 0.5) * self.cell)

    def index(self, x: float, y: float) -> tuple[int, int]:
        return (
            min(max(int(x / self.cell), 0), self.nx - 1),
            min(max(int(y / self.cell), 0), self.ny - 1),
        )

    def is_free(self, x: float, y: float) -> bool:
        i, j = self.index(x, y)
        return self.free[j][i]

    def nearest_free(self, x: float, y: float, max_radius_cells: int = 8):
        """Closest free cell to (x, y), for endpoints inside the inflation margin.

        A spawn or target may legitimately sit within a body-radius of a wall
        (spawn 103 is 0.4 m from the west wall by design). Snapping is honest:
        it moves the endpoint by at most a few centimetres and never through a
        wall, because only free cells are candidates.
        """
        i0, j0 = self.index(x, y)
        if self.free[j0][i0]:
            return (i0, j0)
        for r in range(1, max_radius_cells + 1):
            best = None
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny and self.free[j][i]:
                        d = math.dist((x, y), self.center(i, j))
                        if best is None or d < best[0]:
                            best = (d, (i, j))
            if best is not None:
                return best[1]
        return None

    def path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]] | None:
        """8-connected A* between two world points. ``None`` if unreachable."""
        s = self.nearest_free(*start)
        g = self.nearest_free(*goal)
        if s is None or g is None:
            return None

        def heuristic(a, b):
            return math.dist(self.center(*a), self.center(*b))

        open_heap = [(heuristic(s, g), 0.0, s)]
        came: dict = {}
        best_cost = {s: 0.0}
        diag = math.sqrt(2.0)

        while open_heap:
            _, cost, node = heapq.heappop(open_heap)
            if node == g:
                out = [node]
                while out[-1] in came:
                    out.append(came[out[-1]])
                return [self.center(*n) for n in reversed(out)]
            if cost > best_cost.get(node, math.inf):
                continue
            i, j = node
            for di, dj in (
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1),
            ):
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny) or not self.free[nj][ni]:
                    continue
                # Never cut a corner diagonally: both orthogonal neighbours must
                # be free too, or the "path" clips a wall corner the body cannot
                # actually pass through.
                if di and dj and not (self.free[j][ni] and self.free[nj][i]):
                    continue
                step = diag if (di and dj) else 1.0
                new_cost = cost + step * self.cell
                if new_cost < best_cost.get((ni, nj), math.inf):
                    best_cost[(ni, nj)] = new_cost
                    came[(ni, nj)] = node
                    heapq.heappush(
                        open_heap, (new_cost + heuristic((ni, nj), g), new_cost, (ni, nj))
                    )
        return None


_GRID: FreeSpaceGrid | None = None


def grid() -> FreeSpaceGrid:
    """Shared grid — building it costs ~7k distance queries, so cache it."""
    global _GRID
    if _GRID is None:
        _GRID = FreeSpaceGrid()
    return _GRID


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def oracle_path(
    start: tuple[float, float], goal: tuple[float, float]
) -> list[tuple[float, float]] | None:
    """Shortest achievable collision-free path — SPL's `l` (doc 06 §5.3)."""
    return grid().path(start, goal)


def oracle_length(start: tuple[float, float], goal: tuple[float, float]) -> float | None:
    p = oracle_path(start, goal)
    return None if p is None else path_length(p)


def target_point() -> tuple[float, float]:
    return tuple(LAYOUT["target"]["point"])


def spawn_pose(seed: int) -> tuple[tuple[float, float], float]:
    entry = LAYOUT["spawn_points"][seed]
    return tuple(entry["pos"]), float(entry["heading_deg"])
