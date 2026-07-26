"""Layout invariants — the pre-batch gate on the world AND the answer key.

``apartment_layout.LAYOUT`` is both the scene specification and the scoring
ground truth, so a geometry mistake here does not produce a crash: it produces
a benchmark that measures the wrong thing. Every invariant below exists because
some metric silently breaks without it.

Covers all six invariants doc 06 §9.2 makes mandatory, plus doc 03 §4's
clearance rules, plus a guard that the footprints still match the measured
asset manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from duck_embody.env.apartment_layout import (
    BODY_RADIUS_M,
    LAYOUT,
    adjacency,
    bearing_deg,
    clearance,
    compass_8,
    connecting_rooms,
    grid,
    oracle_length,
    oracle_path,
    path_length,
    room_at,
    room_bounds,
    room_centroid,
    room_path,
    spawn_pose,
    target_point,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = sorted(LAYOUT["spawn_points"])
#: doc 03 §4: every spawn keeps at least this much clearance from walls and
#: furniture. Spawns 103 and 104 sit exactly on the boundary by design, so the
#: comparison needs a floating-point tolerance rather than a strict >.
MIN_SPAWN_CLEARANCE_M = 0.4
EPS = 1e-9


class TestRoomPolygons:
    def test_rooms_are_within_the_apartment_bounds(self):
        w, h = LAYOUT["extents"]
        for name in LAYOUT["rooms"]:
            x0, y0, x1, y1 = room_bounds(name)
            assert 0.0 <= x0 < x1 <= w, name
            assert 0.0 <= y0 < y1 <= h, name

    def test_rooms_do_not_overlap(self):
        names = list(LAYOUT["rooms"])
        for i, a in enumerate(names):
            ax0, ay0, ax1, ay1 = room_bounds(a)
            for b in names[i + 1 :]:
                bx0, by0, bx1, by1 = room_bounds(b)
                overlap_x = min(ax1, bx1) - max(ax0, bx0)
                overlap_y = min(ay1, by1) - max(ay0, by0)
                assert overlap_x <= 0 or overlap_y <= 0, f"{a} overlaps {b}"

    def test_rooms_tile_the_apartment(self):
        """No unreachable dead space: room areas must sum to the footprint."""
        w, h = LAYOUT["extents"]
        total = sum(
            (x1 - x0) * (y1 - y0)
            for x0, y0, x1, y1 in (room_bounds(n) for n in LAYOUT["rooms"])
        )
        assert total == pytest.approx(w * h)

    def test_room_at_is_unambiguous_on_shared_edges(self):
        """A pose on a boundary must resolve to exactly one room, or 'which
        room is the robot in' breaks precisely at the doorways."""
        for x, y in [(1.8, 1.0), (3.3, 1.0), (2.0, 2.7), (0.0, 0.0)]:
            assert room_at(x, y) is not None

    def test_every_room_is_reachable_from_every_other(self):
        graph = adjacency()
        seen, stack = {"hallway"}, ["hallway"]
        while stack:
            for nxt in graph[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        assert seen == set(LAYOUT["rooms"])


class TestDoorways:
    def test_each_doorway_lies_on_the_shared_boundary_of_the_two_rooms_it_claims(self):
        """doc 06 §9.2. A doorway whose adjacency claim does not match its
        geometry would corrupt the map-accuracy answer key."""
        for door in LAYOUT["doorways"]:
            a, b = door["between"]
            cx, cy = door["center"]
            ax0, ay0, ax1, ay1 = room_bounds(a)
            bx0, by0, bx1, by1 = room_bounds(b)

            # The centre must sit on the shared edge: touching both closed
            # rectangles at once.
            on_a = ax0 - EPS <= cx <= ax1 + EPS and ay0 - EPS <= cy <= ay1 + EPS
            on_b = bx0 - EPS <= cx <= bx1 + EPS and by0 - EPS <= cy <= by1 + EPS
            assert on_a and on_b, f"doorway {a}<->{b} at {(cx, cy)} is not on their shared edge"

    def test_doorway_connects_exactly_those_two_rooms(self):
        """The gap must not also open onto a third room."""
        for door in LAYOUT["doorways"]:
            a, b = door["between"]
            cx, cy = door["center"]
            touching = {
                name
                for name in LAYOUT["rooms"]
                if (
                    room_bounds(name)[0] - EPS <= cx <= room_bounds(name)[2] + EPS
                    and room_bounds(name)[1] - EPS <= cy <= room_bounds(name)[3] + EPS
                )
            }
            assert touching == {a, b}, f"doorway {a}<->{b} also touches {touching - {a, b}}"

    def test_each_doorway_is_a_real_gap_in_a_wall(self):
        """The builder splits walls to make gaps; a doorway with no gap is a
        painted-on door the robot cannot walk through."""
        half = LAYOUT["wall_thickness"] / 2.0
        for door in LAYOUT["doorways"]:
            cx, cy = door["center"]
            for seg in LAYOUT["walls"]:
                (x0, y0), (x1, y1) = seg["start"], seg["end"]
                inside = (
                    min(x0, x1) - half < cx < max(x0, x1) + half
                    and min(y0, y1) - half < cy < max(y0, y1) + half
                )
                assert not inside, f"doorway {door['between']} is blocked by wall {seg['name']}"

    def test_a_wall_carrying_D_doorways_yields_D_plus_1_segments(self):
        """doc 03 §4. Wall A carries THREE doorways, so it must be FOUR
        segments — 'two segments each' is arithmetically wrong."""
        wall_a = [s for s in LAYOUT["walls"] if s["name"].startswith("A")]
        doors_on_a = [d for d in LAYOUT["doorways"] if d["center"][1] == 2.7]
        assert len(doors_on_a) == 3
        assert len(wall_a) == len(doors_on_a) + 1 == 4

        wall_b = [s for s in LAYOUT["walls"] if s["name"].startswith("B")]
        doors_on_b = [d for d in LAYOUT["doorways"] if d["center"][0] == 1.8]
        assert len(doors_on_b) == 1
        assert len(wall_b) == len(doors_on_b) + 1 == 2

    def test_every_doorway_is_grid_reachable_with_the_inflated_body(self):
        """doc 06 §9.2. A gap the inflated body cannot fit through is a wall."""
        g = grid()
        for door in LAYOUT["doorways"]:
            cx, cy = door["center"]
            assert g.nearest_free(cx, cy) is not None, f"doorway {door['between']} unreachable"

    def test_doorways_are_wide_enough_for_the_body(self):
        for door in LAYOUT["doorways"]:
            free_width = door["width"] - 2 * BODY_RADIUS_M
            assert free_width > LAYOUT["grid_cell"], (
                f"doorway {door['between']} leaves only {free_width:.3f} m for the body"
            )


class TestTarget:
    def test_target_is_inside_the_kitchen(self):
        assert room_at(*target_point()) == "kitchen"
        assert LAYOUT["target"]["room"] == "kitchen"

    def test_target_disc_stays_inside_the_kitchen(self):
        """A disc spilling into another room would let a robot 'succeed' from
        the living room."""
        x, y = target_point()
        r = LAYOUT["target"]["radius"]
        x0, y0, x1, y1 = room_bounds("kitchen")
        assert x0 <= x - r and x + r <= x1
        assert y0 <= y - r and y + r <= y1

    def test_target_is_standable(self):
        """The scored point must be somewhere the body can actually be — a
        target inside the counter would make success impossible."""
        assert grid().is_free(*target_point())

    def test_target_is_in_front_of_the_counter_not_inside_it(self):
        counters = [f for f in LAYOUT["furniture"] if f["name"].startswith("counter_")]
        assert counters
        tx, ty = target_point()
        for c in counters:
            cx, cy = c["pos"]
            w, d = c["footprint"]
            inside = abs(tx - cx) <= w / 2 and abs(ty - cy) <= d / 2
            assert not inside, f"target sits inside {c['name']}"
        # ...and close enough that "walk to the counter" is a fair description.
        assert min(math.dist((tx, ty), c["pos"]) for c in counters) < 0.8


class TestSpawns:
    def test_every_spawn_is_inside_the_room_it_claims(self):
        for seed, entry in LAYOUT["spawn_points"].items():
            assert room_at(*entry["pos"]) == entry["room"], seed

    def test_every_spawn_is_far_from_the_target(self):
        """doc 06 §9.2: > 3 x goal radius, so d_initial > 0 (progress and SPL
        denominators are well-defined) and stage 1 is never trivially won."""
        limit = 3 * LAYOUT["target"]["radius"]
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert math.dist(pos, target_point()) > limit, seed

    def test_every_spawn_has_clearance_from_walls_and_furniture(self):
        """doc 03 §4: >= 0.4 m. Design review moved the plant and the armchair
        for violating this; the test is what keeps them moved."""
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert clearance(*pos) >= MIN_SPAWN_CLEARANCE_M - EPS, (
                f"seed {seed} has only {clearance(*pos):.3f} m clearance"
            )

    def test_every_spawn_is_standable(self):
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert grid().is_free(*pos), seed

    def test_spawn_headings_are_valid_bearings(self):
        for seed in SEEDS:
            _, heading = spawn_pose(seed)
            assert 0.0 <= heading < 360.0, seed

    def test_spawns_cover_more_than_one_room(self):
        """A seed set confined to one room would not test navigation."""
        assert len({LAYOUT["spawn_points"][s]["room"] for s in SEEDS}) >= 3


class TestOraclePaths:
    def test_a_path_exists_from_every_spawn_to_the_target(self):
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert oracle_path(pos, target_point()) is not None, seed

    def test_a_path_exists_back_from_the_target_to_every_spawn(self):
        """return_home is scored against this; doc 06 §9.2 requires both."""
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert oracle_path(target_point(), pos) is not None, seed

    def test_oracle_length_is_at_least_the_straight_line(self):
        """A shortest path shorter than the straight line means the path is
        cutting through a wall — SPL would then exceed 1."""
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            length = oracle_length(pos, target_point())
            assert length >= math.dist(pos, target_point()) - LAYOUT["grid_cell"] * 2, seed

    def test_oracle_length_is_within_the_policy_seconds_budget(self):
        """doc 03 §2: the longest route must leave budget for exploration, not
        just transit. At 0.2 m/s a 240 policy-second stage covers 48 m."""
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            length = oracle_length(pos, target_point())
            transit_s = length / 0.2
            assert transit_s < 0.25 * 240, f"seed {seed} transit {transit_s:.0f}s is too much"

    def test_oracle_path_never_crosses_a_wall(self):
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            for x, y in oracle_path(pos, target_point()):
                assert grid().is_free(x, y), f"seed {seed} path enters an obstacle"

    def test_bedroom_route_goes_through_the_hallway(self):
        """Wall C has no doorway, so the only bedroom<->kitchen route is via
        the hallway. This is what makes QA question 1 answerable."""
        assert room_path("bedroom", "kitchen") == ["bedroom", "hallway", "kitchen"]


class TestQAGroundTruthIsComputable:
    """doc 06 §9.2 + §5.9 — all five answers must exist and Q1 must be unique."""

    def test_q1_exactly_one_room_connects_bedroom_and_kitchen(self):
        """Q1's uniqueness precondition. Without it the question is unscoreable
        and a headline metric silently loses a fifth of its value."""
        connectors = connecting_rooms("bedroom", "kitchen")
        assert connectors == ["hallway"], connectors

    def test_q2_a_room_route_exists_from_the_sofa_to_the_fridge(self):
        sofa = next(f for f in LAYOUT["furniture"] if f["name"] == "sofa")
        fridge = next(f for f in LAYOUT["furniture"] if f["name"] == "fridge")
        route = room_path(sofa["room"], fridge["room"])
        assert route == ["living_room", "kitchen"]

    def test_q3_visited_rooms_are_derivable_from_a_pose_trace(self):
        trace = [(0.5, 0.5), (0.9, 2.75), (2.55, 2.75), (2.55, 1.0)]
        visited = {room_at(x, y) for x, y in trace}
        assert visited == {"living_room", "hallway", "kitchen"}

    def test_q4_kitchen_bearing_is_defined_for_every_seed(self):
        centroid = room_centroid("kitchen")
        for seed in SEEDS:
            pos, _ = spawn_pose(seed)
            assert compass_8(bearing_deg(pos, centroid)) in {
                "N", "NE", "E", "SE", "S", "SW", "W", "NW"
            }

    def test_q5_every_room_has_at_least_one_landmark(self):
        for name, room in LAYOUT["rooms"].items():
            assert room["landmarks"], f"{name} has no landmark for QA question 5"

    def test_compass_buckets_are_correct_at_the_cardinals(self):
        assert compass_8(0) == "E"
        assert compass_8(90) == "N"
        assert compass_8(180) == "W"
        assert compass_8(270) == "S"
        assert compass_8(45) == "NE"
        assert compass_8(359) == "E"


class TestFurniture:
    def test_every_furniture_footprint_is_inside_its_room(self):
        for item in LAYOUT["furniture"]:
            cx, cy = item["pos"]
            w, d = item["footprint"]
            x0, y0, x1, y1 = room_bounds(item["room"])
            assert x0 - EPS <= cx - w / 2 and cx + w / 2 <= x1 + EPS, item["name"]
            assert y0 - EPS <= cy - d / 2 and cy + d / 2 <= y1 + EPS, item["name"]

    def test_furniture_does_not_overlap(self):
        items = LAYOUT["furniture"]
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                ox = (a["footprint"][0] + b["footprint"][0]) / 2 - abs(a["pos"][0] - b["pos"][0])
                oy = (a["footprint"][1] + b["footprint"][1]) / 2 - abs(a["pos"][1] - b["pos"][1])
                # Decor that shares a footprint with something else is fine when
                # it is on a different level: the rug lies UNDER the coffee
                # table, the microwave sits ON the counter.
                if "blue_rug" in (a["name"], b["name"]):
                    continue
                if a.get("z", 0.0) != b.get("z", 0.0):
                    continue
                assert ox <= EPS or oy <= EPS, f"{a['name']} overlaps {b['name']}"

    def test_no_furniture_blocks_a_doorway(self):
        for door in LAYOUT["doorways"]:
            cx, cy = door["center"]
            for item in LAYOUT["furniture"]:
                if item["collision"] == "none":
                    continue
                ix, iy = item["pos"]
                w, d = item["footprint"]
                blocks = abs(cx - ix) < w / 2 + BODY_RADIUS_M and abs(cy - iy) < d / 2 + BODY_RADIUS_M
                assert not blocks, f"{item['name']} blocks doorway {door['between']}"

    def test_the_rug_is_not_an_obstacle(self):
        """It is 0.002 m tall. A collider there would be an invisible lip that
        deflects the robot and inflates the oracle path.

        Checked by its absence from the obstacle set, NOT by probing its centre:
        the rug deliberately sits under the coffee table, so that cell is
        blocked by the table regardless.
        """
        from duck_embody.env.apartment_layout import furniture_rects

        rug = next(f for f in LAYOUT["furniture"] if f["name"] == "blue_rug")
        assert rug["collision"] == "none"

        cx, cy = rug["pos"]
        w, d = rug["footprint"]
        rug_rect = (cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2)
        assert rug_rect not in furniture_rects()
        assert rug_rect in furniture_rects(include_non_colliding=True)

    def test_footprints_match_the_measured_asset_manifest(self):
        """Guards against a re-fetched or swapped asset silently invalidating
        every clearance in this file."""
        manifest_path = REPO_ROOT / "assets" / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("assets/manifest.json absent — run assets/fetch_assets.sh")
        assets = json.loads(manifest_path.read_text())["assets"]
        for item in LAYOUT["furniture"]:
            measured = assets[item["asset"]]["size_m_at_duck_scale"]
            w, d = item["footprint"]
            # yaw 90/270 swaps the x and y extents.
            expect = (measured[1], measured[0]) if item["yaw_deg"] % 180 == 90 else (
                measured[0], measured[1]
            )
            assert w == pytest.approx(expect[0], abs=1e-3), item["name"]
            assert d == pytest.approx(expect[1], abs=1e-3), item["name"]

    def test_desk_and_visual_only_assets_get_a_proxy(self):
        """T0.2 measured these as having no colliders; declaring them native
        would let the robot walk straight through (doc 03 §7)."""
        by_name = {f["name"]: f for f in LAYOUT["furniture"]}
        for name in ("desk", "bed", "fridge", "stove", "plant"):
            assert by_name[name]["collision"] == "bbox_proxy", name


class TestFreeSpaceGrid:
    def test_body_radius_is_half_the_body_width_not_the_width(self):
        """PLAN CORRECTION: inflating by 0.16 m would leave 0.03 m of free
        doorway — narrower than a grid cell — and every doorway would close."""
        assert BODY_RADIUS_M == pytest.approx(0.08)
        assert 0.35 - 2 * BODY_RADIUS_M > LAYOUT["grid_cell"]

    def test_grid_covers_the_apartment(self):
        g = grid()
        w, h = LAYOUT["extents"]
        assert g.nx == int(round(w / LAYOUT["grid_cell"]))
        assert g.ny == int(round(h / LAYOUT["grid_cell"]))

    def test_cells_inside_walls_are_blocked(self):
        for x, y in [(1.8, 0.5), (3.3, 1.0), (0.5, 2.7)]:
            assert not grid().is_free(x, y), f"({x}, {y}) is inside a wall"

    def test_cells_inside_furniture_are_blocked(self):
        for name in ("sofa", "bed", "fridge"):
            item = next(f for f in LAYOUT["furniture"] if f["name"] == name)
            assert not grid().is_free(*item["pos"]), name

    def test_path_length_of_a_single_point_is_zero(self):
        assert path_length([(1.0, 1.0)]) == 0.0
