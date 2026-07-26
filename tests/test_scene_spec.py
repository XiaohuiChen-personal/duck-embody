"""Scene-builder arithmetic, tested without launching Isaac Sim.

Everything asserted here fails **silently** in-sim if it is wrong: a wall with
no gap looks like a wall, an asset at 100x scale is off-camera, and a
"native" collider that does not exist renders perfectly while the robot walks
through it. Catching these on the pure layer costs milliseconds instead of a
kit launch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from duck_embody.env.apartment_layout import LAYOUT, room_at
from duck_embody.env.scene_builder import (
    CONTACT_OFFSET_M,
    floor_specs,
    furniture_specs,
    layout_to_spec,
    load_manifest,
    spec_summary,
    wall_specs,
    yaw_to_quat,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "assets" / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="assets/manifest.json absent — run assets/fetch_assets.sh"
)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


@pytest.fixture(scope="module")
def specs(manifest):
    return layout_to_spec(LAYOUT, manifest)


class TestWallSplitting:
    def test_a_wall_carrying_D_doorways_yields_D_plus_1_segments(self):
        """doc 03 §4 / PLAN. Wall A carries THREE doorways, so FOUR segments —
        the plan's earlier 'two segments each' was arithmetically wrong."""
        segments_a = [w for w in LAYOUT["walls"] if w["name"].startswith("A")]
        doors_a = [d for d in LAYOUT["doorways"] if d["center"][1] == 2.7]
        assert len(doors_a) == 3
        assert len(segments_a) == len(doors_a) + 1 == 4

    def test_total_segment_count_matches_the_doorway_count(self):
        """Across the three interior walls: segments == walls + doorways."""
        interior = [w for w in LAYOUT["walls"] if not w["name"].startswith("outer")]
        distinct_walls = {w["name"][0] for w in interior}
        assert len(interior) == len(distinct_walls) + len(LAYOUT["doorways"])

    def test_no_wall_segment_spans_a_doorway_interval(self):
        """The gap must be real geometry, not a differently-coloured wall."""
        half = LAYOUT["wall_thickness"] / 2.0
        for door in LAYOUT["doorways"]:
            cx, cy = door["center"]
            for seg in LAYOUT["walls"]:
                (x0, y0), (x1, y1) = seg["start"], seg["end"]
                spans = (
                    min(x0, x1) - half < cx < max(x0, x1) + half
                    and min(y0, y1) - half < cy < max(y0, y1) + half
                )
                assert not spans, f"{seg['name']} spans doorway {door['between']}"

    def test_each_segment_becomes_two_slabs(self):
        """One per face — a cuboid has no two-sided material, so per-room wall
        colour is only possible by splitting."""
        walls = wall_specs()
        assert len(walls) == 2 * len(LAYOUT["walls"])

    def test_slabs_sum_to_the_specified_wall_thickness(self):
        walls = wall_specs()
        by_segment: dict[str, float] = {}
        for w in walls:
            base = w["name"].rsplit("_", 1)[0]
            thin = min(w["size"][0], w["size"][1])
            by_segment[base] = by_segment.get(base, 0.0) + thin
        for base, total in by_segment.items():
            assert total == pytest.approx(LAYOUT["wall_thickness"]), base

    def test_slabs_do_not_overlap_each_other(self):
        """Two slabs of a segment sit either side of the centreline."""
        walls = {w["name"]: w for w in wall_specs()}
        for seg in LAYOUT["walls"]:
            a, b = walls[f"{seg['name']}_a"], walls[f"{seg['name']}_b"]
            horizontal = abs(seg["end"][1] - seg["start"][1]) < 1e-9
            axis = 1 if horizontal else 0
            assert a["pos"][axis] < b["pos"][axis]

    def test_walls_are_full_height_and_collide(self):
        for w in wall_specs():
            assert w["size"][2] == pytest.approx(LAYOUT["wall_height"])
            assert w["collision"] is True
            assert w["pos"][2] == pytest.approx(LAYOUT["wall_height"] / 2)


class TestWallColours:
    def test_each_slab_takes_the_colour_of_the_room_it_faces(self):
        walls = wall_specs()
        interior = [w for w in walls if w["faces_room"] is not None]
        assert interior, "no wall face resolved to a room"
        for w in interior:
            expected = tuple(LAYOUT["rooms"][w["faces_room"]]["wall_color"])
            assert w["color"] == expected, w["name"]

    def test_the_two_faces_of_an_interior_wall_differ(self):
        """Wall A separates a room from the hallway; if both faces were the same
        colour the split would be pointless."""
        walls = {w["name"]: w for w in wall_specs()}
        a, b = walls["A2_a"], walls["A2_b"]
        assert a["faces_room"] != b["faces_room"]
        assert a["color"] != b["color"]

    def test_outward_faces_of_outer_walls_are_neutral(self):
        from duck_embody.env.scene_builder import WALL_NEUTRAL_COLOR

        walls = {w["name"]: w for w in wall_specs()}
        # The south outer wall's -y face looks out of the apartment.
        assert walls["outer_S_a"]["faces_room"] is None
        assert walls["outer_S_a"]["color"] == WALL_NEUTRAL_COLOR

    def test_every_room_has_a_distinct_wall_colour(self):
        """Two rooms sharing a palette would defeat the room-recognition gate."""
        colours = [tuple(r["wall_color"]) for r in LAYOUT["rooms"].values()]
        assert len(set(colours)) == len(colours)

    def test_every_room_has_a_distinct_floor_colour(self):
        colours = [tuple(r["floor_color"]) for r in LAYOUT["rooms"].values()]
        assert len(set(colours)) == len(colours)


class TestFloors:
    def test_one_tile_per_room_covering_it_exactly(self):
        floors = floor_specs()
        assert len(floors) == len(LAYOUT["rooms"])
        for f in floors:
            x0, y0, x1, y1 = (
                min(p[0] for p in LAYOUT["rooms"][f["room"]]["poly"]),
                min(p[1] for p in LAYOUT["rooms"][f["room"]]["poly"]),
                max(p[0] for p in LAYOUT["rooms"][f["room"]]["poly"]),
                max(p[1] for p in LAYOUT["rooms"][f["room"]]["poly"]),
            )
            assert f["size"][0] == pytest.approx(x1 - x0)
            assert f["size"][1] == pytest.approx(y1 - y0)

    def test_floor_tiles_do_not_collide(self):
        """The terrain plane is the physics. A collider here would put a step at
        every room boundary."""
        assert all(f["collision"] is False for f in floor_specs())

    def test_floor_tiles_are_thin_and_sit_on_the_ground(self):
        for f in floor_specs():
            assert f["size"][2] < 0.01
            assert f["pos"][2] == pytest.approx(f["size"][2] / 2)

    def test_tiles_are_centred_in_their_room(self):
        for f in floor_specs():
            assert room_at(f["pos"][0], f["pos"][1]) == f["room"]


class TestFurnitureScale:
    def test_scale_is_per_asset_not_a_blanket_0_4(self, specs):
        """doc 03 §6.4 hardcoded 0.4. ArchVis is authored in CENTIMETRES, so
        that would spawn the 1.87 m fridge at 187 m."""
        by_name = {s["name"]: s for s in specs if s["kind"] == "furniture"}
        assert by_name["sofa"]["scale"] == pytest.approx(0.4)
        assert by_name["fridge"]["scale"] == pytest.approx(0.004)

    def test_every_scale_matches_the_measured_meters_per_unit(self, specs, manifest):
        for s in specs:
            if s["kind"] != "furniture":
                continue
            item = next(f for f in LAYOUT["furniture"] if f["name"] == s["name"])
            expected = manifest[item["asset"]]["metersPerUnit"] * LAYOUT["world_scale"]
            assert s["scale"] == pytest.approx(expected), s["name"]

    def test_scale_is_uniform(self, specs):
        """Non-uniform scale on mesh/convex colliders is a known PhysX trouble
        spot and buys nothing here."""
        for s in specs:
            if s["kind"] == "furniture":
                assert isinstance(s["scale"], float)


class TestCollisionClassification:
    def test_simready_items_carry_their_physics_variant(self, specs):
        """Their PhysicsVariant defaults to None — without this the robot walks
        through the sofa with no error anywhere."""
        by_name = {s["name"]: s for s in specs if s["kind"] == "furniture"}
        assert by_name["sofa"]["variants"] == {"PhysicsVariant": "RigidBody"}

    def test_native_furniture_is_pinned_kinematic(self, specs):
        """RigidBody is the only collider-bearing variant and it makes the asset
        dynamic; without pinning, a leaning duck shoves the sofa away."""
        for s in specs:
            if s["kind"] == "furniture" and s["collision"]:
                assert s["kinematic"] is True, s["name"]

    def test_every_bbox_proxy_item_gets_a_proxy_spec(self, specs):
        """doc 03 §7: collision_props can only MODIFY an existing collider."""
        expected = {f["name"] for f in LAYOUT["furniture"] if f["collision"] == "bbox_proxy"}
        produced = {s["for_item"] for s in specs if s["kind"] == "proxy"}
        assert produced == expected

    def test_proxies_are_invisible_but_solid(self, specs):
        for s in specs:
            if s["kind"] == "proxy":
                assert s["visible"] is False
                assert s["collision"] is True

    def test_proxy_footprint_matches_the_layout(self, specs):
        for s in specs:
            if s["kind"] != "proxy":
                continue
            item = next(f for f in LAYOUT["furniture"] if f["name"] == s["for_item"])
            assert s["size"][0] == pytest.approx(item["footprint"][0])
            assert s["size"][1] == pytest.approx(item["footprint"][1])

    def test_proxy_is_as_tall_as_the_asset(self, specs, manifest):
        """A short proxy under a tall asset lets the robot's head pass through
        the fridge."""
        for s in specs:
            if s["kind"] != "proxy":
                continue
            item = next(f for f in LAYOUT["furniture"] if f["name"] == s["for_item"])
            measured_h = manifest[item["asset"]]["size_m_at_duck_scale"][2]
            assert s["size"][2] >= min(measured_h, 0.05) - 1e-6

    def test_the_rug_gets_neither_native_collision_nor_a_proxy(self, specs):
        rug = next(s for s in specs if s.get("name") == "blue_rug")
        assert rug["collision"] is False
        assert not [s for s in specs if s.get("for_item") == "blue_rug"]

    def test_declaring_a_visual_only_asset_native_is_rejected(self, manifest):
        """The silent trap: it would render perfectly and stop nothing."""
        bad = dict(LAYOUT)
        bad["furniture"] = [
            {
                "name": "fridge", "room": "kitchen", "asset": "fridge",
                "pos": (3.10, 2.30), "yaw_deg": 90, "footprint": (0.280, 0.746),
                "collision": "native",  # the manifest measured bbox_proxy
            }
        ]
        with pytest.raises(ValueError, match="would render and not collide"):
            furniture_specs(bad, manifest)

    def test_unknown_asset_is_rejected(self, manifest):
        bad = dict(LAYOUT)
        bad["furniture"] = [
            {"name": "x", "room": "kitchen", "asset": "not_a_real_asset",
             "pos": (2.0, 1.0), "yaw_deg": 0, "footprint": (0.1, 0.1),
             "collision": "bbox_proxy"}
        ]
        with pytest.raises(KeyError, match="not in the manifest"):
            furniture_specs(bad, manifest)


class TestSemanticsAndPaths:
    def test_every_furniture_spec_has_a_semantic_tag(self, specs):
        for s in specs:
            if s["kind"] == "furniture":
                assert s["semantic"] == s["name"]

    def test_usd_paths_point_into_the_local_mirror(self, specs):
        """The batch must never depend on a live S3 bucket."""
        for s in specs:
            if s["kind"] == "furniture":
                assert "/assets/Assets/" in s["usd_path"], s["name"]

    def test_every_referenced_usd_exists_on_disk(self, specs):
        for s in specs:
            if s["kind"] == "furniture":
                assert Path(s["usd_path"]).exists(), s["usd_path"]


class TestWholeScene:
    def test_spec_counts_are_what_the_layout_implies(self, specs):
        summary = spec_summary(specs)
        assert summary["floor"] == len(LAYOUT["rooms"])
        assert summary["wall"] == 2 * len(LAYOUT["walls"])
        assert summary["furniture"] == len(LAYOUT["furniture"])
        assert summary["proxy"] == len(
            [f for f in LAYOUT["furniture"] if f["collision"] == "bbox_proxy"]
        )

    def test_spec_names_are_unique(self, specs):
        """They become prim paths; a duplicate silently overwrites a prim."""
        names = [f"{s['kind']}/{s['name']}" for s in specs]
        assert len(names) == len(set(names))

    def test_contact_offset_is_small_enough_for_duck_scale(self):
        """Authored offsets are absolute metres and do not shrink with the 0.4x
        scale — a human-scale 2 cm offset is a force field around a 20 cm table."""
        assert CONTACT_OFFSET_M <= 0.005

    def test_no_furniture_is_placed_outside_its_room(self, specs):
        for s in specs:
            if s["kind"] == "furniture":
                assert room_at(s["pos"][0], s["pos"][1]) == s["room"], s["name"]


class TestQuaternion:
    @pytest.mark.parametrize(
        "yaw,expected",
        [(0, (1, 0, 0, 0)), (180, (0, 0, 0, 1)), (90, (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))],
    )
    def test_yaw_to_quat(self, yaw, expected):
        got = yaw_to_quat(yaw)
        for g, e in zip(got, expected):
            assert g == pytest.approx(e, abs=1e-9)

    def test_quaternion_is_unit_length(self):
        for yaw in range(0, 360, 17):
            q = yaw_to_quat(yaw)
            assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0)
