"""``LAYOUT`` -> Isaac Lab scene entries.

Split in two on purpose:

* :func:`layout_to_spec` is **pure** — dicts in, dicts out, no Isaac import. All
  the arithmetic that can be wrong (wall splitting, per-asset scale, proxy
  sizing, wall-face colours) lives here and is unit-tested without paying for a
  kit launch.
* :func:`add_apartment_to_scene` is a thin adapter that turns those dicts into
  ``CuboidCfg`` / ``AssetBaseCfg`` entries.

Four things this file must get right, each of which fails **silently** if it
does not (doc 03 §5–§7, all measured in T0.2):

1. **Per-asset scale, not a blanket 0.4.** SimReady and Isaac Props are authored
   in metres; **all ArchVis assets are centimetres** (`metersPerUnit = 0.01`).
   Isaac Lab does not convert layer units, so doc 03 §6.4's hardcoded
   ``scale=(0.4, 0.4, 0.4)`` would spawn the 1.87 m fridge at **187 m**. The
   manifest publishes ``scale_for_duck_scale`` per asset; we use it.
2. **SimReady colliders are behind a variant.** Their ``PhysicsVariant``
   defaults to ``None``, so a naive spawn has *zero* collision prims and the
   robot walks through the sofa. We pass ``variants`` at spawn.
3. **...and that variant makes them dynamic.** ``RigidBody`` is the only other
   option, so furniture would be pushable. ``kinematic_enabled=True`` pins them:
   solid, immovable obstacles, which is what a navigation task needs.
4. **`collision_props` cannot CREATE a collider**, only modify an existing one.
   Anything the manifest measured as visual-only gets an invisible cuboid proxy
   instead; declaring it "native" would render perfectly and stop nothing.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

from duck_embody.env.apartment_layout import LAYOUT, room_at

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

#: PhysX contact offset for every collider we control (doc 03 §7). Authored
#: offsets are absolute metres and do NOT shrink with the 0.4x scale, so a
#: human-scale 2 cm offset would be an invisible force field around a 20 cm
#: duck-scale table. T0.2 measured that NO mirrored asset authors one, so this
#: is a uniform floor rather than a per-asset rescue. Tuned in T2.4.
CONTACT_OFFSET_M = 0.002
REST_OFFSET_M = 0.0

#: Interior walls are built as two half-thickness slabs so each face can carry
#: the colour of the room it faces — a cuboid cannot have two-sided materials,
#: and at duck height wall colour is one of the strongest room cues in frame.
WALL_NEUTRAL_COLOR = (0.85, 0.85, 0.84)

#: Floor tiles are visual only; the terrain plane provides the physics. A
#: collider here would add a 3 mm step at every room boundary.
FLOOR_THICKNESS_M = 0.003

#: Invisible collider proxies are as tall as the asset they stand in for, so a
#: 0.73 m fridge blocks the robot over its whole height.
PROXY_MIN_HEIGHT_M = 0.05

#: Ceiling, added by T2.3 iteration 2. Without it the survey judge read every
#: room as "an outdoor terrace/rooftop courtyard" — open sky above a 0.7 m wall
#: does not look like a home, and the question the gate asks is literally "what
#: room OF A HOME is this?". Visual only: the duck cannot fly, and a collider
#: here would only add contact-resolution work.
CEILING_THICKNESS_M = 0.02
CEILING_COLOR = (0.93, 0.93, 0.91)
#: The ceiling prim path, so the survey script can hide it for the top-down
#: render and restore it afterwards — doc 03 §3.1 wants walls "low enough for
#: top-down debug renders", and a roof would otherwise end that.
CEILING_PRIM_PATH = "/World/Apartment/ceilings/ceiling"

#: One light per room, just under the ceiling. A sealed room lit only by the
#: scene's dome light is black, so the ceiling and the lights arrive together.
ROOM_LIGHT_INTENSITY = 12000.0
ROOM_LIGHT_RADIUS_M = 0.12


@contextmanager
def ceiling_hidden(stage, renders: int = 3, verbose: bool = True):
    """Hide the ceiling for the duration of the block, then restore it.

    Two different jobs need to see *into* sealed rooms — the T2.3 top-down survey
    render and the T2.4 physics-pass audit video — and both need the ceiling back
    afterwards, because the head-camera renders the models actually see are what
    the ceiling exists for. Toggling beats choosing between them.

    Restoration runs in a ``finally``: an exception mid-render must not leave the
    stage roofless, or every subsequent capture in the session silently reverts
    to the "outdoor terrace" look the T2.3 gate rejected.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(CEILING_PRIM_PATH)
    if not (prim and prim.IsValid()):
        if verbose:
            print(f"  WARNING: no ceiling prim at {CEILING_PRIM_PATH}; nothing hidden")
        yield False
        return

    imageable = UsdGeom.Imageable(prim)
    imageable.MakeInvisible()
    if verbose:
        print("  ceiling hidden")
    try:
        yield True
    finally:
        imageable.MakeVisible()
        if verbose:
            print("  ceiling restored")
        _ = renders  # callers re-render as needed; kept for call-site clarity


def load_manifest(path: Path | None = None) -> dict:
    p = path or MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run assets/fetch_assets.sh then scripts/inspect_assets.sh"
        )
    return json.loads(p.read_text())["assets"]


def yaw_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Yaw about +Z as (w, x, y, z)."""
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


# ---------------------------------------------------------------------------
# The pure layer
# ---------------------------------------------------------------------------


def _wall_face_rooms(seg: dict) -> tuple[str | None, str | None]:
    """Which room lies on each side of a wall segment.

    Returns ``(side_a, side_b)`` where side_a is the -y (or -x) side. ``None``
    means "outside the apartment". Used to colour each face for the room that
    sees it.
    """
    (x0, y0), (x1, y1) = seg["start"], seg["end"]
    eps = 0.05
    if abs(y1 - y0) < 1e-9:  # horizontal wall
        mid_x = (x0 + x1) / 2.0
        return room_at(mid_x, y0 - eps), room_at(mid_x, y0 + eps)
    mid_y = (y0 + y1) / 2.0
    return room_at(x0 - eps, mid_y), room_at(x0 + eps, mid_y)


def _room_color(room: str | None) -> tuple[float, float, float]:
    if room is None:
        return WALL_NEUTRAL_COLOR
    return tuple(LAYOUT["rooms"][room]["wall_color"])


def wall_specs(layout: dict = LAYOUT) -> list[dict]:
    """One spec per wall *face*: two slabs per segment, one per side.

    Splitting each wall into two half-thickness slabs is what makes per-room
    wall colour possible at all. Both slabs collide; at 200 Hz physics the
    robot advances ~1 mm per step against a 15 mm slab, so nothing tunnels.
    """
    thickness = layout["wall_thickness"]
    height = layout["wall_height"]
    half_t = thickness / 2.0
    specs: list[dict] = []

    for seg in layout["walls"]:
        (x0, y0), (x1, y1) = seg["start"], seg["end"]
        horizontal = abs(y1 - y0) < 1e-9
        length = abs(x1 - x0) if horizontal else abs(y1 - y0)
        if length <= 0:
            continue
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        side_a, side_b = _wall_face_rooms(seg)

        for side, room, sign in (("a", side_a, -1.0), ("b", side_b, +1.0)):
            offset = sign * half_t / 2.0
            if horizontal:
                pos = (cx, cy + offset, height / 2.0)
                size = (length, half_t, height)
            else:
                pos = (cx + offset, cy, height / 2.0)
                size = (half_t, length, height)
            specs.append(
                {
                    "kind": "wall",
                    "name": f"{seg['name']}_{side}",
                    "pos": pos,
                    "size": size,
                    "color": _room_color(room),
                    "collision": True,
                    "visible": True,
                    "faces_room": room,
                }
            )
    return specs


def floor_specs(layout: dict = LAYOUT) -> list[dict]:
    """One coloured, VISUAL-ONLY tile per room.

    At a 0.36 m camera height with a 90° field of view the floor occupies much
    of every frame, so per-room floor colour is the most reliable room cue the
    scene can offer a VLM (doc 03 §6).
    """
    specs = []
    for name, room in layout["rooms"].items():
        poly = room["poly"]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        specs.append(
            {
                "kind": "floor",
                "name": f"floor_{name}",
                "pos": (
                    (min(xs) + max(xs)) / 2.0,
                    (min(ys) + max(ys)) / 2.0,
                    FLOOR_THICKNESS_M / 2.0,
                ),
                "size": (max(xs) - min(xs), max(ys) - min(ys), FLOOR_THICKNESS_M),
                "color": tuple(room["floor_color"]),
                # Visual only: the terrain plane is the physics. A collider here
                # would create a step at every room boundary.
                "collision": False,
                "visible": True,
                "room": name,
            }
        )
    return specs


def ceiling_specs(layout: dict = LAYOUT) -> list[dict]:
    """One slab over the whole apartment, plus a light per room.

    They are inseparable: sealing the rooms is what makes them read as interior
    space, and it is also what makes them dark.
    """
    w, h = layout["extents"]
    wall_h = layout["wall_height"]
    specs: list[dict] = [
        {
            "kind": "ceiling",
            "name": "ceiling",
            "pos": (w / 2.0, h / 2.0, wall_h + CEILING_THICKNESS_M / 2.0),
            "size": (w, h, CEILING_THICKNESS_M),
            "color": CEILING_COLOR,
            "collision": False,
            "visible": True,
        }
    ]
    for name, room in layout["rooms"].items():
        poly = room["poly"]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        specs.append(
            {
                "kind": "light",
                "name": f"light_{name}",
                "pos": (
                    (min(xs) + max(xs)) / 2.0,
                    (min(ys) + max(ys)) / 2.0,
                    wall_h - 0.10,
                ),
                "intensity": ROOM_LIGHT_INTENSITY,
                "radius": ROOM_LIGHT_RADIUS_M,
                "room": name,
            }
        )
    return specs


def furniture_specs(layout: dict = LAYOUT, manifest: dict | None = None) -> list[dict]:
    """One USD spec per item, plus a proxy spec for anything not natively solid."""
    manifest = manifest if manifest is not None else load_manifest()
    specs: list[dict] = []

    for item in layout["furniture"]:
        asset = manifest.get(item["asset"])
        if asset is None:
            raise KeyError(
                f"{item['name']} references asset '{item['asset']}' which is not in the "
                "manifest — re-run scripts/inspect_assets.sh"
            )

        native = item["collision"] == "native"
        measured_native = asset["measured_collision"] == "native"
        if native and not measured_native:
            # The trap doc 03 §7 warns about: collision_props can only MODIFY an
            # existing collider. A visual-only asset declared "native" renders
            # perfectly and stops nothing, with no error anywhere.
            raise ValueError(
                f"{item['name']} declares collision='native' but the manifest measured "
                f"'{asset['measured_collision']}'. It would render and not collide."
            )

        specs.append(
            {
                "kind": "furniture",
                "name": item["name"],
                "usd_path": str(ASSETS_DIR / asset["local_path"]),
                # Optional z lets an item sit ON another (the microwave on the
                # counter) instead of on the floor.
                "pos": (item["pos"][0], item["pos"][1], float(item.get("z", 0.0))),
                "yaw_deg": item["yaw_deg"],
                # Per-asset: metersPerUnit x duck scale. 0.4 for metre-authored
                # assets, 0.004 for the centimetre-authored ArchVis ones.
                "scale": asset["scale_for_duck_scale"],
                "variants": asset.get("variants") or {},
                # RigidBody is the only variant that yields colliders, and it
                # also makes the asset dynamic — pin it so a leaning duck cannot
                # shove the sofa across the room.
                "collision": native,
                "kinematic": native,
                "semantic": item["name"],
                "visible": True,
                "room": item["room"],
            }
        )

        if item["collision"] == "bbox_proxy":
            w, d = item["footprint"]
            height = max(asset["size_m_at_duck_scale"][2], PROXY_MIN_HEIGHT_M)
            base_z = float(item.get("z", 0.0))
            specs.append(
                {
                    "kind": "proxy",
                    "name": f"{item['name']}_proxy",
                    "pos": (item["pos"][0], item["pos"][1], base_z + height / 2.0),
                    "size": (w, d, height),
                    "collision": True,
                    # Invisible: the USD above supplies the looks, this supplies
                    # the physics.
                    "visible": False,
                    "for_item": item["name"],
                    "room": item["room"],
                }
            )
        # collision == "none" (the rug) gets neither native collision nor a
        # proxy: it is 0.002 m tall and must not be an obstacle.

    return specs


def layout_to_spec(layout: dict = LAYOUT, manifest: dict | None = None) -> list[dict]:
    """The whole apartment as plain dicts. Kit-free and fully testable."""
    return (
        floor_specs(layout)
        + wall_specs(layout)
        + ceiling_specs(layout)
        + furniture_specs(layout, manifest)
    )


def spec_summary(specs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in specs:
        out[s["kind"]] = out.get(s["kind"], 0) + 1
    return out


# ---------------------------------------------------------------------------
# The Isaac Lab layer
# ---------------------------------------------------------------------------


def add_apartment_to_scene(scene_cfg, layout: dict = LAYOUT, manifest: dict | None = None) -> None:
    """Attach every spec to an ``InteractiveSceneCfg`` as a scene entry.

    Called from ``DuckEmbodyApartmentEnvCfg.__post_init__``. Imports Isaac Lab
    lazily so the pure layer above stays importable without a kit app.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    specs = layout_to_spec(layout, manifest)

    for spec in specs:
        prim_path = f"/World/Apartment/{spec['kind']}s/{spec['name']}"

        if spec["kind"] == "light":
            cfg = AssetBaseCfg(
                prim_path=prim_path,
                init_state=AssetBaseCfg.InitialStateCfg(pos=spec["pos"]),
                spawn=sim_utils.SphereLightCfg(
                    intensity=spec["intensity"],
                    radius=spec["radius"],
                    color=(1.0, 0.97, 0.92),
                ),
            )
        elif spec["kind"] in ("wall", "floor", "proxy", "ceiling"):
            # CuboidCfg DEFINES a collision schema on the prim (box colliders
            # need no mesh cooking and are the most reliable geometry PhysX has).
            collision_props = (
                sim_utils.CollisionPropertiesCfg(
                    contact_offset=CONTACT_OFFSET_M, rest_offset=REST_OFFSET_M
                )
                if spec["collision"]
                else None
            )
            visual_material = (
                sim_utils.PreviewSurfaceCfg(diffuse_color=spec["color"])
                if spec.get("color")
                else None
            )
            cfg = AssetBaseCfg(
                prim_path=prim_path,
                init_state=AssetBaseCfg.InitialStateCfg(pos=spec["pos"]),
                spawn=sim_utils.CuboidCfg(
                    size=spec["size"],
                    collision_props=collision_props,
                    visual_material=visual_material,
                    visible=spec["visible"],
                ),
                # Global collision group: this is what makes scene prims collide
                # with the robot rather than only within their own env copy.
                collision_group=-1,
            )
        else:  # furniture
            scale = spec["scale"]
            rigid_props = (
                sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
                if spec["kinematic"]
                else None
            )
            collision_props = (
                sim_utils.CollisionPropertiesCfg(
                    contact_offset=CONTACT_OFFSET_M, rest_offset=REST_OFFSET_M
                )
                if spec["collision"]
                else None
            )
            cfg = AssetBaseCfg(
                prim_path=prim_path,
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=spec["pos"], rot=yaw_to_quat(spec["yaw_deg"])
                ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=spec["usd_path"],
                    scale=(scale, scale, scale),
                    variants=spec["variants"] or None,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    semantic_tags=[("class", spec["semantic"])],
                ),
                collision_group=-1,
            )

        setattr(scene_cfg, f"apt_{spec['kind']}_{spec['name']}", cfg)
