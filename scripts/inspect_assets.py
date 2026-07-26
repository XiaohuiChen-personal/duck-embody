"""Offline USD inspection of the mirrored assets -> ``assets/manifest.json``.

Answers, per asset, the four questions the scene builder cannot guess
(design doc 03 §6.3, §6.4, §7, §8.1):

1. **How big is it really?** Axis-aligned bounding box at the asset's own
   authored scale, plus the same box converted to metres. Sizes the invisible
   collider proxies for visual-only assets, and lets T2.1 place furniture from
   measured footprints instead of estimates.

2. **What units is it in?** ``metersPerUnit``. The ArchVis catalog is authored in
   **centimetres** (0.01) while Isaac Lab's stage is metres — spawning one of
   those with a naive ``scale=0.4`` makes it 100x too large. The builder needs
   ``effective_scale = duck_scale * metersPerUnit`` per asset, so the manifest
   reports ``scale_for_duck_scale`` ready to use.

3. **Is it actually solid?** Count of prims carrying ``UsdPhysics.CollisionAPI``.
   This is the gate behind doc 03 §7's nastiest trap: ``UsdFileCfg(
   collision_props=...)`` only *modifies* colliders that already exist — it can
   never add one. A visual-only asset declared "native" renders perfectly and
   the robot walks straight through it, with no error and no warning. Assets
   whose measured collision class contradicts ``asset_list.tsv`` are reported as
   a mismatch and make this script exit non-zero.

4. **Will it shove the robot?** Any authored ``contactOffset`` / ``restOffset``.
   Those are absolute metres and do NOT shrink with scale, so a 2 cm offset
   authored for a human-scale table becomes an invisible force field around a
   20 cm duck-scale one. Every authored value is listed so T2.2 can override it.

Requires ``pxr``, importable from NEITHER default interpreter. Run it through the
pinned packman USD runtime (verified 2026-07-26 -> USD (0, 24, 5)):

    USDROOT=$(ls -d ~/.cache/packman/chk/usd.py311.manylinux_2_35_aarch64.stock.release/* | head -1)
    PYTHONPATH=$USDROOT/lib/python LD_LIBRARY_PATH=$USDROOT/lib:$LD_LIBRARY_PATH \
      ~/IsaacLab/_isaac_sim/python.sh scripts/inspect_assets.py

or simply::

    bash scripts/inspect_assets.sh
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSET_LIST = ASSETS_DIR / "asset_list.tsv"
MANIFEST = ASSETS_DIR / "manifest.json"

# Duck scale from design doc 03 §2 — the apartment is built at 0.4x human scale.
DUCK_SCALE = 0.4

# PhysX offset attributes are authored under the physxCollision schema.
OFFSET_ATTRS = ("physxCollision:contactOffset", "physxCollision:restOffset")


def parse_variants(spec: str) -> dict[str, str]:
    """``"PhysicsVariant=RigidBody,Foo=Bar"`` -> ``{...}``; ``"-"`` -> ``{}``."""
    spec = spec.strip()
    if not spec or spec == "-":
        return {}
    out = {}
    for part in spec.split(","):
        vset, _, value = part.partition("=")
        out[vset.strip()] = value.strip()
    return out


def read_asset_list() -> list[tuple[str, str, str, dict[str, str]]]:
    rows = []
    for line in ASSET_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, key, collision, variants = line.split("\t")
        rows.append((name, key, collision, parse_variants(variants)))
    return rows


def apply_variants(stage: Usd.Stage, variants: dict[str, str]) -> list[str]:
    """Select `variants` on every prim that carries the named variant set.

    SimReady props expose ``PhysicsVariant`` on both the root and the inner
    ``*_inst`` prim; selecting only on the root leaves the instanced subtree on
    its default. Applying everywhere is what the scene builder's
    ``UsdFileCfg(variants=...)`` effectively does, so the manifest must measure
    the same state.
    """
    applied = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        vsets = prim.GetVariantSets()
        for vset_name, value in variants.items():
            if vset_name in vsets.GetNames():
                vsets.GetVariantSet(vset_name).SetVariantSelection(value)
                applied.append(f"{prim.GetPath()}:{vset_name}={value}")
    return applied


def inspect_one(name: str, key: str, expected_collision: str, variants: dict[str, str]) -> dict:
    path = ASSETS_DIR / key
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"{name}: could not open {path}")

    applied_variants = apply_variants(stage, variants)
    if variants and not applied_variants:
        raise RuntimeError(f"{name}: variants {variants} requested but no prim carries them")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    default_prim = stage.GetDefaultPrim()
    root = default_prim if default_prim and default_prim.IsValid() else stage.GetPseudoRoot()

    # Include the default AND render purposes: some catalog assets author their
    # visible geometry under `render`, and a default-only cache returns an empty
    # box for them (which would silently size a collider proxy to nothing).
    # `guide`/`proxy` are deliberately EXCLUDED here — collision hulls live there
    # and are usually slightly larger than the visible mesh, which would inflate
    # every bbox proxy and every furniture footprint in the layout.
    def measure(purposes) -> tuple:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
        return cache.ComputeWorldBound(root).ComputeAlignedRange()

    visible_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    rng = measure(visible_purposes)

    # An asset with no visible geometry at all is a collision-only companion
    # (e.g. sektion_cabinet_collisions.usd, whose prims are all purpose=guide).
    # Fall back to every purpose so its extent is still reported, and flag it.
    visual_geometry = not rng.IsEmpty()
    if not visual_geometry:
        rng = measure(
            visible_purposes + [UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
        )

    if rng.IsEmpty():
        aabb_min = aabb_max = (0.0, 0.0, 0.0)
        size_units = (0.0, 0.0, 0.0)
        degenerate = True
    else:
        aabb_min = tuple(float(v) for v in rng.GetMin())
        aabb_max = tuple(float(v) for v in rng.GetMax())
        size_units = tuple(float(b - a) for a, b in zip(aabb_min, aabb_max))
        degenerate = any(s <= 1e-9 for s in size_units)

    size_m = tuple(s * meters_per_unit for s in size_units)

    # Traverse instance proxies: the geometry of an instanceable asset (e.g.
    # sektion_cabinet_instanceable.usd) lives behind instance boundaries and a
    # plain Traverse() would report zero collision prims for it.
    collision_prims: list[str] = []
    authored_offsets: dict[str, dict[str, float]] = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(str(prim.GetPath()))
        for attr_name in OFFSET_ATTRS:
            attr = prim.GetAttribute(attr_name)
            if attr and attr.HasAuthoredValue():
                authored_offsets.setdefault(str(prim.GetPath()), {})[
                    attr_name.split(":")[-1]
                ] = float(attr.Get())

    has_collision = len(collision_prims) > 0
    measured_collision = "native" if has_collision else "bbox_proxy"

    return {
        "name": name,
        "s3_key": key,
        "local_path": key,  # relative to assets/
        "metersPerUnit": meters_per_unit,
        "default_prim": str(root.GetPath()),
        "variants": variants,
        "variants_applied_to": applied_variants,
        "visual_geometry": visual_geometry,
        "aabb": {"min": aabb_min, "max": aabb_max},
        "size_authored_units": size_units,
        "size_m": size_m,
        "size_m_at_duck_scale": tuple(s * DUCK_SCALE for s in size_m),
        # What the scene builder must pass as UsdFileCfg(scale=...): converts the
        # asset's authored units to metres AND applies duck scale in one factor.
        "scale_for_duck_scale": meters_per_unit * DUCK_SCALE,
        "degenerate_aabb": degenerate,
        "expected_collision": expected_collision,
        "measured_collision": measured_collision,
        "collision_matches_expectation": measured_collision == expected_collision,
        "n_collision_prims": len(collision_prims),
        "collision_prims": collision_prims[:20],
        "authored_contact_offsets": authored_offsets,
    }


def main() -> int:
    rows = read_asset_list()
    entries: dict[str, dict] = {}
    problems: list[str] = []

    hdr = f"{'asset':<26}{'mPU':>6}  {'size (m, authored)':<24}{'collision':<20}{'offsets':<10}"
    print(hdr)
    print("-" * len(hdr))

    for name, key, expected, variants in rows:
        if not (ASSETS_DIR / key).exists():
            problems.append(f"{name}: not mirrored ({key}) - run assets/fetch_assets.sh")
            print(f"{name:<26}{'--':>6}  {'MISSING':<24}")
            continue
        info = inspect_one(name, key, expected, variants)
        entries[name] = info

        size_str = " x ".join(f"{v:.2f}" for v in info["size_m"])
        coll = info["measured_collision"]
        if not info["collision_matches_expectation"]:
            coll = f"{coll} != {expected}"
            problems.append(
                f"{name}: expected {expected} but measured {info['measured_collision']} "
                f"({info['n_collision_prims']} collision prims)"
            )
        if info["degenerate_aabb"]:
            problems.append(f"{name}: degenerate AABB {info['size_authored_units']}")

        n_off = len(info["authored_contact_offsets"])
        print(
            f"{name:<26}{info['metersPerUnit']:>6.3g}  {size_str:<24}{coll:<20}"
            f"{(str(n_off) + ' prim(s)') if n_off else '-':<10}"
        )

    manifest = {
        "_comment": (
            "Generated by scripts/inspect_assets.py from the local mirror in assets/. "
            "Consumed by duck_embody/env/scene_builder.py (proxy sizing, per-asset "
            "scale, contact-offset overrides) and by apartment_layout.py authoring. "
            "Regenerate after any change to assets/asset_list.tsv."
        ),
        "duck_scale": DUCK_SCALE,
        "assets": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {MANIFEST.relative_to(REPO_ROOT)} ({len(entries)} assets)")

    # Report every authored offset explicitly: PLAN T0.2's acceptance criterion
    # is that each one is listed with its planned override (doc 03 §7).
    print("\n== authored contact/rest offsets (absolute metres - do NOT scale) ==")
    any_offsets = False
    for name, info in entries.items():
        for prim_path, offsets in info["authored_contact_offsets"].items():
            any_offsets = True
            print(f"  {name}: {prim_path} -> {offsets}")
    if not any_offsets:
        print("  none authored in any mirrored asset")

    if problems:
        print("\n== PROBLEMS ==")
        for p in problems:
            print(f"  {p}")
        return 1
    print("\nOK - every asset mirrored, non-degenerate, and collision class as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
