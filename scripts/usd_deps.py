"""Print the external asset paths a USD file references, one per line.

Used by ``assets/fetch_assets.sh`` to mirror exactly the files an asset needs
(sub-USDs and textures) instead of blindly pulling whole S3 prefixes — the oven
prefix alone is 243 MB for two models we use one of.

Paths are printed **relative to the USD file's own directory**, normalised
(``./Textures/x.png`` → ``Textures/x.png``), so the caller can map them straight
onto S3 keys. Paths that escape the asset root via ``../`` are printed as-is and
the caller resolves them.

Both resolved *and* unresolved references are printed: during a first mirror
nothing is on disk yet, so "unresolved" is the normal state of a file we still
need to fetch, not an error.

Requires ``pxr``, which is importable from NEITHER default interpreter — invoke
through the pinned packman USD runtime (``fetch_assets.sh`` does this for you):

    USDROOT=$(ls -d ~/.cache/packman/chk/usd.py311.manylinux_2_35_aarch64.stock.release/* | head -1)
    PYTHONPATH=$USDROOT/lib/python LD_LIBRARY_PATH=$USDROOT/lib:$LD_LIBRARY_PATH \
      ~/IsaacLab/_isaac_sim/python.sh scripts/usd_deps.py <file.usd>
"""

from __future__ import annotations

import posixpath
import sys

from pxr import UsdUtils


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <file.usd>", file=sys.stderr)
        return 2
    usd_path = argv[1]

    # (sublayers, references, unresolved) — references covers both USD payloads
    # /references and shader texture inputs.
    sublayers, references, unresolved = UsdUtils.ExtractExternalReferences(usd_path)

    # `unresolved` entries are emitted too, NOT skipped. A reference is
    # "unresolved" simply because the target is not on disk yet — which is the
    # normal state during a first mirror, and exactly the file we need to fetch.
    # Dropping them silently lost real assets: sektion_cabinet_instanceable.usd
    # reaches its `configurations/*.usd` (used by its variants) only this way.
    seen: set[str] = set()
    for raw in list(sublayers) + list(references) + list(unresolved):
        if not raw or "://" in raw:
            # Skip empty entries and absolute URIs (none expected in this catalog).
            continue
        rel = posixpath.normpath(raw.lstrip("/"))
        if rel in (".", "..") or rel in seen:
            continue
        seen.add(rel)
        print(rel)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
