"""T0.0 environment-bootstrap smoke test.

Two parts:

1. **Import check (no kit app).** Every package the agent loop and the sim layer
   need must import from the *kit* python — the single interpreter that runs both
   (PLAN "Interpreter policy"). ``isaaclab_tasks`` is deliberately NOT in this
   list: it imports ``pxr`` at module scope, and ``pxr`` only becomes importable
   once ``AppLauncher`` has started the kit app (verified 2026-07-26).

2. **Kit-launch check.** T0.0 installs packages into the kit python's
   site-packages, which can in principle shadow or upgrade something a kit
   extension depends on. Launching the app headless and closing it again proves
   the bootstrap did not break Isaac Sim, and is far cheaper than discovering it
   during T1.3. It also reports whether ``pxr`` and ``isaaclab_tasks`` become
   importable inside the app (needed by T0.2 and T1.2 respectively).

Two kit gotchas this script discovered the hard way (2026-07-26), now recorded in
AGENTS.md §5 and obeyed by every sim script in this repo:

* kit buffers stdout aggressively — run with ``PYTHONUNBUFFERED=1`` or nothing
  printed during the run survives.
* ``SimulationApp.close()`` **terminates the process**; statements after it never
  execute. Print verdicts and write artifacts BEFORE closing.

Run:  PYTHONUNBUFFERED=1 ~/IsaacLab/isaaclab.sh -p scripts/smoke_env.py
"""

from __future__ import annotations

import importlib
import sys

# Packages that must import with no kit app running.
NO_APP_MODULES = [
    "isaaclab",
    "rsl_rl",
    "torch",
    "yaml",
    "dotenv",
    "anthropic",
    "openai",
    "PIL",
    "matplotlib",
    "pytest",
    "numpy",
    "duck_embody",
]

# Packages that are expected to require a running kit app.
APP_MODULES = ["pxr", "isaaclab_tasks"]


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - we report whatever went wrong
        return False, f"{type(exc).__name__}: {exc}"
    return True, getattr(mod, "__version__", "")


def main() -> int:
    failures: list[str] = []

    print("== 1. imports without a kit app ==")
    for name in NO_APP_MODULES:
        ok, info = _try_import(name)
        print(f"  {name:<14} {'OK' if ok else 'FAIL'} {info}")
        if not ok:
            failures.append(f"{name} ({info})")

    import numpy

    # Isaac Sim 5.1.0's compiled extensions are built against the numpy 1.x ABI.
    print(f"  numpy version  {numpy.__version__}")
    if not numpy.__version__.startswith("1."):
        failures.append(f"numpy is {numpy.__version__}, expected 1.x (kit ABI)")

    print("\n== 2. kit app launch (headless) ==")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True)
    app = launcher.app
    print("  AppLauncher started OK")

    for name in APP_MODULES:
        ok, info = _try_import(name)
        print(f"  {name:<14} {'OK' if ok else 'FAIL'} {info}")
        if not ok:
            failures.append(f"{name} inside app ({info})")

    try:
        from pxr import Usd  # noqa: F401

        print(f"  pxr USD version {Usd.GetVersion()}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"pxr.Usd unusable ({exc})")

    # Verdict is printed BEFORE app.close(): close() terminates the process, so
    # anything after it is never reached (verified 2026-07-26 — the first version
    # of this script silently lost its own result section).
    print("\n== result ==")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    else:
        print("  OK - kit python imports everything and Isaac Sim still launches")
    print("  closing app (nothing after this line runs)")

    app.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
