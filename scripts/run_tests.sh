#!/usr/bin/env bash
# Run the test suite with the KIT python — the interpreter policy for this repo.
#
# Two environment hazards this wrapper neutralises, both of which otherwise make
# `pytest` fail in ways that have nothing to do with the code under test:
#
# 1. **A sourced ROS 2 (jazzy) install leaks into PYTHONPATH.** Its
#    `launch_testing` pytest plugin is auto-discovered, imports `lark`, and dies
#    with ModuleNotFoundError during COLLECTION — before a single test runs.
#    (It is also built for python3.12 while the kit python is 3.11.)
#    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 stops the discovery; this project needs no
#    third-party pytest plugins.
#
# 2. **Why the kit python and not system python3:** the two interpreters are
#    disjoint. Only the kit python has isaaclab/rsl_rl AND (after T0.0)
#    anthropic/openai. See AGENTS.md §4.
#
# 3. **Stale bytecode reports green on code that is no longer on disk.** A .pyc
#    is reused when (source mtime truncated to SECONDS, source size) is
#    unchanged — so an edit that keeps the byte count and lands in the same
#    second as the source it replaces silently executes the OLD module.
#    Reproduced during T3.1's review pass with the kit python: a module holding
#    `X = 40`, imported, then rewritten to `X = 25` (identical length, mtime
#    unchanged) still imported as 40; after `rm -rf __pycache__` with this
#    variable set it imported as 25 and wrote no cache. That is exactly the
#    edit-then-test loop an agent runs, and a one-character constant fix is
#    exactly the shape that hits it — a test suite reporting green on code that
#    is no longer on disk is the one failure this repo cannot afford, since the
#    suite is the gate in front of a paid batch (AGENTS.md rule 2).
#    PYTHONDONTWRITEBYTECODE=1 removes the hazard; the suite takes 0.2 s, so the
#    recompile costs nothing.
#
# Usage:  bash scripts/run_tests.sh [pytest args...]
#         bash scripts/run_tests.sh tests/test_layout.py -v
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

exec "$HOME/IsaacLab/isaaclab.sh" -p -m pytest "$@"
