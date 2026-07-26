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
# Usage:  bash scripts/run_tests.sh [pytest args...]
#         bash scripts/run_tests.sh tests/test_layout.py -v
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1

exec "$HOME/IsaacLab/isaaclab.sh" -p -m pytest "$@"
