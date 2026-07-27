"""AGENTS.md rule-1 preflight: refuse to launch beside another GPU/kit job.

Rule 1 ("ONE Isaac Sim / GPU job at a time") was a human checklist item —
"check ``nvidia-smi`` before launching" — and the pre-freeze forensic pass
found it violated twice: two logs end in a 22-minute hang that had to be
SIGKILLed while the dead process held the machine's only GPU, and the T3.5
concurrency probe showed a second kit does NOT reliably die in its init banner
(it can limp into nondeterministic failures at material binding / camera
attach — see PLAN.md T3.5). A batch trial started against a half-dead kit
process would fail minutes into a cold start, or worse, run degraded.

So the check is automated at every entry point that is about to launch a kit
process (``scripts/run_trial.py`` now; the T4.2 ``runner.py`` must call it
too). Two independent detectors, because they fail differently:

* ``nvidia-smi --query-compute-apps`` — anything CURRENTLY holding CUDA
  compute (a running sim, a training job, a wedged kit that still maps GPU
  memory);
* ``pgrep -af`` on the kit interpreter path segment — a kit process that is
  starting up or shutting down may hold the app lock while briefly absent
  from the compute-apps list (the kvdb-contention window the concurrency
  probe hit).

The parsing lives in pure functions so ``tests/test_preflight.py`` can drive
them with captured fake output — a guard that only runs on the real machine
is a guard nobody has ever seen fire until it matters.

Deliberately NON-blocking when a detector binary is missing: refusing to run
because ``pgrep`` is absent would invent a new way for the batch to fail that
has nothing to do with rule 1. Both tools exist on the DGX; the miss is
printed loudly and the launch proceeds.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

#: Regex over the FULL cmdline, anchored so only argv[0] — the interpreter
#: path itself (``.../kit/python/bin/python3``) — can match. A bare substring
#: pattern was MEASURED (2026-07-27, this machine) to false-positive on any
#: process whose command text merely mentions it: a shell running a command
#: containing the words "kit/python" matched itself. The launching wrappers
#: (``isaaclab.sh``, ``_isaac_sim/python.sh``) do not match either way.
KIT_PYTHON_PATTERN = r"^\S*kit/python/bin/python"

#: nounits so a future nvidia-smi cannot append " MiB" into a parsed column.
NVIDIA_SMI_CMD = (
    "nvidia-smi",
    "--query-compute-apps=pid,process_name",
    "--format=csv,noheader,nounits",
)
PGREP_CMD = ("pgrep", "-af", KIT_PYTHON_PATTERN)


@dataclass(frozen=True)
class Violation:
    """One process that makes launching a kit job a rule-1 violation."""

    pid: int
    source: str  # "nvidia-smi" | "pgrep"
    detail: str

    def __str__(self) -> str:
        return f"pid {self.pid} [{self.source}] {self.detail}"


def parse_compute_apps(csv_text: str, self_pid: int) -> list[Violation]:
    """``nvidia-smi --query-compute-apps`` CSV -> violations, excluding self.

    Empty output means no compute apps (the common good case). Lines that do
    not start with an integer PID are skipped rather than crashing: some
    driver builds emit informational text (e.g. "[N/A]") into this query, and
    the preflight must never be the thing that breaks a launch.
    """
    out: list[Violation] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, name = line.partition(",")
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        if pid == self_pid:
            continue
        out.append(Violation(pid=pid, source="nvidia-smi", detail=name.strip() or "<unnamed>"))
    return out


def parse_kit_processes(pgrep_text: str, self_pid: int) -> list[Violation]:
    """``pgrep -af <pattern>`` output ("<pid> <cmdline>") -> violations.

    Excludes ``self_pid`` because the caller IS a kit python (``isaaclab.sh
    -p`` interpreters match the pattern before AppLauncher ever runs) — the
    rule is "no OTHER kit job", not "never run at all".
    """
    out: list[Violation] = []
    for line in pgrep_text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, cmdline = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == self_pid:
            continue
        out.append(Violation(pid=pid, source="pgrep", detail=cmdline or "<no cmdline>"))
    return out


def format_refusal(violations: list[Violation]) -> str:
    """The message printed when the preflight refuses. Names every PID, so the
    operator can decide between waiting and killing — the guard itself never
    kills anything (a 22-minute encode or a colleague's job looks identical to
    a wedged process from here)."""
    lines = [
        "FATAL: rule-1 preflight found another GPU/kit job on this machine "
        "(AGENTS.md rule 1: ONE Isaac Sim / GPU job at a time):"
    ]
    lines += [f"  {v}" for v in violations]
    lines.append(
        "Refusing to launch. Wait for it to finish, or if it is a wedged "
        "shutdown (the exception-exit hang: see results/logs/README.md), "
        "verify and kill it, then re-run."
    )
    return "\n".join(lines)


def _run(cmd: tuple[str, ...]) -> str | None:
    """Run one detector; None means it could not run (missing binary etc.).

    ``pgrep`` exits 1 for "no matches", which is a SUCCESS for our purposes —
    only an execution failure returns None.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):
        return None
    return proc.stdout


def rule1_violations(self_pid: int | None = None) -> list[Violation]:
    """Run both detectors against the live machine. Empty list = clear."""
    pid = os.getpid() if self_pid is None else self_pid
    violations: list[Violation] = []

    smi = _run(NVIDIA_SMI_CMD)
    if smi is None:
        print("  [preflight] WARNING: nvidia-smi unavailable — GPU compute-app check skipped")
    else:
        violations.extend(parse_compute_apps(smi, pid))

    pgrep = _run(PGREP_CMD)
    if pgrep is None:
        print("  [preflight] WARNING: pgrep unavailable — kit-process check skipped")
    else:
        violations.extend(parse_kit_processes(pgrep, pid))

    # One process can appear in both detectors; report it once, nvidia-smi
    # first (its detail names the binary, pgrep's the full cmdline).
    seen: set[int] = set()
    unique: list[Violation] = []
    for violation in violations:
        if violation.pid not in seen:
            seen.add(violation.pid)
            unique.append(violation)
    return unique
