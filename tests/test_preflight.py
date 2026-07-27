"""Rule-1 preflight guard logic (gap G7), driven by captured fake output.

The guard exists because rule 1 was a human checklist item and got violated
twice (the 22-minute wedged-shutdown hang holding the GPU; the concurrency
probe's nondeterministic second-kit failures). A guard that only runs on the
real machine is a guard nobody has seen fire, so the parsing is pure and
these tests feed it the exact output shapes the detectors produce.
"""

from __future__ import annotations

from duck_embody.sim.preflight import (
    Violation,
    format_refusal,
    parse_compute_apps,
    parse_kit_processes,
)

SELF_PID = 40001

#: Shape verified against `nvidia-smi --query-compute-apps=pid,process_name
#: --format=csv,noheader,nounits` (driver 580.x on the DGX Spark).
SMI_TWO_APPS = (
    "179048, /home/x/IsaacSim/_build/linux-aarch64/release/kit/python/bin/python3\n"
    "40001, /home/x/IsaacSim/_build/linux-aarch64/release/kit/python/bin/python3\n"
)

#: Shape of `pgrep -af kit/python`: "<pid> <full cmdline>".
PGREP_TWO_KITS = (
    "179048 /home/x/.../kit/python/bin/python3 scripts/smoke_env.py --headless\n"
    "40001 /home/x/.../kit/python/bin/python3 scripts/run_trial.py --model fable5\n"
)


class TestParseComputeApps:
    def test_another_compute_app_is_a_violation_and_self_is_not(self):
        violations = parse_compute_apps(SMI_TWO_APPS, self_pid=SELF_PID)
        assert [v.pid for v in violations] == [179048]
        assert violations[0].source == "nvidia-smi"

    def test_empty_output_means_clear(self):
        assert parse_compute_apps("", self_pid=SELF_PID) == []
        assert parse_compute_apps("\n\n", self_pid=SELF_PID) == []

    def test_informational_non_pid_lines_are_skipped_not_fatal(self):
        # Some driver builds emit placeholder text into this query; the
        # preflight must never be the thing that breaks a launch.
        out = parse_compute_apps("[N/A]\nno running compute apps\n", self_pid=SELF_PID)
        assert out == []

    def test_unnamed_process_still_reports_its_pid(self):
        out = parse_compute_apps("777,\n", self_pid=SELF_PID)
        assert out[0].pid == 777
        assert out[0].detail == "<unnamed>"


class TestParseKitProcesses:
    def test_another_kit_python_is_a_violation_and_self_is_not(self):
        violations = parse_kit_processes(PGREP_TWO_KITS, self_pid=SELF_PID)
        assert [v.pid for v in violations] == [179048]
        assert "smoke_env.py" in violations[0].detail

    def test_no_matches_is_clear(self):
        assert parse_kit_processes("", self_pid=SELF_PID) == []


class TestRefusalMessage:
    def test_refusal_names_every_pid_and_cites_the_rule(self):
        """'Refuse and print PIDs' is the whole contract: the operator must be
        able to inspect each process before deciding to wait or kill."""
        violations = [
            *parse_compute_apps(SMI_TWO_APPS, self_pid=SELF_PID),
            *parse_kit_processes(PGREP_TWO_KITS, self_pid=SELF_PID),
        ]
        message = format_refusal(violations)
        assert "rule 1" in message.lower()
        assert "179048" in message
        assert "Refusing to launch" in message
        assert "kill" in message  # the operator's options are stated

    def test_violation_str_is_greppable(self):
        v = Violation(pid=123, source="pgrep", detail="python3 foo.py")
        assert str(v) == "pid 123 [pgrep] python3 foo.py"
