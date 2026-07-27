# results/logs — log policy, benign-noise allowlist, batch canary checklist

Committed per AGENTS.md rule 3 (every number names its source) and the
pre-freeze forensic pass (gaps G13/G22/G23). Every count below was measured
against the committed logs in this directory (grep commands shown), not
inherited from memory.

## Log policy (G13)

- **One log file per invocation.** Every sim run logs to a fresh
  timestamp-suffixed path (`<name>_<YYYYmmdd-HHMMSS>.log`). Never reuse a
  `results/logs/` path for a rerun of a changed script: `t3_5_contact_side.log`
  once recorded `exit=0` for an invocation whose log a later crashed rerun
  overwrote — the recorded verdict pointed at evidence that no longer existed.
- **Exit codes are trustworthy but coarse.** `isaaclab.sh` propagates 0/1 only
  (measured); per-trial success is judged from the trial JSON (`final` present,
  doc 06 §9.1), never from the shell exit code.
- Always run sim scripts with `PYTHONUNBUFFERED=1` and under
  `timeout --kill-after=...` with a budget derived from the script's own step
  count (see `scripts/smoke_gap_hunt.py`'s run line for the pattern) — the
  exception-exit hang below is why.

## Benign-noise allowlist (G22)

Expected severity lines **per apartment-scene launch**, byte-stable across all
8 committed apartment-launch logs (verify with the greps; a deviation in COUNT
is the signal, not the lines themselves):

| Count | Pattern (grep) | Why benign |
|---|---|---|
| 45 | `CreateJoint - cannot create a joint between static bodies` | The 9 authored Sektion joints × 5 counters. Static bodies cannot be jointed, which is exactly what keeps drawers/doors rigid. Doc 03 §11 (RESOLVED), T2.4 `counter_bump` PASS. |
| 3 | `MDLC:COMPILER.*comp error` | SimPBR.mdl's sibling modules were never mirrored; part of the 6-line degraded-materials signature recorded in doc 03 §5 (G4). Frames are judge-gate-validated in this state. |
| 2 | `rtx.neuraylib.*MdlModule` (`Loading MdlModule to DB … failed` / `NOT in the DB`) | Same G4 signature. |
| 1 | `USD_MDL.*MdlModuleId.*is Invalid` | Same G4 signature. |
| 12 | `Unable to find SdrShaderNode` | Hydra falling back after the MDL failure above; same G4 signature. |
| 5 | `Could not perform 'modify_collision_properties'` | The 5 Sektion counters' collision prims are inside an instanceable subtree; offsets validated anyway by T2.4 (doc 03 §7 caveat, G21). |
| 1 | `DLSS increasing input dimensions: Render resolution of (297, 297)` | The render pipeline natively renders ~297×297 and DLSS upscales to 512 (doc 04 §4 correction, G19). |
| 1 | `Seed not set for the environment` | Emitted at env construction; `SimSession.reset()` seeds explicitly afterwards, per trial. |

Non-apartment launches (empty-plane smoke tests) produce none of the scene
lines; the DLSS and seed lines appear on any rendering launch.

## Signal patterns — never allowlist these

1. **Exception-exit hang.** A `Traceback (most recent call last):` in a sim
   log, especially followed by `python.sh: line 73: … Killed` (the shell had
   to SIGKILL a kit that threw during shutdown and never exited — it holds
   the machine's only GPU meanwhile). Committed examples:
   `t3_5_contact_side.log` (NameError → 22-min hang → Killed),
   `t2_4_viewer.log` (teardown AttributeError, log ends mid-shutdown).
2. **kvdb contention.** Any line mentioning `kvdb` — the T3.5 concurrency
   probe's marker for two kit processes fighting over the app database, i.e.
   an AGENTS.md rule-1 violation in progress (see `docs/PLAN.md` T3.5).
   `duck_embody/sim/preflight.py` exists to prevent it; this grep catches
   what the preflight raced.

## T4.3 batch canary checklist (run per trial log, before trusting the batch)

```
grep -c "CreateJoint - cannot create"    <log>   # expect 45 per apartment launch
grep -c "MDLC:COMPILER"                  <log>   # expect 3
grep -c "Unable to find SdrShaderNode"   <log>   # expect 12
grep -c "modify_collision_properties"    <log>   # expect 5
grep -ci "kvdb"                          <log>   # expect 0 — rule-1 violation detector
grep -c  "Traceback"                     <log>   # expect 0 — exception-exit detector
grep -Ein "cut.?off|truncated" <trial>.json      # expect no NEW hits (G23: a model
    # remarking its budget section was "cut off" is likeliest its own summarized-
    # thinking echo — model difficulty, LEFT alone — but grep so a real harness
    # truncation cannot hide behind that explanation)
```

Any off-list severity line, or any allowlisted line at the wrong count, stops
the batch until explained. `scripts/smoke_gap_hunt.py` S0 automates the
allowlist diff for the pre-freeze smoke.
