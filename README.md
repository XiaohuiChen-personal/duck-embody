# Duck Embody 🦆

**LLM-as-SLAM: can a language model navigate a walking robot through an unknown
apartment — with no mapping system except its own memory?**

> 🚧 **Work in progress** (started 2026-07-26). Skeleton + design phase; results land here.

An LLM (Claude / GPT / open-weight VLM) controls a simulated 42 cm bipedal robot
([Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini)) in NVIDIA Isaac Sim
through tool calls: it sees egocentric camera frames, issues velocity commands to a
pretrained RL locomotion policy, and must **find the kitchen in an apartment it has
never seen**. There is no SLAM, no occupancy grid, no depth sensor. The model
authors its own map as text — rooms, exits, landmarks — dead-reckons its position,
and corrects drift by recognizing places it has seen before ("cognitive loop
closure"). The harness stores and formats that memory; every spatial fact in it was
asserted by the model from its own observations.

## Why this is interesting

- Anthropic's ["How Claude Performs on Robotics Tasks"](https://www.anthropic.com/research/claude-plays-robotics)
  (Jul 2026) found frontier models drive robots well through high-level velocity
  commands but *"fail at tasks that require stable spatial memory, self-localization,
  or long open-loop plans."* This project asks: **is that a model limitation, or a
  harness limitation?** We give the model the memory scaffolding their harness
  lacked and measure what changes.
- A prior-art sweep (~40 repos/papers, 2026-07-26) found no published system where
  the language model is sole holder of both the map and the position estimate:
  existing work either lets classical geometry build the map (VLMaps, SG-Nav, VLFM…),
  gets nodes free from a discrete simulator graph (MapGPT), or keeps no memory at
  all (VLMnav). This slot appears to be unoccupied.
- The embodiment is harsher than the paper's Go2 quadruped: a biped that can
  actually fall, a 0.22 m/s velocity envelope, and a bobbing head-mounted camera
  0.36 m above the floor.

## What gets measured

Several frontier models run the identical task set — same apartment, same tools,
same prompt, fixed seeds: **find the kitchen and reach the counter**, then
**return to the start** (the direct test of whether the self-built map is real).
Metrics: success rate, time-to-kitchen, path efficiency (SPL), falls/bumps, and —
because the harness stores the model's map — **map accuracy scored against ground
truth**, something aggregate task metrics can't see.

## Architecture (one paragraph)

A persistent Isaac Sim process runs the duck under a pretrained PPO policy
(59-dim proprioceptive observations, 50 Hz control; trained in Isaac Lab in the
[parent robot project](https://github.com/XiaohuiChen-personal/Open_Duck_Mini_Jetson)).
The sim **pauses while the model thinks** (measuring capability, not API latency —
the paper's protocol). Per turn the model receives one camera frame, compass
heading, a drifting dead-reckoned position, and its own re-injected map; it acts
through closed-loop motion macros (`turn_to_heading`, `move`) and memory tools
(`update_room`, `mark_exit`, `correct_position`, …). Details: [`AGENTS.md`](AGENTS.md)
(design decisions + full technical context) and [`docs/PLAN.md`](docs/PLAN.md).

## Results

*Coming — table + videos per model after the benchmark batch runs.*

## Honest framing & caveats

- Replication-and-extension of the embody *paradigm* on a novel embodiment; the
  original harness is unreleased — this is a from-scratch implementation, not a fork.
- Single embodiment, simulation only, small N per model, one prompt template, one
  max-scaffold memory configuration (no ablation yet). No claims beyond the data.
- Sensor honesty: camera + compass + dead reckoning mirror the real robot's actual
  hardware (head CSI camera, BNO055 IMU, no depth/lidar). Ground-truth position is
  never given to the model.

## Process & attribution

Interface, tasks, metrics, and analysis designed by me; implementation is
AI-assisted (Claude Code) under my direction. The locomotion policy, simulation
stack, and robot model come from my Open Duck Mini Jetson project (Isaac Lab PPO
training on a DGX Spark). This project doubles as the sim-side prototype for that
robot's Phase 5 (on-board VLM navigation with Cosmos Reason2).
