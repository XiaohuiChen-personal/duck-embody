# sonnet5_seed102 — audit

turns 40 | bumps 7 | cost $1.37
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 7: believed 1.96 m vs true 1.89 m -> **+0.07 m**
- of which BUMPED 7: believed 1.96 m vs true 1.89 m -> **+0.07 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

**Locomotion: HEALTHY under sustained contact.** 12 frames: spawns in the
bedroom, drives to a wall, and spends frames 5-12 pressed against / working
along it. Trunk upright in every frame, legs alternating, no fall, no crumple.
This is the scenario v4 could not survive (it fell on its first bump).

**Navigation: poor.** It wall-followed without finding a doorway and declared
done elsewhere at the 40-turn cap.

**Why this trial matters most for the redesign:** ALL 7 motion calls were
bumped — the precise class that generated +25.10 m of phantom distance in the
v4 batch. Inflation here: **+0.07 m**. A trial spent wedged against a wall no
longer teaches the model that it walked across the apartment.
