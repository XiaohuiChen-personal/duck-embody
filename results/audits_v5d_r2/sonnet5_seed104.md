# sonnet5_seed104 — audit

turns 40 | bumps 4 | cost $1.16
outcome {'find_kitchen': 'timeout_turns', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'turn_cap', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 9: believed 5.37 m vs true 4.98 m -> **+0.39 m**
- of which BUMPED 4: believed 1.95 m vs true 1.80 m -> **+0.15 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
