# gpt56sol_seed104 — audit

turns 40 | bumps 9 | cost $1.45
outcome {'find_kitchen': 'timeout_turns', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'turn_cap', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 14: believed 6.05 m vs true 5.61 m -> **+0.44 m**
- of which BUMPED 9: believed 2.43 m vs true 2.25 m -> **+0.17 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
