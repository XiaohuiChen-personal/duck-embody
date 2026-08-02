# gpt56sol_seed102 — audit

turns 40 | bumps 16 | cost $1.37
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 18: believed 7.20 m vs true 6.95 m -> **+0.24 m**
- of which BUMPED 16: believed 6.39 m vs true 6.18 m -> **+0.22 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
