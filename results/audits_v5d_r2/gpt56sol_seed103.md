# gpt56sol_seed103 — audit

turns 40 | bumps 14 | cost $1.58
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 20: believed 5.24 m vs true 5.03 m -> **+0.22 m**
- of which BUMPED 14: believed 1.19 m vs true 1.14 m -> **+0.05 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
