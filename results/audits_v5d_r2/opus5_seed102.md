# opus5_seed102 — audit

turns 40 | bumps 26 | cost $4.36
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 35: believed 17.30 m vs true 16.73 m -> **+0.57 m**
- of which BUMPED 26: believed 9.13 m vs true 8.84 m -> **+0.29 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
