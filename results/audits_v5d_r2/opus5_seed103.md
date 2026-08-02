# opus5_seed103 — audit

turns 37 | bumps 16 | cost $3.72
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 25: believed 19.71 m vs true 18.90 m -> **+0.81 m**
- of which BUMPED 16: believed 8.86 m vs true 8.48 m -> **+0.38 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
