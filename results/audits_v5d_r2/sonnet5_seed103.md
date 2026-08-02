# sonnet5_seed103 — audit

turns 40 | bumps 9 | cost $1.55
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 16: believed 13.53 m vs true 12.97 m -> **+0.56 m**
- of which BUMPED 9: believed 3.67 m vs true 3.52 m -> **+0.15 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
