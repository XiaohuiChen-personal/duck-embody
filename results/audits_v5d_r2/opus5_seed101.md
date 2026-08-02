# opus5_seed101 — audit

turns 23 | bumps 7 | cost $1.24
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 10: believed 4.59 m vs true 4.71 m -> **-0.12 m**
- of which BUMPED 7: believed 2.46 m vs true 2.52 m -> **-0.06 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
