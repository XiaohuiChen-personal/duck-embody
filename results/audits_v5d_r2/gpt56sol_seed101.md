# gpt56sol_seed101 — audit

turns 24 | bumps 4 | cost $0.87
outcome {'find_kitchen': 'declared_elsewhere', 'return_home': 'not_run'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'not_run'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 7: believed 2.41 m vs true 2.46 m -> **-0.05 m**
- of which BUMPED 4: believed 0.85 m vs true 0.86 m -> **-0.02 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=False drift=None
- return_home: success=False drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
