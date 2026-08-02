# opus5_seed104 — audit

turns 31 | bumps 6 | cost $2.33
outcome {'find_kitchen': 'success', 'return_home': 'success'}
end_reason {'find_kitchen': 'declare_done', 'return_home': 'declare_done'}

## Odometry (leg-odometry redesign, 2026-07-30)

- motion calls 13: believed 8.54 m vs true 7.91 m -> **+0.63 m**
- of which BUMPED 6: believed 4.13 m vs true 3.82 m -> **+0.31 m**  (the class that produced +25.10 m in the v4 batch)
- correct_position calls: 0
- find_kitchen: success=True drift=None
- return_home: success=True drift=None

## Frame audit

_pending visual pass — sheet in sheets/_
