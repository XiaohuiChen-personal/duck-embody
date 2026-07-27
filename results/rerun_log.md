# Rerun log — doc 06 §7

Every resume move and infra rerun in the batch, appended by
`duck_embody/runner.py`. It ships with the results: reruns are visible, not
silent. Model failures (cap / fall / wrong `declare_done`) are final results
and never appear here — the only legitimate rerun is a logged infra failure.
T4.3's restart branch, when taken, is also recorded here
(a: fix touches non-frozen code -> keep the freeze commit, resume;
b: fix touches any frozen file -> new freeze commit, new batch directory,
restart from zero).

| trial id | timestamp (UTC) | cause | evidence |
|---|---|---|---|
| opus5_seed101 | 2026-07-27T08:51:25Z | infra failure (attempt 1): anthropic.OverloadedError: Error code: 529 - {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}, 'request_id': 'req_011CdSH1VvSXFUyxu4NQrHDb'} | results/incomplete/opus5_seed101.20260727-085125.json |
