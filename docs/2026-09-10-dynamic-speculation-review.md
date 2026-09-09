# Dynamic speculation: current implementation boundaries

Inspection of the installed `0.29.0+strix.rocm100` source found an existing
`num_speculative_tokens_per_batch_size` schedule. The scheduler selects its
entry using the number of scheduled requests, not their observed acceptance.
At steady C1, a batch-size-only table therefore does not adapt draft depth to
an individual answer's acceptance history.

`enable_adaptive_verification` is explicitly restricted to DSpark. It is not
an existing adaptive-MTP switch for this checkpoint. In addition, the current
configuration handler downgrades dynamic speculation from full graphs to
PIECEWISE unless the V2 model runner is enabled. Merely enabling the schedule
would therefore change a critical execution setting at the same time.

Relevant locations in the inspected files: `config/speculative.py` around
lines 472 and 533, `v1/core/sched/scheduler.py` around line 1319 and
`config/vllm.py` around line 926. Source SHA256s are recorded in
`bench/records/2026-09-10-dynamic-speculation-source.json`.

An acceptance-driven MTP policy remains a possible implementation project,
but requires graph-shape support, recurrent-state/rollback validation and
matched output/performance checks. No model, runner or speculative setting
was changed during this source review.
