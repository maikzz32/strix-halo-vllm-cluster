# Dynamic speculation: current implementation boundaries

Inspection of the installed `0.29.0+strix.rocm100` source found an existing
`num_speculative_tokens_per_batch_size` schedule. The scheduler selects its
entry using the number of scheduled requests, not their observed acceptance.
At steady C1, a batch-size-only table therefore does not adapt draft depth to
an individual answer's acceptance history.

`enable_adaptive_verification` is explicitly restricted to DSpark. It is not
an existing adaptive-MTP switch for this checkpoint. The configuration handler downgrades dynamic speculation from full graphs to
PIECEWISE only when V2 is not enabled. **Correction:** the current service
already uses V2, confirmed by all four worker startup logs for run
`594dd64087594e7eb373d1bf6bdc6593`. The earlier conclusion that the current
runner would force PIECEWISE was incorrect; it applied a V1 condition to V2.
See `bench/records/2026-09-10-v2-runner-proof.json` for logs and source hashes.

Relevant locations in the inspected files: `config/speculative.py` around
lines 472 and 533, `v1/core/sched/scheduler.py` around line 1319 and
`config/vllm.py` around line 926. Source SHA256s are recorded in
`bench/records/2026-09-10-dynamic-speculation-source.json`.

An acceptance-driven MTP policy remains a possible implementation project,
but requires graph-shape support, recurrent-state/rollback validation and
matched output/performance checks. No model, runner or speculative setting
was changed during this source review.

V2 graph setup (`v1/worker/gpu/cudagraph_utils.py`) explicitly enumerates
verification lengths from the dynamic batch-size schedule. Its hybrid model
state passes per-request draft and accepted-token counts into the GDN metadata.
These mechanisms are a better starting point than a runner migration. They do
not by themselves prove correctness of a new acceptance-driven policy.

The inspected autoregressive speculator still loops to its configured
`num_speculative_steps`. Reducing the number of verified tokens is not proof
that fewer draft forwards run. Any proposed adaptive policy must verify both
scheduling and actual draft execution, and capture every length it can select.
No dynamic policy has been enabled in the service.
