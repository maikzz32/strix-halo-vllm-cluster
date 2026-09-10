# Current TP4 decode profile

The restored vLLM 0.29 TP4/MTP3 runtime was profiled for 16 active iterations
on all four nodes, with the existing INT4 weights, W1 expert ordering, serial
W2 and AVX512 UMA exchange. No model arithmetic was changed. Profiling run:
`b8d7e4c467aa4b6d9136ea003fe1bdf5`. One 2048-token answer completed; both
profiling API calls returned HTTP 200 and cleanup succeeded.

Each trace contains 16 complete M4 target graphs with 2646 kernels and 16
additional 112-kernel graphs. Target identity uses CPU scope containment and
same-file correlation IDs. Attribution checks all 48 adjacent
W1/reduce/SiLU/W2/sum sequences, all 97 HC down/SiLU/up/gate sequences and
98 UMA exchange kernels in each target graph. The former 2694-kernel count
is obsolete after removal of the separate W2 reduction.

Mean disjoint target-graph time, milliseconds:

| Component | Node 1 | Node 2 | Node 3 | Node 4 |
|---|---:|---:|---:|---:|
| Full target span | 35.93 | 36.05 | 36.02 | 36.00 |
| HC down BF16 | 3.90 | 3.90 | 3.89 | 3.90 |
| HC up BF16 | 3.17 | 3.17 | 3.20 | 3.18 |
| Routed W1 | 3.58 | 3.60 | 3.62 | 3.58 |
| Routed W2 | 2.18 | 2.57 | 2.41 | 2.19 |
| UMA exchange, including waits | 7.36 | 7.32 | 5.67 | 7.53 |
| Intervals without a traced kernel | 6.13 | 5.90 | 7.43 | 6.00 |
| Dense/router/shared INT4 kernels | 3.59 | 3.60 | 3.61 | 3.57 |

Remaining categories and every individual graph are recorded in
`bench/records/2026-09-09-decode-attribution.json`, with raw trace SHA256s.
The inventory and attribution scripts are under `tools/`. Source sequence
checks are intentionally strict and must be updated from evidence when kernels
change. Raw timestamps are not aligned across hosts.

These are **profiled target-graph durations**, not end-to-end iteration time
or serving throughput. Drafting, scheduling and sampling add work outside
the target graph. Exchange duration includes waiting; intervals without a
traced kernel cannot simply be labeled CPU overhead. Profiling itself can
alter execution. Any optimization requires unprofiled serving measurements.

The HC matrices remain a sizeable, consistent target at about 7.1 ms per
verification graph. Earlier transpose, AITER and lossless packing experiments
are documented separately and must not be repeated as new discoveries.
Further work should examine the current native BF16 kernel and its memory
traffic while retaining all weights and computation. Communication and gaps
also remain substantial, but their rank-dependent split needs careful
attribution before selecting another transport change.

The profiling processes were stopped cleanly. The canonical runtime is
restarting as `e00892710abd4eccb50391edfaa5e9f7`; no kernel source changed.
