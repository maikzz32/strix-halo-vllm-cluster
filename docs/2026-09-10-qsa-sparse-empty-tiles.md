# Sparse QSA empty-tile screening

An isolated Node 4 experiment conditionally skips K/V loads and the attention
update when a selection tile has no valid physical cache entries. It preserves
the original tile geometry, split boundaries and arithmetic for populated tiles.
The installed module and serving processes were not modified.

The test uses TP4-shaped BF16 queries (6 heads, head dimension 256), one KV head,
2051 selection slots, and synthetic paged caches. Four row counts and five mask
patterns cover empty selections, causal prefixes, alternating empty tiles, full
selections and invalid logical pages/request IDs. All 20 outputs match the
original bit for bit. Changed-input HIP graph replay also matches candidate
eager execution. This is finite synthetic coverage, not full-model validation.

| Rows | Selection | Original (us) | Skip empty tiles (us) |
| --- | --- | ---: | ---: |
| 4 | Causal | 38.77 | 20.34 |
| 8 | Causal | 70.03 | 37.55 |
| 504 | Causal | 4916.13 | 988.08 |
| 504 | Full | 5219.80 | 5928.61 |
| 1024 | Causal | 9892.57 | 3212.60 |
| 1024 | Full | 10509.98 | 11742.56 |

Measurements alternate original/candidate order over six samples of eight
graph replays. The eight-row empty case has large original outliers; its median
must not be used as a reliable speedup estimate. Raw samples are retained.
Serving request counters remained unchanged throughout the experiment.

The short-prefix result justifies further work, but unconditional use regresses
fully populated prefill by about 12–14%. A production candidate needs a
graph-compatible path choice based on available context metadata, or another
implementation that avoids this regression. Row count alone is insufficient:
a small chunk can belong to a long context. Actual indexer-generated masks,
multiple requests, cache page permutations and additional numerical cases
should be validated before a matched TP4 serving trial. No serving gain or
60-token/s result is claimed.

Test: `tests/bench_qsa_sparse_empty_tiles.py`. Evidence:
`bench/records/2026-09-10-qsa-sparse-empty-tiles.json`.
