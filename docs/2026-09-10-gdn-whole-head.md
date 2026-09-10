# Whole-head GDN recurrence feasibility

The installed speculative recurrence partitions each 128-value head into
four BV32 programs. Output RMS normalization spans the whole head. A direct
single-program recurrence/norm fusion therefore needs a different work
partition or additional synchronization.

An isolated Node 4 test changes only BV32 to BV128 in the hash-checked installed
Triton source, retaining four warps. The source is imported from a temporary
file; no serving source or environment changes. All 16 cases (four seeds and
accepted-token counts 1 through 4) produce identical BF16 outputs and FP32
state tensors. Changed-input graph replay also matches each variant's eager
output and state. These are bounded operator checks, not model parity.

| Variant | Median recurrence replay, us |
| --- | ---: |
| Original BV32 | 14.7395 |
| Whole-head BV128 | 19.2097 |

Timing alternates variants over eight rounds, each taking the median of 32
individual HIP-event measurements. State is restored outside each timed replay
to avoid benchmarking an evolving recurrence. Shapes are C1, four verification
tokens, four Q/K heads and twelve V heads, K=V=128, BF16 inputs and FP32 state.
Both variants use the same input/state reset protocol. Serving request counters
remain unchanged throughout the test.

The whole-head variant adds about 4.47 microseconds before adding normalization.
This is a feasibility cost, not evidence for promotion. A subsequent fusion
would need to recover that cost plus its own norm work. The test does not
identify whether register pressure, occupancy or another resource causes the
regression; no resource attribution is claimed. It also does not test other
warp counts or a cooperative multi-program design.

Evidence: `tests/bench_gdn_whole_head.py` and
`bench/records/2026-09-10-gdn-whole-head.json`, including source hashes,
per-case differences and timing samples. Production retains BV32.
