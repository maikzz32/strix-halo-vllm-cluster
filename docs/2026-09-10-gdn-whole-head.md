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

## Matched recurrence-plus-norm controls

The actual checkpoint configuration specifies sigmoid output gating and RMS
epsilon 1e-6. The follow-up captures recurrence and the installed `rmsnorm_fn`
in the same graph, alongside recurrence-only controls. Synthetic BF16 norm
weights and gates are used, not extracted checkpoint activations. Four variants
are timed in alternating forward/reverse order, with state resets outside timing.

| Whole-head warps | Original recurrence | Whole-head recurrence | Original + norm | Whole-head + norm |
| --- | ---: | ---: | ---: | ---: |
| 4 | 14.760 | 19.049 | 17.160 | 21.479 |
| 8 | 14.960 | 16.879 | 17.319 | 19.279 |
| 16 | 14.830 | 16.740 | 17.179 | 19.149 |

All values are microseconds. All 16 cases per configuration match in recurrence
outputs, normalized outputs and the compared recurrence states. Changed-input
graph outputs and states match eager execution for each of the four variants.
Request counters remain unchanged. This is not full-model validation.

Increasing the whole-head warp count reduces its penalty, but its recurrence
alone remains about 1.9 us slower than the original. The norm adds about 2.4 us
to the original graph. Thus only approximately 0.44 us separates the best
whole-head recurrence-only timing from its matched original-plus-norm timing,
before adding fused normalization work. This is a screening comparison, not a
rigorous bound: fusion can change compiler resource allocation and scheduling.
It does not justify a production trial of this direct whole-head design as the
next substantial speedup. No fused operator was implemented or promoted.

The test is `tests/bench_gdn_norm_chain.py --warps 4|8|16`; records are
`bench/records/2026-09-10-gdn-norm-chain-w{4,8,16}.json`. The initial four-warp
record predates the explicit warp CLI metadata; its candidate source hash
matches the four-warp whole-head source above.
