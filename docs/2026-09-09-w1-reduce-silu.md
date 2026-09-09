# W1 reduction and SiLU fusion: isolated candidate

This candidate combines the four-part FP32 W1 reduction and the following
SiLU/multiply operation. It leaves the W1 GEMM intact, unlike the previously
rejected serial W1 GEMM experiment. No serving source has been changed.

For 40 rows, N320 and Split-K4, the installed two-operator sequence measured
5.767 microseconds versus 3.836 for the fused Triton kernel. Timing used six
alternating rounds with 32 graph replays each. Forty-eight synthetic banks
at four scales checked 307200 output values bitwise, with no mismatches;
a changed-input graph replay also matched.

An initial candidate preserved the BF16 cast after reduction but omitted the
BF16 rounding after SiLU, before multiplication. It failed 73847 values.
Adding that intermediate rounding made this test exact. This is evidence
for these inputs, not a proof for all possible activation values; native
activation source and broader numerical cases still need checking.

The isolated saving is about 1.93 microseconds per pair. Across 48 target-layer
pairs that suggests only about 0.093 ms before integration overhead, not a
prediction of serving TPS and not enough alone to reach 60 tokens/s.
Full-model integration and unchanged-output serving validation remain pending.

Test: `tests/bench_w1_reduce_silu.py`. Evidence:
`bench/records/2026-09-09-w1-reduce-silu.json`.
