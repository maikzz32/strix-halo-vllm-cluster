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

The subsequent finite-gate sweep checked all 65280 finite BF16 gate values
with four multipliers (261120 outputs), including exact signed output bits:
zero mismatches. This does not exhaust all four-part FP32 reduction inputs.
See `tests/check_w1_reduce_silu_edges.py`.

Call-site inspection found that production uses reduction BLOCK512, whereas
the initial benchmark used BLOCK128. The corrected benchmark uses BLOCK512
for the reference and retains BLOCK128 for the fused candidate. It again
passes all 307200 random values and changed-input graph checks; medians are
5.772 versus 3.821 microseconds. Both runs are preserved in the record.
Integration must target the WNA16 expert's invocation followed by
`self.activation`, rather than assuming the legacy `fused_experts_impl` path.

The isolated saving is about 1.93 microseconds per pair. Across 48 target-layer
pairs that suggests only about 0.093 ms before integration overhead, not a
prediction of serving TPS and not enough alone to reach 60 tokens/s.
Full-model integration and unchanged-output serving validation remain pending.

Test: `tests/bench_w1_reduce_silu.py`. Evidence:
`bench/records/2026-09-09-w1-reduce-silu.json`.
