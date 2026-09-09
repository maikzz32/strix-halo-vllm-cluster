# Shape-specific dense INT4 launch geometry

The installed RDNA hybrid path passes `num_compute_units()` (20 on these
APUs) into `wvSplitK_int4_g` for small decode batches. This experiment varies
that argument locally; it does not change reported hardware properties or
claim that 20 physical CUs exist. Strix Halo has 40 physical CUs.

The native op and its arithmetic are unchanged. Each node independently tested
M1/M4 and three observed model shapes with 24 rotating synthetic asymmetric
INT4 group-32 weight/scale/zero-point banks, grids 10/20/30/40/60, six alternating
measurement rounds and 20 graph replays per round. Timings include the complete
HIP op, reported per invocation. All eager outputs and changed-input graph
outputs were bitwise equal to grid20. No model requests overlapped the tests,
according to unchanged success counters and idle before/after queue counts.

## Selected results

| Shape [M,N,K] | Grid20 to grid40 latency change across four nodes | Decision |
| --- | ---: | --- |
| [1,512,2560] router | -14.0% to -23.3% | Candidate for isolated dispatch integration |
| [4,2560,160] shared down | -14.4% to -16.2% | Candidate for isolated dispatch integration |
| [4,512,2560] router | -3.6% to +1.6% | Keep original |
| [1,320,2560] shared gate/up | -0.1% to +1.9% | Keep original |
| [4,320,2560] shared gate/up | +1.4% to +1.6% | Keep original |
| [1,2560,160] shared down | -0.3% to -0.1% | Keep original |

These are microbenchmarks with synthetic weights, not serving gains. Rotating
banks do not reproduce full-model cache pressure. Correctness evidence covers
the sampled eager/graph inputs, not every possible input. No arithmetic rewrite
or new quantization was used, and no production source was modified.

Both installed Python and native-library SHA256 match across all four nodes;
the evidence and individual samples are in
[the record](../bench/records/2026-09-10-dense-int4-grid.json).
The test is `tests/bench_dense_int4_grid.py`; the controller is
`tools/run_dense_grid.py --host 192.168.1.18` (sequential runs only).
The controller refuses non-idle service and bounds the remote process to150s.
Use a fresh result directory for a repeat; existing results are not overwritten.

Next: preserve the original dispatch for all other shapes, test the actual
shape-restricted wrapper, then compare the candidate against restored original
ShareGPT48/C1 runs with output and Hermes parity. Do not promote from these
kernel timings. Original TP4/MTP3 serving remains active at the previous
approximately 56.3 tokens/s baseline.


## Actual dispatch integration

`patches/dense_int4_grid.py` generates a source-hash-guarded candidate with
`STRIX_DENSE_INT4_GRID=1`. It changes only the two selected shapes, only on
gfx1151 with the original grid20, asymmetric group32, BF16 activations/scales/
zero points, INT8 packed weights and no bias. All other dispatch remains original.

`tests/check_dense_int4_dispatch.py` extracts and executes the actual generated
function with the installed module dependencies. On node4, all six forms passed
flag-off and flag-on output parity. Observed native-call arguments prove the
selected grids. HIP graph replay with three input changes passed exact parity;
adding bias selected the original grid and passed parity. The test does not
change installed source. See `2026-09-10-dense-int4-dispatch.json` in the records.

The source candidate has SHA256
`bdacb1fe96faa57b1b7153b95712da927a1c9f6c6a960b013694be6877b767d2`.
Original and candidate source are staged separately under
`/opt/strix-halo-next/dense-grid-20260910-r1` on each node. After stopping all
original workers, the same candidate hash was activated on all four nodes.
Run `183be1d308ae4636a16bdea946a6b1d6` completed the serving trial with the
option enabled, original UMA transport and GPU high mode.

## Serving trial: no gain over the prior control

| Configuration | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| Prior original control | 56.272 | 565.697 | 15.382 |
| Restricted dense INT4 candidate | 55.599 | 580.107 | 15.509 |

The candidate is 1.20% slower in this comparison. All 48 texts, output lengths
and speculation statistics are identical (11,839 output tokens, 14,767 input
tokens). Hermes auto tools, tool roundtrip, streaming and default thinking pass.
The isolated speedups did not establish a serving gain. No cause for the
regression is established; do not attribute it to a particular kernel or cache
mechanism without further evidence.

All candidate workers were stopped, and the original source SHA256 was restored
on all four nodes. Run `594dd64087594e7eb373d1bf6bdc6593` is loading the original
configuration for the fresh restored control. The candidate is not promoted.
The restored control result is still pending.

