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
