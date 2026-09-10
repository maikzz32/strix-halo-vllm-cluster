# INT4 vocabulary projection launch geometry

The current checkpoint config uses hidden size 2560 and vocabulary 248320,
with `lm_head` included in asymmetric INT4 group-32 quantization. The nominal
TP4 shard shape is N62080/K2560. Prior launch-grid testing covered router and
shared-expert linears, not this larger shape.

On idle Node 4, four rotating synthetic packed weight banks were tested at
M1 and M4 with the installed `wvSplitK_int4_g` operator. Six alternating rounds
of 20 graph replays provide the following median operator times. All tested
choices match the original output exactly in eager and changed-input graph
execution. Serving counters stayed unchanged and the bounded process exited 0.

| Launch argument | M1 us | M4 us |
| --- | ---: | ---: |
| 20 (installed value) | 442.797 | 457.742 |
| 10 | 624.803 | 559.573 |
| 30 | 439.323 | 473.147 |
| 40 | 435.087 | 476.082 |
| 60 | 443.389 | 474.001 |

The best M1 observation is about 1.74% faster; M4 retains the original value.
This is one seed on one node, not a repeatable model gain. An approximately
7.7-us M1 operator difference is insufficient evidence for another serving
restart. The running MTP3 service remains unchanged.

Each synthetic bank contains 99,328,000 bytes of packed weights, BF16 scales
and BF16 zero points. Dividing this nominal payload by the M1 base time gives
about 224 GB/s. This is payload/time, not a hardware-counter measurement of
DRAM traffic, and does not prove a strict bandwidth ceiling. Cache behavior,
actual model layout and surrounding operators can differ.

Reproduce with `python tools/run_head_grid.py` on an idle cluster. Full samples
are in `bench/records/2026-09-10-head-int4-grid.json`; the benchmark is
`tests/bench_head_int4_grid.py`. No serving patch was generated or promoted.
