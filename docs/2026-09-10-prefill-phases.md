# Request-phase latency audit

The unchanged TP4 service completed twelve C1 completion requests, three per
exact prompt length, with up to 32 output tokens. Inputs use a unique prefix
and the same token IDs for repeats. Each request advances all six collected
phase-histogram counts by exactly one; running/waiting gauges are zero after
completion. No concurrent GPU microbenchmark ran.

| Prompt tokens | Repeat | Client TTFT, ms | Server prefill phase, ms | Prefix hits |
| --- | --- | ---: | ---: | ---: |
| 256 | First use | 475.84 | 461.29 | 0 |
| 256 | First reuse | 421.70 | 395.31 | 0 |
| 256 | Second reuse | 407.79 | 398.37 | 0 |
| 512 | First use | 632.40 | 603.75 | 0 |
| 512 | First reuse | 620.55 | 597.94 | 0 |
| 512 | Second reuse | 612.31 | 598.07 | 0 |
| 1024 | First use | 1399.49 | 1385.61 | 0 |
| 1024 | First reuse | 1044.32 | 1025.44 | 0 |
| 1024 | Second reuse | 1044.57 | 1025.63 | 0 |
| 2048 | First use | 2010.12 | 1979.46 | 0 |
| 2048 | First reuse | 1999.80 | 1993.74 | 0 |
| 2048 | Second reuse | 1234.07 | 1218.58 | 800 |

Queue time is 0.005–0.039 ms. The server's prefill phase dominates TTFT;
queue tuning cannot explain the hundreds of milliseconds here. The phase
metric includes work within vLLM's phase boundaries and is not pure GPU kernel
time. First-use differences without prefix hits must not be labeled cache
savings. All three output hashes match within each prompt length in this run;
this does not establish general prefix-cache correctness.

The existing four-rank September 9 profiler capture cannot attribute this
prefill cost: profiling was requested only after first output, and the trace
contains the M4 generation scopes. A separate pre-request capture is needed
before selecting prefill kernels for optimization. Existing negative prefill
MoE tile experiments remain valid and should not be repeated solely because
prefill is now the focus.

`bench/context_latency.py --phase-metrics` now records sums/counts for TTFT,
queue, inference, prefill, decode and computed prefill KV tokens. Evidence is
`bench/records/2026-09-10-prefill-phases.json`. This is a diagnostic workload,
not a new ShareGPT benchmark or a model configuration change.
