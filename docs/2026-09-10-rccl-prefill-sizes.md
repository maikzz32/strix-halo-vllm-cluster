# Isolated RCCL prefill-size comparison

The original compiled backbone contains 97 annotated all-reduce calls over
BF16 `[tokens, 2560]`. This implies 40,960 bytes for eight tokens and
2,580,480 bytes for 504 tokens. It does not establish that all 98 RCCL kernel
events in each profiled scope are those same calls; the additional event has
not been attributed to an exact payload.

Four separate processes use PyNcclCommunicator with the original workers'
readable NCCL environment, four channels, a separate bootstrap port, and an
explicit Gloo interface from the cluster configuration. The first attempt
failed during Gloo bootstrap because some readable worker environments lacked
GLOO_SOCKET_IFNAME and Gloo selected loopback. All processes exited; no RCCL
latency was measured in that attempt. The corrected run completed on all ranks.

| Rank | 20,480 bytes, us | 40,960 bytes, us | 2,580,480 bytes, us |
| --- | ---: | ---: | ---: |
| 0 | 97.025 | 127.229 | 1340.135 |
| 1 | 103.749 | 129.123 | 1337.577 |
| 2 | 99.044 | 125.105 | 1343.620 |
| 3 | 104.520 | 128.320 | 1339.214 |

Each result is the median of five samples of 32 graph replays, with rank
barriers outside timing. Two changed integer input patterns produce exact
sums. This is not cancellation-rich BF16 reduction-order calibration. All
graphs, NCCL communicators and CPU groups were destroyed; no owned process
remained, and serving request counters were unchanged.

For context, a bandwidth-optimal four-rank ring moves `2*(4-1)/4 = 1.5`
payload bytes per rank. At a nominal 25 Gbit/s, 1.5 times 2,580,480 bytes takes
1.239 ms before protocol overhead. This is a theoretical ring comparison,
not proof of the measured algorithm or actual wire utilization. It indicates
limited room for a large-payload transport-only speedup at unchanged precision.
Our small UMA transport sends each full input to three peers: naively scaling
that method doubles the per-rank bytes relative to the ring reference and is
not justified by these results.

The profiled model RCCL category also includes arrival skew and synchronization.
Its sum cannot be replaced by this isolated timing or treated entirely as
avoidable host staging. Further transport work needs a bandwidth-efficient
algorithm and per-size numerical validation; lifting the 10240-value guard
alone is neither correct nor a supported optimization.

Evidence: `bench/records/2026-09-10-rccl-prefill-sizes.json`,
`tests/bench_rccl_prefill_sizes.py`, and `tools/run_rccl_prefill_sizes.py`.
Production is restored and unchanged, with all Hermes checks passing.
