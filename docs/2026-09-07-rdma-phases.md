# CPU transport phase measurements

The existing transport already sends the complete 20 KiB input directly to each of its three peers. `RCCL_RING4_BF16` chooses the calibrated arithmetic order after reception; it is not a three-round network ring. Replacing a presumed network ring with direct exchange would duplicate the current implementation.

Two CPU-only experiments measured an isolated copy of `cpu_rdma_transport.c` with monotonic timestamps around send posting, completion polling and reduction. All ranks used the existing Intel RoCE interfaces, separate bootstrap port 29679 and malloc-backed arenas. No HIP calls, model restart, package installation or serving modification occurred. API request counters remained unchanged and idle before/after both jobs.

## Alternating measurement result

The second run alternates tracing on/off in 32-call blocks, so each state sees all 32 input banks. Three phases each contain 64 warmups and 256 retained samples. Every collective is checked against the unchanged host Ring4 reference, including BF16 cancellation patterns. Printing occurs only after all timed loops.

| Rank | Untraced total median (us) | Traced total median | Send posting median | Completion wait median | Reduction median |
|---|---:|---:|---:|---:|---:|
| 0 | 47.039 | 47.024 | 0.580 | 37.019 | 8.919 |
| 1 | 47.103 | 47.058 | 0.580 | 37.254 | 9.010 |
| 2 | 46.769 | 46.829 | 0.630 | 33.179 | 12.790 |
| 3 | 47.019 | 47.024 | 0.580 | 37.479 | 8.859 |

These are medians of separate distributions and must not be added as an exact decomposition. Means and raw samples are also retained. The wait includes all send/receive completions, wire serialization, peer arrival skew and effects of the other ranks' previous validation/reduction work. It is not a measurement of switch latency alone. The rank2 difference does not establish a particular CPU scheduling or frequency cause.

The first run used longer untraced/traced/untraced blocks. Its traced medians were 56.6–58.1 us versus 44.7–47.4 us for controls, so it did not cleanly separate timestamp overhead from drift. That result is retained rather than replaced silently. The alternating run shows nearly equal traced/untraced totals, providing a more useful phase estimate.

Both runs completed 960 exact collectives per rank and cleaned up all owned processes and verbs contexts. This validates the unchanged arithmetic implementation in the instrumented transport; it is not a new GPU-vs-RCCL numerical proof.

## Implication

Posting three verbs requests costs less than 1 us, so optimizing that alone has little opportunity. CPU BF16 summation remains a measurable fraction of the total. A bounded next experiment can test wider CPU vector arithmetic while preserving the existing per-add BF16 rounding, scalar handling of exceptional values and calibrated rank order. No such replacement is enabled by this study.

The malloc arena and CPU-driven cadence differ from the serving GPU-mapped arena. These timings cannot simply be subtracted from prior HIP collective timings to infer GPU overhead or full-model speedup.

[All samples and source identities](../bench/records/2026-09-07-rdma-phases.json) are recorded. Both generated source versions are archived in the local result directories with hash checks. The driver `tools/run_cpu_rdma_phases.py --output NEW_DIRECTORY` defaults to dry run; `--execute` runs the bounded four-node CPU test only when serving request gauges are idle.
