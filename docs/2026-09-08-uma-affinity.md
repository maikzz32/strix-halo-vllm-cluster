# UMA progress-thread affinity on Strix Halo

A bounded isolated trial finds no compelling reason to bind the communication thread in serving. Canonical W1 expert scheduling, W2 serial and AVX512 remain active and unchanged. The current full-model C1 result remains56.0396 output token/s; the60 target is not met.

Each host has two CPU-L3 groups, `0-7,16-23` and `8-15,24-31`; the NIC reports no preferred NUMA group. Five-second read-only observations show one continuously busy worker thread per node. These likely correspond to UMA progress, but the production library does not export their identity, so this attribution is not proven. During active generation, these threads show9/0/14/0 migrations on nodes15/16/17/18. All have CPU0-31 permitted.

The isolated backend exports the actual progress-thread TID, without changing transport arithmetic. The test saves its affinity, selects the least busy physical core (including its SMT sibling's load) in each CPU-L3 group, and alternates unbound/LLC0/LLC1 and reverse order for five rounds. Each measurement covers32 graph replays of32 collectives of20KiB. Core selection comes from a250ms `/proc/stat` sample; this is a heuristic, not CPU isolation. Original affinity is restored before teardown. No production threads or IRQs are changed.

| Rank | Selected CPUs LLC0/LLC1 | Unbound median us | LLC0 median us | LLC1 median us |
|---|---|---:|---:|---:|
| 0 | 1 / 8 | 48.466 | 48.283 | 49.636 |
| 1 | 5 / 8 | 48.292 | 48.235 | 49.761 |
| 2 | 1 / 8 | 48.580 | 48.192 | 49.660 |
| 3 | 1 / 8 | 48.399 | 48.266 | 49.778 |

The first-group improvement is0.12–0.80%; the second group is slower. These are isolated HIP-event means per collective, summarized by sample median, not token/s or per-operation tail percentiles. Five samples in one trial are insufficient for statistical confidence. Serving remains idle throughout the isolated trial.

All four native processes pass fresh32-bank eager/graph parity against active PyNccl and10240-element BF16 checks, plus1088 captured output checks per rank across changed inputs and affinity variants. Graphs and communicators are destroyed, original affinity restored, and no owned processes remain.

The first orchestration exits1 because its inherited verifier expects one variant and one check per sample. Native ranks all exit0. The corrected verifier requires exactly three ordered variants, two checks per variant, matching timing arrays and restored affinity, and successfully revalidates the unchanged saved results. The original failure is retained; no successful rerun is claimed.

The six512-token model streams taken before the isolated trial are observational samples, not an affinity treatment comparison. No full-model affinity test or production configuration change is justified by this result.

[Machine-readable record](../bench/records/2026-09-08-uma-affinity.json). Next priorities are lossless W1 layout and measured GPU epilogue overhead; those remain hypotheses.
