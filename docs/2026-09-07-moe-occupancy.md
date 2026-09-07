# Same-split MoE tile and warp experiments

The W1 BN32/two-warp candidate reduces isolated rotating-expert time by 3.5–6.3% while preserving all tested BF16 output bits. It is a candidate for a subsequent model trial, not an enabled serving change. The earlier W1 study varied split-K substantially; this experiment keeps split count, K tile, quantization and accumulation order fixed.

## W1

The installed baseline uses SPLITK4, BK128, BN64, four warps, two stages. The candidate changes only BN to32 and warps to2. BN64/two warps and BN64/four warps/one stage lose in the preliminary fixed-expert test. The four configurations pass 40 independent-reference checks, including missing experts and trimmed/native routing capacities. The selected candidate also matches the baseline bit for bit in all ten preliminary routing/layout cases.

| Rotating routing pattern | Installed W1 (us) | BN32 / two warps (us) |
|---|---:|---:|
| 10 shared experts per invocation | 38.900 | 36.440 |
| 25 overlapping experts per invocation | 82.686 | 79.819 |
| 40 distinct experts per invocation | 114.760 | 109.956 |

The raw W1 rotating record calls its middle pattern `overlap20`, but its actual offsets are 0/5/10/15, hence **25 experts**, unlike the earlier fixed-expert test's offsets 0/5/10/0. This table reports the actual pattern. Across 24 calls the patterns touch respectively 240/462/477 expert matrices, totaling 113.7/218.8/225.9 MB of packed weights, scales and zero points. These are working-set sizes, not measured cache-miss rates.

All 72 rotating W1 comparisons and captured replays after changing inputs match bit for bit. The timings use five samples with method order alternating, twenty replays per sample and 24 calls per graph. Matrix weights and routing are synthetic; actual model routing and activation traces have not been tested here. Baseline drift within the disjoint sample is retained in the record.

## W2

A separate rotating-expert test retains SPLITK5/BK32/stages1 and compares output tile/warp configurations. Every variant matches the installed baseline on 72 synthetic inputs and changed-input graph replay.

| Pattern | Installed BN64/w4 (us) | BN32/w2 | BN64/w2 | BN128/w4 |
|---|---:|---:|---:|---:|
| Shared10 | 31.842 | 31.280 | 30.740 | 31.574 |
| Overlap25 | 61.888 | 61.672 | 62.520 | 60.890 |
| Disjoint40 | 81.608 | 80.350 | 77.751 | 77.088 |

The W2 improvements depend on routing. They are much smaller than a major full-model speedup, and the existing baseline remains active. No model-level speedup is inferred by summing independent microbenchmark percentages.

## Evidence and next step

[All samples, source hashes, numerical checks and process/service checks](../bench/records/2026-09-07-moe-occupancy.json) are retained. All three bounded Node18 jobs exited0 with no owned processes remaining. Idle serving counters stayed6 before and after every job. The canonical TP4/UMA run was not restarted or edited.

The drivers `tools/run_moe_w1_occupancy.py`, `tools/run_moe_w1_rotating.py` and `tools/run_moe_w2_rotating.py` default to dry run; each takes `--output NEW_DIRECTORY` and requires `--execute` for GPU execution. They use the existing patched container and reject an unexpected fused-MoE source hash.

A subsequent W1 model trial should limit the change to the verified M4/top10/N320/K2560/GS32 path, preserve the original default, and compare identical requests, outputs and MTP acceptance against the canonical UMA baseline. The small W1 gain alone does not establish that the single-answer throughput target has been reached.

The subsequent [bracketed model trial](2026-09-07-moe-w1-model.md) preserved all outputs but showed only 0.57% generation gain against the stronger control. The candidate was not promoted.
