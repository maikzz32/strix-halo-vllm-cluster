# W1 serial partial fusion: rejected

The W2 fusion's benefit does not transfer to the tested W1 shapes. This isolated
candidate retains the four original interleaved Split-K accumulators, each with
five K128 dot steps, then adds them sequentially in FP32 before the BF16 cast.
An opaque FP32 move separates partials from the outer reduction. It uses the
same asymmetric INT4 group-32 dequantization and BF16 dot inputs. No installed
source or serving configuration was changed.

Tests ran in a separate process in the existing Node18 container while the
model was idle. The source guard checks the current W2-enabled installation
(`25ac1a38...`); its W1 implementation is unchanged. Dimensions are four input
rows, top-k 10, 512 experts, N320/K2560 and BM16. Each routing pattern rotates
through 24 banks. Both BN64/four-warps and BN32/two-warps candidates are tested.

| Pipeline stages / seed | Routing | Original microseconds | Best serial microseconds | Extra time |
|---|---|---:|---:|---:|
| 1 / 9340 | reuse10 | 40.04 | 69.12 | 72.6% |
| 1 / 9340 | overlap25 | 79.57 | 125.74 | 58.0% |
| 1 / 9340 | disjoint40 | 112.61 | 160.92 | 42.9% |
| 2 / 9341 | reuse10 | 38.26 | 55.13 | 44.1% |
| 2 / 9341 | overlap25 | 83.55 | 119.85 | 43.4% |
| 2 / 9341 | disjoint40 | 121.34 | 142.14 | 17.1% |

Each run passes all 72 routing-bank comparisons and graph replay with changed
inputs for both candidates. Timing uses alternating method order, five samples
and 20 graph replays per sample. These are synthetic operator tests, not a
model-quality proof or model throughput measurement. The runs use different
seeds, so their cross-run difference is not an isolated pipeline-stage effect.

Reducing the number of independent workgroups is a plausible explanation for
the slowdown, but no register/spill or hardware-counter attribution was made.
Both tested configurations are rejected without a full-model deployment.

Both bounded processes exited successfully and left no owned processes.
The model's success counter stayed at 60, with no running or waiting requests.
Canonical run `032b7174e0404a96926d8847bdc94ccf` remains active with serial W2,
original W1 and AVX2 UMA. Its last measured ShareGPT48 C1 throughput remains
53.85 tokens/s; the goal of at least 60 remains open.

[Raw summaries and source hashes](../bench/records/2026-09-07-moe-w1-serial.json).
The committed benchmark uses two stages. The one-stage run's exact source is
archived locally with a matching manifest hash; it differs only in `num_stages`
and the two method labels. `tools/run_moe_w1_serial.py` defaults to dry run and
requires `--execute` plus a new output directory for the isolated test.
