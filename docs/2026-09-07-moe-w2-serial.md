# W2 fusion with separately rounded split products

An isolated candidate computes the five existing K32 W2 dot products in one workgroup and adds their FP32 results sequentially before applying the routing weight and casting to BF16. This removes the 2 MiB partial-output workspace and the separate split-reduction launch at the tested M40/N2560/K160 shape. It does not fuse the subsequent expert sum or alter the weights.

The first attempt failed bitwise comparison: all three tile variants differed in 210 output elements across 24 routing banks. It was rejected before timing. An opaque FP32 register move between each dot result and its addition prevents compiler combination across this boundary. That corrected candidate passes all 72 routing-bank cases per seed, plus graph replays with changed activations, for two seeds. This is empirical parity for those tests, not an exhaustive proof for arbitrary inputs. The precise compiler transformation responsible for the first failure has not been established from disassembly.

| Routing pattern | Installed, seed9340 (us) | BN32/2 warps (us) | Installed, seed9341 (us) | BN32/2 warps (us) |
|---|---:|---:|---:|---:|
| Reuse10 | 32.763 | 23.416 | 32.684 | 23.706 |
| Overlap25 | 62.775 | 41.828 | 64.493 | 43.653 |
| Disjoint40 | 82.523 | 59.162 | 85.486 | 62.874 |

These are medians of five GPU-event samples, each covering 20 replays of a 24-bank graph. Method order alternates by sample. The first installed reuse sample is substantially slower in both runs; it is retained in the evidence. Routing banks rotate over more than the GPU cache capacity. BN64/4 and BN128/4 were also tested and were slower than BN32/2. The measured time reductions of roughly 26–33% apply to this operator, not full-model token throughput.

All three experiments terminate and leave no owned process. Serving stays idle with its success counter unchanged at 54 throughout. Each exact source version is retrieved from its isolated remote directory and verified against its manifest hash. The failed attempt is retained alongside the successful tests in [the evidence record](../bench/records/2026-09-07-moe-w2-serial.json).

The candidate source is staged separately on all four nodes at `/opt/strix-halo-next/moe-w2-serial-20260907-r1`, hash `25ac1a380733b9f3ee3735cc58779e2d1215d5f87ed9e601cfc839d49c1403b7`. Installed source remains original `4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070`; no model restart or activation occurred. The opt-in patch is restricted to the tested tensor shapes, contiguous BF16 activations/scales, UINT8 packed weights/zero-points, FP32 routing weights and BM16. Other shapes retain the original path. The kernel source is hash-guarded in addition to the baseline runtime source.

The actual dispatch guard now passes CPU tests: one intended shape accepted, 24 alternatives rejected, and two modified-source inputs rejected. This checks routing conditions, not GPU arithmetic. The [completed full-model comparison](2026-09-07-moe-w2-serial-model.md) now verifies identical answers, token lengths and speculative statistics, with 2.70% faster decoding against the stronger control and 3.15% in matched short streams. Original TP4/UMA AVX2 is restored as run `8e0242c35b064a8baeb94f6debb4a088`; permanent promotion remains pending.
