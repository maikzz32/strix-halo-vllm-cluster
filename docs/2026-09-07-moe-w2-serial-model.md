# W2 serial fusion: completed model comparison

The candidate preserves all 48 ShareGPT answer texts, per-request token lengths and speculative decoding statistics across an original/candidate/original comparison. Weights, context, MTP3, QSA and the existing AVX2 UMA transport remain fixed. Candidate source is `25ac1a380733b9f3ee3735cc58779e2d1215d5f87ed9e601cfc839d49c1403b7`, enabled through `STRIX_MOE_W2_SERIAL=1` only for its bounded shape.

| Configuration | Output tokens/s | Mean TPOT (ms) | Mean TTFT (ms) |
|---|---:|---:|---:|
| Original before | 52.3103 | 16.9845 | 540.59 |
| Serial W2 | 52.7070 | 16.5383 | 600.62 |
| Original after | 51.4522 | 17.4248 | 568.70 |

Against the stronger original control, the candidate provides **2.70% faster decoding** from mean TPOT, but only **0.76% greater overall output throughput**. Initial waiting times vary, including a slower first request in the after control; all observations remain in the results. No cause for these delays is asserted. The two unchanged controls differ by 1.64% in overall throughput, so the small overall gain alone should not be treated as strong evidence.

Six matched 512-token streams with `ignore_eos=true` provide an additional check: all request bodies, contents, reasoning, usage and finish reasons match. Candidate decoding improves by 2.98–3.32%, median **3.15%**, against the after-control streams. Together with the isolated operator tests this supports a small practical decoding improvement, not the large speed increase sought by the overall goal or a formal statistical confidence bound.

All three ShareGPT runs complete 48 requests without failures. Independent server counters increase by exactly 48 and queues are idle before and after. There is no additional warmup request: the benchmark skips endpoint-ready checking. Runtime evidence on every candidate/restored rank records the source hash, W2 environment flag, actual mapped UMA library and fresh 32-bank active-PyNccl parity. Runtime source hashing verifies the installed file, not Python bytecode introspection.

The candidate run `3e59a51755ce4688b1542eac7dd116a7` was stopped cleanly on all ranks, and the original source was restored for the after control. Subsequent promotion and verification are complete as described below. AVX-512 and UCCL were not activated in these model measurements.

Two bounded UCCL CPU builds ran during the after-control model loading period and finished before its benchmark started. This detail is retained as context; it is not evidence that they caused any measured variation. UCCL was never loaded into the serving processes. Later isolated plugin tests failed before any collective completed; see [the independent UCCL review/build evidence](2026-09-07-uccl-review.md).

[Full model/stream hashes, metrics checks and runtime evidence](../bench/records/2026-09-07-moe-w2-serial-model.json).

## Promotion and final verification

Canonical run `032b7174e0404a96926d8847bdc94ccf` uses the serial W2 source on all four nodes, the original W1 kernel and the existing AVX2 UMA library. The only semantic configuration change is `STRIX_MOE_W2_SERIAL=1`. Canonical configuration SHA-256 is `e516f0bf5eea420292d967795442261801741008bc759145e1d8e558d5307304`. All four ranks passed fresh 32-bank active-PyNccl startup parity, and read-only runtime evidence confirms the expected installed source, environment and mapped library.

The post-promotion ShareGPT48 C1 run achieves **53.8508 output tokens/s**, **16.4336 ms mean TPOT** and **547.31 ms mean TTFT**. All 48 answers, input/output lengths and speculative decoding statistics match both the earlier serial candidate and original before control. Against the original before control, this run is 2.95% faster overall and 3.35% faster in decoding; this is one repeat, not a confidence interval. The bracketed trial above remains the main comparison for attributing the W2 effect.

Six matched 512-token streams preserve request bodies, contents, reasoning, usage and finish reasons. Six additional C2 smoke requests each complete 512 tokens successfully; C1/C2 answer equivalence is not asserted. Independent service counters increase from 0 to 48 for ShareGPT and to 60 after the stream/smoke checks, with empty queues. The service remains running. [Final verification record](../bench/records/2026-09-07-moe-w2-production.json).

The explicit success target is at least **60 output tokens/s in this ShareGPT48 C1 benchmark**, with unchanged weights and computational quality. It remains unmet: approximately 11.4% additional speed is needed. Short-prompt decode rates and reciprocal mean TPOT do not substitute for this target.

For exact backup and rollback instructions, see the [native runbook](../scripts/native/README.md#w2-serial-canonical-configuration).
