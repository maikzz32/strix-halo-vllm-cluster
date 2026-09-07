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

The candidate run `3e59a51755ce4688b1542eac7dd116a7` was stopped cleanly on all ranks. Original source `4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070` is restored on all four nodes. Canonical TP4/UMA is active as `8e0242c35b064a8baeb94f6debb4a088`, idle after the six concluding streams. The candidate is eligible for a controlled promotion with post-promotion checks, but has not been permanently enabled yet. AVX-512 and UCCL were not activated in these measurements.

Two bounded UCCL CPU builds ran during the after-control model loading period and finished before its benchmark started. This detail is retained as context; it is not evidence that they caused any measured variation. UCCL was never loaded into a process or used for transport. See [the independent UCCL review/build evidence](2026-09-07-uccl-review.md).

[Full model/stream hashes, metrics checks and runtime evidence](../bench/records/2026-09-07-moe-w2-serial-model.json).
