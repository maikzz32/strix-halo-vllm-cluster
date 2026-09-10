# RDMA posting order and early reduction experiments

Two isolated changes preserve peer payloads, receive slots and the existing
ordered AVX512/BF16 reduction. Both passed the eager and graph parity checks
against active PyNccl on all four nodes. Neither has been promoted to serving.

## Cyclic posting: no demonstrated gain

The original loop posts peers in ascending rank order. A cyclic variant posts
`(rank + step) % 4` for steps 1–3. This visits every other peer once without
changing how inputs are summed. The hypothesis was less simultaneous traffic
to the same switch output; no switch-level congestion measurement was made.

Maximum rank median complete HIP-event latency was 49.294 / 49.109 / 48.152
microseconds for before / cyclic / after. The candidate is within control
variation and is not promoted. Each run used three samples of eight replays,
with 32 collectives per graph. See `2026-09-10-rdma-cyclic.json` in the records.

## Early reduction: isolated improvement

The existing code waits for all receives and all send completions before
reducing. The candidate starts the identical reduction once all receives are
available, then continues polling until all sends finish. It preserves the
acquire fence before reading received inputs and the final release ordering.
Current receive slots cannot be overwritten by the next sequence, which uses
the other slot. Input reuse still waits for every send completion.

Maximum rank median HIP-event latency was 48.007 / 45.822 / 48.116 microseconds
for before / candidate / after. All four candidate rank medians were below
their controls. Each run used three samples of 32 replays and 32 collectives
per graph. This roughly 4.7% isolated improvement is not a serving TPS claim.
The diagnostic model benchmark completed separately at 54.82 tokens/s; it
includes the trace library and is not a matched uninstrumented control.
The candidate uses the original uninstrumented source plus early reduction.
The full serving comparison below is complete; the original transport remains active.

`tools/build_rdma_early_reduce.py` generates the candidate in a separate source;
`--early-reduce` enables it only in the isolated UMA driver. It is deliberately
incompatible with the current trace insertion generator, whose phase boundaries
assume reduction follows the completion loop. Source hashes and parity evidence
are in `bench/records/2026-09-10-rdma-early-reduce.json`.

The validated candidate was staged at
`/opt/strix-halo-next/rdma-early-reduce-20260910-r1`, identical SHA256 on all
four nodes: `f8e3fddbdad358892e3e90ffed8f1cda00fce8cc0f1a699450191079e744079b`.
## Serving result: not promoted

ShareGPT48/C1, unchanged checkpoint, MTP3 and GPU high mode:

| Configuration | Output tokens/s | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: |
| Earlier original transport control | 56.364 | 545.612 | 15.414 |
| Early reduction candidate | 56.446 | 562.404 | 15.343 |
| Fresh restored original transport | 56.272 | 565.697 | 15.382 |

The candidate is 0.15% above the earlier control and 0.31% above the fresh
restored control. These single runs do not establish a repeatable improvement;
the approximately 4.7% isolated collective gain did not translate into a
comparable serving gain. TTFT is worse than the earlier control and slightly
better than the restored control. No latency improvement is established.

All 48 texts, output lengths and speculative statistics match both controls:
11,839 output tokens and 14,767 input tokens. Auto tool choice, tool-result
roundtrip, streaming tool calls and default thinking passed for both runs.
The benchmark metric includes prefill and is not pure decode speed.

Candidate run `4960cc5a07094fa18bd949bb58484670` was stopped. Restored run
`845e5a91e47946948f446cc13b03f9e2` is ready on all four nodes with the original
uninstrumented UMA transport, GPU high mode and unchanged model weights.
No RDMA trace library is active. The graph inventory preload remains common
to these comparisons. Exported serving records, comparisons and hashed tool
check evidence are in `bench/records/2026-09-10-rdma-early-*.json`.
