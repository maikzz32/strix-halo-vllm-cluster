# TP2 versus TP4: the same single-request benchmark

At the user's request, the existing Qwen3.8-Flash-Next checkpoint was served
on nodes 15/16 with TP2. All weights, quantization settings, QSA, MTP3,
262144-token context limit, batching limits and 0.85 memory utilization were
retained. The native controller derived TP2 and two ranks from the node list.
The opt-in UMA path was explicitly disabled because it is qualified for TP4;
TP2 used the installed RCCL implementation.

The same ShareGPT seed 42 benchmark issues 48 requests, one at a time:

| Configuration | Output tokens/s | Mean TPOT (ms) | Mean TTFT (ms) | MTP acceptance |
|---|---:|---:|---:|---:|
| TP2 / RCCL, new test | **39.8221** | 21.9810 | 834.371 | 53.857% |
| TP4 / RCCL, earlier control | 50.4409 | 17.6429 | 547.795 | 54.143% |
| TP4 / UMA, earlier validated run | **52.0387** | **17.0053** | 552.796 | 54.143% |

Relative to TP2, TP4/UMA provides **30.68% greater overall output throughput**
and **29.26% faster decoding**, inferred from mean TPOT. Even the TP4/RCCL
control is 26.67% faster overall and 24.59% faster in decoding. Doubling the
node count therefore does not double single-answer performance, but these
settings clearly favor TP4 for the user's single-answer speed objective.

Every run completed all 48 requests without failures. Input token counts and
all per-request input lengths match (14767 total). TP2 generated 11894 tokens
versus 11839 with TP4; 47 of 48 output lengths match, and 11 of 48 complete texts
match. The differing outputs and slightly different speculative acceptance
mean this is a practical same-request comparison, not bit-identical execution
or a general quality evaluation. No lossy weight conversion was introduced.

The six fixed 512-token stream requests also run slower on TP2, by 18.8-25.4%
against the matched TP4/UMA recordings, median 21.15%. Their texts differ
between TP counts, but each TP2 prompt produces the same text on its repeat.
The first TP2 coding request has a 2.482s TTFT; its repeat has 0.463s. Both are
retained. These short workloads must not be compared directly with the 48-
request output-throughput numbers above.

TP2 loaded successfully without lowering the context limit. Startup reported
7.02GiB available KV-cache memory and capacity for 470442 tokens (1.79 times
the configured maximum context per request). It ran as
`1c271addf47e43cf8f845b5e3e03c23a`. The controller stopped both owned ranks
cleanly after the test. The canonical four-node UMA configuration was retained
throughout and was restored successfully.

The [TP2 configuration](../bench/records/tp2-compare-20260907.json) is archived
for the preserved installation. Copy the same file to all participating hosts
before passing its absolute path to the native controller. Stop the existing
four-rank service before changing the node list; stopping only a two-node
configuration would leave the other ranks behind. The standard host config
remains `/home/cluster-user/strix-halo-next/config/cluster.json`.

[Raw metrics and response hashes](../bench/records/2026-09-07-tp2-comparison.json).
Full answer texts remain in the local benchmark artifacts.

## Recovery verification

The canonical TP4/UMA run `4905dc1980324c08842710c58fb900af` reached READY
and remains active. All six matched 512-token outputs and token counts are
identical to the previous TP4/UMA run. Median paired decode speed differs by
-0.0272%. Five requests match or exceed the prior speed closely; the second
German request is slower (47.912 versus 59.187 tokens/s), with multiple SSE
gaps and a maximum 221.928ms gap. Its result is retained, and the cause is
unassigned. This verification confirms restoration; it is not a new 48-request
throughput measurement or proof of uniform latency.
