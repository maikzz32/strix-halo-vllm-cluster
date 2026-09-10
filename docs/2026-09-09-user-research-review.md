# Review of proposed optimization directions

The proposed directions are evaluated against the current TP4/MTP3 runtime,
not the older RCCL baseline. Keep the original weights and arithmetic.

## Communication

GPU-mapped registered UMA, a GPU-owned replay epoch and a native CPU RDMA
progress thread are already implemented and enabled. See
[the implementation and measured model comparison](2026-09-07-hip-uma.md).
The complete isolated UMA collective measured 53.93 microseconds, not the
approximately 20-microsecond CPU transport-only figure. Multiplying the latter
by the old collective count omits GPU handshakes, reduction and rank waiting.
The proposed 66 tokens/s is therefore not a prediction for the current runtime.
[Triton-distributed](https://arxiv.org/abs/2504.19442) provides useful overlapping
communication primitives, but does not establish a drop-in gfx1151/Intel-RoCE
speedup here.

## Graph nodes and fusion: measured follow-up

The older observation of 630 CPU-side copy operations does not establish
630 captured HIP memcpy nodes. Conversely, small payloads alone do not prove
that graph dispatch is free. The earlier PERFORMANCE.md conclusion is too
strong for this specific hypothesis.

Count node types and dependencies in the actual target and draft graphs using
HIP graph introspection. Separate copies outside capture from graph nodes.
Then measure bounded replay controls preserving copy sizes, memory types and
dependencies; do not assume a universal 10–20 microseconds per node. A synthetic
copy-only graph is diagnostic, not an additive estimate of model savings.
The current target-sized graph has 2646 nodes: 2558 kernels and 88 D2D copies.
The earlier profiler's 2646 kernel events included copy kernels. Roughly 6–7 ms
without a traced kernel remain unattributed; these intervals cannot all be
assigned to dispatch or copies.

The [direct graph inventory and replay control](2026-09-09-graph-inventory.md)
completed on all four ranks. A serial synthetic graph with the same 88 copy
sizes measured 0.179 ms in total. Its memory/cache behavior differs from the
model, so this is not an additive model-time estimate, but the proposed
630-copy, 6–13 ms calculation is not supported by the actual captured graph.

Use this evidence to choose a small adjacent operator fusion. Preserve BF16
rounding boundaries, reduction order and FP32 recurrent state. Kernel-count
reduction alone cannot justify the suggested 13.5-to-4 ms extrapolation.

The [W1 reduction/SiLU fusion](2026-09-09-w1-reduce-silu.md) subsequently
removed 48 kernel nodes with unchanged tested outputs. ShareGPT48/C1 measured
55.579 / 55.161 / 55.422 tokens/s before / candidate / restored control.
There was no demonstrated serving benefit; the candidate was reverted.
The current trace already contains a fused GDN gating/recurrent-update kernel,
so a new proposal must identify the remaining unfused boundaries explicitly.

## Distributed speculation and changed model assumptions

[DSI](https://arxiv.org/abs/2405.14105) trades extra target instances for latency;
its abstract reports 1.29–1.92x in simulations, not measurements on this cluster.
Its theoretical guarantee cannot be applied without the scheduling and resource
assumptions. [PEARL](https://arxiv.org/abs/2408.11850) and
[PipeSpec](https://arxiv.org/abs/2505.01572) motivate overlap experiments, but
require runner integration and correct recurrent-state rollback.

Four TP1 replicas cannot be assumed for the current approximately 170.55-GiB
checkpoint on 128-GB nodes. The suggested 88.7-GiB group-128 checkpoint is a
different model artifact from the current group-32 weights. Discuss that change
with Maik before testing it. llama.cpp timing does not establish equal TP1 and
TP4 forward latency for this vLLM runtime. Linear catch-up cost and multiplicative
DSI/DFlash gains are hypotheses, not established results.

MTP retraining changes weights and is outside the current experiment constraint.
Tree verification needs branch-specific recurrent state and correct attention;
vocabulary pruning needs a mathematically correct verifier and measured overhead.
No acceptance gain or 3% saving is assumed from either idea.

## Priorities after the measured follow-up

1. Attribute collective time to local GPU work, CPU progress and waiting for
   other ranks. The current profile's approximately 5.7–7.5 ms per-rank UMA
   category includes waiting, not just transport. Do not subtract a CPU-only
   probe from this total or compare unsynchronized timestamps across hosts.
   Optimize the work delaying arrival if rank imbalance dominates.
2. Continue source-guided HC and dense INT4 kernel work with matched controls.
   Smaller LDS allocation and cached-load variants have already failed to
   improve their matched native controls; do not repeat them as new gains.
   Require a serving comparison before promotion, even for exact microkernel
   improvements.
3. Investigate draft/verification overlap as runner design work. Check MTP
   hidden-state dependencies, recurrent-state snapshots, rollback correctness
   and contention on the same four APUs before implementing concurrency.
   PEARL is motivation, not evidence that its reported gains apply to TP4/MTP.
4. Keep four-replica DSI and MTP retraining outside the current deployment:
   the former needs a different memory plan/model artifact and the latter
   changes weights. Discuss any new checkpoint with Maik first.

The latest restored ShareGPT48/C1 control is 55.933 tokens/s with mean TTFT
561.396 ms, using GPU high mode. Reaching 60 at the same output count requires
about 6.78% less benchmark elapsed time (7.27% more throughput). Fixed-prompt
decode rates must not replace the ShareGPT metric when reporting completion.

The [early RDMA reduction trial](2026-09-10-rdma-overlap.md) improved isolated
collective latency by about 4.7%, but serving reached only 56.446 tokens/s,
0.31% above the fresh restored control. It was not promoted. The existing
[dynamic speculation implementation](2026-09-10-dynamic-speculation-review.md)
also is not an acceptance-adaptive MTP switch: its batch-size policy is constant
at C1. The current service already uses V2, so the previously stated V1
full-graph downgrade does not apply; that earlier conclusion was corrected.


## Follow-up: close optimized-path coverage before another depth comparison

The installed W1/W2 guards only select the promoted kernels at MTP3's four
verification rows. MTP2 and MTP4 therefore also change optimized-path coverage.
[Neighbor-depth operator tests](2026-09-10-moe-neighbor-depths.md) now pass exact
checks with two seeds for both operators at three/five rows. W2 improves roughly
23-32%; W1 improves 4-10% for overlap/disjoint routing, with mixed reuse results.
These are operator measurements, not model speedups. Actual dispatch validation
and a matched model trial remain outstanding; production MTP3 is unchanged.

This is a bounded prerequisite for evaluating draft length. It neither implements
PEARL nor proves acceptance-adaptive speculation will improve the service.
The earlier MTP2 run produced different text on 35 of 48 prompts, so it cannot
serve as an identical-output estimate of the cost of one fewer draft step.
