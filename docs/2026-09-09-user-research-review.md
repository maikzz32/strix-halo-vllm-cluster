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

## Graph nodes and fusion: next diagnostic priority

The older observation of 630 CPU-side copy operations does not establish
630 captured HIP memcpy nodes. Conversely, small payloads alone do not prove
that graph dispatch is free. The earlier PERFORMANCE.md conclusion is too
strong for this specific hypothesis.

Count node types and dependencies in the actual target and draft graphs using
HIP graph introspection. Separate copies outside capture from graph nodes.
Then measure bounded replay controls preserving copy sizes, memory types and
dependencies; do not assume a universal 10–20 microseconds per node. A synthetic
copy-only graph is diagnostic, not an additive estimate of model savings.
The current target has 2646 kernels and roughly 6–7 ms without a traced kernel;
those intervals are not yet attributed to dispatch or copies.

Use this evidence to choose a small adjacent operator fusion. Preserve BF16
rounding boundaries, reduction order and FP32 recurrent state. Kernel-count
reduction alone cannot justify the suggested 13.5-to-4 ms extrapolation.

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
