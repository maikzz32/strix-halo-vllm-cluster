# Dynamo review for four Strix Halo nodes

Reviewed upstream main at `41a0faa0f12f68d63bcf2260e48abcd5c18c5002`.
Dynamo coordinates inference engines rather than replacing their compute
kernels. Its relevant capabilities are KV-aware routing, separate prefill and
decode pools, cache management, scaling and faster replica startup.
[Pinned overview](https://github.com/ai-dynamo/dynamo/blob/41a0faa0f12f68d63bcf2260e48abcd5c18c5002/README.md).

For this user's specific target—at least 60 output tokens/s on ShareGPT48 C1
with the same weights and computational quality—there is no measured Dynamo
speedup. All four nodes already participate in one TP4 engine. Adding a router
does not itself replace the MoE kernels or the per-layer TP all-reduce.

Separate prefill/decode pools can reduce interference when concurrent requests
mix these phases. The documented vLLM flow still waits for synchronous prefill
and then transfers cache data to decode. With one active request, the next token
continues to depend on that request's prior computation. Our existing TP2
measurement is 39.82 tokens/s versus 52.04 for the then-current TP4/UMA engine.
Splitting these four nodes into TP2 prefill and TP2 decode groups therefore has
an unfavorable measured starting point for single-answer speed. This is an
inference from our TP2 test, not a Dynamo benchmark or proof that every possible
disaggregated configuration is slower.
[Transfer sequence and tradeoffs](https://github.com/ai-dynamo/dynamo/blob/41a0faa0f12f68d63bcf2260e48abcd5c18c5002/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md),
[local TP2 evidence](2026-09-07-tp2-comparison.md).

AMD is not categorically excluded: current official documentation describes
AMD GPU support, and release notes include ROCm import fixes. That does not
establish compatibility of this exact gfx1151 runtime, Intel RoCE transport,
hybrid QSA/GDN cache state and MTP3 configuration. Cache transfer through NIXL
is a separate compatibility question from our tested direct UMA TP collective.
[Official platform overview](https://docs.nvidia.com/dynamo/),
[ROCm release fixes](https://github.com/ai-dynamo/dynamo/blob/41a0faa0f12f68d63bcf2260e48abcd5c18c5002/docs/fern/pages/reference/general/releases/dynamo-v1-3-0.mdx).

The feature matrix documents vLLM speculative decoding with Eagle3; that is
not evidence for this checkpoint's MTP3 plus QSA/GDN state transfer. Native KV
offloading separately lists sliding-window and Mamba models validated with
Gemma-2/Falcon-H1, and states that the router sees only full-attention KV groups.
It should not be treated as validation of our model's disaggregated state.
[Feature matrix](https://github.com/ai-dynamo/dynamo/blob/41a0faa0f12f68d63bcf2260e48abcd5c18c5002/docs/fern/pages/reference/general/compatibility.mdx),
[native cache offloading scope](https://github.com/ai-dynamo/dynamo/blob/41a0faa0f12f68d63bcf2260e48abcd5c18c5002/docs/fern/pages/developer-guide/knowledge-base/modular-components/backends/vllm/native-kv-offloading.md).

Decision: retain Dynamo as a candidate for later multiple-client serving,
prefix reuse, replica management and deployment resilience. For the current C1
speed target, prioritize direct kernel/state-path improvements such as the
bounded AITER GDN investigation. No Dynamo package, service or replacement
container was installed; the current model comparison remained isolated from
this source review.
