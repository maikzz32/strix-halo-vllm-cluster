# TP4 latency research: source review and test decisions

Target: Qwen3.8 Flash Next, existing asymmetric INT4 GS32 checkpoint, four
gfx1151 APUs, one running answer. Reviewed September 9, 2026. Changing model
weights or quantization requires a separate discussion with the operator.

Repository heads retrieved during this review are recorded in
[the source manifest](../bench/records/2026-09-09-research-sources.json).

## Community projects

| Source | Transferable idea | Decision for this cluster |
|---|---|---|
| [kyuz0 / amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) | gfx1151 runtime patches and RDMA deployment | Existing foundation; compare source revisions before importing additional patches. |
| [ayysasha / Strix-halo-dual-optimized](https://github.com/ayysasha/Strix-halo-dual-optimized) | Verify device-name lookup and tune actual small-batch MoE shapes | Its MiniMax E256/N768 tuning is not our E512/N160 configuration. Our custom W1/W2 path chooses its own tiles; copying the JSON would not tune it. |
| [leonyurko / vllm-fp8-strix-halo-kernel-support](https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support) | Specialized kernels for hardware without native FP8 matrix operations | Our target already uses INT4 W4A16. FP8 emulation comparisons do not establish a gain here; retain weights and reduction arithmetic. |
| [MiaAI-Lab / Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | Model-specific MTP and QSA investigation, reproducible launch configuration | GB10/NVFP4 and concurrency differ. Previously reviewed vocabulary slicing is less attractive for our already quantized draft head. No directly comparable speedup established. |
| [YanissAmz / strix-halo-llm-serving](https://github.com/YanissAmz/strix-halo-llm-serving) | Paired comparisons, context-depth measurements, small-batch kernel thresholds | Apply the measurement method. Its Qwen Flash Next MTP result is explicitly a single-prompt llama.cpp result, not a TP4 vLLM comparison. |
| [bitserv-ai / _gfx115x_](https://github.com/bitserv-ai/_gfx115x_) | Reproducible source builds for Zen5/gfx115x | Useful build reference. Compiler flags alone do not prove speedup in already compiled GPU kernels. |

## Papers and their applicability

[SwiftSpec](https://arxiv.org/abs/2506.11309) separates draft and target work,
uses asynchronous tree generation and latency-oriented kernels. Its
[artifact](https://github.com/ByteDance-Seed/SwiftSpec) targets CUDA/H800 and
other model families. The useful hypothesis is reducing draft work on the
critical path, not importing its throughput number. A Flash Next port would
need correct hybrid recurrent-state rollback and an appropriate drafter;
the current native MTP path cannot simply enable tree generation.

[Speculative Decoding in Decentralized LLM Inference](https://arxiv.org/abs/2511.11733)
motivates measuring speculation together with network latency. Here the
decision variable is accepted output tokens per complete iteration, including
drafts, target verification and collectives. Increasing draft depth without
measuring these costs is not an optimization.

[SP-MoE](https://arxiv.org/abs/2510.10302) combines speculation with expert
prefetch and offload scheduling. Our TP4 weights already reside in unified
memory; adding CPU/GPU offload would introduce a different bottleneck.
Expert locality remains useful for kernel experiments, but its reported
offload gains are not a forecast for this system.

[MoE-Spec](https://arxiv.org/abs/2602.16052) limits verification expert work by
dropping low-contribution experts. This changes target computation and does
not satisfy the retained-computation requirement. It is not a candidate for
the current deployment.

## Current engineering experiments

1. **MoE output alias:** preserve every arithmetic operation while writing the
   final expert sum directly into the caller's output. The installed ROCm
   guard otherwise permits this only for AITER. The candidate is restricted
   to the actual WNA16 TP4 shapes, BF16, small batches, no LoRA and NoDPEP
   finalize. Eager and changed-input graph tests pass, but the first model
   comparison shows only +0.064% throughput and no decode improvement.
   [Experiment report](2026-09-09-moe-output-alias.md). This targets dispatch
   overhead, not memory capacity, and is not promoted.
2. **Lossless W1 tile packing:** earlier isolated tests showed only a small
   advantage and require extra resident weights for fallback. Do not claim
   a model gain before load-time integration and bracketed measurements.
3. **Refresh the critical-path profile:** the older profile identifies MoE,
   hyper-connections and collective waits, but predates the latest UMA/W1/W2
   improvements. Use the current runtime for the next substantial kernel
   decision. Profiled timings must not be presented as unprofiled throughput.
4. **Prefill and context depth:** measure first-use and reused 2K/8K/32K
   prefixes using exact token IDs and cache counters. The current runtime
   already enables prefix caching. Use the observed prefill shapes to decide
   whether its WNA16 fallback needs separate tuning; the small-batch decode
   kernel is not the prefill path. A faster cached request alone does not
   prove a kernel improvement or cache correctness.

Known negative results, including TP2, EP4, several W1/W2 tile variants,
HC packing and CPU thread placement, remain in the individual experiment
reports. UCCL's SRQ requirement is incompatible with the measured `max_srq=0`
on these NICs in the tested implementation; see the existing UCCL report.
