# Additional gfx1151 projects: source-level applicability

Two additional repositories were reviewed against the current TP4 asymmetric
INT4 GS32/BF16 arithmetic constraint. Source revisions and downloaded-file
SHA256 values are in `bench/records/2026-09-10-community-refresh.json`.
No remote code was executed or installed.

## hec-ovi/vllm-awq4-qwen

Pinned [HIP kernel](https://github.com/hec-ovi/vllm-awq4-qwen/blob/1452a9ff2bac1e422c9c1667dd5a8b4045e6d8b9/csrc/awq_mmq_gfx1151/awq_mmq_gfx1151_kernel.hip)
and [vLLM adapter](https://github.com/hec-ovi/vllm-awq4-qwen/blob/1452a9ff2bac1e422c9c1667dd5a8b4045e6d8b9/csrc/awq_mmq_gfx1151/awq_mmq_gfx1151/vllm_kernel.py)
were read directly. Its MMQ route quantizes activations to INT8, then uses
integer WMMA with FP32 scaling. The adapter accepts BF16 through an FP16 cast;
it is not a drop-in equivalent to the current BF16-activation path. The current
adapter does support explicit asymmetric zero points, so lack of zero-point
support is not the reason to reject it. The activation conversion changes the
computation and fails the retained-arithmetic constraint. No port is proposed
based on its prefill throughput claims.

## KYmidnight/strix-halo-qwen38-flash

The pinned [configuration and report](https://github.com/KYmidnight/strix-halo-qwen38-flash/blob/24e6c3d4762c76dbccb5ecf1df30477ef734392b/README.md)
uses llama.cpp's Vulkan backend with Q4_0_ROCMFP4_FAST, despite the ROCm name in
the weight format. This is a different runtime and model artifact from the
current TP4 vLLM checkpoint. Its configuration/ledger repository does not contain
a direct HIP kernel replacement for our HC or MoE paths. Reported long-context
decode figures cannot be compared directly to ShareGPT48/C1 output throughput.
Its paired-build measurements and production-prompt checks are useful evaluation
practices; the report itself notes that some faster builds fail its production
workload. No kernel command-line or model-format change was applied here.

## Decision

Neither reviewed source provides a demonstrated same-arithmetic acceleration
for this deployment. Existing local argmax reduction, global grid changes and
other negative experiments should not be repeated merely because a different
repository reports a faster unrelated workload. The active MTP3 service remains
at its last validated 56.286 tokens/s control; the 60 tokens/s target is open.
