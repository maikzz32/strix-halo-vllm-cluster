# Qwen3.8 / GLM5.3 and DGX Spark comparison

Research date: 2026-09-06. Web findings are publisher measurements; they are not measurements of this cluster. Remote inspection below used read-only source/config/header reads in `ray-head` on `maik@192.168.1.15`. No remote runtime file or model was changed by this research agent.

## Model identity

The requested models exist. Qwen3.8-Flash-Next was released on 2026-08-26, with a 125B main model, 51B additional N-gram embeddings, 6B active parameters, and native 262,144-token context. Qwen3.8-Flash is the production API name. [Qwen announcement](https://qwen.ai/blog?id=qwen3.8-flash-next), [official model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

GLM-5.3-Flash is a 320B / 18B-active multimodal model with sparse/linear attention. [Official model](https://huggingface.co/zai-org/GLM-5.3-Flash). The official vLLM ROCm recipe currently targets gfx950 and says ROCm MTP is unsupported in that image; it does not establish gfx1151 compatibility. [vLLM recipe, updated 2026-09-05](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).

## DGX measurements for exactly Flash-Next

| Publisher | Configuration | Single stream | Aggregate | Prefill |
|---|---|---:|---:|---:|
| MiaAI-Lab | 2x128GB GB10, ConnectX RoCE/IB, vLLM NVFP4 TP2+EP, MTP3, 262k context | 24.5 without MTP; 52.1 with MTP; 54.4 after FP8 KV + smaller draft vocabulary | 207 t/s at C8 | 2962 t/s, 32,806-token input |
| beastllama | 2xGB10, ConnectX-7 single-rail RoCE, SGLang with SM121 QSA patch, RadixArk NVFP4 TP2, NEXTN 3/1/4 | ~63 t/s for natural 10.7k-token HTML output, thinking off, temperature0.3 | 306.6 t/s at C6 | 3050 t/s, ~7450-token unique prompts, n=6, cache excluded |

[MiaAI-Lab source](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks). [beastllama source](https://github.com/beastllama/dgx-spark-qwen38-flash-next-recipe). The latter's C6 result predates its pinned 8-request/600k-KV default and used 12 requests/850,816 KV tokens. Its difficult forced-output test produced 47.6 t/s. These are prompt- and configuration-dependent results. No exact nominal link rate was found in these benchmark descriptions; do not infer it from NIC branding. Aggregate throughput is not per-request throughput.

## Reduced draft vocabulary: what the DGX patch actually does

[MiaAI-Lab patch source](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/main/files/patch_mtp_draft_vocab.py): after weight loading, copy selected rows of each rank's plain BF16 `lm_head.weight` into a smaller buffer, preserving global token IDs. A draft-only `get_top_tokens` computes the local maximum, then gathers value/ID pairs across TP ranks. `compute_logits` remains full-vocabulary. Empty ranks contribute negative infinity. The shortcut requires greedy drafting and `use_local_argmax_reduction`. It uses ordinary Torch operations, but assumes an unpacked two-dimensional weight. Its fallback preserves normal behavior if that weight is absent. It does not preserve an INT4 quantization scheme automatically.

[Vocabulary builder](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/main/files/build_draft_vocab.py) ranks tokens from generated output, retaining every special/added token. Its optional fill heuristic pads undersized corpora with low IDs and balances padding over TP vocabulary ranges. This is not an empirical frequency ranking for unseen tokens. For our TP4 build, use the actual padded shard boundaries, not assumed TP2 equal ranges.

The publisher's isolated 1k-context/400-output greedy comparison changed prose/code/entropy/copy rates from 40.0/43.2/40.1/59.1 to 42.5/44.5/45.4/56.5 t/s with a balanced65,536 vocabulary. Acceptance fell56.5% to47.8%. The corpus had49,141 token occurrences and only8305 distinct IDs, so most of the65,536 entries were padded. This is an acceptance/bandwidth tradeoff, not a demonstrated large universal gain. [Measured comparison](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks#reduced-vocabulary-mtp-drafting).

The publisher's [CPU test](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/blob/main/files/test_draft_vocab.py) checks shard-row mapping, cross-rank ID selection and an empty rank against restricted full-vocabulary argmax. It does not prove end-to-end numerical identity or stochastic sampling correctness by itself.

## What is already installed in ray-head

Root: `/usr/local/lib64/python3.12/site-packages/vllm`.

| Item | Read-only finding |
|---|---|
| `models/qwen4_exp/amd/mtp.py:368` | Actual model is `Qwen4ExpMTP`; the DGX path/class names differ. |
| Same file:404 | Existing gfx1151 patch69 passes quant_config into the MTP `ParallelLMHead`. |
| Same file:449 | `compute_logits` calls `self.logits_processor(self.lm_head, hidden_states)`. |
| Same file:468 | Loader returns `loader.load_weights(remap_weight_names(), mapper=mapper)`, differing from MiaAI's exact string anchor. |
| `model_executor/layers/logits_processor.py:201` | Existing `get_top_tokens` performs quant-aware local argmax, masks padding, translates rank-local IDs, and gathers FP32 value/ID pairs. |
| Same file:136 | `_apply_head` retains `lm_head.quant_method.apply`; this supports the current packed INT4 path. |
| `v1/spec_decode/llm_base_proposer.py:438` | Flag dispatches only draft sampling to `self.model.get_top_tokens`. |
| Same file:1629 | Startup rejects the flag if the draft model lacks that method. |
| `v1/worker/gpu/spec_decode/speculator.py:339` | Alternate speculator validates the same draft hook and rejects probabilistic draft sampling with the flag. |
| `config/speculative.py:439` | Flag exists, default false; documented for greedy non-tree drafting. |
| Same file:582 | Draft sampling defaults to greedy; standard rejection treats its distribution as one-hot. |

Whole-package text search found no `VLLM_MTP_DRAFT_VOCAB` or `_attach_draft_vocab`. `Qwen4ExpMTP` has no `get_top_tokens` before our proposed patch. Therefore merely setting the existing flag fails. The flag is not used to replace target-model sampling; the target needs no new method.

The inspected process served `/home/maik/qwen38_rest` with TP4/4 nodes, FULL_DECODE_ONLY, MTP3, 262144 context, async scheduling,8192 batched tokens,8 sequences. No local-argmax flag was supplied.

### The important quantization difference

The local config says248,320 vocabulary rows, hidden width2560, untied embeddings. Its compressed-tensors group targets both Linear and lm_head with asymmetric INT4, group32. The checkpoint header confirms:

| Tensor | Stored dtype and shape |
|---|---|
| `lm_head.weight_packed` | I32 `[248320,320]` |
| `lm_head.weight_scale` | BF16 `[248320,80]` |
| `lm_head.weight_zero_point` | I32 `[31040,80]` |
| `mtp.fc_embedding.weight` | BF16 `[2560,2560]` |
| `mtp.fc_hidden.weight` | BF16 `[2560,2560]` |

Consequently MiaAI's BF16-head bandwidth argument does not transfer unchanged. Arithmetic from these tensor shapes: packed weights+scales+zero points total~350.49MiB globally,~87.62MiB/rank at TP4 before any runtime kernel repacking. A65,536-row BF16 subset would itself be80MiB/rank with balancedTP4, almost as much as the entire current INT4 head. Reconstructing a BF16 subset can lose the existing optimization. An eventual shortlist should retain quantized row/scales/zero-point semantics and respect the runtime kernel's repacked layout; raw on-disk row slicing alone is insufficient.

MTP fc_embedding/fc_hidden are much smaller dense matrices than this head and remain BF16. Profiling is needed before changing them; their two `ColumnParallelLinear(gather_output=True)` calls also communicate. This report does not claim their share of actual runtime.

## Recommended immediate change

Add only this method to the AMD MTP class and enable the existing speculative flag:

```python
def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
```

```json
{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}
```

This retains the full vocabulary and existing INT4 matrix multiplication, while replacing draft-logit communication with small value/ID pairs. At TP4 with C1, the logical reduction payload is8 FP32 values after gathering versus248,320 logits; actual NCCL/RCCL wire traffic depends on the chosen collective. It does not reduce weight bandwidth, target logits, or target sampling. Existing positive scaling, softcap and padding handling remain in vLLM.

`patches/mtp_local_argmax.py` is a local, reversible, opt-in script. Default is dry-run; `--apply` makes a SHA-tagged original backup first; `--restore --apply` restores only when the backup and exact patch still match. AST checks reject a changed compute_logits anchor or a conflicting method. Root owns remote application and restart.

Before-change SHA256s:

```
models/qwen4_exp/amd/mtp.py
abdaeadf54ddbd8c8bf7f10321fdcac48792cb51e7fa45a1a6c0460f0c7244b7
model_executor/layers/logits_processor.py
6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f
v1/spec_decode/llm_base_proposer.py
7b404e8de2a0068510cbda29d3b36c16fb7515a5e5260ab26be9f381692d3c91
```

## Correctness and validation

Full-vocabulary local argmax should select the same draft ID as gathered-logits argmax for the current head: identical quant method, global ID offsets, padding masked, same scale/softcap, lowest-rank/lowest-local-ID tie order. FP32 exactly represents token IDs below248,320. No vocabulary restriction or approximate acceptance is introduced.

Actual validation completed:

- Local patch self-test passed: changed-anchor refusal, dry-run, idempotence, backup creation and exact restore.
- `tests/test_mtp_local_argmax.py` executed the installed LogitsProcessor methods and the proposed in-memory MTP delegation using CPU only, one thread, with no model loading, GPU allocation, serving requests or installed-file mutation.
-30 cases passed: TP1/2/4, FP32/BF16, real248,320-token global indexing, non-divisible/padded vocab, padding that would otherwise win, cross-rank ties, positive scaling, softcap, and quant-method dispatch. A fake quant method deliberately provides no `.weight` to catch bypasses.
- This test simulates collectives and uses CPU-decoded quantized values. It does not validate live RCCL, actual gfx1151 INT4 kernels, HIP graph capture or throughput. Root's controlled restart/A-B benchmark remains necessary.

## Vocabulary recommendation after the immediate test

Keep the full vocabulary as the default. If live profiling still makes the MTP head a meaningful bottleneck, test a quantization-preserving64k shortlist as an experimental variant. Build it from representative model-generated German/English prose, code, JSON/tool output, quoted text and rarer scripts; keep special tokens; report held-out coverage and per-rank counts. Compare full/64k/32k on identical prompts, context, natural output, TTFT, decode rate, acceptance and correctness. Do not label coverage on the training corpus as a quality evaluation. Do not adopt a low-ID padded English-centric shortlist as a general German-language default.

For standard exact rejection sampling, arbitrary greedy draft proposals can preserve the target distribution because the verifier corrects them; an excluded desired token reduces acceptance rather than excluding target output. That statement assumes the verifier remains exact, target logits stay full-vocabulary, mapping is correct, and no synthetic acceptance is enabled. It does not promise bit-identical generated text after changing floating-point execution or RNG-consumption order. The immediate full-vocabulary patch avoids that additional shortlist tradeoff entirely.
