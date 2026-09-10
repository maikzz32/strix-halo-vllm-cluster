# AITER module review for the current Qwen cluster

Reviewed upstream main at commit `e8eac17259db6efcb48eba2aa0d9a3bf422e7b35`.
This extends the earlier isolated HC GEMM test; it is not an AITER installation
or a performance result for these additional modules.

Upstream now explicitly lists gfx1151 as experimental. It describes Triton,
most FlyDSL and many HIP operators as RDNA-capable, while most CK/ASM operators
remain CDNA-specific. This is a useful starting point, not proof that every
operator supports our shapes, quantization and MTP state handling.
[Hardware scope](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/README.md).

| Module family | Fit to this cluster | Next decision |
|---|---|---|
| Triton GDN recurrence and convolution fusion | Existing Qwen AMD model uses `QwenGatedDeltaNetAttention`. AITER exposes packed QKV recurrence with accepted-token/state-index arguments. | Highest-priority isolated comparison: outputs **and mutated recurrent states**, accepted-token rollback, graph replay and actual MTP shapes. |
| BF16 RMSNorm and QK norm/RoPE/cache fusion | Avoids requantization, but our QSA uses Gemma norm and 1D/MRoPE composition plus specialized caches. | Compare weight convention, intermediate rounding, position layout and cache writes before adopting a fused path. |
| SiLU/multiply | Shape potentially matches MoE, but AITER explicitly rounds SiLU to input dtype before multiplication. | Check against the current fused operator; do not assume bit equivalence from the same formula. |
| `moe_op_gemm_a16w4` | The reviewed kernel uses MXFP4 `e2m1` and uint8 microscaling with `tl.dot_scaled`. | Not a replacement for the checkpoint's asymmetric integer INT4/GS32 with zero-points. |
| INT4 conversion helpers | `int4_utils.py` explicitly ignores zero-points and assumes symmetric signed INT4. | Do not use for the existing checkpoint. |
| Fused routing from top-k | Produces `matmul_ogs` routing structures; current INT4 kernel consumes block-aligned expert/token arrays. | Requires an adapter and ordering/parity checks, not a direct switch. |
| HC GEMM | Previously tested actual small-M BF16 shapes with an extracted upstream tile. | That tile was slower; retain the existing HC path. This does not reject all AITER GEMMs. |

Relevant pinned sources:

- [GDN wrapper](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/gated_delta_net/fused_rearrange_sigmoid_gdr.py) and [kernel](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_rearrange_sigmoid_gdr.py).
- [QKV/norm/RoPE/cache wrapper](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/rope/fused_qkv_split_qk_norm_rope_cache.py).
- [SiLU intermediate cast](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/_triton_kernels/activation.py).
- [A16W4 arithmetic](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a16w4.py) and [symmetric INT4 helper](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/int4_utils.py).
- [Routing output contract](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/fusions/fused_routing_from_topk.py).

The runtime source snapshot already contains a gfx1151-specific
`VLLM_GFX1X_AITER_TRITON` gate, independent of the master AITER switch, which
feeds the GDN availability check. Its method docstring still says RDNA4, but
the implementation explicitly includes gfx1151. Simply assuming that method
excludes this GPU would be incorrect. Before enabling it, recheck the installed
file and all paths it affects; a global enable could change more than GDN.

Further tracing finds a material limitation: `_forward_core_with_packed_inputs`
dispatches to `_forward_core_decode_aiter` only with interleaved GQA layout,
`spec_sequence_masks is None`, no prefills and at least one decode. Otherwise
it unpacks inputs and calls the existing core. The currently exposed fast path
therefore does not accelerate the speculative verification step simply by
turning on the flag. AITER's accepted-token-capable recurrence wrapper is a
candidate for an explicit integration, not evidence that this integration
already exists. Its single-token conv wrapper also rejects accepted-token
arguments, so it cannot be substituted for MTP convolution unchanged.

Upstream supports `AITER_TRITON_ONLY=1`, which skips top-level CK/HIP imports and
their JIT build. Its compatibility map also retains the old conv1d module name
used by this vLLM snapshot, despite the file moving into `conv/`. Neither a
missing old-path file nor a successful source inspection proves import/runtime
compatibility. A separate bounded import and operator test is the next step.
[Package mode](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/__init__.py),
[module compatibility map](https://github.com/ROCm/aiter/blob/e8eac17259db6efcb48eba2aa0d9a3bf422e7b35/aiter/ops/triton/__init__.py).

No activation quantization, weight conversion or serving switch is authorized
by a module name alone. The existing unchanged-quality requirement still
governs the tests. The separate W2/AVX-512 model trial continues independently;
no AITER GPU work ran concurrently with its throughput measurement.

## Bounded import results

After all model measurements completed, three isolated Node18 probes used a
hash-checked archive of 1,277 pinned Python/config files, without installing a
package. With `AITER_TRITON_ONLY=1`, the old conv module alias and packed GDN
recurrence wrapper both import successfully. Importing `aiter.ops.triton.quant`
fails because the top-level package lacks `dtypes` in this mode.

An isolated attempt to export upstream `utility.dtypes` first failed on its
absolute `build_targets` import. Adding the upstream `jit/utils` path advanced
further but triggered the native `module_aiter_core` build through enum setup;
that failed because the deliberately Python/config-only archive lacks its C++
sources. Thus Triton-only top-level initialization does not guarantee that
arbitrary additional imports avoid native JIT work. No successful full AITER
integration is claimed. Triton's own HIP utility helper also compiled during
import; no explicit model or operator benchmark was launched by these probes.

All three probes terminated and left no owned processes. The serving counter
stayed at 54 with empty queues. Direct GDN import is sufficient to pursue a
bounded operator/state-parity test; making the unrelated quant availability
check pass is not a prerequisite for that experiment.
[Import records](../bench/records/2026-09-07-aiter-imports.json).

Packed GDN follow-up: [isolated recurrence correctness and timing](2026-09-07-aiter-gdn.md). The upstream zero-state sentinel required adaptation; no serving activation yet.
