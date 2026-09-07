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
