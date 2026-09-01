#!/usr/bin/env python3
"""Runtime patch: split-K skinny int4 dequant GEMV for the gfx1x MoE decode.

GLM-5.3-Flash (compressed-tensors pack-quantized int4, group 32, asymmetric
zero points) routes decode through vLLM's WNA16 Triton MoE backend
(fused_moe_kernel_gptq_awq). Measured on the C=1 decode profile (tp4, eager):
384 us/call average, ~32 ms/step over 84 calls — 4-6x off the memory
roofline. Two structural problems at M<=8:

  1. the packed-weight tile is loaded as [BLOCK_K, BLOCK_N] bytes where
     consecutive N elements sit K//2 bytes apart AND the byte index is
     computed as offs_k//2, which defeats Triton's load vectorizer — the
     effective bandwidth caps at ~55-60 GB/s;
  2. the serial K loop (K/BLOCK_K dependent iterations per program) leaves
     the latency-bound decode grid starved.

This patch injects a two-phase split-K variant and reroutes the wna16 Triton
launch to it when the shape fits (env gate, see below):

  - phase 1 reads the packed weights as [BLOCK_N, BLOCK_K//2] CONTIGUOUS byte
    rows, unpacks nibbles in-register via tl.interleave (LSB-first along K,
    matching the kernel layout), broadcasts group scales/zps, and accumulates
    fp32 partials over a K-split grid (default SPLIT_K=4 for K>=2048, else 2);
  - phase 2 deterministically reduces the partials, applies the routed weight
    (w2 call) and casts once to the output dtype.

Numerics: identical dequant math ((nibble - zp) * scale, fp32 -> bf16 ->
fp32-accum dot, single final bf16 round); the only difference is the K-split
summation order. Standalone A/B (bench/moe_int4_ab.py, production-faithful
two-call flow, E=288/top-8/group-32/asymmetric): max rel diff vs the stock
kernel 6.5e-3..7.8e-3 (bf16 rounding-boundary class; SPLIT_K=1 is bit-exact);
stock kernel itself sits 3.7e-3 off an fp32 reference.

Measured (node1 gfx1151, per-layer w13+w2, E=288, shard intermediate 512):
  M=1: 0.421 -> 0.229 ms (1.84x)   M=6 (MTP5 verify): 2.71 -> 1.78 (1.52x)
  M=16: 5.32 -> 4.01 (1.33x)       M=32: 7.70 -> 6.37 (1.21x)
  w13 call alone at M=1: 58 -> 131 GB/s packed-weight traffic.

v1.1: the partial kernel's M-block size and grid now come from the call's
config (BLOCK_SIZE_M), exactly mirroring the stock launch — a hardcoded
BM=16 misindexed expert_ids (one entry per align block) whenever the default
config picked BM=32/64 (M>20), reading garbage expert ids -> OOB weight
reads -> VM fault at MTP verify C=8 (M=48, BM=64). Found via gate
instrumentation on the production path; standalone tests had always aligned
with BM=16.

Gate: VLLM_GFX1X_MOE_INT4_GEMV=1 on (default), =0 off; resolved lazily per
call (Ray applies worker env after import). Applies only when:
use_int4_w4a16 + asymmetric zp + bf16 compute + uint8 packed weights +
num_valid (= M*top_k) <= 512 + tile divisibility holds; anything else falls
back to the stock kernel launch unchanged. Idempotent, fail-closed
(ast.parse before write).

Target: vllm/model_executor/layers/fused_moe/fused_moe.py
"""

import ast
import io
import sys

P = "/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py"

HELPER = '''
# glm53 moe int4 gemv lane version: v1.3 (gfx1x split-K skinny dequant GEMV)


def _glm53_moe_int4_enabled():
    """Gate for the split-K skinny int4 MoE GEMV: VLLM_GFX1X_MOE_INT4_GEMV,
    default ON. Lazy per call (Ray applies worker env after module import)."""
    import os

    return os.environ.get("VLLM_GFX1X_MOE_INT4_GEMV", "1") == "1"


@_glm53_triton.jit
def _glm53_moe_int4_gemv_partial(
    x_ptr,  # [R, K] bf16
    b_ptr,  # uint8 [E, N, K//2], 2 nibbles/byte LSB-first along K
    bs_ptr,  # [E, N, K//GS] (any fp dtype)
    bz_ptr,  # uint8 [E, N//2, K//GS]; byte n//2 holds channel n (low) and n+1
    part_ptr,  # fp32 [SPLITK, num_valid, N]
    sorted_ids_ptr,
    expert_ids_ptr,
    ntp_ptr,
    num_valid,
    stride_xm,
    N: _glm53_tl.constexpr,
    K: _glm53_tl.constexpr,
    GS: _glm53_tl.constexpr,
    TOPK: _glm53_tl.constexpr,
    BM: _glm53_tl.constexpr,
    BN: _glm53_tl.constexpr,
    BK: _glm53_tl.constexpr,
    SPLITK: _glm53_tl.constexpr,
):
    """Split-K dequant GEMV partials. Same semantics as
    fused_moe_kernel_gptq_awq's K-loop body ((nibble - zp) * scale in fp32,
    bf16 cast, fp32-acc dot), with coalesced packed loads and the K range
    interleaved across SPLITK programs. K % (BK*SPLITK) == 0 and BK % GS == 0
    are guaranteed by the driver."""
    pid_m = _glm53_tl.program_id(0)
    pid_n = _glm53_tl.program_id(1)
    pid_k = _glm53_tl.program_id(2)

    ntp = _glm53_tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return

    offs_tm = pid_m * BM + _glm53_tl.arange(0, BM)
    offs_token = _glm53_tl.load(sorted_ids_ptr + offs_tm).to(_glm53_tl.int64)
    token_mask = offs_token < num_valid
    expert = _glm53_tl.load(expert_ids_ptr + pid_m).to(_glm53_tl.int64)
    offs_n = pid_n * BN + _glm53_tl.arange(0, BN)

    if expert < 0:
        # EP remap miss: parity with the stock kernel's write_zeros_to_output
        # (zero partials -> zero output after the reduce).
        zero = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
        _glm53_tl.store(
            part_ptr
            + pid_k.to(_glm53_tl.int64) * (num_valid * N)
            + offs_token[:, None] * N
            + offs_n[None, :],
            zero,
            mask=token_mask[:, None],
        )
        return

    x_row = offs_token // TOPK

    BKW: _glm53_tl.constexpr = BK // 2
    BKG: _glm53_tl.constexpr = BK // GS

    acc = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
    k_iters = _glm53_tl.cdiv(K, BK * SPLITK)
    for i in range(k_iters):
        kb = pid_k * BK + i * BK * SPLITK  # interleaved splits
        offs_k = kb + _glm53_tl.arange(0, BK)
        a = _glm53_tl.load(
            x_ptr + x_row[:, None] * stride_xm + offs_k[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        # [BN, BK//2]: contiguous byte rows (vectorizable), vs the stock
        # kernel's BN-strided single-byte gather through an unvectorizable
        # offs_k//2 index.
        offs_kw = kb // 2 + _glm53_tl.arange(0, BKW)
        b8 = _glm53_tl.load(
            b_ptr
            + expert * (N * (K // 2))
            + offs_n[:, None] * (K // 2)
            + offs_kw[None, :]
        )
        lo = (b8 & 0xF).to(_glm53_tl.float32)
        hi = (b8 >> 4).to(_glm53_tl.float32)
        nib = _glm53_tl.interleave(lo, hi)  # [BN, BK], LSB-first along K
        offs_g = kb // GS + _glm53_tl.arange(0, BKG)
        scg = _glm53_tl.load(
            bs_ptr
            + expert * (N * (K // GS))
            + offs_n[:, None] * (K // GS)
            + offs_g[None, :]
        ).to(_glm53_tl.float32)
        zg = _glm53_tl.load(
            bz_ptr
            + expert * ((N // 2) * (K // GS))
            + (offs_n[:, None] // 2) * (K // GS)
            + offs_g[None, :]
        )
        zpg = ((zg >> ((offs_n[:, None] % 2) * 4)) & 0xF).to(_glm53_tl.float32)
        sc = _glm53_tl.reshape(
            _glm53_tl.broadcast_to(scg[:, :, None], (BN, BKG, GS)), (BN, BK)
        )
        zp = _glm53_tl.reshape(
            _glm53_tl.broadcast_to(zpg[:, :, None], (BN, BKG, GS)), (BN, BK)
        )
        b = _glm53_tl.trans(((nib - zp) * sc).to(_glm53_tl.bfloat16))
        acc = _glm53_tl.dot(a, b, acc=acc)

    _glm53_tl.store(
        part_ptr
        + pid_k.to(_glm53_tl.int64) * (num_valid * N)
        + offs_token[:, None] * N
        + offs_n[None, :],
        acc,
        mask=token_mask[:, None],
    )


@_glm53_triton.jit
def _glm53_moe_int4_gemv_reduce(
    part_ptr,  # fp32 [SPLITK, num_valid, N]
    c_ptr,  # output [num_valid, N] (flat view of the caller's [M, topk, N])
    w_ptr,  # fp32 [num_valid] routed weights (dummy when not MUL_ROUTED)
    num_valid,
    N: _glm53_tl.constexpr,
    SPLITK: _glm53_tl.constexpr,
    MUL_ROUTED: _glm53_tl.constexpr,
    BLOCK: _glm53_tl.constexpr,
):
    """Deterministic partials reduce + routed weight + single output cast."""
    pid = _glm53_tl.program_id(0)
    num_n = _glm53_tl.cdiv(N, BLOCK)
    row = pid // num_n
    nb = pid % num_n
    if row >= num_valid:
        return
    offs_n = nb * BLOCK + _glm53_tl.arange(0, BLOCK)
    nmask = offs_n < N
    acc = _glm53_tl.zeros((BLOCK,), dtype=_glm53_tl.float32)
    base = row.to(_glm53_tl.int64) * N + offs_n
    for s in _glm53_tl.static_range(SPLITK):
        acc += _glm53_tl.load(
            part_ptr + s * (num_valid * N) + base, mask=nmask, other=0.0
        )
    if MUL_ROUTED:
        acc = acc * _glm53_tl.load(w_ptr + row)
    _glm53_tl.store(c_ptr + base, acc.to(c_ptr.dtype.element_ty), mask=nmask)


def _glm53_moe_int4_gemv(
    A,
    B,
    C,
    B_scale,
    B_zp,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    compute_type,
    use_int4_w4a16,
    use_int8_w8a16,
    block_shape,
    config,
):
    """Split-K skinny int4 dequant GEMV for decode shapes. Returns True when
    the call was handled (C written), False to run the stock kernel."""
    if not _glm53_moe_int4_enabled():
        return False
    if (not use_int4_w4a16) or use_int8_w8a16 or B_zp is None:
        return False
    if compute_type is not _glm53_tl.bfloat16:
        return False
    if B.dtype != torch.uint8 or B_scale.dim() != 3 or B_zp.dim() != 3:
        return False
    M = A.size(0)
    num_valid = M * top_k
    if num_valid > 512:
        return False
    if not A.is_contiguous() or not C.is_contiguous():
        return False

    N, K = B.size(1), A.size(1)
    GS = block_shape[1]
    # The split-K GEMV wins where the align used BM 16/32 (M <= 40 by the
    # default config heuristic); at BM=64 (M>40, e.g. MTP verify at C>=8) the
    # padding-heavy big tiles lose to the stock kernel — leave those alone.
    if config["BLOCK_SIZE_M"] > 32:
        return False
    if K >= 2048:
        SPLITK, BN, BK, num_warps, num_stages = 4, 64, 128, 4, 2
    else:
        SPLITK, BN, BK, num_warps, num_stages = 2, 64, 256, 4, 1
    if K % (BK * SPLITK) or BK % GS or N % BN or N % 2:
        return False

    # CRITICAL parity with the stock kernel: moe_align_block_size padded the
    # token runs to the config's BLOCK_SIZE_M and expert_ids has one entry
    # per such block. The grid MUST use that block size; a hardcoded smaller
    # BM misindexes expert_ids (reads past the written entries -> garbage
    # expert id -> OOB weight read -> VM fault at MTP verify C=8; found via
    # gate instrumentation, fixed in v1.1). v1.3 trims unconditionally:
    # ntp <= num_valid + (BM-1)*min(num_valid, E) <= num_valid*BM always
    # (each active expert holds >=1 valid token), so no valid block is
    # dropped, while no-op tail blocks beyond ntp are never launched (the
    # stock kernel launches and early-exits them; at C=8 the w2 call alone
    # paid ~210 no-op programs per layer).
    BM = config["BLOCK_SIZE_M"]
    EM = min(sorted_token_ids.size(0), num_valid * BM)
    part = torch.empty((SPLITK, num_valid, N), dtype=torch.float32,
                       device=A.device)
    num_pid_m = _glm53_triton.cdiv(EM, BM)
    grid = (num_pid_m, _glm53_triton.cdiv(N, BN), SPLITK)
    _glm53_moe_int4_gemv_partial[grid](
        A,
        B,
        B_scale,
        B_zp,
        part,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        A.stride(0),
        N=N,
        K=K,
        GS=GS,
        TOPK=top_k,
        BM=BM,
        BN=BN,
        BK=BK,
        SPLITK=SPLITK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    tw = (
        topk_weights.reshape(-1)
        if (mul_routed_weight and topk_weights is not None)
        else part
    )
    BLOCK = 512
    _glm53_moe_int4_gemv_reduce[(num_valid * _glm53_triton.cdiv(N, BLOCK),)](
        part,
        C,
        tw,
        num_valid,
        N=N,
        SPLITK=SPLITK,
        MUL_ROUTED=mul_routed_weight,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return True

'''

MARKER = "# glm53 moe int4 gemv lane version: v1.3"

# Injection point: immediately before the wna16 Triton launcher def.
ANCHOR_FN = "def invoke_fused_moe_wna16_triton_kernel(\n"

# Dispatch anchor: the stock kernel launch (must occur exactly once).
ANCHOR_LAUNCH = "    fused_moe_kernel_gptq_awq[grid](\n"

DISPATCH = (
    "    if _glm53_moe_int4_gemv(\n"
    "        A, B, C, B_scale, B_zp, topk_weights, sorted_token_ids,\n"
    "        expert_ids, num_tokens_post_padded, mul_routed_weight, top_k,\n"
    "        compute_type, use_int4_w4a16, use_int8_w8a16, block_shape, config\n"
    "    ):\n"
    "        return\n"
    + ANCHOR_LAUNCH
)

# The helper block needs the module's triton imports under private names.
IMPORT_LINE = "from vllm.triton_utils import tl as _glm53_tl\nfrom vllm.triton_utils import triton as _glm53_triton\n"


def main():
    s = io.open(P, encoding="utf-8").read()

    if ANCHOR_FN not in s:
        print(f"   ERROR: no invoke_fused_moe_wna16_triton_kernel anchor in {P}")
        return 42

    # Replace an older helper block (any version) in place.
    if "# glm53 moe int4 gemv lane version:" in s:
        start = s.find("\n# glm53 moe int4 gemv lane version:")
        end = s.find(ANCHOR_FN, start)
        if start < 0 or end < 0:
            print(f"   ERROR: helper span not found in {P}")
            return 42
        s = s[:start] + HELPER + s[end:]
    else:
        i = s.find(ANCHOR_FN)
        s = s[:i] + IMPORT_LINE + HELPER + s[i:]

    # Normalize the dispatch: remove ANY previously injected dispatch
    # block(s) between the first dispatch line and the stock launch anchor
    # (covers the v1.3 form and the nested v1.3+v1.3 state produced by
    # applying v1.3 over v1.3), then install the current dispatch.
    first = s.find("    if _glm53_moe_int4_gemv(\n")
    if first >= 0:
        launch_at = s.find(ANCHOR_LAUNCH, first)
        if launch_at < 0:
            print(f"   ERROR: dispatch present but launch anchor gone in {P}")
            return 42
        s = s[:first] + s[launch_at:]
    if s.count(ANCHOR_LAUNCH) != 1:
        print(
            f"   ERROR: expected 1 fused_moe_kernel_gptq_awq launch anchor, "
            f"found {s.count(ANCHOR_LAUNCH)}"
        )
        return 42
    s = s.replace(ANCHOR_LAUNCH, DISPATCH)

    ast.parse(s)  # fail closed: never write an unparseable tree
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("   v1.3 injected (split-K skinny int4 MoE GEMV for gfx1x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
