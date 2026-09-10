"""Isolated W2 split-preserving fusion, derived from pinned installed vLLM source.
See tests/third_party/VLLM-LICENSE. No serving patch.
"""
import triton as _glm53_triton
import triton.language as _glm53_tl
@_glm53_triton.jit
def w2_serial(
    x_ptr,  # [R, K] bf16
    b_ptr,  # uint8 [E, N, K//2], 2 nibbles/byte LSB-first along K
    bs_ptr,  # [E, N, K//GS] (any fp dtype)
    bz_ptr,  # uint8 [E, N//2, K//GS]; byte n//2 holds channel n (low) and n+1
    part_ptr,  # BF16 output [num_valid, N]
    w_ptr,
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
    pid_k = 0

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
            + offs_token[:, None] * N
            + offs_n[None, :],
            zero,
            mask=token_mask[:, None],
        )
        return

    x_row = offs_token // TOPK

    BKW: _glm53_tl.constexpr = BK // 2
    BKG: _glm53_tl.constexpr = BK // GS

    total = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
    for i in _glm53_tl.static_range(SPLITK):
        kb = i * BK
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
        partial = _glm53_tl.dot(a, b, acc=_glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32))
        partial = _glm53_tl.inline_asm_elementwise("v_mov_b32 $0, $1;", constraints="=v,v", args=[partial], dtype=_glm53_tl.float32, is_pure=True, pack=1)
        total = total + partial

    weights = _glm53_tl.load(w_ptr + offs_token, mask=token_mask, other=0).to(_glm53_tl.float32)
    acc = total * weights[:, None]
    _glm53_tl.store(
        part_ptr
        + offs_token[:, None] * N
        + offs_n[None, :],
        acc,
        mask=token_mask[:, None],
    )
