# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from the pinned installed W1 partial kernel; arithmetic unchanged.
import triton as _glm53_triton
import triton.language as _glm53_tl


@_glm53_triton.jit
def w1_expert_kfast(x_ptr, b_ptr, bs_ptr, bz_ptr, part_ptr, sorted_ids_ptr, expert_ids_ptr, ntp_ptr, num_valid, stride_xm, N: _glm53_tl.constexpr, K: _glm53_tl.constexpr, GS: _glm53_tl.constexpr, TOPK: _glm53_tl.constexpr, BM: _glm53_tl.constexpr, BN: _glm53_tl.constexpr, BK: _glm53_tl.constexpr, SPLITK: _glm53_tl.constexpr):
    """Split-K dequant GEMV partials. Same semantics as
    fused_moe_kernel_gptq_awq's K-loop body ((nibble - zp) * scale in fp32,
    bf16 cast, fp32-acc dot), with coalesced packed loads and the K range
    interleaved across SPLITK programs. K % (BK*SPLITK) == 0 and BK % GS == 0
    are guaranteed by the driver."""
    linear = _glm53_tl.program_id(0) + _glm53_tl.num_programs(0) * (_glm53_tl.program_id(1) + _glm53_tl.num_programs(1) * _glm53_tl.program_id(2))
    pid_m = linear // (N // BN * SPLITK)
    pid_n = linear // SPLITK % (N // BN)
    pid_k = linear % SPLITK
    ntp = _glm53_tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return
    offs_tm = pid_m * BM + _glm53_tl.arange(0, BM)
    offs_token = _glm53_tl.load(sorted_ids_ptr + offs_tm).to(_glm53_tl.int64)
    token_mask = offs_token < num_valid
    expert = _glm53_tl.load(expert_ids_ptr + pid_m).to(_glm53_tl.int64)
    offs_n = pid_n * BN + _glm53_tl.arange(0, BN)
    if expert < 0:
        zero = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
        _glm53_tl.store(part_ptr + pid_k.to(_glm53_tl.int64) * (num_valid * N) + offs_token[:, None] * N + offs_n[None, :], zero, mask=token_mask[:, None])
        return
    x_row = offs_token // TOPK
    BKW: _glm53_tl.constexpr = BK // 2
    BKG: _glm53_tl.constexpr = BK // GS
    acc = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
    k_iters = _glm53_tl.cdiv(K, BK * SPLITK)
    for i in range(k_iters):
        kb = pid_k * BK + i * BK * SPLITK
        offs_k = kb + _glm53_tl.arange(0, BK)
        a = _glm53_tl.load(x_ptr + x_row[:, None] * stride_xm + offs_k[None, :], mask=token_mask[:, None], other=0.0)
        offs_kw = kb // 2 + _glm53_tl.arange(0, BKW)
        b8 = _glm53_tl.load(b_ptr + expert * (N * (K // 2)) + offs_n[:, None] * (K // 2) + offs_kw[None, :])
        lo = (b8 & 15).to(_glm53_tl.float32)
        hi = (b8 >> 4).to(_glm53_tl.float32)
        nib = _glm53_tl.interleave(lo, hi)
        offs_g = kb // GS + _glm53_tl.arange(0, BKG)
        scg = _glm53_tl.load(bs_ptr + expert * (N * (K // GS)) + offs_n[:, None] * (K // GS) + offs_g[None, :]).to(_glm53_tl.float32)
        zg = _glm53_tl.load(bz_ptr + expert * (N // 2 * (K // GS)) + offs_n[:, None] // 2 * (K // GS) + offs_g[None, :])
        zpg = (zg >> offs_n[:, None] % 2 * 4 & 15).to(_glm53_tl.float32)
        sc = _glm53_tl.reshape(_glm53_tl.broadcast_to(scg[:, :, None], (BN, BKG, GS)), (BN, BK))
        zp = _glm53_tl.reshape(_glm53_tl.broadcast_to(zpg[:, :, None], (BN, BKG, GS)), (BN, BK))
        b = _glm53_tl.trans(((nib - zp) * sc).to(_glm53_tl.bfloat16))
        acc = _glm53_tl.dot(a, b, acc=acc)
    _glm53_tl.store(part_ptr + pid_k.to(_glm53_tl.int64) * (num_valid * N) + offs_token[:, None] * N + offs_n[None, :], acc, mask=token_mask[:, None])

@_glm53_triton.jit
def w1_expert_nfast(x_ptr, b_ptr, bs_ptr, bz_ptr, part_ptr, sorted_ids_ptr, expert_ids_ptr, ntp_ptr, num_valid, stride_xm, N: _glm53_tl.constexpr, K: _glm53_tl.constexpr, GS: _glm53_tl.constexpr, TOPK: _glm53_tl.constexpr, BM: _glm53_tl.constexpr, BN: _glm53_tl.constexpr, BK: _glm53_tl.constexpr, SPLITK: _glm53_tl.constexpr):
    """Split-K dequant GEMV partials. Same semantics as
    fused_moe_kernel_gptq_awq's K-loop body ((nibble - zp) * scale in fp32,
    bf16 cast, fp32-acc dot), with coalesced packed loads and the K range
    interleaved across SPLITK programs. K % (BK*SPLITK) == 0 and BK % GS == 0
    are guaranteed by the driver."""
    linear = _glm53_tl.program_id(0) + _glm53_tl.num_programs(0) * (_glm53_tl.program_id(1) + _glm53_tl.num_programs(1) * _glm53_tl.program_id(2))
    pid_m = linear // (N // BN * SPLITK)
    pid_n = linear % (N // BN)
    pid_k = linear // (N // BN) % SPLITK
    ntp = _glm53_tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return
    offs_tm = pid_m * BM + _glm53_tl.arange(0, BM)
    offs_token = _glm53_tl.load(sorted_ids_ptr + offs_tm).to(_glm53_tl.int64)
    token_mask = offs_token < num_valid
    expert = _glm53_tl.load(expert_ids_ptr + pid_m).to(_glm53_tl.int64)
    offs_n = pid_n * BN + _glm53_tl.arange(0, BN)
    if expert < 0:
        zero = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
        _glm53_tl.store(part_ptr + pid_k.to(_glm53_tl.int64) * (num_valid * N) + offs_token[:, None] * N + offs_n[None, :], zero, mask=token_mask[:, None])
        return
    x_row = offs_token // TOPK
    BKW: _glm53_tl.constexpr = BK // 2
    BKG: _glm53_tl.constexpr = BK // GS
    acc = _glm53_tl.zeros((BM, BN), dtype=_glm53_tl.float32)
    k_iters = _glm53_tl.cdiv(K, BK * SPLITK)
    for i in range(k_iters):
        kb = pid_k * BK + i * BK * SPLITK
        offs_k = kb + _glm53_tl.arange(0, BK)
        a = _glm53_tl.load(x_ptr + x_row[:, None] * stride_xm + offs_k[None, :], mask=token_mask[:, None], other=0.0)
        offs_kw = kb // 2 + _glm53_tl.arange(0, BKW)
        b8 = _glm53_tl.load(b_ptr + expert * (N * (K // 2)) + offs_n[:, None] * (K // 2) + offs_kw[None, :])
        lo = (b8 & 15).to(_glm53_tl.float32)
        hi = (b8 >> 4).to(_glm53_tl.float32)
        nib = _glm53_tl.interleave(lo, hi)
        offs_g = kb // GS + _glm53_tl.arange(0, BKG)
        scg = _glm53_tl.load(bs_ptr + expert * (N * (K // GS)) + offs_n[:, None] * (K // GS) + offs_g[None, :]).to(_glm53_tl.float32)
        zg = _glm53_tl.load(bz_ptr + expert * (N // 2 * (K // GS)) + offs_n[:, None] // 2 * (K // GS) + offs_g[None, :])
        zpg = (zg >> offs_n[:, None] % 2 * 4 & 15).to(_glm53_tl.float32)
        sc = _glm53_tl.reshape(_glm53_tl.broadcast_to(scg[:, :, None], (BN, BKG, GS)), (BN, BK))
        zp = _glm53_tl.reshape(_glm53_tl.broadcast_to(zpg[:, :, None], (BN, BKG, GS)), (BN, BK))
        b = _glm53_tl.trans(((nib - zp) * sc).to(_glm53_tl.bfloat16))
        acc = _glm53_tl.dot(a, b, acc=acc)
    _glm53_tl.store(part_ptr + pid_k.to(_glm53_tl.int64) * (num_valid * N) + offs_token[:, None] * N + offs_n[None, :], acc, mask=token_mask[:, None])
