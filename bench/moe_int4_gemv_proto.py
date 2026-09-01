#!/usr/bin/env python3
"""Prototype: split-K skinny int4 dequant MoE GEMV for gfx1151 (M<=8 decode).

Same I/O contract as vLLM's fused_moe_kernel_gptq_awq (uint8 N-first packed
weights [E, N, K//2] LSB-first along K, scales bf16 [E, N, K//GS], zero points
uint8 [E, N//2, K//GS] with byte n//2 holding channel n (low) and n+1 (high)),
but the K loop is split across programs (fp32 partials + deterministic
reduce) so the serial K chain no longer limits occupancy at M=1..8.

Phase 2 applies the routed weight and casts to the output dtype, mirroring
invoke_fused_moe_wna16_triton_kernel semantics (single final bf16 round).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_int4_gemv_partial(
    x_ptr,  # [M, K] bf16
    b_ptr,  # uint8 [E, N, K//2]
    bs_ptr,  # bf16 [E, N, K//GS]
    bz_ptr,  # uint8 [E, N//2, K//GS]
    part_ptr,  # fp32 [SPLITK, M*TOPK, N]
    sorted_ids_ptr,
    expert_ids_ptr,
    ntp_ptr,
    num_valid,
    stride_xm,
    N: tl.constexpr,
    K: tl.constexpr,
    GS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITK: tl.constexpr,
    BTRANS: tl.constexpr,  # load B as [BN, BK] contiguous rows + transpose
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    ntp = tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return

    offs_tm = pid_m * BM + tl.arange(0, BM)
    offs_token = tl.load(sorted_ids_ptr + offs_tm).to(tl.int64)
    token_mask = offs_token < num_valid
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    x_row = offs_token // TOPK

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_iters = tl.cdiv(K, BK * SPLITK)
    for i in range(k_iters):
        kb = pid_k * BK + i * BK * SPLITK  # interleaved splits
        offs_k = kb + tl.arange(0, BK)
        kmask = offs_k < K
        a = tl.load(
            x_ptr + x_row[:, None] * stride_xm + offs_k[None, :],
            mask=token_mask[:, None] & kmask[None, :],
            other=0.0,
        )
        if BTRANS:
            # [BN, BK]: each row is BK/2 CONTIGUOUS bytes (coalesced); the
            # vLLM layout [BK, BN] gathers BN bytes strided K//2 apart.
            bq = tl.load(
                b_ptr
                + expert * (N * (K // 2))
                + offs_n[:, None] * (K // 2)
                + (offs_k[None, :] // 2),
                mask=kmask[None, :],
                other=0,
            )
            nib = ((bq >> ((offs_k[None, :] % 2) * 4)) & 0xF).to(tl.float32)
            sc = tl.load(
                bs_ptr
                + expert * (N * (K // GS))
                + offs_n[:, None] * (K // GS)
                + (offs_k[None, :] // GS),
                mask=kmask[None, :],
                other=0.0,
            ).to(tl.float32)
            zp = tl.load(
                bz_ptr
                + expert * ((N // 2) * (K // GS))
                + (offs_n[:, None] // 2) * (K // GS)
                + (offs_k[None, :] // GS),
                mask=kmask[None, :],
                other=0,
            )
            zpn = ((zp >> ((offs_n[:, None] % 2) * 4)) & 0xF).to(tl.float32)
            b = tl.trans(((nib - zpn) * sc).to(tl.bfloat16))  # [BK, BN]
        else:
            bq = tl.load(
                b_ptr
                + expert * (N * (K // 2))
                + (offs_k[:, None] // 2)
                + offs_n[None, :] * (K // 2),
                mask=kmask[:, None],
                other=0,
            )
            nib = ((bq >> ((offs_k[:, None] % 2) * 4)) & 0xF).to(tl.float32)
            sc = tl.load(
                bs_ptr
                + expert * (N * (K // GS))
                + offs_n[None, :] * (K // GS)
                + (offs_k[:, None] // GS),
                mask=kmask[:, None],
                other=0.0,
            ).to(tl.float32)
            zp = tl.load(
                bz_ptr
                + expert * ((N // 2) * (K // GS))
                + (offs_n[None, :] // 2) * (K // GS)
                + (offs_k[:, None] // GS),
                mask=kmask[:, None],
                other=0,
            )
            zpn = ((zp >> ((offs_n[None, :] % 2) * 4)) & 0xF).to(tl.float32)
            b = ((nib - zpn) * sc).to(tl.bfloat16)
        acc = tl.dot(a, b, acc=acc)

    tl.store(
        part_ptr
        + pid_k.to(tl.int64) * (num_valid * N)
        + offs_token[:, None] * N
        + offs_n[None, :],
        acc,
        mask=token_mask[:, None],
    )


@triton.jit
def _moe_int4_gemv_reduce(
    part_ptr,  # fp32 [SPLITK, M*TOPK, N]
    c_ptr,  # bf16 [M, TOPK, N]
    w_ptr,  # fp32 [M, TOPK] routed weights or dummy
    num_valid,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITK: tl.constexpr,
    MUL_ROUTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK)
    row = pid // num_n
    nb = pid % num_n
    if row >= num_valid:
        return
    offs_n = nb * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLITK):
        acc += tl.load(part_ptr + s * (num_valid * N) + row * N + offs_n)
    if MUL_ROUTED:
        w = tl.load(w_ptr + row)
        acc = acc * w
    tl.store(
        c_ptr + row.to(tl.int64) * N + offs_n,
        acc.to(c_ptr.dtype.element_ty),
    )


@triton.jit
def _moe_int4_gemv_partial_v2(
    x_ptr,  # [R, K] bf16
    b_ptr,  # uint8 [E, N, K//2]
    bs_ptr,  # bf16 [E, N, K//GS]
    bz_ptr,  # uint8 [E, N//2, K//GS]
    part_ptr,  # fp32 [SPLITK, num_valid, N]
    sorted_ids_ptr,
    expert_ids_ptr,
    ntp_ptr,
    num_valid,
    stride_xm,
    N: tl.constexpr,
    K: tl.constexpr,
    GS: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITK: tl.constexpr,
):
    """v2: coalesced loads. The packed B tile is read as [BN, BK//2]
    contiguous byte rows (the [BK, BN] nibble layout of the vLLM kernel
    gathers BN bytes strided K//2 apart, which also defeats Triton's
    vectorizer through the //2). Nibbles are split in-register via
    tl.interleave (LSB-first along K). Scales/zp are loaded once per
    GS-group and broadcast. Requires K % (BK*SPLITK) == 0 and BK % GS == 0
    (the driver guarantees both or falls back to the vLLM kernel)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    ntp = tl.load(ntp_ptr)
    if pid_m * BM >= ntp:
        return

    offs_tm = pid_m * BM + tl.arange(0, BM)
    offs_token = tl.load(sorted_ids_ptr + offs_tm).to(tl.int64)
    token_mask = offs_token < num_valid
    expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    x_row = offs_token // TOPK

    BKW: tl.constexpr = BK // 2
    BKG: tl.constexpr = BK // GS

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_iters = tl.cdiv(K, BK * SPLITK)
    for i in range(k_iters):
        kb = pid_k * BK + i * BK * SPLITK  # interleaved splits
        offs_k = kb + tl.arange(0, BK)
        a = tl.load(
            x_ptr + x_row[:, None] * stride_xm + offs_k[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        offs_kw = kb // 2 + tl.arange(0, BKW)
        b8 = tl.load(
            b_ptr
            + expert * (N * (K // 2))
            + offs_n[:, None] * (K // 2)
            + offs_kw[None, :]
        )
        lo = (b8 & 0xF).to(tl.float32)
        hi = (b8 >> 4).to(tl.float32)
        nib = tl.interleave(lo, hi)  # [BN, BK], LSB-first along K
        offs_g = kb // GS + tl.arange(0, BKG)
        scg = tl.load(
            bs_ptr
            + expert * (N * (K // GS))
            + offs_n[:, None] * (K // GS)
            + offs_g[None, :]
        ).to(tl.float32)
        zg = tl.load(
            bz_ptr
            + expert * ((N // 2) * (K // GS))
            + (offs_n[:, None] // 2) * (K // GS)
            + offs_g[None, :]
        )
        zpg = ((zg >> ((offs_n[:, None] % 2) * 4)) & 0xF).to(tl.float32)
        sc = tl.reshape(tl.broadcast_to(scg[:, :, None], (BN, BKG, GS)), (BN, BK))
        zp = tl.reshape(tl.broadcast_to(zpg[:, :, None], (BN, BKG, GS)), (BN, BK))
        b = tl.trans(((nib - zp) * sc).to(tl.bfloat16))  # [BK, BN]
        acc = tl.dot(a, b, acc=acc)

    tl.store(
        part_ptr
        + pid_k.to(tl.int64) * (num_valid * N)
        + offs_token[:, None] * N
        + offs_n[None, :],
        acc,
        mask=token_mask[:, None],
    )


def moe_int4_gemv_v2(x, wq, topk_ids, topk_weights, mul_routed, sorted_ids,
                     expert_ids, ntp, N, K, GS, TOPK, SPLITK=4, BN=64, BK=128,
                     num_warps=4, num_stages=2):
    """v2 driver; falls back to the v1 kernel when K % (BK*SPLITK) != 0."""
    R = x.shape[0]
    num_valid = R * TOPK
    part = torch.empty((SPLITK, num_valid, N), dtype=torch.float32, device=x.device)
    BM = 16
    num_pid_m = triton.cdiv(min(sorted_ids.shape[0], num_valid * BM), BM)
    grid = (num_pid_m, triton.cdiv(N, BN), SPLITK)
    _moe_int4_gemv_partial_v2[grid](
        x, wq[0], wq[1], wq[2], part,
        sorted_ids, expert_ids, ntp, num_valid,
        x.stride(0),
        N=N, K=K, GS=GS, TOPK=TOPK,
        BM=BM, BN=BN, BK=BK, SPLITK=SPLITK,
        num_warps=num_warps, num_stages=num_stages,
    )
    c = torch.empty((num_valid, N), dtype=torch.bfloat16, device=x.device)
    BLOCK = 512
    _moe_int4_gemv_reduce[(num_valid * triton.cdiv(N, BLOCK),)](
        part, c,
        topk_weights if mul_routed else part,
        num_valid, N=N, TOPK=TOPK, SPLITK=SPLITK,
        MUL_ROUTED=mul_routed, BLOCK=BLOCK,
        num_warps=4,
    )
    return c


def moe_int4_gemv(x, wq, topk_ids, topk_weights, mul_routed, sorted_ids,
                  expert_ids, ntp, N, K, GS, TOPK, SPLITK=4, BN=64, BK=128,
                  num_warps=4, num_stages=2, btrans=False):
    """Two-phase split-K int4 MoE GEMV (v1 kernel). x [R, K] bf16 (R = M rows
    for the w13 call with TOPK=top_k, or M*top_k rows for the w2 call with
    TOPK=1, exactly mirroring invoke_fused_moe_wna16_triton_kernel); returns
    C [R*TOPK, N] bf16 (routed weight applied when mul_routed)."""
    R = x.shape[0]
    num_valid = R * TOPK
    part = torch.empty((SPLITK, num_valid, N), dtype=torch.float32, device=x.device)
    BM = 16
    num_pid_m = triton.cdiv(min(sorted_ids.shape[0], num_valid * BM), BM)
    grid = (num_pid_m, triton.cdiv(N, BN), SPLITK)
    _moe_int4_gemv_partial[grid](
        x, wq[0], wq[1], wq[2], part,
        sorted_ids, expert_ids, ntp, num_valid,
        x.stride(0),
        N=N, K=K, GS=GS, TOPK=TOPK,
        BM=BM, BN=BN, BK=BK, SPLITK=SPLITK, BTRANS=btrans,
        num_warps=num_warps, num_stages=num_stages,
    )
    c = torch.empty((num_valid, N), dtype=torch.bfloat16, device=x.device)
    BLOCK = 512
    _moe_int4_gemv_reduce[(num_valid * triton.cdiv(N, BLOCK),)](
        part, c,
        topk_weights if mul_routed else part,
        num_valid, N=N, TOPK=TOPK, SPLITK=SPLITK,
        MUL_ROUTED=mul_routed, BLOCK=BLOCK,
        num_warps=4,
    )
    return c
