# SPDX-License-Identifier: Apache-2.0
"""Triton hot-path kernels for the GLM-5.3-Flash kpool sparse indexer (gfx1x).

Standalone copy of the helper block injected by
patches/runtime_glm53_kpool_triton.py (regenerate with a text extraction from
that patch; the module header below stands in for the imports the target
module already provides). Validated by bench/kpool_triton_validate.py.
"""

import torch

from vllm.platforms import current_platform

# kpool triton lane version: v1.0 (gfx1x Triton hot paths inside the torch lane)
from vllm.triton_utils import tl as _glm53_tl
from vllm.triton_utils import triton as _glm53_triton


def _glm53_kpool_triton_enabled():
    """Gate for the Triton hot-path lane inside the gfx1x kpool torch lane:
    VLLM_GFX1X_KPOOL_TRITON=1 on, default off. Lazy per call (Ray applies
    worker env after module import)."""
    import os

    return os.environ.get("VLLM_GFX1X_KPOOL_TRITON", "0") == "1"


@_glm53_triton.jit
def _glm53_kpool_gather_kernel(
    cache_fp8_ptr,  # flat fp8 element view of the uint8 cache
    cache_f32_ptr,  # flat fp32 element view of the uint8 cache
    block_table_ptr,  # [R, W] int32, NUM_STATES-pool units
    cu_seq_lens_ptr,  # [R + 1] int32, pool-granular
    token_to_seq_ptr,  # [total] int32
    k_out_ptr,  # [total, HEAD_DIM] fp8
    s_out_ptr,  # [total] fp32
    bt_stride,
    total,
    PAGE_SIZE: _glm53_tl.constexpr,  # pools per cache page (kv_cache.shape[1])
    NUM_STATES: _glm53_tl.constexpr,  # pools per block_table column unit
    HEAD_DIM: _glm53_tl.constexpr,
):
    """One program per pool entry: de-shuffled gather of fp8 K + fp32 scale."""
    tid = _glm53_tl.program_id(0).to(_glm53_tl.int64)
    if tid >= total:
        return
    t2s = _glm53_tl.load(token_to_seq_ptr + tid).to(_glm53_tl.int64)
    start = _glm53_tl.load(cu_seq_lens_ptr + t2s).to(_glm53_tl.int64)
    local = tid - start
    entry = _glm53_tl.load(
        block_table_ptr + t2s * bt_stride + local // NUM_STATES
    ).to(_glm53_tl.int64)
    slot = entry * NUM_STATES + local % NUM_STATES
    page = slot // PAGE_SIZE
    off = slot % PAGE_SIZE

    d = _glm53_tl.arange(0, HEAD_DIM)
    page_bytes: _glm53_tl.constexpr = PAGE_SIZE * (HEAD_DIM + 4)
    # ROCm 16x16 preshuffle byte offset of (off, d) within the page.
    shuf = (
        (off // 16) * (16 * HEAD_DIM)
        + (d // 16) * 256
        + (off % 16) * 16
        + d % 16
    )
    vals = _glm53_tl.load(cache_fp8_ptr + page * page_bytes + shuf)
    _glm53_tl.store(k_out_ptr + tid * HEAD_DIM + d, vals)

    scale_off = page * (page_bytes // 4) + (PAGE_SIZE * HEAD_DIM) // 4 + off
    _glm53_tl.store(s_out_ptr + tid, _glm53_tl.load(cache_f32_ptr + scale_off))


def _glm53_kpool_cache_gather_triton(
    kv_cache, head_dim, block_table, cu_seq_lens, token_to_seq, total, num_states
):
    """Triton port of _glm53_kpool_cache_gather: [total, head_dim] fp8 values
    + [total] fp32 scales from the paged pool cache."""
    device = kv_cache.device
    nb, bs, width = kv_cache.shape
    fp8_dtype = current_platform.fp8_dtype()
    k_vals = torch.empty((total, head_dim), dtype=fp8_dtype, device=device)
    k_scales = torch.empty((total,), dtype=torch.float32, device=device)
    if total == 0:
        return k_vals, k_scales
    flat = kv_cache.view(nb, -1)
    _glm53_kpool_gather_kernel[(total,)](
        flat.view(fp8_dtype),
        flat.view(torch.float32),
        block_table,
        cu_seq_lens,
        token_to_seq,
        k_vals,
        k_scales,
        block_table.stride(0),
        total,
        PAGE_SIZE=bs,
        NUM_STATES=num_states,
        HEAD_DIM=head_dim,
        num_warps=1,
    )
    return k_vals, k_scales


@_glm53_triton.jit
def _glm53_kpool_mqa_logits_kernel(
    q_ptr,  # [M, H, D] fp8
    k_ptr,  # [T, D] fp8
    s_ptr,  # [T] fp32
    w_ptr,  # [M, H] fp32
    ks_ptr,  # [M] int32
    ke_ptr,  # [M] int32
    out_ptr,  # [M, T] fp32
    M,
    T,
    q_stride_m,
    w_stride_m,
    HEAD_DIM: _glm53_tl.constexpr,
    H: _glm53_tl.constexpr,
    BLOCK_M: _glm53_tl.constexpr,
    BLOCK_T: _glm53_tl.constexpr,
):
    """Prefill MQA logits: out[m, t] = sum_h relu(q[m,h].k[t]) * s[t] * w[m,h].

    2-D tiling: each K tile is loaded once per BLOCK_M query rows, so K reads
    scale as (M/BLOCK_M) x T x HEAD_DIM bytes instead of M x T x HEAD_DIM."""
    pid_m = _glm53_tl.program_id(0)
    pid_t = _glm53_tl.program_id(1)
    rows = pid_m * BLOCK_M + _glm53_tl.arange(0, BLOCK_M)
    cols = pid_t * BLOCK_T + _glm53_tl.arange(0, BLOCK_T)
    m_mask = rows < M
    t_mask = cols < T
    ks = _glm53_tl.load(ks_ptr + rows, mask=m_mask, other=0)
    ke = _glm53_tl.load(ke_ptr + rows, mask=m_mask, other=0)

    d = _glm53_tl.arange(0, HEAD_DIM)
    kt = _glm53_tl.load(
        k_ptr + cols[:, None] * HEAD_DIM + d[None, :],
        mask=t_mask[:, None],
        other=0.0,
    ).to(_glm53_tl.bfloat16)
    ktT = _glm53_tl.trans(kt)  # [HEAD_DIM, BLOCK_T]
    scale = _glm53_tl.load(s_ptr + cols, mask=t_mask, other=0.0)

    acc = _glm53_tl.zeros((BLOCK_M, BLOCK_T), dtype=_glm53_tl.float32)
    for h in _glm53_tl.static_range(H):
        qh = _glm53_tl.load(
            q_ptr + rows[:, None] * q_stride_m + h * HEAD_DIM + d[None, :],
            mask=m_mask[:, None],
            other=0.0,
        ).to(_glm53_tl.bfloat16)  # [BLOCK_M, HEAD_DIM]
        # bf16 rounding after the dot replicates the golden torch lane, whose
        # torch.matmul(bf16, bf16) materializes a bf16 tensor before .float().
        s_h = _glm53_tl.dot(qh, ktT).to(_glm53_tl.bfloat16).to(_glm53_tl.float32)
        s_h = s_h * scale[None, :]
        s_h = _glm53_tl.maximum(s_h, 0.0)
        wh = _glm53_tl.load(w_ptr + rows * w_stride_m + h, mask=m_mask, other=0.0)
        acc += s_h * wh[:, None]

    valid = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
    acc = _glm53_tl.where(valid, acc, float("-inf"))
    _glm53_tl.store(
        out_ptr + rows[:, None] * T + cols[None, :],
        acc,
        mask=m_mask[:, None] & t_mask[None, :],
    )


def _glm53_kpool_mqa_logits_triton(q_fp8, k_vals, k_scales, weights, ks, ke):
    """Triton port of _glm53_kpool_mqa_logits_torch: [M, T] fp32 logits,
    -inf outside [ks[m], ke[m])."""
    M, H, D = q_fp8.shape
    T = k_vals.shape[0]
    out = torch.empty((M, T), dtype=torch.float32, device=q_fp8.device)
    if M == 0 or T == 0:
        return out
    # gfx1151-tuned (sweep on Strix Halo 8060S, M=2048 T=8192): BM=16/BT=128
    BLOCK_M, BLOCK_T = 16, 128
    grid = (_glm53_triton.cdiv(M, BLOCK_M), _glm53_triton.cdiv(T, BLOCK_T))
    _glm53_kpool_mqa_logits_kernel[grid](
        q_fp8,
        k_vals,
        k_scales,
        weights,
        ks,
        ke,
        out,
        M,
        T,
        q_fp8.stride(0),
        weights.stride(0),
        HEAD_DIM=D,
        H=H,
        BLOCK_M=BLOCK_M,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return out


@_glm53_triton.jit
def _glm53_kpool_paged_logits_kernel(
    q_ptr,  # [B, N, H, D] fp8
    cache_fp8_ptr,  # flat fp8 element view of the uint8 cache
    cache_f32_ptr,  # flat fp32 element view of the uint8 cache
    w_ptr,  # [B, N, H] fp32
    ctx_ptr,  # [B, N] int64, pool-granular context length
    bt_ptr,  # [B, W] int32, NUM_STATES-pool units
    out_ptr,  # [B*N, S] fp32
    N,
    S,
    bt_stride,
    q_stride_b,
    q_stride_n,
    w_stride_b,
    w_stride_n,
    ctx_stride_b,
    PAGE_SIZE: _glm53_tl.constexpr,
    NUM_STATES: _glm53_tl.constexpr,
    HEAD_DIM: _glm53_tl.constexpr,
    H: _glm53_tl.constexpr,
    BLOCK_T: _glm53_tl.constexpr,
):
    """Decode paged MQA logits over pool-granular pages (fused gather+MQA):
    out[row, c] = sum_h relu(q[row,h].k[c]) * k_scale[c] * w[row,h], fp32,
    -inf at/after the row's context length."""
    row = _glm53_tl.program_id(0).to(_glm53_tl.int64)
    pid_t = _glm53_tl.program_id(1)
    b = row // N
    n = row % N
    ctx = _glm53_tl.load(ctx_ptr + b * ctx_stride_b + n).to(_glm53_tl.int64)

    cols = (pid_t * BLOCK_T + _glm53_tl.arange(0, BLOCK_T)).to(_glm53_tl.int64)
    in_range = cols < ctx
    entry = _glm53_tl.load(
        bt_ptr + b * bt_stride + cols // NUM_STATES, mask=in_range, other=0
    ).to(_glm53_tl.int64)
    slot = entry * NUM_STATES + cols % NUM_STATES
    page = slot // PAGE_SIZE
    off = slot % PAGE_SIZE

    d = _glm53_tl.arange(0, HEAD_DIM)
    page_bytes: _glm53_tl.constexpr = PAGE_SIZE * (HEAD_DIM + 4)
    # ROCm 16x16 preshuffle byte offsets of (off, d) within each page.
    shuf = (
        (off // 16)[:, None] * (16 * HEAD_DIM)
        + (d // 16)[None, :] * 256
        + (off % 16)[:, None] * 16
        + d[None, :] % 16
    )
    kt = _glm53_tl.load(
        cache_fp8_ptr + page[:, None] * page_bytes + shuf,
        mask=in_range[:, None],
        other=0.0,
    ).to(_glm53_tl.bfloat16)  # [BLOCK_T, HEAD_DIM]
    scale = _glm53_tl.load(
        cache_f32_ptr + page * (page_bytes // 4) + (PAGE_SIZE * HEAD_DIM) // 4 + off,
        mask=in_range,
        other=0.0,
    )  # [BLOCK_T] fp32

    h = _glm53_tl.arange(0, H)
    qT = _glm53_tl.load(
        q_ptr + b * q_stride_b + n * q_stride_n + h[None, :] * HEAD_DIM + d[:, None]
    ).to(_glm53_tl.bfloat16)  # [HEAD_DIM, H]

    scores = _glm53_tl.dot(kt, qT)  # [BLOCK_T, H] fp32
    # bf16 rounding replicates the golden torch lane's torch.matmul bf16
    # output before its .float() cast.
    scores = scores.to(_glm53_tl.bfloat16).to(_glm53_tl.float32)
    scores = scores * scale[:, None]
    scores = _glm53_tl.maximum(scores, 0.0)
    w = _glm53_tl.load(w_ptr + b * w_stride_b + n * w_stride_n + h)  # [H] fp32
    out = _glm53_tl.sum(scores * w[None, :], axis=1)  # [BLOCK_T] fp32
    out = _glm53_tl.where(in_range, out, float("-inf"))
    _glm53_tl.store(out_ptr + row * S + cols, out, mask=cols < S)


def _glm53_kpool_paged_logits_triton(
    q_fp8, kv_cache, head_dim, weights, ctx_lens, block_table, num_states
):
    """Triton port of _glm53_kpool_paged_logits_torch: [B, N, S] fp32 with
    S = pages*num_states rounded to whole manager blocks, -inf at/after each
    row's pool-granular context length."""
    B, N, H, D = q_fp8.shape
    nb, bs, width = kv_cache.shape
    device = q_fp8.device
    ctx = ctx_lens.contiguous()
    max_ctx = int(ctx.max().item())
    nbp = max(1, (max_ctx + num_states - 1) // num_states)
    S = nbp * num_states
    out = torch.empty((B, N, S), dtype=torch.float32, device=device)
    flat = kv_cache.view(nb, -1)
    # gfx1151-tuned (sweep on Strix Halo 8060S, B=32, S=2304): BT=64/nw=2
    BLOCK_T = 64
    grid = (B * N, _glm53_triton.cdiv(S, BLOCK_T))
    _glm53_kpool_paged_logits_kernel[grid](
        q_fp8,
        flat.view(current_platform.fp8_dtype()),
        flat.view(torch.float32),
        weights,
        ctx,
        block_table,
        out,
        N,
        S,
        block_table.stride(0),
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        ctx.stride(0),
        PAGE_SIZE=bs,
        NUM_STATES=num_states,
        HEAD_DIM=head_dim,
        H=H,
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out


def _glm53_kpool_cache_gather_dispatch(
    kv_cache, head_dim, block_table, cu_seq_lens, token_to_seq, total, num_states
):
    if _glm53_kpool_triton_enabled():
        return _glm53_kpool_cache_gather_triton(
            kv_cache, head_dim, block_table, cu_seq_lens, token_to_seq, total,
            num_states
        )
    return _glm53_kpool_cache_gather(
        kv_cache, head_dim, block_table, cu_seq_lens, token_to_seq, total,
        num_states
    )


def _glm53_kpool_mqa_logits_dispatch(q_fp8, k_vals, k_scales, weights, ks, ke):
    if _glm53_kpool_triton_enabled():
        return _glm53_kpool_mqa_logits_triton(
            q_fp8, k_vals, k_scales, weights, ks, ke
        )
    return _glm53_kpool_mqa_logits_torch(q_fp8, k_vals, k_scales, weights, ks, ke)


def _glm53_kpool_paged_logits_dispatch(
    q_fp8, kv_cache, head_dim, weights, ctx_lens, block_table, num_states
):
    if _glm53_kpool_triton_enabled():
        return _glm53_kpool_paged_logits_triton(
            q_fp8, kv_cache, head_dim, weights, ctx_lens, block_table, num_states
        )
    return _glm53_kpool_paged_logits_torch(
        q_fp8, kv_cache, head_dim, weights, ctx_lens, block_table, num_states
    )


def _glm53_kpool_expand_tail_dispatch(pool_ids, seq_lens_tokens, kpool):
    """Expand selected pools to token ids + append the incomplete tail pool.
    Triton path: the shipped kpool_compress expand kernel (pure integer
    gather, no AITER) — bit-exact vs the ~25-op torch chain."""
    if _glm53_kpool_triton_enabled() and pool_ids.shape[0] > 0:
        return kpool_ops.expand_pools_and_append_tail(
            pool_ids, seq_lens_tokens, kpool
        )
    return _glm53_kpool_expand_tail_torch(pool_ids, seq_lens_tokens, kpool)

