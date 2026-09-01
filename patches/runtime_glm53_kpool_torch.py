#!/usr/bin/env python3
"""Runtime patch: torch-native lane for the kpool sparse-attention indexer.

GLM-5.3-Flash (index_kpool=4, index_topk=2048) on gfx1151 runs with
VLLM_ROCM_USE_AITER=0 (aiter crashes on gfx1x), so SparseAttnIndexerKpool.
forward_hip dies with "Sparse attention indexer ROCm path is only supported
on AITER". This patch reroutes that dead end to a pure-PyTorch
reimplementation of the module-level sparse_attn_indexer_kpool() semantics:

  - prefill pool compress: per-dim softmax(gate+ape) over each pool of
    index_kpool consecutive tokens -> bf16 -> Hadamard-128 -> bf16 ->
    per-vector absmax fp8 quant (ue8m0 power-of-2 scale rounding when
    scale_fmt is set) -> paged cache write with the ROCm 16x16 preshuffle.
  - prefill tail seed: last-kpool tokens of each request -> paged bf16 tail
    cache (raw K at half 0, gate score at half 1, slot = pos % kpool).
  - decode tail update: stash every token into its request's circular tail
    block; on pool completion (slot_mapping >= 0 at pos % kpool == kpool-1)
    compress ring + current token and write the pool slot.
  - prefill logits: gather pool K from the paged cache (deshuffled, fp8 x
    fp32 scale), logits[m, t] = sum_h relu(q[m,h] . k[t]) * k_scale[t] *
    weights[m,h], masked to [cu_seqlen_ks, cu_seqlen_ke); top-(topk/kpool)
    pools per row (ascending, -1 padded), expand pools to tokens and append
    the incomplete tail pool.
  - decode logits: paged MQA over pool-granular block_table with per-row
    context lens, same top-k + expand + tail.

Validated bit-exact against the shipped Triton kernels
(vllm.models.glm5next.amd.ops.kpool_compress) on gfx1151 for compress /
cache write+read / tail seed / decode update.

v1.5: the fp32 [tokens, cols] logits are built with a head-batched bf16
matmul per head chunk (bit-identical to v1.4's per-head matmuls on this
ROCm/torch stack) + per-head fp32 addcmul accumulation in the original
head order, and the decode paged gather reads whole pages through a
strided 6D view + permute deshuffle of the 16x16 preshuffle (bitwise
identical to the v1.4 index gather) instead of materializing [B, S, D]
int64 indices. fp32 logits reassociate only in the final dequant-scale
multiply (~1e-7 rel on realistic magnitudes); topk/expand integer outputs
stay bit-identical to v1.4.

Gate: VLLM_GFX1X_KPOOL_TORCH=1 forces on, =0 forces off; default auto =
gfx1x without aiter. Idempotent, fail-closed (ast.parse before write).

Target: vllm/model_executor/layers/sparse_attn_indexer_kpool.py
"""

import ast
import io
import sys

P = "/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"

HELPER = '''
# kpool torch lane version: v1.5 (gfx1x no-AITER torch-native reference,
# batched-head logits + page-gather deshuffle)

_GLM53_KPOOL_HADAMARD = {}
_GLM53_KPOOL_SHUF_IDX = {}
_GLM53_KPOOL_LOGGED = False

# Element budget for the head-batched [tokens, heads, cols] bf16 matmul
# temporary (2**27 elems = 256 MiB bf16): bounds the v1.5 temporary while
# keeping the single-batched-matmul fast path for all realistic
# decode/prefill-chunk shapes.
_GLM53_KPOOL_CHUNK_ELEMS = 1 << 27


def _glm53_kpool_dbg(stage, **kv):
    """Debug stage markers for the torch lane; active only with
    VLLM_GFX1X_KPOOL_DEBUG=1 (note: serve.sh forwards only registry env into
    the containers, so this must be set via models/registry.yaml to take
    effect cluster-wide). Host-syncs the passed tensors so the last printed
    marker localizes a device abort."""
    import os

    if os.environ.get("VLLM_GFX1X_KPOOL_DEBUG", "0") != "1":
        return
    parts = []
    for name, v in kv.items():
        if isinstance(v, torch.Tensor):
            parts.append(
                f"{name}={tuple(v.shape)}/{v.dtype}/max={int(v.max().item()) if v.numel() else 'empty'}"
            )
        else:
            parts.append(f"{name}={v}")
    torch.cuda.synchronize()
    print(f"[gfx1x_kpool][{stage}] " + " ".join(parts), flush=True)


def _glm53_kpool_torch_enabled():
    """Gate for the torch-native kpool indexer lane."""
    import os

    env = os.environ.get("VLLM_GFX1X_KPOOL_TORCH", "")
    if env == "1":
        return True
    if env == "0":
        return False
    try:
        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return False
    except Exception:
        return False
    return not (rocm_aiter_ops.is_enabled() or rocm_aiter_ops.is_rdna_aiter_enabled())


def _glm53_kpool_hadamard128(device):
    """Sylvester Hadamard-128 matrix x 1/sqrt(128); equals the Triton
    butterfly chain (_hadamard128) up to fp32 summation order."""
    m = _GLM53_KPOOL_HADAMARD.get(device)
    if m is None:
        h = torch.ones((1, 1), dtype=torch.float32, device=device)
        for _ in range(7):
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        m = h * (128**-0.5)
        _GLM53_KPOOL_HADAMARD[device] = m
    return m


def _glm53_kpool_shuf_idx(block_size, head_dim, device):
    """Byte offset of (token_offset, dim_offset) inside one cache page under
    the ROCm 16x16 preshuffle (matches _cache_k_offset PRESHUFFLE=True)."""
    key = (block_size, head_dim, device)
    idx = _GLM53_KPOOL_SHUF_IDX.get(key)
    if idx is None:
        t = torch.arange(block_size, device=device)[:, None]
        d = torch.arange(head_dim, device=device)[None, :]
        idx = (t // 16) * (16 * head_dim) + (d // 16) * 256 + (t % 16) * 16 + d % 16
        _GLM53_KPOOL_SHUF_IDX[key] = idx
    return idx


def _glm53_kpool_compress_torch(kp, gp, ape, round_scale):
    """Pool compress: [n, kpool, D] bf16 K + gate, ape [kpool, D] fp32 ->
    (fp8 [n, D], fp32 scale [n]). Per-dim softmax over pool slots, bf16
    rounding before/after Hadamard-128, per-vector absmax fp8 quant."""
    score = gp.float() + ape.unsqueeze(0)
    p = torch.exp(score - score.amax(dim=1, keepdim=True))
    x = (kp.float() * p).sum(dim=1) / p.sum(dim=1)
    x = x.to(torch.bfloat16).float()
    x = (x @ _glm53_kpool_hadamard128(x.device)).to(torch.bfloat16).float()
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    absmax = x.abs().amax(dim=1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / fp8_max))))
    else:
        scale = absmax * (1.0 / fp8_max)
    q = (x / scale.unsqueeze(1)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale


def _glm53_kpool_cache_write(kv_cache, locs, q_fp8, scales, head_dim):
    """Write fp8 K + fp32 scale at flat slots ``locs`` (int64 [n]).
    Cache: [num_blocks, block_size, head_dim+4] uint8; fp8 values in the
    leading block_size*head_dim bytes (16x16 preshuffled when block_size>1),
    fp32 scales at byte block_size*head_dim + 4*token_offset."""
    nb, bs, width = kv_cache.shape
    flat = kv_cache.view(nb, -1)
    page = locs // bs
    off = locs % bs
    _glm53_kpool_dbg("cache.write", locs=locs, page=page, nb=nb, bs=bs)
    if bs > 1:
        col = _glm53_kpool_shuf_idx(bs, head_dim, kv_cache.device)[off]
    else:
        col = off[:, None] * head_dim + torch.arange(
            head_dim, device=kv_cache.device
        )[None, :]
    flat.view(current_platform.fp8_dtype())[page[:, None], col] = q_fp8
    flat.view(torch.float32)[page, (bs * head_dim) // 4 + off] = scales


def _glm53_kpool_prefill_insert(k, gate_score, ape, kv_cache, slot_mapping,
                                kpool, head_dim, round_scale):
    """Torch port of _kpool_compress_insert: pool kpool consecutive prefill
    tokens; only pool-completion rows (slot_mapping >= 0 at pool-local index
    kpool-1) are written. Assumes pool-aligned chunk starts."""
    n = slot_mapping.shape[0]
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    write_mask = valid & (pos >= kpool - 1)
    if not bool(write_mask.any()):
        return
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    q, scale = _glm53_kpool_compress_torch(k[idx], gate_score[idx], ape,
                                           round_scale)
    locs = slot_mapping.to(torch.int64)
    _glm53_kpool_cache_write(kv_cache, locs[write_mask], q[write_mask],
                             scale[write_mask], head_dim)


def _glm53_kpool_seed_tail(tail_kv_cache, k, gate_score, tslot, kpool):
    """Torch port of kpool_seed_tail_cache: keep each request's last-kpool
    tokens (token i+kpool lands in a different tail block or past the batch)."""
    n = tslot.shape[0]
    if n == 0:
        return
    t = tslot.to(torch.int64)
    valid = t >= 0
    ahead = torch.full_like(t, -1)
    if n > kpool:
        ahead[:-kpool] = t[kpool:]
    blk = torch.where(valid, t // kpool, t)
    ahead_blk = torch.where(ahead >= 0, ahead // kpool, ahead)
    is_tail = valid & (ahead_blk != blk)
    if not bool(is_tail.any()):
        return
    blk_t = blk[is_tail]
    sl = t[is_tail] % kpool
    tail_kv_cache[blk_t, 0, sl] = k[is_tail]
    tail_kv_cache[blk_t, 1, sl] = gate_score[is_tail]


def _glm53_kpool_decode_tail_update(kv_cache, tail_kv_cache, tail_slot, kk, gg,
                                    ape, slot_mapping, positions, kpool,
                                    head_dim, round_scale):
    """Torch port of kpool_decode_update_and_maybe_write_cache_batched.
    Per request, tokens are processed in position order: pool completion
    reads the tail ring with the current token substituted at its slot, then
    the current token is stashed. All inputs are [B, N, ...] per-request."""
    B, N = kk.shape[0], kk.shape[1]
    rows = torch.arange(B, device=kk.device)
    for t in range(N):
        pos = positions[:, t].to(torch.int64)
        sm = slot_mapping[:, t].to(torch.int64)
        ts = tail_slot[:, t].to(torch.int64)
        slot_in_pool = pos.clamp_min(0) % kpool
        block = ts.clamp_min(0) // kpool
        done = (sm >= 0) & (pos >= 0) & (slot_in_pool == kpool - 1)
        if bool(done.any()):
            ring_k = tail_kv_cache[block, 0].clone()
            ring_g = tail_kv_cache[block, 1].clone()
            ring_k[rows, slot_in_pool] = kk[:, t]
            ring_g[rows, slot_in_pool] = gg[:, t]
            q, scale = _glm53_kpool_compress_torch(ring_k, ring_g, ape,
                                                   round_scale)
            _glm53_kpool_cache_write(kv_cache, sm[done], q[done], scale[done],
                                     head_dim)
        stash = (pos >= 0) & (ts >= 0)
        if bool(stash.any()):
            tail_kv_cache[block[stash], 0, slot_in_pool[stash]] = kk[stash, t]
            tail_kv_cache[block[stash], 1, slot_in_pool[stash]] = gg[stash, t]


def _glm53_kpool_cache_gather(kv_cache, head_dim, block_table, cu_seq_lens,
                              token_to_seq, total, num_states):
    """Gather ``total`` pool K entries (fp8 values + fp32 scales) from the
    paged cache.

    Addressing mirrors the write side (_compressed_slot_mapping_kernel):
    block_table columns are ``num_states``-pool units (one cache block of
    block_size tokens = block_size/kpool pools), so
    ``slot = table[seq, pool // num_states] * num_states + pool % num_states``
    and the tensor page/offset split is ``slot // page_size`` /
    ``slot % page_size``. Indexing columns by ``pool // page_size`` directly
    is only correct when num_states == page_size; with block_size 1152 and
    32-pool pages that runs past the table width and aborts the GPU."""
    nb, bs, width = kv_cache.shape
    device = kv_cache.device
    flat = kv_cache.view(nb, -1)
    t2s = token_to_seq[:total].to(torch.int64)
    local = (
        torch.arange(total, device=device, dtype=torch.int64)
        - cu_seq_lens.to(torch.int64)[t2s]
    )
    entry = block_table[t2s, local // num_states].to(torch.int64)
    slot = entry * num_states + (local % num_states)
    page = slot // bs
    off = slot % bs
    _glm53_kpool_dbg("cache.gather.idx", total=total, bt_cols=block_table.shape[1],
                     col=(local // num_states), page=page, nb=nb, bs=bs,
                     num_states=num_states)
    if bs > 1:
        col = _glm53_kpool_shuf_idx(bs, head_dim, device)[off]
    else:
        col = off[:, None] * head_dim + torch.arange(head_dim, device=device)[
            None, :
        ]
    vals = flat.view(current_platform.fp8_dtype())[page[:, None], col]
    scales = flat.view(torch.float32)[page, (bs * head_dim) // 4 + off]
    return vals, scales


def _glm53_kpool_head_chunk(total_heads, rows, cols):
    """Head chunk size for the batched kpool matmul: a single shot when the
    [rows, total_heads, cols] bf16 temporary fits the budget, else the
    largest chunk that does (per-head fallback at the extreme)."""
    if rows * cols * total_heads <= _GLM53_KPOOL_CHUNK_ELEMS:
        return total_heads
    return max(1, _GLM53_KPOOL_CHUNK_ELEMS // max(rows * cols, 1))


def _glm53_kpool_mqa_logits_torch(q_fp8, k_vals, k_scales, weights, ks, ke):
    """Prefill MQA logits: logits[m, t] = sum_h relu(q[m,h].k[t]) *
    k_scale[t] * weights[m,h], -inf outside [ks[m], ke[m]).

    v1.5: one head-batched bf16 matmul per head chunk (bit-identical to
    v1.4's per-head matmuls on this ROCm/torch stack), relu on the bf16
    result (exact in any dtype), fp32 accumulation via addcmul per head in
    the original head order, and the fp8 dequant scale applied once at the
    end: k_scale > 0 always (compress-side clamp_min(1e-4)), so
    relu(x*s)*w == s*(relu(x)*w). Only the final scale multiply
    reassociates vs v1.4 (~1e-7 rel on realistic magnitudes); integer
    topk outputs are bit-identical.
    """
    M, H, D = q_fp8.shape
    T = k_vals.shape[0]
    device = q_fp8.device
    kT = k_vals.to(torch.bfloat16).t()
    q_bf16 = q_fp8.to(torch.bfloat16)
    acc = torch.zeros((M, T), dtype=torch.float32, device=device)
    if M > 0 and T > 0:
        chunk = _glm53_kpool_head_chunk(H, M, T)
        for h0 in range(0, H, chunk):
            h1 = min(H, h0 + chunk)
            if h1 - h0 == 1:
                # 2D matmul: the [M,1,D] batched form is far slower on ROCm
                s = torch.matmul(q_bf16[:, h0], kT).unsqueeze(1)
            else:
                s = torch.matmul(q_bf16[:, h0:h1], kT)
            s.relu_()
            for i, h in enumerate(range(h0, h1)):
                acc.addcmul_(s[:, i], weights[:, h].unsqueeze(1))
    acc *= k_scales.float()
    cols = torch.arange(T, device=device)
    mask = (cols[None, :] >= ks[:, None].to(torch.int64)) & (
        cols[None, :] < ke[:, None].to(torch.int64)
    )
    acc.masked_fill_(~mask, float("-inf"))
    return acc


def _glm53_kpool_paged_logits_torch(q_fp8, kv_cache, head_dim, weights,
                                    ctx_lens, block_table, num_states):
    """Decode paged MQA logits over pool-granular pages.
    q_fp8 [B, N, H, D] fp8; weights [B, N, H] fp32; ctx_lens [B, N] pool
    counts; block_table [B, max_pages] in num_states-pool units (same
    addressing as _glm53_kpool_cache_gather). Returns [B, N, S] fp32 with
    -inf at/after each row's context length, S = pages*num_states rounded to
    whole cache blocks.

    v1.5: page-granular gather + permute deshuffle, then the same
    batched-head logits as the prefill path."""
    B, N, H, D = q_fp8.shape
    nb, bs, width = kv_cache.shape
    device = q_fp8.device
    max_ctx = int(ctx_lens.max().item())
    nbp = max(1, (max_ctx + num_states - 1) // num_states)
    entries = block_table[:B, :nbp].to(torch.int64)
    S = nbp * num_states
    _glm53_kpool_dbg("paged.logits.idx", nbp=nbp, bt_cols=block_table.shape[1],
                     nb=nb, bs=bs, num_states=num_states, S=S)
    fp8_dtype = current_platform.fp8_dtype()
    if bs > 1 and bs == num_states and bs % 16 == 0 and D % 16 == 0:
        # Fast gather: one page-level gather + a pure permute. Requires
        # bs == num_states (one cache page holds exactly one block-table
        # column's pools — true for this cache: page_size == num_states, see
        # kpool_compress_and_write_cache), so page = slot//bs == the block
        # number and pool s <-> page entries[:, s//num_states], row
        # s%num_states, preserving the [B, S] column order exactly. The ROCm
        # 16x16 preshuffle stores a page as [bs//16, D//16, 16, 16] tiles
        # (byte strides 16*D, 256, 16, 1 over the leading bs*D bytes), so
        # logical [pool, dim] = tile[t//16, d//16, t%16, d%16] — bitwise
        # identical to the _glm53_kpool_shuf_idx gather, without the
        # [B, S, D] int64 index tensors.
        row = bs * width
        tiles = kv_cache.view(nb, row).as_strided(
            (nb, bs // 16, D // 16, 16, 16), (row, 16 * D, 256, 16, 1)
        )
        pages = tiles[entries]  # [B, nbp, t16, d16, tm, dm]
        vals = (
            pages.permute(0, 1, 2, 4, 3, 5)
            .reshape(B, S, D)
            .view(fp8_dtype)
        )
        f32 = kv_cache.view(nb, -1).view(torch.float32)
        scales = (
            f32[:, (bs * D) // 4 : (bs * D) // 4 + bs][entries]
            .reshape(B, S)
            .float()
        )
    else:
        # Generic v1.4 gather (unpreshuffled bs == 1 caches).
        s = torch.arange(S, device=device)
        slot = entries[:, s // num_states] * num_states + (s % num_states)[
            None, :
        ]
        page = slot // bs
        off = slot % bs
        flat = kv_cache.view(nb, -1)
        if bs > 1:
            col = _glm53_kpool_shuf_idx(bs, D, device)[off]
        else:
            col = off[:, :, None] * D + torch.arange(D, device=device)[
                None, None, :
            ]
        vals = flat.view(fp8_dtype)[page[:, :, None], col]
        f32 = flat.view(torch.float32)
        scales = f32[page, (bs * D) // 4 + off].float()
    kT = vals.to(torch.bfloat16).transpose(1, 2)  # [B, D, S]
    scales = scales[:, None, :]
    q_bf16 = q_fp8.to(torch.bfloat16)
    acc = torch.zeros((B, N, S), dtype=torch.float32, device=device)
    if B * N > 0 and S > 0:
        chunk = _glm53_kpool_head_chunk(H, B * N, S)
        for h0 in range(0, H, chunk):
            h1 = min(H, h0 + chunk)
            if h1 - h0 == 1:
                # v1.4's exact per-head call: [B,N,D] @ [B,D,S]
                s = torch.matmul(q_bf16[:, :, h0], kT).unsqueeze(2)
            else:
                s = torch.matmul(q_bf16[:, :, h0:h1], kT.unsqueeze(1))
            s.relu_()
            for i, h in enumerate(range(h0, h1)):
                acc.addcmul_(s[:, :, i], weights[:, :, h].unsqueeze(-1))
    acc *= scales
    cols = torch.arange(S, device=device)
    acc.masked_fill_(
        cols[None, None, :] >= ctx_lens[:, :, None].to(torch.int64),
        float("-inf"),
    )
    return acc


def _glm53_kpool_topk_rows(logits, select_k, row_start=None):
    """Top-``select_k`` columns per row of fp32 logits (-inf never selected),
    sequence-relative when ``row_start`` is given, sorted ascending by column
    index, -1 padded to width select_k. Matches the gfx1x radix-topk output
    contract (patch 51) and _C.top_k_per_row_* selection semantics."""
    R, W = logits.shape
    k_eff = min(select_k, W)
    idx = torch.empty((R, 0), dtype=torch.int64, device=logits.device)
    valid = torch.zeros((R, 0), dtype=torch.bool, device=logits.device)
    if k_eff > 0:
        vals, idx = torch.topk(logits, k_eff, dim=1)
        valid = torch.isfinite(vals)
    if row_start is not None:
        idx = idx - row_start[:, None].to(torch.int64)
    big = torch.iinfo(torch.int64).max
    key = torch.where(valid, idx, torch.full_like(idx, big))
    order = key.argsort(dim=1, stable=True)
    idx_s = idx.gather(1, order)
    valid_s = valid.gather(1, order)
    out = torch.where(valid_s, idx_s, torch.full_like(idx_s, -1))
    if k_eff < select_k:
        pad = torch.full(
            (R, select_k - k_eff), -1, dtype=torch.int64, device=logits.device
        )
        out = torch.cat([out, pad], dim=1)
    return out.to(torch.int32)


def _glm53_kpool_expand_tail_torch(pool_ids, seq_lens_tokens, kpool):
    """Torch port of expand_pools_and_append_tail (identity path):
    [rows, n_groups] pool ids -> [rows, n_groups*kpool + kpool - 1] token
    ids; invalid pools -> -1; trailing incomplete pool appended from
    token-granular seq lens."""
    rows, n_groups = pool_ids.shape
    device = pool_ids.device
    topk = n_groups * kpool
    out_cols = topk + kpool - 1
    offs = torch.arange(kpool, device=device, dtype=torch.int64)
    pid = pool_ids.to(torch.int64)
    hist = pid.unsqueeze(-1) * kpool + offs
    hist = torch.where(pid.unsqueeze(-1) >= 0, hist, torch.full_like(hist, -1))
    hist = hist.reshape(rows, topk)
    seq = seq_lens_tokens.to(torch.int64)
    tail_start = (seq // kpool) * kpool
    tail_count = seq - tail_start
    cols = torch.arange(out_cols, device=device, dtype=torch.int64)
    tail_off = cols[None, :] - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])
    tail_val = tail_start[:, None] + tail_off
    out = torch.cat(
        [hist, torch.full((rows, kpool - 1), -1, dtype=torch.int64, device=device)],
        dim=1,
    )
    out = torch.where(is_tail, tail_val, out)
    return out.to(torch.int32)


def _sparse_attn_indexer_kpool_torch(
    op,
    hidden_states,
    q_quant,
    k,
    weights,
    *,
    gate_score,
    compress_ape,
    index_kpool,
    positions,
):
    """Torch-native reference for the kpool sparse indexer on gfx1x (no
    AITER, no DeepGEMM, no _C topk). Mirrors the module-level
    sparse_attn_indexer_kpool() semantics; correctness-first."""
    global _GLM53_KPOOL_LOGGED
    head_dim = op.head_dim
    topk_tokens = op.topk_tokens
    topk_indices_buffer = op.topk_indices_buffer
    kv_cache = op.k_cache.kv_cache
    tail_kv_cache = op.tail_cache.kv_cache if op.tail_cache is not None else None
    tail_prefix = op.tail_cache.prefix if op.tail_cache is not None else None

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        # Profiling / dummy run: reserve the memory the real path allocates
        # (fp8 gather + fp32 scales + logits + per-head temporaries + the
        # decode paged gather) so the memory planner accounts for it.
        fp8_dtype = current_platform.fp8_dtype()
        values_spec, scales_spec = _gather_workspace_shapes(
            op.max_total_seq_len, head_dim, fp8_dtype, False
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        decode_logits_elems = worst_decode_tokens * op.max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )
        # torch lane extras: ~3x logits temporaries + fp32 gather + decode read
        _ = torch.empty(
            max_logits_elems * 3, dtype=torch.uint8, device=hidden_states.device
        )
        _ = torch.empty(
            op.max_total_seq_len * head_dim * 3,
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        pools = (op.max_model_len + index_kpool - 1) // max(index_kpool, 1)
        _ = torch.empty(
            worst_decode_tokens * pools * head_dim * 3,
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        return topk_indices_buffer

    if not _GLM53_KPOOL_LOGGED:
        _GLM53_KPOOL_LOGGED = True
        print(
            "[gfx1x_kpool] torch-native kpool sparse-indexer lane active "
            f"(kpool={index_kpool}, topk={topk_tokens}, head_dim={head_dim})",
            flush=True,
        )

    if head_dim != 128:
        raise RuntimeError(
            f"gfx1x kpool torch lane: head_dim {head_dim} != 128 (Hadamard-128)"
        )
    if index_kpool <= 1:
        raise RuntimeError(
            "gfx1x kpool torch lane: index_kpool<=1 (unfused per-token insert) "
            "is not implemented; GLM-5.3-Flash uses index_kpool>1"
        )

    k_cache_prefix = _resolve_layer_name(op.k_cache.prefix)
    meta = attn_metadata[k_cache_prefix]
    assert isinstance(meta, DeepseekV32IndexerMetadata)
    slot_mapping = meta.slot_mapping
    has_decode = meta.num_decodes > 0
    has_prefill = meta.num_prefills > 0
    num_decode_tokens = meta.num_decode_tokens
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    round_scale = op.scale_fmt is not None
    # Pools per cache block (the block_table column unit on both the slot
    # mapping and the gather): cache block_size tokens / kpool tokens per pool.
    num_states = op.k_cache.cache_config.block_size // index_kpool
    if num_states < 1:
        raise RuntimeError(
            f"gfx1x kpool torch lane: bad num_states={num_states} "
            f"(block_size={op.k_cache.cache_config.block_size}, "
            f"kpool={index_kpool})"
        )

    if not op.skip_k_cache_insert:
        if gate_score is None or compress_ape is None:
            raise RuntimeError(
                "gfx1x kpool torch lane: gate_score/compress_ape required for "
                "index_kpool>1"
            )
        n_prefill = num_tokens - num_decode_tokens
        if n_prefill > 0:
            ps = slice(num_decode_tokens, num_tokens)
            _glm53_kpool_dbg("prefill.insert", kv_cache=kv_cache,
                             slots=slot_mapping[ps], n=n_prefill)
            _glm53_kpool_prefill_insert(
                k[ps],
                gate_score[ps],
                compress_ape,
                kv_cache,
                slot_mapping[ps],
                index_kpool,
                head_dim,
                round_scale,
            )
            _glm53_kpool_dbg("prefill.insert.done")
            if tail_kv_cache is not None and tail_prefix is not None:
                tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_meta is not None:
                    assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                    _glm53_kpool_dbg("prefill.tail_seed",
                                     tail=tail_kv_cache,
                                     tslot=tail_meta.slot_mapping[ps])
                    _glm53_kpool_seed_tail(
                        tail_kv_cache,
                        k[ps],
                        gate_score[ps],
                        tail_meta.slot_mapping[ps],
                        index_kpool,
                    )
                    _glm53_kpool_dbg("prefill.tail_seed.done")

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if has_prefill:
        prefill_metadata = meta.prefill
        assert prefill_metadata is not None
        n_prefill_sf = num_tokens - num_decode_tokens
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_causal_indices(_buf, _pos)
        else:
            if positions is None:
                raise RuntimeError(
                    "gfx1x kpool torch lane: positions required for sparse "
                    "prefill topk"
                )
            select_k = topk_tokens // index_kpool
            k_vals = None
            k_scales = None
            for chunk in prefill_metadata.chunks:
                total = chunk.total_seq_lens
                if not chunk.skip_kv_gather:
                    if total > 0:
                        _glm53_kpool_dbg(
                            "prefill.gather",
                            kv_cache=kv_cache,
                            block_table=chunk.block_table,
                            cu_seq_lens=chunk.cu_seq_lens,
                            token_to_seq=chunk.token_to_seq,
                            total=total,
                            tokens=f"{chunk.token_start}:{chunk.token_end}",
                        )
                        k_vals, k_scales = _glm53_kpool_cache_gather(
                            kv_cache,
                            head_dim,
                            chunk.block_table,
                            chunk.cu_seq_lens,
                            chunk.token_to_seq,
                            total,
                            num_states,
                        )
                        _glm53_kpool_dbg("prefill.gather.done")
                    else:
                        k_vals = None
                        k_scales = None
                if total == 0 or k_vals is None:
                    # no context in this chunk: rows stay -1 (buffer cleared)
                    continue
                q_slice = q_quant[chunk.token_start : chunk.token_end]
                w_slice = weights[chunk.token_start : chunk.token_end]
                logits = _glm53_kpool_mqa_logits_torch(
                    q_slice,
                    k_vals,
                    k_scales,
                    w_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
                _glm53_kpool_dbg("prefill.logits.done")
                pool_ids = _glm53_kpool_topk_rows(
                    logits, select_k, row_start=chunk.cu_seqlen_ks
                )
                q_seq = positions[chunk.token_start : chunk.token_end] + 1
                expanded = _glm53_kpool_expand_tail_torch(
                    pool_ids, q_seq, index_kpool
                )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded
                _glm53_kpool_dbg("prefill.topk.done")

    if has_decode:
        decode_metadata = meta.decode
        assert decode_metadata is not None

        # 1) tail update + completed-pool compress/write
        if (
            gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not op.skip_k_cache_insert
        ):
            num_requests = meta.num_decodes
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            if not use_uniform:
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(
                    shape2
                )
            if dec_tail_slot is not None:
                _glm53_kpool_dbg("decode.tail", kv_cache=kv_cache,
                                 tail=tail_kv_cache, tslot=dec_tail_slot,
                                 sm=dec_slot, pos=dec_pos)
                _glm53_kpool_decode_tail_update(
                    kv_cache,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale,
                )
                _glm53_kpool_dbg("decode.tail.done")

        # 2) short decode: select every token, skip sparse scoring
        if _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            meta.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer

        # 3) sparse decode: paged pool logits -> top-k pools -> expand + tail
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens, pad_value=0
            )
            batch_size = padded_q.shape[0]
            next_n = padded_q.shape[1]
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(batch_size, next_n, *weights.shape[1:])
        else:
            batch_size = decode_lens.shape[0]
            next_n = num_decode_tokens // batch_size
            padded_q = q_quant[:num_decode_tokens].reshape(
                batch_size, next_n, *q_quant.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens].reshape(
                batch_size, next_n, *weights.shape[1:]
            )
        seq_lens = decode_metadata.seq_lens[:batch_size]
        if seq_lens.ndim == 1:
            ctx_rows = seq_lens.to(torch.int64)[:, None].expand(
                batch_size, next_n
            )
        else:
            ctx_rows = seq_lens[:, :next_n].to(torch.int64)
        _glm53_kpool_dbg("decode.logits", kv_cache=kv_cache,
                         block_table=decode_metadata.block_table,
                         ctx=ctx_rows, B=batch_size, N=next_n)
        logits = _glm53_kpool_paged_logits_torch(
            padded_q,
            kv_cache,
            head_dim,
            padded_weights,
            ctx_rows,
            decode_metadata.block_table,
            num_states,
        )
        _glm53_kpool_dbg("decode.logits.done")
        select_k = topk_tokens // index_kpool
        pool_ids = _glm53_kpool_topk_rows(
            logits.reshape(batch_size * next_n, -1), select_k
        )
        if positions is not None:
            dec_seq = _decode_topk_seq_lens(
                positions,
                decode_lens,
                num_decode_tokens,
                batch_size,
                next_n,
                decode_metadata.requires_padding,
            )
        else:
            dec_seq = decode_metadata.seq_lens[: batch_size * next_n]
            if dec_seq.ndim == 2:
                dec_seq = dec_seq[:, -1]
            dec_seq = dec_seq.to(torch.int32)
        out = _glm53_kpool_expand_tail_torch(pool_ids, dec_seq, index_kpool)
        if decode_metadata.requires_padding:
            req_id, intra = _build_decode_scatter_indices(
                decode_lens, batch_size, num_decode_tokens
            )
            out = out.reshape(batch_size, next_n, -1)[req_id, intra]
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer

'''

ANCHOR_RAISE = (
    '        raise RuntimeError(\n'
    '            "Sparse attention indexer ROCm path is only supported on AITER. "\n'
    '            "Please enable aiter with VLLM_ROCM_USE_AITER=1"\n'
    '        )'
)
REPLACEMENT_RAISE = (
    '        if _glm53_kpool_torch_enabled():\n'
    '            return _sparse_attn_indexer_kpool_torch(\n'
    '                self,\n'
    '                hidden_states,\n'
    '                q_quant,\n'
    '                k,\n'
    '                weights,\n'
    '                gate_score=gate_score,\n'
    '                compress_ape=compress_ape,\n'
    '                index_kpool=index_kpool,\n'
    '                positions=positions,\n'
    '            )\n'
    + ANCHOR_RAISE
)

ANCHOR_REGISTER = (
    '\ndirect_register_custom_op(\n    op_name="sparse_attn_indexer_kpool",'
)


def find_helper_span(s):
    """Locate an already-injected helper block (any version): from the version
    marker up to the direct_register_custom_op call."""
    start = s.find("\n# kpool torch lane version:")
    if start < 0:
        return None
    end = s.find(ANCHOR_REGISTER, start)
    if end < 0:
        return None
    return start, end + 1


def main():
    s = io.open(P, encoding="utf-8").read()

    if "kpool torch lane version: v1.5" in s:
        print("   v1.5 already applied")
        return 0
    if ANCHOR_REGISTER not in s:
        print(f"   ERROR: no direct_register_custom_op anchor in {P}")
        return 42
    if s.count(ANCHOR_RAISE) != 1:
        print(f"   ERROR: expected 1 forward_hip raise anchor, found {s.count(ANCHOR_RAISE)}")
        return 42

    span = find_helper_span(s)
    if span is not None:
        # older version present: replace the helper block, keep the call site
        s = s[: span[0]] + HELPER + s[span[1]:]
    else:
        i = s.find(ANCHOR_REGISTER)
        s = s[:i] + HELPER + s[i:]
    if "if _glm53_kpool_torch_enabled():" not in s:
        s = s.replace(ANCHOR_RAISE, REPLACEMENT_RAISE)

    ast.parse(s)  # fail closed: never write an unparseable tree
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("   v1.5 injected (torch-native kpool indexer lane for gfx1x, batched-head logits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
