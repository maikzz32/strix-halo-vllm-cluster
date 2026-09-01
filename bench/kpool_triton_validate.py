#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone validation + microbench for the gfx1x Triton kpool indexer lane.

Golden reference: the v1.4 torch lane helpers injected into the installed
vllm/model_executor/layers/sparse_attn_indexer_kpool.py
(patches/runtime_glm53_kpool_torch.py). Under test: the Triton lane
(bench/kpool_triton_lane.py, or the functions injected by
patches/runtime_glm53_kpool_triton.py when --installed is passed).

Contract-accurate synthetic inputs (GLM-5.3-Flash, index_kpool=4,
index_topk=2048, index_n_heads=32, index_head_dim=128):
  - cache tensor uint8 [num_blocks, 32, 132]: 32-pool physical pages
    (storage blocks), fp8 e4m3fn K under the ROCm 16x16 preshuffle + fp32
    scale per pool slot;
  - block_table int32 [R, W] in 288-pool manager-block units
    (num_states = 1152-token manager block / kpool 4); manager block m owns
    physical pages 9m..9m+8;
  - the synthetic cache is BUILT through the torch lane's own compress +
    cache-write helpers, so the read side is exercised against the exact
    production layout.

Checks (hard requirements):
  - gather: fp8 values + fp32 scales BIT-exact vs torch gather;
  - logits: fp32 max abs/rel diff (report; rel must stay << 1e-3) and the
    -inf mask pattern must match exactly;
  - topk pool ids (torch _glm53_kpool_topk_rows fed from each lane's logits)
    and expanded token ids (torch _glm53_kpool_expand_tail_torch) must match
    EXACTLY.

Usage (on a cluster node, inside the dev container):
  python3 kpool_triton_validate.py --lane /tmp/kpool_triton_lane.py
  python3 kpool_triton_validate.py --installed   # use the patched module
  python3 kpool_triton_validate.py --lane ... --bench-only
"""

import argparse
import importlib.util
import math
import sys

import torch

import vllm.models.glm5next  # noqa: F401  (import order: breaks a circular
# import between the layer module and the glm5next package __init__)
import vllm.model_executor.layers.sparse_attn_indexer_kpool as GL
from vllm.models.glm5next.amd.ops import kpool_compress as kpool_ops

KPOOL = 4
HEAD_DIM = 128
H = 32
TOPK_TOKENS = 2048
SELECT_K = TOPK_TOKENS // KPOOL  # 512 pools
NUM_STATES = 288  # pools per manager block (block_size 1152 / kpool 4)
PAGE_SIZE = 32  # pools per physical cache page (storage block 128 tokens)
FP8_MAX = 448.0


def load_lane(path):
    spec = importlib.util.spec_from_file_location("kpool_triton_lane", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# synthetic contract-accurate inputs
# --------------------------------------------------------------------------- #
def make_cache(num_manager_blocks, seed, device):
    """Random paged pool cache: returns (kv_cache, per-slot (pool K fp32))
    built through the torch lane's compress + write helpers."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    nb = num_manager_blocks * (NUM_STATES // PAGE_SIZE)
    kv_cache = torch.zeros((nb, PAGE_SIZE, HEAD_DIM + 4), dtype=torch.uint8,
                           device=device)
    n_slots = num_manager_blocks * NUM_STATES
    # raw pool material: kpool tokens per pool, compressed through the lane
    k_raw = (torch.randn((n_slots, KPOOL, HEAD_DIM), generator=rng)
             .to(torch.bfloat16).to(device))
    g_raw = (torch.randn((n_slots, KPOOL, HEAD_DIM), generator=rng)
             .to(torch.bfloat16).to(device))
    ape = torch.randn((KPOOL, HEAD_DIM), generator=rng).float().to(device)
    q_fp8, scales = GL._glm53_kpool_compress_torch(k_raw, g_raw, ape, True)
    locs = torch.arange(n_slots, device=device, dtype=torch.int64)
    GL._glm53_kpool_cache_write(kv_cache, locs, q_fp8, scales, HEAD_DIM)
    return kv_cache, nb


def make_block_table(pool_lens, num_manager_blocks, seed, device):
    """Random distinct manager-block assignment for each request."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    width = max(math.ceil(p / NUM_STATES) for p in pool_lens)
    width = max(width, 1)
    perm = torch.randperm(num_manager_blocks, generator=rng)
    bt = torch.zeros((len(pool_lens), width), dtype=torch.int32)
    i = 0
    for r, p in enumerate(pool_lens):
        n = math.ceil(p / NUM_STATES)
        bt[r, :n] = perm[i:i + n].to(torch.int32)
        i += n
    assert i <= num_manager_blocks
    return bt.to(device)


def quantize_q(rows, seed, device):
    """Per-(row, head) absmax ue8m0 fp8 quant of random q: [rows, H, D]."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((rows, H, HEAD_DIM), generator=rng).to(device)
    absmax = q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_MAX)))
    return (q / scale).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn), scale.squeeze(-1)


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name} {detail}")
        if not ok:
            self.failures.append(name)


def logits_diff(ref, got):
    """max abs / rel diff over entries finite in BOTH, plus mask agreement."""
    ref_fin = torch.isfinite(ref)
    got_fin = torch.isfinite(got)
    mask_ok = bool((ref_fin == got_fin).all().item())
    both = ref_fin & got_fin
    if not bool(both.any()):
        return mask_ok, 0.0, 0.0
    d = (ref[both] - got[both]).abs()
    rel = d / ref[both].abs().clamp_min(1e-6)
    return mask_ok, d.max().item(), rel.max().item()


def validate_gather(lane, kv_cache, block_table, pool_lens, device):
    print("== gather (prefill cache read) ==")
    cu = torch.tensor([0] + list(torch.tensor(pool_lens).cumsum(0)),
                      dtype=torch.int32, device=device)
    total = int(cu[-1].item())
    t2s = torch.cat([
        torch.full((p,), i, dtype=torch.int32) for i, p in enumerate(pool_lens)
    ]).to(device)
    kv_ref, ks_ref = GL._glm53_kpool_cache_gather(
        kv_cache, HEAD_DIM, block_table, cu, t2s, total, NUM_STATES)
    kv_new, ks_new = lane._glm53_kpool_cache_gather_triton(
        kv_cache, HEAD_DIM, block_table, cu, t2s, total, NUM_STATES)
    rep = Report()
    rep.check("gather fp8 values bit-exact",
              bool((kv_ref.view(torch.uint8) == kv_new.view(torch.uint8)).all()))
    rep.check("gather fp32 scales bit-exact",
              bool((ks_ref.view(torch.int32) == ks_new.view(torch.int32)).all()))
    return rep, (kv_ref, ks_ref), (cu, t2s, total)


def validate_prefill_logits(lane, kv_ref, ks_ref, pool_lens, seed, device,
                            chunk2=False):
    """Prefill MQA logits over the gathered cache; rows of the requests with
    causal ke; optionally ks>0 to emulate a later query chunk."""
    print(f"== prefill MQA logits (chunk2={chunk2}) ==")
    T = kv_ref.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed)
    rows_per_req = [min(96, p * KPOOL) for p in pool_lens]
    ks_l, ke_l = [], []
    base = 0
    for p, rows in zip(pool_lens, rows_per_req):
        start_pool = (p // 2) if chunk2 else 0
        pos = torch.randint(start_pool * KPOOL, p * KPOOL, (rows,),
                            generator=rng).sort()[0]
        ke_l.extend((pos // KPOOL + 1 + base).tolist())
        ks_l.extend([start_pool + base] * rows)
        base += p
    M = sum(rows_per_req)
    ks = torch.tensor(ks_l, dtype=torch.int32, device=device)
    ke = torch.tensor(ke_l, dtype=torch.int32, device=device)
    q_fp8, _ = quantize_q(M, seed + 1, device)
    weights = torch.softmax(
        torch.randn((M, H), generator=rng).float(), dim=1).to(device)

    ref = GL._glm53_kpool_mqa_logits_torch(q_fp8, kv_ref, ks_ref, weights,
                                           ks, ke)
    got = lane._glm53_kpool_mqa_logits_triton(q_fp8, kv_ref, ks_ref, weights,
                                              ks, ke)
    rep = Report()
    mask_ok, dabs, drel = logits_diff(ref, got)
    rep.check("prefill logits -inf mask exact", mask_ok)
    rep.check("prefill logits rel diff < 1e-3", drel < 1e-3,
              f"max_abs={dabs:.4e} max_rel={drel:.4e}")

    # topk pool ids (row-relative when ks given) + expanded tokens: EXACT
    seq_tokens = (ke * KPOOL).to(torch.int32)
    ids_ref = GL._glm53_kpool_topk_rows(ref, SELECT_K, row_start=ks)
    ids_new = GL._glm53_kpool_topk_rows(got, SELECT_K, row_start=ks)
    rep.check("prefill topk pool ids exact",
              bool((ids_ref == ids_new).all()))
    ex_ref = GL._glm53_kpool_expand_tail_torch(ids_ref, seq_tokens, KPOOL)
    ex_new = GL._glm53_kpool_expand_tail_torch(ids_new, seq_tokens, KPOOL)
    rep.check("prefill expanded token ids exact",
              bool((ex_ref == ex_new).all()))
    # the dispatch's Triton path (shipped kpool_compress expand kernel):
    # pure integer gather, must be bit-exact vs the torch chain
    ex_triton = kpool_ops.expand_pools_and_append_tail(ids_ref, seq_tokens,
                                                       KPOOL)
    rep.check("prefill expand triton kernel exact",
              bool((ex_ref == ex_triton).all()))
    return rep


def validate_decode(lane, kv_cache, block_table, pool_lens, seed, device,
                    next_n=1):
    print(f"== decode paged MQA logits (next_n={next_n}) ==")
    B = len(pool_lens)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ctx = torch.tensor(pool_lens, dtype=torch.int64)[:, None]
    if next_n > 1:
        grow = torch.arange(next_n)[None, :] // KPOOL
        ctx = ctx + grow  # per-row pool ctx for verify tokens
    ctx = ctx.to(device)
    q_fp8, _ = quantize_q(B * next_n, seed + 2, device)
    q_fp8 = q_fp8.view(B, next_n, H, HEAD_DIM)
    weights = torch.softmax(
        torch.randn((B, next_n, H), generator=rng).float(), dim=2).to(device)

    ref = GL._glm53_kpool_paged_logits_torch(
        q_fp8, kv_cache, HEAD_DIM, weights, ctx, block_table, NUM_STATES)
    got = lane._glm53_kpool_paged_logits_triton(
        q_fp8, kv_cache, HEAD_DIM, weights, ctx, block_table, NUM_STATES)
    rep = Report()
    rep.check("decode logits shape match", tuple(ref.shape) == tuple(got.shape),
              f"ref={tuple(ref.shape)} got={tuple(got.shape)}")
    mask_ok, dabs, drel = logits_diff(ref, got)
    rep.check("decode logits -inf mask exact", mask_ok)
    rep.check("decode logits rel diff < 1e-3", drel < 1e-3,
              f"max_abs={dabs:.4e} max_rel={drel:.4e}")

    R = B * next_n
    ids_ref = GL._glm53_kpool_topk_rows(ref.reshape(R, -1), SELECT_K)
    ids_new = GL._glm53_kpool_topk_rows(got.reshape(R, -1), SELECT_K)
    rep.check("decode topk pool ids exact", bool((ids_ref == ids_new).all()))
    seq_tokens = (ctx.reshape(R) * KPOOL).to(torch.int32)
    ex_ref = GL._glm53_kpool_expand_tail_torch(ids_ref, seq_tokens, KPOOL)
    ex_new = GL._glm53_kpool_expand_tail_torch(ids_new, seq_tokens, KPOOL)
    rep.check("decode expanded token ids exact",
              bool((ex_ref == ex_new).all()))
    ex_triton = kpool_ops.expand_pools_and_append_tail(ids_ref, seq_tokens,
                                                       KPOOL)
    rep.check("decode expand triton kernel exact",
              bool((ex_ref == ex_triton).all()))
    return rep


# --------------------------------------------------------------------------- #
# microbench
# --------------------------------------------------------------------------- #
def _time(fn, iters=25, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ev0[i].record()
        fn()
        ev1[i].record()
    torch.cuda.synchronize()
    ts = sorted(ev0[i].elapsed_time(ev1[i]) for i in range(iters))
    return ts[len(ts) // 2]


def bench_decode(lane, device):
    print("== bench decode: paged logits (torch vs triton) ==")
    print(f"  {'rows':>5} {'ctx_tok':>8} {'pools':>6} {'torch_ms':>9} "
          f"{'triton_ms':>10} {'speedup':>8}")
    for rows, ctx_tok in [(1, 1024), (1, 8192), (8, 1024), (8, 8192),
                          (32, 1024), (32, 8192)]:
        pools = ctx_tok // KPOOL
        nman = math.ceil(pools / NUM_STATES) * rows + 2
        kv, _nb = make_cache(nman, 7, device)
        bt = make_block_table([pools] * rows, nman, 8, device)
        ctx = torch.full((rows, 1), pools, dtype=torch.int64, device=device)
        q, _s = quantize_q(rows, 9, device)
        q = q.view(rows, 1, H, HEAD_DIM)
        w = torch.softmax(torch.randn((rows, 1, H)), dim=2).float().to(device)
        t_torch = _time(lambda: GL._glm53_kpool_paged_logits_torch(
            q, kv, HEAD_DIM, w, ctx, bt, NUM_STATES))
        t_triton = _time(lambda: lane._glm53_kpool_paged_logits_triton(
            q, kv, HEAD_DIM, w, ctx, bt, NUM_STATES))
        print(f"  {rows:>5} {ctx_tok:>8} {pools:>6} {t_torch:>9.3f} "
              f"{t_triton:>10.3f} {t_torch / t_triton:>7.2f}x")
        del kv
        torch.cuda.empty_cache()


def bench_prefill(lane, device):
    print("== bench prefill: gather + MQA logits (torch vs triton) ==")
    print(f"  {'M':>5} {'T_pool':>7} {'gather_t':>9} {'gather_tr':>10} "
          f"{'logits_t':>9} {'logits_tr':>10} {'tot_speed':>9}")
    for M, T in [(512, 2048), (2048, 8192), (4096, 8192), (2048, 32768),
                 (4096, 32768)]:
        pool_lens = [min(T, 4096)] * math.ceil(T / 4096)
        pool_lens[-1] = T - 4096 * (len(pool_lens) - 1)
        nman = sum(math.ceil(p / NUM_STATES) for p in pool_lens) + 2
        kv, _nb = make_cache(nman, 11, device)
        bt = make_block_table(pool_lens, nman, 12, device)
        cu = torch.tensor([0] + list(torch.tensor(pool_lens).cumsum(0)),
                          dtype=torch.int32, device=device)
        t2s = torch.cat([
            torch.full((p,), i, dtype=torch.int32)
            for i, p in enumerate(pool_lens)
        ]).to(device)
        q, _s = quantize_q(M, 13, device)
        w = torch.softmax(torch.randn((M, H)), dim=1).float().to(device)
        ks = torch.zeros(M, dtype=torch.int32, device=device)
        ke = torch.full((M,), T, dtype=torch.int32, device=device)

        def torch_gather():
            return GL._glm53_kpool_cache_gather(
                kv, HEAD_DIM, bt, cu, t2s, T, NUM_STATES)

        def triton_gather():
            return lane._glm53_kpool_cache_gather_triton(
                kv, HEAD_DIM, bt, cu, t2s, T, NUM_STATES)

        kv_ref, ksc_ref = torch_gather()
        t_gt = _time(torch_gather)
        t_gtr = _time(triton_gather)
        t_lt = _time(lambda: GL._glm53_kpool_mqa_logits_torch(
            q, kv_ref, ksc_ref, w, ks, ke), iters=15, warmup=3)
        t_ltr = _time(lambda: lane._glm53_kpool_mqa_logits_triton(
            q, kv_ref, ksc_ref, w, ks, ke), iters=15, warmup=3)
        print(f"  {M:>5} {T:>7} {t_gt:>9.3f} {t_gtr:>10.3f} {t_lt:>9.3f} "
              f"{t_ltr:>10.3f} {(t_gt + t_lt) / (t_gtr + t_ltr):>8.2f}x")
        del kv, kv_ref, ksc_ref
        torch.cuda.empty_cache()


def bench_expand(device):
    print("== bench expand+tail (torch chain vs shipped Triton kernel) ==")
    print(f"  {'rows':>6} {'torch_us':>9} {'triton_us':>10} {'speedup':>8}")
    for rows in (32, 512, 2048):
        ids = torch.randint(0, 4096, (rows, SELECT_K), dtype=torch.int32,
                            device=device)
        ids[:, -8:] = -1  # some padding
        seq = torch.full((rows,), 8192, dtype=torch.int32, device=device)
        t_t = _time(lambda: GL._glm53_kpool_expand_tail_torch(ids, seq, KPOOL))
        t_tr = _time(lambda: kpool_ops.expand_pools_and_append_tail(
            ids, seq, KPOOL))
        print(f"  {rows:>6} {t_t * 1000:>9.1f} {t_tr * 1000:>10.1f} "
              f"{t_t / t_tr:>7.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default=None,
                    help="path to the standalone lane module")
    ap.add_argument("--installed", action="store_true",
                    help="use the Triton lane injected into the installed "
                         "sparse_attn_indexer_kpool.py")
    ap.add_argument("--bench-only", action="store_true")
    args = ap.parse_args()

    if args.installed:
        lane = GL  # the patch injects _glm53_kpool_*_triton into the module
        for name in ("_glm53_kpool_cache_gather_triton",
                     "_glm53_kpool_mqa_logits_triton",
                     "_glm53_kpool_paged_logits_triton"):
            if not hasattr(GL, name):
                print(f"ERROR: installed module lacks {name} — apply "
                      "patches/runtime_glm53_kpool_triton.py first")
                return 1
    else:
        lane = load_lane(args.lane or "/tmp/kpool_triton_lane.py")

    device = "cuda"
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}")
    print(f"golden torch lane: {GL.__file__}")

    if not args.bench_only:
        all_rep = Report()
        # mixed-length requests, incl. > manager-block (288 pools) and
        # non-multiple-of-page lengths
        pool_lens = [37, 288 + 41, 1024, 300, 7]
        nman = sum(math.ceil(p / NUM_STATES) for p in pool_lens) + 2
        kv_cache, _ = make_cache(nman, 1, device)
        bt = make_block_table(pool_lens, nman, 2, device)

        rep, (kv_ref, ks_ref), _meta = validate_gather(
            lane, kv_cache, bt, pool_lens, device)
        all_rep.failures += rep.failures
        rep = validate_prefill_logits(lane, kv_ref, ks_ref, pool_lens, 3,
                                      device, chunk2=False)
        all_rep.failures += rep.failures
        rep = validate_prefill_logits(lane, kv_ref, ks_ref, pool_lens, 4,
                                      device, chunk2=True)
        all_rep.failures += rep.failures
        rep = validate_decode(lane, kv_cache, bt, pool_lens, 5, device,
                              next_n=1)
        all_rep.failures += rep.failures
        rep = validate_decode(lane, kv_cache, bt, pool_lens, 6, device,
                              next_n=4)
        all_rep.failures += rep.failures

        if all_rep.failures:
            print(f"VALIDATION FAILED: {all_rep.failures}")
            return 1
        print("VALIDATION PASSED")

    bench_decode(lane, device)
    bench_prefill(lane, device)
    bench_expand(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
