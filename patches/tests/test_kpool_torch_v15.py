#!/usr/bin/env python3
"""Standalone validation + microbench for the v1.5 torch kpool lane.

Compares the v1.5 helpers (extracted from the HELPER block of
patches/runtime_glm53_kpool_torch.py) against the v1.4 reference already
injected in the installed
vllm/model_executor/layers/sparse_attn_indexer_kpool.py.

Fixtures are contract-accurate: the paged pool cache is filled through the
golden v1.4 compress + preshuffled write pipeline (fp8 ue8m0-quantized K +
fp32 scales, ROCm 16x16 preshuffle), gathers use the golden
_glm53_kpool_cache_gather addressing, and all shapes exercise the masking /
topk / expand-tail contract (bit-identical integer outputs required).

Run inside a cluster container, e.g. on node4:

  podman cp patches/tests/test_kpool_torch_v15.py ray-worker:/tmp/
  podman cp patches/runtime_glm53_kpool_torch.py ray-worker:/tmp/
  podman exec ray-worker python3 /tmp/test_kpool_torch_v15.py \
      --patch /tmp/runtime_glm53_kpool_torch.py --device cpu
  podman exec ray-worker python3 /tmp/test_kpool_torch_v15.py \
      --patch /tmp/runtime_glm53_kpool_torch.py --device cuda --bench

Exit code 0 = all correctness checks passed.
"""

import argparse
import importlib.util
import math
import sys
import time

import torch


def load_golden():
    """The v1.4 reference helpers from the installed vLLM module."""
    import vllm  # noqa: F401  full chain first: the kpool module has a
    import vllm.models.glm5next  # noqa: F401  circular import otherwise
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as m

    return m


def load_v15(patch_path, golden):
    """Exec the patch's HELPER block on top of the golden module namespace so
    the v1.5 helpers resolve the same globals (torch, current_platform, ...)."""
    spec = importlib.util.spec_from_file_location("kpool_patch", patch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "kpool torch lane version: v1.5" in mod.HELPER, (
        "patch file does not carry the v1.5 marker"
    )
    ns = dict(vars(golden))
    exec(mod.HELPER, ns)

    class V15:
        pass

    for k, v in ns.items():
        if k.startswith("_glm53_kpool"):
            setattr(V15, k, staticmethod(v) if callable(v) else v)
    V15._ns = ns  # globals dict of the v1.5 helpers (budget knob for tests)
    return V15


def golden_version(golden):
    try:
        with open(golden.__file__, encoding="utf-8") as f:
            s = f.read()
        for v in ("v1.5", "v1.4", "v1.3", "v1.2", "v1.1"):
            if f"kpool torch lane version: {v}" in s:
                return v
    except OSError:
        pass
    return "unknown"


def to_fp8(x, fp8_dtype):
    """Cast to the platform fp8 dtype; fnuz casts may be CUDA-only."""
    try:
        return x.to(fp8_dtype)
    except Exception:
        return x.cuda().to(fp8_dtype).to(x.device)


def build_filled_cache(golden, nb, bs, D, kpool, device, gen):
    """[nb, bs, D+4] uint8 cache with every one of the nb*bs pool slots
    written through the golden compress+write pipeline (realistic fp8 values
    and positive ue8m0-rounded fp32 scales)."""
    from vllm.platforms import current_platform

    cache = torch.zeros(nb, bs, D + 4, dtype=torch.uint8, device=device)
    n = nb * bs
    k = torch.randn(n, kpool, D, generator=gen, device=device).to(torch.bfloat16)
    gate = torch.randn(n, kpool, D, generator=gen, device=device) * 0.5
    ape = torch.randn(kpool, D, generator=gen, device=device) * 0.5
    q, scale = golden._glm53_kpool_compress_torch(k, gate, ape, True)
    assert q.dtype == current_platform.fp8_dtype()
    locs = torch.randperm(n, generator=gen, device=device)
    golden._glm53_kpool_cache_write(cache, locs, q, scale, D)
    return cache


def rand_block_table(gen, device, cols_per_row, nb):
    """[rows, max(cols_per_row)] int64 block table; every entry a valid block
    (< nb), the used prefix of each row unique."""
    rows = len(cols_per_row)
    max_cols = max(max(cols_per_row), 1)
    perm = torch.randperm(nb, generator=gen, device=device)
    bt = torch.zeros(rows, max_cols, dtype=torch.int64, device=device)
    pos = 0
    for i, c in enumerate(cols_per_row):
        for j in range(max_cols):
            bt[i, j] = perm[(pos + j) % nb]
        pos += c
    return bt


def diff_stats(name, ref, out, rel_tol=1e-5, assert_rel=True):
    """Compare fp32 logits: identical finite masks, max abs/rel diff on the
    finite entries. Returns (ok, max_abs, max_rel)."""
    fin_ref = torch.isfinite(ref)
    fin_out = torch.isfinite(out)
    mask_same = torch.equal(fin_ref, fin_out)
    if not mask_same:
        print(f"    {name}: FINITE-MASK MISMATCH")
        return False, float("inf"), float("inf")
    fin = fin_ref
    if not bool(fin.any()):
        print(f"    {name}: all -inf (trivially equal)")
        return True, 0.0, 0.0
    d = (out - ref).abs()[fin]
    max_abs = d.max().item()
    rel = d / ref.abs()[fin].clamp_min(1.0)
    max_rel = rel.max().item()
    ok = mask_same and (not assert_rel or max_rel <= rel_tol)
    print(f"    {name}: max abs {max_abs:.3e}  max rel {max_rel:.3e}")
    return ok, max_abs, max_rel


def check_prefill(golden, v15, device, gen, seq_lens, rows_per_seq, kpool, bs,
                  D, H, select_k, tag, mixed_w=False):
    num_states = bs  # cache page == one block-table column of pools
    pools = [L // kpool for L in seq_lens]
    cols = [max(1, math.ceil(p / num_states)) for p in pools]
    nb = sum(cols) + 8
    cache = build_filled_cache(golden, nb, bs, D, kpool, device, gen)
    bt = rand_block_table(gen, device, cols, nb)
    cu = torch.zeros(len(pools), dtype=torch.int64, device=device)
    total = 0
    for i, p in enumerate(pools):
        cu[i] = total
        total += p
    t2s = torch.cat(
        [torch.full((p,), i, dtype=torch.int64, device=device)
         for i, p in enumerate(pools)]
    ) if total > 0 else torch.zeros(0, dtype=torch.int64, device=device)
    if total > 0:
        k_vals, k_scales = golden._glm53_kpool_cache_gather(
            cache, D, bt, cu, t2s, total, num_states
        )
    else:
        from vllm.platforms import current_platform

        k_vals = torch.zeros(0, D, dtype=current_platform.fp8_dtype(),
                             device=device)
        k_scales = torch.zeros(0, dtype=torch.float32, device=device)
    # rows: random token positions per sequence; ks=0, ke = complete pools
    pos = []
    ks_l, ke_l = [], []
    for i, L in enumerate(seq_lens):
        for _ in range(rows_per_seq):
            p = int(torch.randint(0, max(L, 1), (1,), generator=gen,
                                  device=device).item())
            pos.append(p)
            ks_l.append(0)
            ke_l.append(min((p + 1) // kpool, pools[i]))
    M = len(pos)
    from vllm.platforms import current_platform

    q = to_fp8(torch.randn(M, H, D, generator=gen, device=device) * 4.0,
               current_platform.fp8_dtype())
    if mixed_w:
        weights = torch.randn(M, H, generator=gen, device=device)
    else:
        weights = torch.rand(M, H, generator=gen, device=device)
    ks = torch.tensor(ks_l, dtype=torch.int32, device=device)
    ke = torch.tensor(ke_l, dtype=torch.int32, device=device)
    l14 = golden._glm53_kpool_mqa_logits_torch(q, k_vals, k_scales, weights,
                                               ks, ke)
    l15 = v15._glm53_kpool_mqa_logits_torch(q, k_vals, k_scales, weights,
                                            ks, ke)
    ok, ma, mr = diff_stats(f"prefill[{tag}] logits", l14, l15,
                            assert_rel=not mixed_w)
    ids14 = golden._glm53_kpool_topk_rows(l14, select_k, row_start=ks)
    ids15 = v15._glm53_kpool_topk_rows(l15, select_k, row_start=ks)
    ids_same = torch.equal(ids14, ids15)
    print(f"    prefill[{tag}] topk ids bit-identical: {ids_same}")
    q_seq = torch.tensor([p + 1 for p in pos], dtype=torch.int32,
                         device=device)
    e14 = golden._glm53_kpool_expand_tail_torch(ids14, q_seq, kpool)
    e15 = v15._glm53_kpool_expand_tail_torch(ids15, q_seq, kpool)
    exp_same = torch.equal(e14, e15)
    print(f"    prefill[{tag}] expanded ids bit-identical: {exp_same}")
    return ok and ids_same and exp_same, ma, mr, not mixed_w


def check_decode(golden, v15, device, gen, ctx_pools, N, kpool, bs, D, H,
                 select_k, tag, mixed_w=False):
    num_states = bs
    B = len(ctx_pools)
    # later speculative tokens see up to N-1 extra pools; size the table for it
    cols = [max(1, math.ceil((c + N - 1) / num_states)) for c in ctx_pools]
    nb = sum(cols) + 8
    cache = build_filled_cache(golden, nb, bs, D, kpool, device, gen)
    bt = rand_block_table(gen, device, cols, nb)
    # ctx per (b, n): later speculative tokens see one more pool
    ctx = torch.tensor(ctx_pools, dtype=torch.int64, device=device)[:, None]
    ctx = ctx + torch.arange(N, dtype=torch.int64, device=device)[None, :]
    ctx = ctx.clamp_min(0)
    from vllm.platforms import current_platform

    q = to_fp8(torch.randn(B, N, H, D, generator=gen, device=device) * 4.0,
               current_platform.fp8_dtype())
    if mixed_w:
        weights = torch.randn(B, N, H, generator=gen, device=device)
    else:
        weights = torch.rand(B, N, H, generator=gen, device=device)
    l14 = golden._glm53_kpool_paged_logits_torch(q, cache, D, weights, ctx,
                                                 bt, num_states)
    l15 = v15._glm53_kpool_paged_logits_torch(q, cache, D, weights, ctx,
                                              bt, num_states)
    ok, ma, mr = diff_stats(f"decode[{tag}] logits", l14, l15,
                            assert_rel=not mixed_w)
    S = l14.shape[-1]
    ids14 = golden._glm53_kpool_topk_rows(l14.reshape(B * N, S), select_k)
    ids15 = v15._glm53_kpool_topk_rows(l15.reshape(B * N, S), select_k)
    ids_same = torch.equal(ids14, ids15)
    print(f"    decode[{tag}] topk ids bit-identical: {ids_same}")
    tail = torch.randint(0, kpool, (B * N,), generator=gen, device=device)
    dec_seq = (ctx.reshape(-1) * kpool + tail).to(torch.int32)
    e14 = golden._glm53_kpool_expand_tail_torch(ids14, dec_seq, kpool)
    e15 = v15._glm53_kpool_expand_tail_torch(ids15, dec_seq, kpool)
    exp_same = torch.equal(e14, e15)
    print(f"    decode[{tag}] expanded ids bit-identical: {exp_same}")
    return ok and ids_same and exp_same, ma, mr, not mixed_w


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_prefill(golden, v15, gen, M, T, H=32, D=128, iters=30):
    from vllm.platforms import current_platform

    dev = "cuda"
    k_vals = torch.randint(0, 255, (T, D), generator=gen, device=dev,
                           dtype=torch.uint8).view(current_platform.fp8_dtype())
    k_scales = torch.rand(T, generator=gen, device=dev) + 1e-3
    q = to_fp8(torch.randn(M, H, D, generator=gen, device=dev) * 4.0,
               current_platform.fp8_dtype())
    w = torch.rand(M, H, generator=gen, device=dev)
    ks = torch.zeros(M, dtype=torch.int32, device=dev)
    ke = torch.full((M,), T, dtype=torch.int32, device=dev)
    t14 = bench(lambda: golden._glm53_kpool_mqa_logits_torch(
        q, k_vals, k_scales, w, ks, ke), iters)
    t15 = bench(lambda: v15._glm53_kpool_mqa_logits_torch(
        q, k_vals, k_scales, w, ks, ke), iters)
    print(f"  prefill M={M:<5} T={T:<5} H={H}: v1.4 {t14:8.3f} ms  "
          f"v1.5 {t15:8.3f} ms  speedup {t14 / t15:5.2f}x")
    return t14, t15


def bench_decode(golden, v15, gen, B, N, S, H=32, D=128, bs=32, iters=30):
    from vllm.platforms import current_platform

    dev = "cuda"
    nbp = S // bs
    nb = B * nbp
    cache = torch.randint(0, 255, (nb, bs, D + 4), generator=gen, device=dev,
                          dtype=torch.uint8)
    bt = torch.randperm(nb, generator=gen, device=dev).to(torch.int32)
    bt = bt.view(B, nbp)
    ctx = torch.full((B, N), S, dtype=torch.int64, device=dev)
    q = to_fp8(torch.randn(B, N, H, D, generator=gen, device=dev) * 4.0,
               current_platform.fp8_dtype())
    w = torch.rand(B, N, H, generator=gen, device=dev)
    t14 = bench(lambda: golden._glm53_kpool_paged_logits_torch(
        q, cache, D, w, ctx, bt, bs), iters)
    t15 = bench(lambda: v15._glm53_kpool_paged_logits_torch(
        q, cache, D, w, ctx, bt, bs), iters)
    print(f"  decode  B={B:<3} N={N} S={S:<5} H={H}: v1.4 {t14:8.3f} ms  "
          f"v1.5 {t15:8.3f} ms  speedup {t14 / t15:5.2f}x")
    return t14, t15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True,
                    help="path to runtime_glm53_kpool_torch.py (v1.5)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bench", action="store_true",
                    help="run the microbenchmarks (requires --device cuda)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    golden = load_golden()
    gver = golden_version(golden)
    print(f"golden reference: {golden.__file__} (lane version {gver})")
    if gver != "v1.4":
        print("WARNING: golden reference is not v1.4 — comparisons are not "
              "against the v1.4 contract baseline")
    v15 = load_v15(args.patch, golden)

    dev = args.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)

    # GLM-5.3-Flash geometry: index_kpool=4, index_n_heads=32,
    # index_head_dim=128, index_topk=2048 -> 512 pools selected.
    D, H, kpool, select_k = 128, 32, 4, 512

    ok_all = True
    worst_rel = 0.0

    def track(res):
        nonlocal ok_all, worst_rel
        ok, _, mr, asserted = res
        ok_all = ok_all and ok
        if asserted:
            worst_rel = max(worst_rel, mr)

    print(f"-- prefill correctness ({dev}) --")
    track(check_prefill(golden, v15, dev, gen, [512, 300, 37], 8, kpool, 32,
                        D, H, select_k, "mixed-lens/bs32"))
    track(check_prefill(golden, v15, dev, gen, [2048, 1024], 16, kpool, 32,
                        D, H, select_k, "long-ctx/bs32"))
    track(check_prefill(golden, v15, dev, gen, [8192], 64, kpool, 32, D, H,
                        select_k, "8k-ctx/bs32"))
    track(check_prefill(golden, v15, dev, gen, [16, 3], 8, kpool, 32, D, H,
                        select_k, "short-seq/bs32"))  # ke=0 rows, T<select_k
    track(check_prefill(golden, v15, dev, gen, [512, 511], 8, kpool, 16, D,
                        H, select_k, "bs16"))
    track(check_prefill(golden, v15, dev, gen, [1024], 16, kpool, 32, D, H,
                        select_k, "mixed-sign-weights", mixed_w=True))

    print(f"-- decode correctness ({dev}) --")
    track(check_decode(golden, v15, dev, gen, [8192] * 4, 1, kpool, 32, D, H,
                       select_k, "B4-S8k/bs32"))
    track(check_decode(golden, v15, dev, gen,
                       [8192, 4096, 1000, 128, 4, 0], 1, kpool, 32, D, H,
                       select_k, "skewed/bs32"))  # incl. ctx=0 row
    track(check_decode(golden, v15, dev, gen, [4096] * 8, 2, kpool, 32, D, H,
                       select_k, "B8-N2/bs32"))
    track(check_decode(golden, v15, dev, gen, [2048] * 4, 1, kpool, 16, D, H,
                       select_k, "bs16"))
    track(check_decode(golden, v15, dev, gen, [64, 33], 1, kpool, 1, D, H,
                       select_k, "bs1-unpreshuffled"))
    track(check_decode(golden, v15, dev, gen, [2048] * 4, 1, kpool, 32, D, H,
                       select_k, "mixed-sign-weights", mixed_w=True))

    print(f"-- forced head-chunking correctness ({dev}) --")
    ns = v15._ns
    saved_budget = ns["_GLM53_KPOOL_CHUNK_ELEMS"]
    try:
        # budget below M*T -> per-head 2D-matmul fallback; below M*T*H ->
        # chunked batches with a remainder chunk
        ns["_GLM53_KPOOL_CHUNK_ELEMS"] = 1 << 14
        track(check_prefill(golden, v15, dev, gen, [2048, 1024], 16, kpool,
                            32, D, H, select_k, "forced-per-head"))
        ns["_GLM53_KPOOL_CHUNK_ELEMS"] = 1 << 17
        track(check_prefill(golden, v15, dev, gen, [2048, 1024], 16, kpool,
                            32, D, H, select_k, "forced-chunk5"))
        ns["_GLM53_KPOOL_CHUNK_ELEMS"] = 1 << 13
        track(check_decode(golden, v15, dev, gen, [1500, 700, 64], 1, kpool,
                           32, D, H, select_k, "forced-per-head"))
        ns["_GLM53_KPOOL_CHUNK_ELEMS"] = 1 << 17
        track(check_decode(golden, v15, dev, gen, [1500, 700, 64], 1, kpool,
                           32, D, H, select_k, "forced-chunk29"))
    finally:
        ns["_GLM53_KPOOL_CHUNK_ELEMS"] = saved_budget

    print(f"worst rel diff (asserted cases): {worst_rel:.3e} "
          f"(tolerance 1e-5)")

    if args.bench:
        if dev != "cuda":
            print("--bench requires --device cuda", file=sys.stderr)
            return 2
        print("-- prefill bench (cuda) --")
        bench_prefill(golden, v15, gen, 512, 128)
        bench_prefill(golden, v15, gen, 512, 2048)
        bench_prefill(golden, v15, gen, 512, 8192)
        bench_prefill(golden, v15, gen, 2048, 2048)
        bench_prefill(golden, v15, gen, 2048, 8192, iters=20)
        print("-- decode bench (cuda) --")
        bench_decode(golden, v15, gen, 1, 1, 2048)
        bench_decode(golden, v15, gen, 8, 1, 8192)
        bench_decode(golden, v15, gen, 32, 1, 2048)
        bench_decode(golden, v15, gen, 32, 1, 8192)
        bench_decode(golden, v15, gen, 32, 2, 8192)
        bench_decode(golden, v15, gen, 64, 1, 8192, iters=20)

    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
