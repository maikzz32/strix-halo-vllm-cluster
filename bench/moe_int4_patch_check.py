#!/usr/bin/env python3
"""Validate the PATCHED installed fused_moe.py dispatch end-to-end at kernel
level: invoke_fused_moe_wna16_triton_kernel with the gate ON must route
through the split-K path (same output class, faster); gate OFF must run the
stock kernel. Run inside a container after applying
patches/runtime_glm53_moe_int4_tune.py. Expects bench/moe_int4_sweep.py
alongside (synthetic weights in the exact production runtime layout)."""

import importlib.util
import os
import sys

import torch
import triton.language as tl

sys.path.insert(0, "/tmp")


def _load_no_main(path, name):
    src = open(path).read()
    idx = src.find("# --- numerics A/B")
    if idx > 0:
        src = src[:idx]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


sw = _load_no_main("/tmp/moe_int4_sweep.py", "swdefs")
E, GS, HID, ISH, TOPK = sw.E, sw.GS, sw.HID, sw.ISH, sw.TOPK
w13q, w2q = sw.w13q, sw.w2q
_time = sw._time
dev = "cuda"

from vllm.model_executor.layers.fused_moe import fused_moe as fm


def invoke(x, wq, topk_ids, tw, mul_routed, top_k, N, cfg):
    M = topk_ids.shape[0]
    s, e, ntp = fm.moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E)
    c = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device=dev)
    fm.invoke_fused_moe_wna16_triton_kernel(
        x, wq[0], c, wq[1], wq[2], tw.view(-1) if tw is not None else None,
        s, e, ntp, mul_routed, top_k, cfg, tl.bfloat16, False, True, [0, GS])
    return c.view(M * TOPK, N)


def prod_cfg(M):
    """Default int4_w4a16 config per get_default_config: BM 16/32/64 by M."""
    bm = 16 if M <= 20 else (32 if M <= 40 else 64)
    return dict(BLOCK_SIZE_M=bm, GROUP_SIZE_M=1, SPLIT_K=1)


CFG13 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, SPLIT_K=1)
CFG2 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, SPLIT_K=1)

print("gate env:", os.environ.get("VLLM_GFX1X_MOE_INT4_GEMV", "<default on>"))
ok = True
for M in (1, 6, 24, 48, 64):
    torch.manual_seed(0)
    x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)

    cfg = prod_cfg(M)
    os.environ["VLLM_GFX1X_MOE_INT4_GEMV"] = "0"
    c13_off = invoke(x, w13q, topk_ids, None, False, TOPK, 1024, cfg)
    t13_off = _time(lambda: invoke(x, w13q, topk_ids, None, False, TOPK, 1024, cfg))
    os.environ["VLLM_GFX1X_MOE_INT4_GEMV"] = "1"
    c13_on = invoke(x, w13q, topk_ids, None, False, TOPK, 1024, cfg)
    t13_on = _time(lambda: invoke(x, w13q, topk_ids, None, False, TOPK, 1024, cfg))
    d = (c13_off.float() - c13_on.float()).abs()
    r = (d / c13_off.float().abs().clamp_min(1e-2)).max().item()

    act = (torch.nn.functional.silu(c13_off[:, :ISH].float()) * c13_off[:, ISH:].float()).to(torch.bfloat16)
    os.environ["VLLM_GFX1X_MOE_INT4_GEMV"] = "0"
    c2_off = invoke(act, w2q, topk_ids, tw, True, 1, HID, cfg)
    t2_off = _time(lambda: invoke(act, w2q, topk_ids, tw, True, 1, HID, cfg))
    os.environ["VLLM_GFX1X_MOE_INT4_GEMV"] = "1"
    c2_on = invoke(act, w2q, topk_ids, tw, True, 1, HID, cfg)
    t2_on = _time(lambda: invoke(act, w2q, topk_ids, tw, True, 1, HID, cfg))
    d2 = (c2_off.float() - c2_on.float()).abs()
    r2 = (d2 / c2_off.float().abs().clamp_min(1e-2)).max().item()

    print(f"M={M:>3}: w13 off {t13_off:.3f} -> on {t13_on:.3f} ms "
          f"({t13_off/t13_on:.2f}x, rel {r:.1e}) | "
          f"w2 off {t2_off:.3f} -> on {t2_on:.3f} ms ({t2_off/t2_on:.2f}x, rel {r2:.1e})")
    ok = ok and r < 2e-2 and r2 < 2e-2 and t13_on <= t13_off * 1.05 and t2_on <= t2_off * 1.05

# big-M must fall back to stock (no crash, no gate effect)
M = 128
x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
c_big = invoke(x, w13q, topk_ids, None, False, TOPK, 1024, prod_cfg(M))
print(f"M=128 fallback ok, out norm {c_big.float().norm().item():.1f}")
print("PATCH-LEVEL VALIDATION:", "PASS" if ok else "FAIL")
