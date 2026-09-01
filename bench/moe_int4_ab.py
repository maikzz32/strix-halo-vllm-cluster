#!/usr/bin/env python3
"""Final-config check: M sweep incl. MTP-verify M=6 and larger batches."""

import importlib.util
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
proto = _load_no_main("/tmp/moe_int4_gemv_proto.py", "proto")

E, GS, HID, ISH, TOPK = sw.E, sw.GS, sw.HID, sw.ISH, sw.TOPK
w13q, w2q = sw.w13q, sw.w2q
_time = sw._time
dev = "cuda"

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
    moe_align_block_size,
)

CFG13 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, SPLIT_K=1)
CFG2 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, SPLIT_K=1)
# v2 finals
V2_13 = (4, 64, 128, 4, 2)   # SPLITK, BN, BK, nw, ns
V2_2 = (2, 64, 256, 4, 1)


def vllm_call(x, wq, topk_ids, tw, mul_routed, top_k, N, cfg):
    M = topk_ids.shape[0]
    s, e, ntp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], E)
    c = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device=dev)
    invoke_fused_moe_wna16_triton_kernel(
        x, wq[0], c, wq[1], wq[2], tw.view(-1) if tw is not None else None,
        s, e, ntp, mul_routed, top_k, cfg, tl.bfloat16, False, True, [0, GS])
    return c.view(M * TOPK, N)


def proto_call(x, wq, topk_ids, tw, mul_routed, top_k, N, K, cfg):
    SPLITK, BN, BK, nw, ns = cfg
    s, e, ntp = moe_align_block_size(topk_ids, 16, E)
    return proto.moe_int4_gemv_v2(
        x, wq, topk_ids, tw.view(-1) if tw is not None else None, mul_routed,
        s, e, ntp, N, K, GS, top_k,
        SPLITK=SPLITK, BN=BN, BK=BK, num_warps=nw, num_stages=ns)


print(f'{"M":>4} {"nv13":>5} {"vllm_13":>8} {"v2_13":>8} {"vllm_2":>8} {"v2_2":>8} {"tot_v":>8} {"tot_p":>8} {"layer x":>7}')
for M in (1, 2, 6, 16, 32, 64):
    torch.manual_seed(0)
    x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)

    c1v = vllm_call(x, w13q, topk_ids, None, False, TOPK, 1024, CFG13)
    t13v = _time(lambda: vllm_call(x, w13q, topk_ids, None, False, TOPK, 1024, CFG13))
    c1p = proto_call(x, w13q, topk_ids, None, False, TOPK, 1024, HID, V2_13)
    t13p = _time(lambda: proto_call(x, w13q, topk_ids, None, False, TOPK, 1024, HID, V2_13))
    d = (c1v.float() - c1p.float()).abs()
    rel = (d / c1v.float().abs().clamp_min(1e-2)).max().item()

    act = (torch.nn.functional.silu(c1v[:, :ISH].float()) * c1v[:, ISH:].float()).to(torch.bfloat16)
    c2v = vllm_call(act, w2q, topk_ids, tw, True, 1, HID, CFG2)
    t2v = _time(lambda: vllm_call(act, w2q, topk_ids, tw, True, 1, HID, CFG2))
    c2p = proto_call(act, w2q, topk_ids, tw, True, 1, HID, ISH, V2_2)
    t2p = _time(lambda: proto_call(act, w2q, topk_ids, tw, True, 1, HID, ISH, V2_2))
    d2 = (c2v.float() - c2p.float()).abs()
    rel2 = (d2 / c2v.float().abs().clamp_min(1e-2)).max().item()

    tot_v, tot_p = t13v + t2v, t13p + t2p
    print(f"{M:>4} {M*TOPK:>5} {t13v:8.3f} {t13p:8.3f} {t2v:8.3f} {t2p:8.3f} "
          f"{tot_v:8.3f} {tot_p:8.3f} {tot_v/tot_p:6.2f}x  rel13={rel:.1e} rel2={rel2:.1e}")
    sys.stdout.flush()
