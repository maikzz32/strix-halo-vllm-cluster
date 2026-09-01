#!/usr/bin/env python3
"""Minimal w2 debug: M=1, one expert, compare proto vs vllm vs fp32 ref."""

import importlib.util
import sys

import torch
import triton.language as tl

sys.path.insert(0, "/tmp")


def _load_no_main(path, name):
    src = open(path).read()
    marker = "# --- numerics A/B"
    idx = src.find(marker)
    if idx > 0:
        src = src[:idx]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


sw = _load_no_main("/tmp/moe_int4_sweep.py", "swdefs")
proto = _load_no_main("/tmp/moe_int4_gemv_proto.py", "proto")

E, GS, HID, ISH, TOPK = sw.E, sw.GS, sw.HID, sw.ISH, sw.TOPK
w2q = sw.w2q
dev = "cuda"

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
    moe_align_block_size,
)

w2_ref = sw.dequant_ref(*w2q, GS)  # [E, 4096, 512] fp32

M = 1
torch.manual_seed(7)
# act rows: M*TOPK = 8 rows of 512
act = torch.randn(M * TOPK, ISH, device=dev).to(torch.bfloat16)
topk_ids = torch.arange(TOPK, device=dev, dtype=torch.int32)[None, :]  # experts 0..7
tw = torch.ones(M, TOPK, device=dev)

CFG2 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, SPLIT_K=1)
s, e, ntp = moe_align_block_size(topk_ids, CFG2["BLOCK_SIZE_M"], E)
c_v = torch.zeros(M, TOPK, HID, dtype=torch.bfloat16, device=dev)
invoke_fused_moe_wna16_triton_kernel(
    act, w2q[0], c_v, w2q[1], w2q[2], tw.view(-1), s, e, ntp, True, 1,
    CFG2, tl.bfloat16, False, True, [0, GS])
c_v = c_v.view(M * TOPK, HID)

c_p = proto.moe_int4_gemv(act, w2q, topk_ids, tw.view(-1), True, s, e, ntp,
                          HID, ISH, GS, 1, SPLITK=4, BN=64, BK=128)

for k in (0, 3, 7):
    e_ = int(topk_ids[0, k])
    ref = act[k].float() @ w2_ref[e_].t()
    gv = c_v[k].float()
    gp = c_p[k].float()
    dv = ((ref - gv).abs() / ref.abs().clamp_min(1e-2)).max().item()
    dp = ((ref - gp).abs() / ref.abs().clamp_min(1e-2)).max().item()
    dvp = ((gv - gp).abs() / gv.abs().clamp_min(1e-2)).max().item()
    print(f"row {k} expert {e_}: vllm_vs_ref={dv:.2e} proto_vs_ref={dp:.2e} proto_vs_vllm={dvp:.2e}")
    # locate worst element of proto vs ref
    d = (ref - gp).abs()
    i = int(d.argmax())
    print(f"   worst n={i}: ref={ref[i].item():.4f} vllm={gv[i].item():.4f} proto={gp[i].item():.4f}")
    # check a few nibbles around that n for expert e_
    n = i
    q8 = w2q[0][e_]
    nib_lo = int(q8[n, 0] & 0xF)
    nib_hi = int(q8[n, 0] >> 4)
    zp8 = w2q[2][e_]
    z_lo = int(zp8[n // 2, 0] & 0xF) if n % 2 == 0 else int(zp8[n // 2, 0] >> 4)
    sc = float(w2q[1][e_][n, 0])
    print(f"   n={n}: packed byte0={int(q8[n,0]):#04x} nib(k0,k1)=({nib_lo},{nib_hi}) zp={z_lo} scale0={sc:.4f}")
