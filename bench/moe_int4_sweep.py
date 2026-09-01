#!/usr/bin/env python3
"""Standalone sweep of the WNA16 (int4 gptq/awq) fused-MoE Triton kernel at
GLM-5.3-Flash tp4 decode shapes: E=288 (all experts local at tp4), per-expert
intermediate shard 512, hidden 4096, group 32, asymmetric zp, top_k=8.

Runtime layouts (from convert_to_wna16_moe_kernel_format TRITON branch,
compressed-tensors source):
  w13 uint8 [E, 1024, 2048] (N-first, 2 nibbles/byte along K=4096, LSB-first)
  w13_scale bf16 [E, 1024, 128] (K/32 groups)
  w13_zp uint8 [E, 512, 128] (2 nibbles/byte along N; byte n//2 holds channel
    n (low nibble) and n+1 (high))
  w2  uint8 [E, 4096, 256]  (K=512)
  w2_scale bf16 [E, 4096, 16]
  w2_zp uint8 [E, 2048, 16]
"""

import sys

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_wna16_triton_kernel,
    moe_align_block_size,
)

dev = "cuda"
E, HID, ISH, GS, TOPK = 288, 4096, 512, 32, 8
torch.manual_seed(0)


def pack_gptq_uint8(w_bf16, gs):
    """[E, N, K] bf16 -> (uint8 [E, N, K//2], scale bf16 [E, N, K//gs],
    zp uint8 [E, N//2, K//gs]) — GPTQ LSB-first along K, asymmetric."""
    E, N, K = w_bf16.shape
    g = K // gs
    wg = w_bf16.float().reshape(E, N, g, gs)
    mn = wg.amin(-1, keepdim=True)
    mx = wg.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp_min(1e-6)
    zp = torch.round(-mn / scale).clamp(0, 15)
    q = torch.round(wg / scale + zp).clamp(0, 15).to(torch.uint8)
    q = q.reshape(E, N, K)
    q8 = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).contiguous()  # [E, N, K//2]
    zp4 = zp.reshape(E, N, g).to(torch.uint8)
    zp8 = (zp4[:, 0::2, :] | (zp4[:, 1::2, :] << 4)).contiguous()  # [E, N//2, g]
    return q8, scale.reshape(E, N, g).to(torch.bfloat16).contiguous(), zp8


def dequant_ref(q8, scale, zp8, gs):
    """Torch reference: unpack -> (nibble - zp) * scale -> bf16 -> float.
    Returns [E, N, K] fp32."""
    E, N, K2 = q8.shape
    K = K2 * 2
    q = torch.empty(E, N, K, dtype=torch.int16, device=q8.device)
    q[:, :, 0::2] = (q8 & 0xF).to(torch.int16)
    q[:, :, 1::2] = (q8 >> 4).to(torch.int16)
    zpn = torch.empty(E, N, K // gs, dtype=torch.int16, device=q8.device)
    zpn[:, 0::2, :] = (zp8 & 0xF).to(torch.int16)
    zpn[:, 1::2, :] = (zp8 >> 4).to(torch.int16)
    w = (q - zpn.repeat_interleave(gs, dim=2)).float() * scale.float().repeat_interleave(gs, dim=2)
    return w.to(torch.bfloat16).float()


w13 = (torch.randn(E, 1024, HID, device=dev) * 0.05).to(torch.bfloat16)
w2 = (torch.randn(E, HID, ISH, device=dev) * 0.05).to(torch.bfloat16)
w13q = pack_gptq_uint8(w13, GS)
w2q = pack_gptq_uint8(w2, GS)


def run_call(x, wq, topk_ids, topk_weights, mul_routed, config, N):
    M = x.shape[0]
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )
    c = torch.zeros(M, TOPK, N, dtype=torch.bfloat16, device=dev)
    invoke_fused_moe_wna16_triton_kernel(
        x,
        wq[0],
        c,
        wq[1],
        wq[2],
        topk_weights if mul_routed else None,
        sorted_ids,
        expert_ids,
        ntp,
        mul_routed,
        TOPK,
        config,
        tl.bfloat16,
        False,
        True,
        [0, GS],
    )
    return c


def _time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e1 = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        e0[i].record()
        fn()
        e1[i].record()
    torch.cuda.synchronize()
    ts = sorted(e0[i].elapsed_time(e1[i]) for i in range(iters))
    return ts[len(ts) // 2]


# --- numerics A/B (M=4, current default configs) vs fp32 dequant reference ---
M = 4
x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
cfg_def13 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, SPLIT_K=1)
cfg_def2 = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=1, SPLIT_K=1)
c1 = run_call(x, w13q, topk_ids, None, False, cfg_def13, 1024)
w13_ref = dequant_ref(*w13q, GS)  # [E, 1024, 4096] fp32
max_rel = 0.0
max_abs = 0.0
for t in range(M):
    for k in range(TOPK):
        e = int(topk_ids[t, k])
        ref = x[t].float() @ w13_ref[e].t()
        got = c1[t, k].float()
        max_abs = max(max_abs, (ref - got).abs().max().item())
        max_rel = max(max_rel, ((ref - got).abs() / ref.abs().clamp_min(1e-3)).max().item())
print(f"numerics w13 (M=4, default cfg): max_abs={max_abs:.3e} max_rel={max_rel:.3e}")

# --- sweep ---
bytes13 = 8 * (1024 * HID // 2 + 1024 * 128 * 2 + 512 * 128)  # 8 active experts
bytes2 = 8 * (HID * ISH // 2 + HID * 16 * 2 + 2048 * 16)
print(f"roofline refs: w13 {bytes13/1e6:.1f}MB, w2 {bytes2/1e6:.1f}MB per call (8 experts)")
print(f'{"M":>3} {"call":>4} {"cfg":>28} {"ms":>8} {"GB/s":>7}')
for M in (1, 4, 8, 32):
    x = torch.randn(M, HID, device=dev).to(torch.bfloat16)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, TOPK, device=dev).softmax(-1)
    for tag, N, wq, byt, dcfg in (
        ("w13", 1024, w13q, bytes13, cfg_def13),
        ("w2", HID, w2q, bytes2, cfg_def2),
    ):
        results = []
        for BN in (32, 64, 128):
            if BN > N:
                continue
            for BK in (64, 128):
                for nw in (4, 8):
                    for ns in (1, 2):
                        cfg = dict(
                            BLOCK_SIZE_M=16, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
                            GROUP_SIZE_M=1, SPLIT_K=1, num_warps=nw, num_stages=ns,
                        )
                        try:
                            t = _time(lambda: run_call(x, wq, topk_ids, tw, True, cfg, N),
                                      iters=20, warmup=3)
                            results.append((t, BN, BK, nw, ns))
                        except Exception:
                            pass
        results.sort()
        tb = _time(lambda: run_call(x, wq, topk_ids, tw, True, dict(dcfg), N),
                   iters=20, warmup=3)
        print(f"{M:>3} {tag:>4} {'DEFAULT':>28} {tb:8.3f} {byt/tb/1e6:7.0f}")
        for t, BN, BK, nw, ns in results[:4]:
            print(f"{M:>3} {tag:>4} BN={BN:<4}BK={BK:<4}w={nw}s={ns}{'':>3} {t:8.3f} {byt/t/1e6:7.0f}")
        sys.stdout.flush()
