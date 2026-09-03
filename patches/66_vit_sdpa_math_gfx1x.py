#!/usr/bin/env python3
"""Patch 66: pin the vision encoder's torch SDPA to the MATH backend on gfx1x.

Enabling images for Qwen3.8-Flash-Next on the gfx1151 cluster (2026-09-03)
killed the encoder profiling run with

    torch.AcceleratorError: CUDA error: invalid argument
      vit_attn_wrappers.py:284 torch_sdpa_wrapper -> apply_sdpa

Isolated in the ray-head container (torch 2.13.0+rocm10.0.0, HIP 7.15):
every SDPA backend works on its own (MATH / EFFICIENT / FLASH, L=256..4096),
but once a fused (aotriton) SDPA call has run, the NEXT MATH call in the same
process fails with that error and the HIP context stays poisoned. With
torch.backends.cuda.enable_flash_sdp(False) + enable_mem_efficient_sdp(False)
the same sequence (default, math, matmul, 50x default) runs clean. The ViT
path mixes backends across its calls, so it trips over this every time.

Fix: on ROCm gfx1x run apply_sdpa() under sdpa_kernel(SDPBackend.MATH). The
ViT is a few hundred tokens per image; MATH is fast enough there. The text
model is unaffected (QSA/GDN are Triton kernels, not torch SDPA).

Gate: VLLM_GFX1X_VIT_SDPA_MATH=1 (default) / 0 = upstream behaviour, read
lazily per call.

Usage: python3 66_vit_sdpa_math_gfx1x.py --src <site-packages|/opt/vllm> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse, sys
from pathlib import Path

MARKER = "gfx1151-patch: 66_vit_sdpa_math_gfx1x"
REL = "vllm/v1/attention/ops/vit_attn_wrappers.py"
OLD = '''    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output
'''
NEW = f'''    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
    # {MARKER}
    # On gfx1151 a MATH SDPA call issued after a fused (aotriton) one fails with
    # "CUDA error: invalid argument" and poisons the HIP context; the ViT mixes
    # backends across calls. Pin the encoder to MATH (env gate, lazy).
    import os

    if current_platform.is_rocm() and os.environ.get(
        "VLLM_GFX1X_VIT_SDPA_MATH", "1"
    ) != "0":
        from vllm.platforms.rocm import on_gfx1x

        if on_gfx1x():
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                output = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
                )
            return einops.rearrange(output, "b h s d -> b s h d ")
    output = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    output = einops.rearrange(output, "b h s d -> b s h d ")
    return output
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/opt/vllm")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text()
    if a.check:
        ok = MARKER in s
        print(("OK: patch 66 present in " if ok else "FAIL: patch 66 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 66 already applied to {f}")
        return 0
    if s.count(OLD) != 1:
        print(f"ERROR: apply_sdpa anchor found {s.count(OLD)} times in {f}; re-audit patch 66", file=sys.stderr)
        return 42
    if "current_platform" not in s:
        print(f"ERROR: {f} no longer imports current_platform; re-audit patch 66", file=sys.stderr)
        return 42
    f.write_text(s.replace(OLD, NEW, 1))
    print(f"OK: patch 66 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
