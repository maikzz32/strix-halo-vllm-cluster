#!/usr/bin/env python3
"""Patch 67: make the GDN decode kernels' launch geometry tunable on gfx1151.

vLLM v0.29.0rc1 hard-codes the Gated-DeltaNet decode kernels to
``num_warps=1, num_stages=3`` and ``BV = min(next_power_of_2(V), 32)``
(flash_linear_attention/ops/fused_recurrent.py). Those values were chosen for
CDNA/Hopper. On gfx1151 the resulting grid is tiny: with V=128 -> BV=32 -> NV=4
and HV=24 (TP2) the launch is 96 workgroups of a single wave32 each, against 160
SIMDs -- one wave per SIMD, so no latency hiding, while each wave holds
BV*BK = 32*128 fp32 state values.

This patch reads the three constants from the environment so they can be
A/B tested without rebuilding:

    VLLM_GFX1X_GDN_BV      (default 32)  - value-dim tile, lower = more workgroups
    VLLM_GFX1X_GDN_WARPS   (default 1)   - waves per workgroup
    VLLM_GFX1X_GDN_STAGES  (default 3)   - software pipelining depth

Defaults reproduce upstream behaviour exactly, so applying the patch alone
changes nothing.

Usage: python3 67_gdn_decode_geometry.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
"""
import argparse, sys
from pathlib import Path

MARKER = "gfx1151-patch: 67_gdn_decode_geometry"
REL = "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py"

HELPER = f'''
# {MARKER}
import os as _gfx_os


def _gfx_gdn_geom(default_bv=32):
    """Launch geometry for the GDN decode kernels, tunable via env on gfx1151."""
    return (
        int(_gfx_os.environ.get("VLLM_GFX1X_GDN_BV", default_bv)),
        int(_gfx_os.environ.get("VLLM_GFX1X_GDN_WARPS", 1)),
        int(_gfx_os.environ.get("VLLM_GFX1X_GDN_STAGES", 3)),
    )

'''

OLD_A = """    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)"""
NEW_A = f"""    # {MARKER}
    _gfx_bv, _gfx_warps, _gfx_stages = _gfx_gdn_geom()
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), _gfx_bv)"""

OLD_B = """    BV = min(triton.next_power_of_2(V), 32)"""
NEW_B = f"""    # {MARKER}
    _gfx_bv, _gfx_warps, _gfx_stages = _gfx_gdn_geom()
    BV = min(triton.next_power_of_2(V), _gfx_bv)"""

OLD_C = """    num_stages = 3
    num_warps = 1"""
NEW_C = """    num_stages = _gfx_stages
    num_warps = _gfx_warps"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/usr/local/lib64/python3.12/site-packages")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text(encoding="utf-8")
    if a.check:
        ok = MARKER in s
        print(("OK: patch 67 present in " if ok else "FAIL: patch 67 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 67 already applied to {f}")
        return 0
    n_a, n_b, n_c = s.count(OLD_A), s.count(OLD_B), s.count(OLD_C)
    if n_a != 1 or n_b != 1 or n_c != 2:
        print(f"ERROR: anchors found {n_a}/{n_b}/{n_c} times (want 1/1/2); re-audit patch 67", file=sys.stderr)
        return 42
    # insert the helper after the import block (first blank line after last import)
    lines = s.splitlines(keepends=True)
    last_import = max(i for i, l in enumerate(lines[:80])
                      if l.startswith("import ") or l.startswith("from "))
    s = "".join(lines[: last_import + 1]) + HELPER + "".join(lines[last_import + 1:])
    s = s.replace(OLD_A, NEW_A, 1).replace(OLD_B, NEW_B, 1).replace(OLD_C, NEW_C, 2)
    f.write_text(s, encoding="utf-8")
    print(f"OK: patch 67 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
