#!/usr/bin/env python3
"""Patch 60: mul+sum fast path for the m=1 skinny GEMM on gfx1x.

vllm#52631: on gfx1151 the ROCm unquantized-GEMM dispatch ladder in
vllm/model_executor/layers/utils.py (rocm_unquantized_gemm_impl) covers
m > 8 (wvSplitK) and m % 4 == 0 (LLMM1), so m == 1 falls through to
torch.nn.functional.linear -> rocBLAS, which picks a pathological
Cijk_..._MT128x32x16_GSU1 kernel (one workgroup walking k=2048): 137.5
us/call. The shape is shared_expert_gate [1, 2048], hit once per layer
per token on Qwen2/3.5-MoE models (~40 calls/decode step = 5.48 ms of a
40.06 ms step, 13.7%). A [1, K] weight is a dot product; the issue's
fix, (x * w).sum(-1), measures 6.82 us (20.5x) and improved decode
40.06 -> 34.79 ms/step (24.53 -> 28.32 tok/s, +15.5%).

One edit: insert an `elif m == 1 and on_gfx1x():` branch between the
wvSplitK and LLMM1 branches of the skinny ladder, computing
(x_view * weight.reshape(-1)).sum(-1, keepdim=True). The branch sits
inside `if use_skinny:`, which already gates on fp16/bf16, contiguous
operands and (on_gfx9() or on_gfx1x()); the extra on_gfx1x() narrows it
to RDNA (gfx1*) only, so MI300-class gfx9 behavior is untouched.

Numeric caveat (from the issue): bf16 * bf16 rounds each of the K
products to bf16 before the fp32 sum, so the output is NOT bitwise
identical to F.linear (max abs diff 3.1e-02..2.5e-1 at K=2048). This is
the accumulation-noise class vLLM already accepts by shape-based kernel
selection, but output parity must be validated on hardware.

Upstream status (2026-08-30): issue open; vllm#53283 (open, unmerged)
proposes widening the wvSplitK guard to `m == 1 or m > 8` instead, which
is bitwise-cleaner. If that lands, the anchor moves; this patch detects
the widened guard (or any `elif m == 1` branch) and SKIPs cleanly
instead of double-patching.

Kill switch: VLLM_GFX1X_SKINNY_M1=0 restores the stock fallback. Env is
resolved lazily inside the impl (Ray applies worker env after import).

STATUS: expected-to-need-adjustment. Anchor verified identical on vLLM
main today and on dev tag v0.28.1rc0 (79651d6). If vllm#53283 lands or
the ladder is restructured, the anchor moves and this patch exits 42
for re-audit - then it is likely obsolete.

Usage:
    python3 60_skinny_gemm_m1.py --src /opt/vllm          # apply
    python3 60_skinny_gemm_m1.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 60_skinny_gemm_m1"
EXIT_REAUDIT = 42

UTILS_REL = "vllm/model_executor/layers/utils.py"

# Signatures of an upstream fix for vllm#52631: vllm#53283 widens the
# wvSplitK guard, or an m==1 branch appears in the ladder directly.
UPSTREAM_FIXED_PATTERNS = (
    "m == 1 or m > 8",
    "elif m == 1",
)


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise KeyError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def find_target(src: Path, rel: str) -> Path | None:
    cand = src / rel
    if cand.is_file():
        return cand
    matches = sorted(
        p for p in src.rglob("utils.py")
        if "model_executor/layers" in str(p).replace("\\", "/")
        and "site-packages" not in str(p)
    )
    return matches[0] if matches else None


def upstream_fixed(source: str) -> str | None:
    for pattern in UPSTREAM_FIXED_PATTERNS:
        if pattern in source:
            return pattern
    return None


M1_BRANCH = '''            return out.reshape(*x.shape[:-1], weight.shape[0])
        elif m == 1 and on_gfx1x():
            # {marker}
            # vllm#52631: a [1, K] weight (e.g. shared_expert_gate,
            # [1, 2048], ~40 calls per decode step on Qwen MoE models)
            # falls through the skinny ladder to a pathological rocBLAS
            # kernel (MT128x32x16 GSU1, ~140 us/call on gfx1151). A
            # [1, K] weight is a dot product: mul+sum is ~20x faster
            # (6.82 us). bf16 products round each term to bf16 before
            # the fp32 sum, so the output is not bitwise-identical to
            # F.linear (max abs diff ~2.5e-1 at K=2048). Env resolved
            # lazily: Ray applies worker env after import time.
            import os

            if os.environ.get("VLLM_GFX1X_SKINNY_M1", "1") != "0":
                out = (x_view * weight.reshape(-1)).sum(-1, keepdim=True)
                if bias is not None:
                    out = out + bias
                return out.reshape(*x.shape[:-1], weight.shape[0])
        elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:
'''.replace("{marker}", MARKER)


def patch_utils_py(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 60 already applied to {path}")
        return False
    fixed = upstream_fixed(source)
    if fixed is not None:
        print(f"SKIP: upstream already handles m==1 in {path} "
              f"(found {fixed!r}); patch 60 is obsolete, not applying.")
        return False
    source = replace_once(
        source,
        "            return out.reshape(*x.shape[:-1], weight.shape[0])\n"
        "        elif m % 4 == 0 and n == 1 and k <= 8192 and bias is None:\n",
        M1_BRANCH,
        "skinny GEMM ladder anchor (wvSplitK return -> LLMM1 elif)",
    )
    # Fail closed: never write a file that does not parse.
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as e:
        print(f"ERROR: patched {path} would not parse ({e}); aborting "
              f"without writing. Re-audit this patch.", file=sys.stderr)
        sys.exit(EXIT_REAUDIT)
    path.write_text(source)
    print(f"OK: patch 60 applied to {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    utils_py = find_target(src, UTILS_REL)
    if utils_py is None:
        print(f"ERROR: {UTILS_REL} not found under {src}. Upstream moved "
              f"the unquantized GEMM dispatch; re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT

    if args.check:
        content = utils_py.read_text(errors="ignore")
        if MARKER in content:
            print(f"OK: patch 60 present in {utils_py}")
            return 0
        fixed = upstream_fixed(content)
        if fixed is not None:
            print(f"OK: patch 60 not needed in {utils_py}: upstream "
                  f"already handles m==1 (found {fixed!r}).")
            return 0
        print(f"FAIL: patch 60 marker missing in {utils_py}",
              file=sys.stderr)
        return 1

    try:
        patch_utils_py(utils_py)
    except KeyError as exc:
        print(f"ERROR: {exc}. Upstream restructured the file; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
