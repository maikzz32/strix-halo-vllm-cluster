#!/usr/bin/env python3
"""Patch 53: cached-BF16 W8A8 skinny-GEMM decode path for gfx1151.

Ported from kyuz0/amd-strix-halo-vllm-toolboxes (scripts/patch_dsv4_gfx1x.py,
patchers `patch_block_scaled_fp8_linear` and `patch_cached_bf16_w8a8_linear`)
and AlexKGwyn/ds4-vllm-public. gfx1151 (RDNA 3.5) has no native FP8 matrix
cores, so block-scaled FP8 GEMMs are slow. Two complementary mechanisms:

1. vllm/model_executor/layers/quantization/utils/fp8_utils.py: add a
   USE_BF16_DOT constexpr to the Triton block-FP8 kernel; on gfx1x the
   operands are cast to BF16 before tl.dot (WMMA supports BF16 natively).
2. vllm/model_executor/kernels/linear/scaled_mm/triton.py: override
   apply_weights on the Triton FP8 block-scaled kernel to route small-M
   decode through a cached dequantized BF16 weight copy and vLLM's ROCm
   unquantized dispatcher (hipblaslt wvSplitK skinny GEMM at M<=5, ~24x vs
   the untuned Triton block-scaled GEMM at M=1). Installs the vendored
   helper patches/vendor/gfx1x_w8a8_bf16.py into the same package.

Env knobs (resolved lazily - Ray applies worker env after import):
  VLLM_GFX1X_W8A8_BF16=1          master switch for the cached-BF16 path
  VLLM_GFX1X_W8A8_BF16_DIRECT=1   skip the activation FP8 quantize/dequantize
                                  round trip (BF16 activation in, BF16 GEMM)
  VLLM_GFX1X_W8A8_BF16_PREFILL=0  disable BF16 cache use for large-M prefill
                                  (consumed by the vendored helper; default 1)
  VLLM_GFX1X_W8A8_BF16_MIN_TP=N   minimum tensor-parallel size (default 2).
                                  The BF16 weight duplicate costs memory;
                                  kyuz0 enables this only for TP2 profiles
                                  because TP1 exceeds memory headroom.

STATUS: expected-to-need-adjustment. Anchors verified against vLLM dev tag
v0.28.1rc0 (79651d6). Note the upstream layout difference vs. kyuz0's PR
base: the block-scaled FP8 kernel is TritonFp8BlockScaledMMKernel and its
apply_weights lives in the Fp8BlockScaledMMLinearKernel parent class.

Usage:
    python3 53_w8a8_bf16_skinny.py --src /opt/vllm          # apply
    python3 53_w8a8_bf16_skinny.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 53_w8a8_bf16_skinny"
EXIT_REAUDIT = 42

FP8_UTILS_REL = "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
SCALED_MM_REL = "vllm/model_executor/kernels/linear/scaled_mm/triton.py"
VENDOR_NAME = "gfx1x_w8a8_bf16.py"
VENDOR_DEST_REL = f"vllm/model_executor/kernels/linear/scaled_mm/{VENDOR_NAME}"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    """Anchor-based single replacement; raises KeyError if not exactly one."""
    count = source.count(old)
    if count != 1:
        raise KeyError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def find_target(src: Path, rel: str, name_hint: str) -> Path | None:
    cand = src / rel
    if cand.is_file():
        return cand
    matches = sorted(p for p in src.rglob(name_hint) if "site-packages" not in str(p))
    return matches[0] if matches else None


# --- Part A: BF16 tl.dot in the block-scaled FP8 Triton kernel -------------

def patch_fp8_utils(path: Path) -> str:
    """Return patched content; KeyError propagates as exit 42."""
    source = path.read_text()
    if MARKER in source:
        return source

    # A1: extra constexpr on the block-scaled FP8 GEMM kernel.
    source = replace_once(
        source,
        "    # Meta-parameters\n"
        "    BLOCK_SIZE_M: tl.constexpr,\n",
        "    # Meta-parameters\n"
        f"    USE_BF16_DOT: tl.constexpr,  # {MARKER}\n"
        "    BLOCK_SIZE_M: tl.constexpr,\n",
        "block-FP8 kernel constexpr",
    )
    # A2: BF16 dot on gfx1x (no native FP8 matrix cores on RDNA 3.5).
    source = replace_once(
        source,
        "        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n",
        f"        # {MARKER}: gfx1x has no native FP8 matrix-core dot;\n"
        "        # cast the operands to BF16 for tl.dot instead.\n"
        "        if USE_BF16_DOT:\n"
        "            dot = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))\n"
        "        else:\n"
        "            dot = tl.dot(a, b)\n"
        "        accumulator += dot * a_s[:, None] * b_s[None, :]\n",
        "block-FP8 dot implementation",
    )
    # A3: resolve the flag at call time (current_platform is module-level here).
    source = replace_once(
        source,
        "    assert len(block_size) == 2\n",
        f"    # {MARKER}\n"
        "    _gfx1x_bf16_dot = False\n"
        "    if current_platform.is_rocm():\n"
        "        try:\n"
        "            from vllm.platforms.rocm import on_gfx1x\n\n"
        "            _gfx1x_bf16_dot = on_gfx1x()\n"
        "        except Exception:\n"
        "            pass\n"
        "    assert len(block_size) == 2\n",
        "gfx helper import",
    )
    # A4: pass the flag into the kernel launch config.
    source = replace_once(
        source,
        "    def grid(META):\n"
        "        return (\n",
        "    config = dict(config)\n"
        '    config["USE_BF16_DOT"] = _gfx1x_bf16_dot\n\n'
        "    def grid(META):\n"
        "        return (\n",
        "block-FP8 gfx1x dispatch",
    )
    return source


# --- Part B: cached-BF16 apply_weights override on the Triton kernel -------

ENV_HELPER = '''
# {marker}
def _gfx1x_w8a8_bf16_flags() -> "tuple[bool, bool]":
    """Lazily resolve the gfx1x cached-BF16 W8A8 decode path.

    Returns (enabled, direct). Lazy by design: Ray applies worker env vars
    only after import time, so module-level latching would miss them.

    The cached BF16 weight duplicate costs memory, so the path is TP-aware:
    it activates only when tensor parallelism >= VLLM_GFX1X_W8A8_BF16_MIN_TP
    (default 2; kyuz0 enables it only for TP2 profiles because TP1 exceeds
    memory headroom).
    """
    import os

    if os.environ.get("VLLM_GFX1X_W8A8_BF16", "0") != "1":
        return False, False
    if not current_platform.is_rocm():
        return False, False
    from vllm.platforms.rocm import on_gfx1x

    if not on_gfx1x():
        return False, False
    try:
        min_tp = int(os.environ.get("VLLM_GFX1X_W8A8_BF16_MIN_TP", "2"))
    except ValueError:
        min_tp = 2
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
    except Exception:
        tp = 1  # TP state not initialized yet: fail safe (path stays off)
    if tp < min_tp:
        return False, False
    return True, os.environ.get("VLLM_GFX1X_W8A8_BF16_DIRECT", "0") == "1"


_GFX1X_W8A8_DIAGNOSTIC = [False, False]

'''.replace("{marker}", MARKER)

APPLY_WEIGHTS_OVERRIDE = '''    # {marker}
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        w8a8_bf16, w8a8_bf16_direct = _gfx1x_w8a8_bf16_flags()
        if w8a8_bf16_direct and x.dtype == torch.bfloat16:
            from .gfx1x_w8a8_bf16 import w8a8_block_bf16_direct

            params = self._get_layer_params(layer)
            weight_scale = (
                params.weight_scale
                if params.weight_scale_inv is None
                else params.weight_scale_inv
            )
            input_2d = x.view(-1, x.shape[-1])
            output = w8a8_block_bf16_direct(
                input_2d,
                params.weight,
                weight_scale,
                list(self.weight_group_shape),
            )
            if output is not None:
                if not _GFX1X_W8A8_DIAGNOSTIC[0]:
                    _GFX1X_W8A8_DIAGNOSTIC[0] = True
                    print(
                        f"[gfx1x_w8a8] BF16 direct path active "
                        f"(M={input_2d.shape[0]}, N={params.weight.shape[0]})",
                        flush=True,
                    )
                if bias is not None:
                    output = output + bias
                output_shape = [*x.shape[:-1], params.weight.shape[0]]
                return output.to(dtype=self.config.out_dtype).view(*output_shape)

        if w8a8_bf16_direct and not _GFX1X_W8A8_DIAGNOSTIC[1]:
            _GFX1X_W8A8_DIAGNOSTIC[1] = True
            rows = x.numel() // x.shape[-1]
            print(
                f"[gfx1x_w8a8] BF16 direct path deferred (M={rows})",
                flush=True,
            )
        return super().apply_weights(layer, x, bias, **kwargs)

'''.replace("{marker}", MARKER)

QUANTIZED_FALLBACK = '''    # {marker}
    if _gfx1x_w8a8_bf16_flags()[0]:
        from .gfx1x_w8a8_bf16 import w8a8_block_fp8_bf16

        output = w8a8_block_fp8_bf16(
            qx, weight, x_scale, weight_scale, block_size, output_dtype
        )
        if output is not None:
            return output

'''.replace("{marker}", MARKER)


def patch_scaled_mm_triton(path: Path) -> str:
    source = path.read_text()
    if MARKER in source:
        return source

    # B1: lazy env helper before the first kernel class.
    source = replace_once(
        source,
        "from .ScaledMMLinearKernel import (\n"
        "    Int8ScaledMMLinearLayerConfig,\n"
        ")\n\n\n"
        "class TritonInt8ScaledMMLinearKernel",
        "from .ScaledMMLinearKernel import (\n"
        "    Int8ScaledMMLinearLayerConfig,\n"
        ")\n\n"
        + ENV_HELPER +
        "class TritonInt8ScaledMMLinearKernel",
        "cached-BF16 environment policy",
    )
    # B2: apply_weights override on TritonFp8BlockScaledMMKernel (inserted
    # before its apply_block_scaled_mm; the inherited apply_weights lives in
    # Fp8BlockScaledMMLinearKernel and is what super() reaches).
    source = replace_once(
        source,
        "    def apply_block_scaled_mm(\n"
        "        self,\n"
        "        A: torch.Tensor,\n",
        APPLY_WEIGHTS_OVERRIDE +
        "    def apply_block_scaled_mm(\n"
        "        self,\n"
        "        A: torch.Tensor,\n",
        "cached-BF16 direct linear override",
    )
    # B3: cached-BF16 fallback when the activation was already quantized.
    source = replace_once(
        source,
        "    from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
        "        w8a8_triton_block_scaled_mm,\n"
        "    )\n\n"
        "    return w8a8_triton_block_scaled_mm(\n",
        QUANTIZED_FALLBACK +
        "    from vllm.model_executor.layers.quantization.utils.fp8_utils import (\n"
        "        w8a8_triton_block_scaled_mm,\n"
        "    )\n\n"
        "    return w8a8_triton_block_scaled_mm(\n",
        "cached-BF16 quantized-input fallback",
    )
    return source


# --- Part C: install the vendored helper module ----------------------------

def install_vendor_module(src: Path) -> Path | None:
    """Copy patches/vendor/gfx1x_w8a8_bf16.py next to the patched triton.py."""
    vendor_src = Path(__file__).resolve().parent / "vendor" / VENDOR_NAME
    if not vendor_src.is_file():
        print(f"ERROR: vendored {VENDOR_NAME} not found at {vendor_src}; "
              f"the patches/ tree is incomplete.", file=sys.stderr)
        return None
    dest_dir = src / "vllm/model_executor/kernels/linear/scaled_mm"
    if not dest_dir.is_dir():
        matches = sorted(p.parent for p in src.rglob("ScaledMMLinearKernel.py"))
        if not matches:
            return None
        dest_dir = matches[0]
    dest = dest_dir / VENDOR_NAME
    content = vendor_src.read_text()
    install_marker = f"\n# {MARKER}: installed vendor module\n"
    if dest.is_file() and MARKER in dest.read_text(errors="ignore"):
        print(f"SKIP: vendor module already installed at {dest}")
        return dest
    dest.write_text(content.rstrip("\n") + install_marker)
    print(f"OK: installed vendor module {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    fp8_utils = find_target(src, FP8_UTILS_REL, "fp8_utils.py")
    scaled_mm = find_target(src, SCALED_MM_REL, "triton.py")
    vendor_dest = src / VENDOR_DEST_REL
    for path, rel in ((fp8_utils, FP8_UTILS_REL), (scaled_mm, SCALED_MM_REL)):
        if path is None:
            print(f"ERROR: {rel} not found under {src}. Upstream moved the "
                  f"W8A8 kernels; re-audit this patch.", file=sys.stderr)
            return EXIT_REAUDIT

    if args.check:
        ok = True
        for path, name in ((fp8_utils, "fp8_utils"), (scaled_mm, "scaled_mm triton"),
                           (vendor_dest, "vendor module")):
            if path.is_file() and MARKER in path.read_text(errors="ignore"):
                print(f"OK: patch 53 present in {path}")
            else:
                print(f"FAIL: patch 53 marker missing in {name} ({path})",
                      file=sys.stderr)
                ok = False
        return 0 if ok else 1

    changed = False
    for path, patcher, name in ((fp8_utils, patch_fp8_utils, "fp8_utils"),
                                (scaled_mm, patch_scaled_mm_triton,
                                 "scaled_mm triton")):
        content = path.read_text()
        if MARKER in content:
            print(f"SKIP: patch 53 already applied to {path} ({name})")
            continue
        try:
            patched = patcher(path)
        except KeyError as exc:
            print(f"ERROR: {exc} in {path}. Upstream restructured the file; "
                  f"re-audit this patch.", file=sys.stderr)
            return EXIT_REAUDIT
        if patched != content:
            path.write_text(patched)
            changed = True
            print(f"OK: patch 53 applied to {path} ({name})")

    if install_vendor_module(src) is None:
        print(f"ERROR: could not install vendor module under {src}; "
              f"re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    if not changed:
        print("SKIP: patch 53 source edits already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
