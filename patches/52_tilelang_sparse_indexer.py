#!/usr/bin/env python3
"""Patch 52: TileLang sparse-indexer MQA kernels for gfx1x.

Routes the paged (decode) and non-paged (prefill) FP8 MQA logits paths in
``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`` through the vendored
TileLang kernels (``gfx1x_tilelang_mqa``) on gfx1x. Measured ~47x on the
indexer kernel; the TileLang GEMM runs in BF16 because gfx1151 has no native
FP8 matrix-core dot (FP8 stays the storage format).

Sources:
  - vendored kernels: kyuz0/amd-strix-halo-vllm-toolboxes
    scripts/gfx1x_tilelang_mqa.py @ 4802c7be1fbadec96a54525526ea5111276a1480
    (adapted from AlexKGwyn/ds4-vllm-public's ds4_tl_indexer.py; the kyuz0
    adaptation adds de-shuffle of the ROCm 16x16 paged-cache layout),
    Apache-2.0.
  - dispatch logic ported from kyuz0's patch_dsv4_gfx1x.py
    (patch_sparse_indexer_mqa), with per-path env gates added.

HARD REQUIREMENTS (kept in the vendored module, restated here):
  - decode kernel threads MUST be 256 (128 gives wrong logits);
  - block_N >= 256 overflows the 64KB LDS budget on gfx1151;
  - KV-length bucketing (512 decode / 8192 prefill) is what prevents
    mid-decode JIT stalls (measured 16 -> 1.5 tok/s without it).

Gates: VLLM_GFX1X_TL_DECODE / VLLM_GFX1X_TL_PREFILL (default 1 = ON;
rocm_aiter_mla_sparse.py only serves DeepSeek-style sparse indexers). The
kernel semantics (sum_h relu(q.k) * w * scale, logits over [ks, ke)) are
DeepSeek-style-generic, but Qwen3.8 QSA and GLM-5.3 sparse-MLA indexers must
be validated per model (head count, index dim, page layout differ) before
enabling; set the env to 0 to keep the upstream path. See
patches/manifest.d/52.md.

Requires the ``tilelang`` package in the image; if it (or the vendored
module) fails to import, the dispatch falls back to the upstream path with a
printed warning. The upstream AITER/torch path is retained in all cases.

STATUS: ported, not yet validated on this hardware. Anchors verified against
vllm main @ 56058fd572f6a7fec6899385f4a4ed7f4b964477; PR-branch checkouts
with a different rocm_aiter_mla_sparse.py layout exit 42 for re-audit.

Usage:
    python3 52_tilelang_sparse_indexer.py --src /opt/vllm          # apply
    python3 52_tilelang_sparse_indexer.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 52_tilelang_sparse_indexer"
EXIT_REAUDIT = 42

REL_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
VENDOR_REL = "vllm/v1/attention/ops/gfx1x_tilelang_mqa.py"
VENDOR_SRC = Path(__file__).resolve().parent / "vendor" / "gfx1x_tilelang_mqa.py"

IMPORTS_ANCHOR = "import functools\nimport importlib\nimport math\n"

# Injected before the paged-MQA function (unique in the file).
HELPER_ANCHOR = "def rocm_fp8_paged_mqa_logits(\n"

HELPER_BLOCK = '''# {marker}
# TileLang sparse-indexer MQA kernels for gfx1x (measured ~47x on the indexer
# kernel; source: kyuz0/amd-strix-halo-vllm-toolboxes, adapted from
# AlexKGwyn/ds4-vllm-public with de-shuffle of the ROCm 16x16 paged-cache
# layout).
# HARD REQUIREMENTS (see the vendored module):
#   - decode kernel threads MUST be 256; 128 produces wrong logits.
#   - block_N >= 256 overflows the 64KB LDS budget on gfx1151.
#   - KV-length bucketing (512 decode / 8192 prefill) prevents mid-decode JIT
#     stalls (measured 16 -> 1.5 tok/s without it).
# Semantics (sum_h relu(q.k) * w * scale, logits over [ks, ke)) are
# DeepSeek-style-generic; Qwen3.8 QSA and GLM-5.3 sparse-MLA indexers must be
# validated per model (head count, index dim, page layout differ) before
# enabling. Gates: VLLM_GFX1X_TL_DECODE / VLLM_GFX1X_TL_PREFILL, default 1 =
# ON for this DeepSeek-style backend (set 0 to keep the upstream path).
# Resolved lazily: Ray applies worker env after module import.
_GFX1X_TL_MQA = {}
_GFX1X_TL_MQA_IMPORT_FAILED = False


def _gfx1x_tl_mqa_impl(kind):
    """Return the TileLang MQA function for *kind* ("decode"/"prefill").

    Returns None (caller keeps the upstream path) when the gate is off, the
    arch is not gfx1x, or tilelang / the vendored module is unavailable.
    """
    global _GFX1X_TL_MQA_IMPORT_FAILED
    if _GFX1X_TL_MQA_IMPORT_FAILED:
        return None
    env = ("VLLM_GFX1X_TL_DECODE" if kind == "decode"
           else "VLLM_GFX1X_TL_PREFILL")
    if os.environ.get(env, "1") != "1":
        return None
    try:
        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return None
    except Exception:
        return None
    if not _GFX1X_TL_MQA:
        try:
            from vllm.v1.attention.ops import gfx1x_tilelang_mqa as _tl

            _GFX1X_TL_MQA["decode"] = _tl.fp8_paged_mqa_logits_tilelang
            _GFX1X_TL_MQA["prefill"] = _tl.fp8_mqa_logits_tilelang
            print("[gfx1x_tl] TileLang sparse-indexer MQA active", flush=True)
        except Exception as exc:
            _GFX1X_TL_MQA_IMPORT_FAILED = True
            print(
                f"[gfx1x_tl] unavailable; using upstream MQA logits: {exc}",
                flush=True,
            )
            return None
    return _GFX1X_TL_MQA[kind]


'''.replace("{marker}", MARKER)

# Dispatch anchors, verbatim from vllm main @
# 56058fd572f6a7fec6899385f4a4ed7f4b964477. If a PR branch reshapes these,
# the patch exits 42 for re-audit.
PAGED_OLD = "    aiter_paged_mqa_logits_module = None\n"
PAGED_NEW = """    _tl_paged = _gfx1x_tl_mqa_impl("decode")
    if _tl_paged is not None:
        # {marker}: TileLang paged MQA (schedule_metadata unused by it)
        return _tl_paged(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    aiter_paged_mqa_logits_module = None
""".replace("{marker}", MARKER)

PREFILL_OLD = "    aiter_mqa_logits_module = None\n"
PREFILL_NEW = """    _tl_prefill = _gfx1x_tl_mqa_impl("prefill")
    if _tl_prefill is not None:
        # {marker}: TileLang non-paged MQA
        return _tl_prefill(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)

    aiter_mqa_logits_module = None
""".replace("{marker}", MARKER)


def find_target(src: Path) -> Path | None:
    cand = src / REL_PATH
    if cand.is_file():
        return cand
    matches = sorted(src.rglob("rocm_aiter_mla_sparse.py"))
    return matches[0] if matches else None


def replace_once(content: str, old: str, new: str, description: str,
                 target: Path) -> str:
    count = content.count(old)
    if count != 1:
        print(f"ERROR: {description}: expected one anchor in {target}, "
              f"found {count}. Upstream reshaped rocm_aiter_mla_sparse.py; "
              f"re-audit this patch.", file=sys.stderr)
        sys.exit(EXIT_REAUDIT)
    return content.replace(old, new, 1)


def ensure_import_os(content: str, target: Path) -> str:
    if re.search(r"^import os$", content, flags=re.MULTILINE):
        return content
    return replace_once(content, IMPORTS_ANCHOR, IMPORTS_ANCHOR + "import os\n",
                        "stdlib import block", target)


def vendor_installed(src: Path) -> bool:
    dest = src / VENDOR_REL
    return dest.is_file() and dest.read_bytes() == VENDOR_SRC.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    if not VENDOR_SRC.is_file():
        print(f"ERROR: vendored source {VENDOR_SRC} missing from the repo.",
              file=sys.stderr)
        return 1

    target = find_target(src)
    if target is None:
        print(f"ERROR: {REL_PATH} not found under {src}. The PR branch being "
              f"built has a different sparse-indexer layout; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT

    content = target.read_text()

    if args.check:
        if MARKER in content and vendor_installed(src):
            print(f"OK: patch 52 present in {target} and {VENDOR_REL} installed")
            return 0
        print(f"FAIL: patch 52 incomplete (marker: {MARKER in content}, "
              f"vendor file installed: {vendor_installed(src)})", file=sys.stderr)
        return 1

    # (a) install the vendored kernel module (idempotent).
    dest = src / VENDOR_REL
    if vendor_installed(src):
        print(f"SKIP: {VENDOR_REL} already installed and identical")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(VENDOR_SRC, dest)
        print(f"OK: installed {VENDOR_REL}")

    # (b) rewire the MQA logits dispatch (idempotent via marker).
    if MARKER in content:
        print(f"SKIP: patch 52 already applied to {target}")
        return 0

    content = ensure_import_os(content, target)
    content = replace_once(content, HELPER_ANCHOR,
                           HELPER_BLOCK + HELPER_ANCHOR,
                           "TileLang helper insertion point", target)
    content = replace_once(content, PAGED_OLD, PAGED_NEW,
                           "paged gfx1x dispatch", target)
    content = replace_once(content, PREFILL_OLD, PREFILL_NEW,
                           "prefill gfx1x dispatch", target)

    target.write_text(content)
    print(f"OK: patch 52 applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
