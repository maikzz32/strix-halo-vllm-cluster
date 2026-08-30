#!/usr/bin/env python3
"""Patch 51: deterministic radix top-k for the gfx1x sparse indexer.

Routes the ``top_k_per_row_prefill`` / ``top_k_per_row_decode`` calls in
``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`` through the vendored
``gfx1x_radix_topk.select_topk`` (Triton radix-select over integer histograms,
deterministic, ascending output, no shape-dependent JIT key). Measured
2.2-8.6x vs the two-sort torch path on gfx1151.

Sources:
  - vendored kernel: kyuz0/amd-strix-halo-vllm-toolboxes
    scripts/gfx1x_radix_topk.py @ 614c91789e4609601e618a8d967390c04896acab
    (itself adapted from AlexKGwyn/ds4-vllm-public @
    95c45bb94f324fcf3f58ec1f5eaf2d1aaceb87ff), Apache-2.0.
  - dispatch logic ported from kyuz0's patch_dsv4_gfx1x.py
    (patch_sparse_indexer_topk), adapted to current upstream: the
    ``torch.ops._C.top_k_per_row_*`` calls now live in the ``else:`` branch
    of the AITER top-k dispatch. On gfx1x the AITER master toggle is gated
    off (patch 40), so this else branch is where decode/prefill land; if a
    future build enables an AITER top-k kernel on gfx1x, that path takes
    precedence over this patch.

Gate: VLLM_GFX1X_RADIX_TOPK (default 1 = ON; rocm_aiter_mla_sparse.py only
serves DeepSeek-style sparse indexers). Qwen3.8 QSA and GLM-5.3 sparse-MLA
indexers must be validated per model before enabling; set the env to 0 to
keep the upstream path. See patches/manifest.d/51.md.

The patch (a) copies patches/vendor/gfx1x_radix_topk.py into the vLLM tree
and (b) rewrites the two top-k call sites. The upstream torch.ops call is
retained as fallback whenever the gate is off, the arch is not gfx1x, or the
vendored module fails to import.

STATUS: ported, not yet validated on this hardware. Anchors verified against
vllm main @ 56058fd572f6a7fec6899385f4a4ed7f4b964477; PR-branch checkouts
with a different rocm_aiter_mla_sparse.py layout exit 42 for re-audit.

Usage:
    python3 51_radix_topk.py --src /opt/vllm          # apply
    python3 51_radix_topk.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 51_radix_topk"
EXIT_REAUDIT = 42

REL_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"
VENDOR_REL = "vllm/v1/attention/ops/gfx1x_radix_topk.py"
VENDOR_SRC = Path(__file__).resolve().parent / "vendor" / "gfx1x_radix_topk.py"

IMPORTS_ANCHOR = "import functools\nimport importlib\nimport math\n"

# Injected before the fake-registration function (unique in the file).
DISPATCHER_ANCHOR = "def rocm_aiter_sparse_attn_indexer_fake(\n"

DISPATCHER_BLOCK = '''# {marker}
# Deterministic radix top-k for the gfx1x sparse indexer (measured 2.2-8.6x
# vs the two-sort torch path; source: kyuz0/amd-strix-halo-vllm-toolboxes,
# adapted from AlexKGwyn/ds4-vllm-public).
# Semantics (top-k over [ks, ke) of the indexer logits, ascending, -1 padded)
# are DeepSeek-style-generic; Qwen3.8 QSA and GLM-5.3 sparse-MLA indexers
# must be validated per model (head count / index dim / page layout differ)
# before enabling. Gate: VLLM_GFX1X_RADIX_TOPK, default 1 = ON for this
# DeepSeek-style backend (set 0 to keep the upstream path). Resolved lazily:
# Ray applies worker env after module import.
_GFX1X_RADIX_TOPK_IMPL = None
_GFX1X_RADIX_TOPK_IMPORT_FAILED = False


def _gfx1x_radix_topk_impl():
    global _GFX1X_RADIX_TOPK_IMPL, _GFX1X_RADIX_TOPK_IMPORT_FAILED
    if _GFX1X_RADIX_TOPK_IMPORT_FAILED:
        return None
    if os.environ.get("VLLM_GFX1X_RADIX_TOPK", "1") != "1":
        return None
    try:
        from vllm.platforms.rocm import on_gfx1x

        if not on_gfx1x():
            return None
    except Exception:
        return None
    if _GFX1X_RADIX_TOPK_IMPL is None:
        try:
            from vllm.v1.attention.ops.gfx1x_radix_topk import select_topk

            _GFX1X_RADIX_TOPK_IMPL = select_topk
            print("[gfx1x_topk] deterministic radix path active", flush=True)
        except Exception as exc:
            _GFX1X_RADIX_TOPK_IMPORT_FAILED = True
            print(
                f"[gfx1x_topk] unavailable; using upstream top-k: {exc}",
                flush=True,
            )
    return _GFX1X_RADIX_TOPK_IMPL


def _gfx1x_radix_topk(
    logits, topk_tokens, row_starts=None, row_ends=None, out=None
) -> bool:
    impl = _gfx1x_radix_topk_impl()
    if impl is None:
        return False
    impl(
        logits,
        topk_tokens,
        row_starts=row_starts,
        row_ends=row_ends,
        out=out,
    )
    return True


'''.replace("{marker}", MARKER)

# Call-site anchors, verbatim from vllm main @
# 56058fd572f6a7fec6899385f4a4ed7f4b964477 (else-branch of the AITER top-k
# dispatch). If a PR branch reshapes these, the patch exits 42 for re-audit.
PREFILL_OLD = """            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
"""
PREFILL_NEW = """            else:
                if not _gfx1x_radix_topk(
                    logits,
                    topk_tokens,
                    row_starts=chunk.cu_seqlen_ks,
                    row_ends=chunk.cu_seqlen_ke,
                    out=topk_indices,
                ):
                    torch.ops._C.top_k_per_row_prefill(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_indices,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )
"""

DECODE_OLD = """        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
"""
DECODE_NEW = """        else:
            if not _gfx1x_radix_topk(
                logits, topk_tokens, out=topk_indices
            ):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode_metadata.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
"""


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
            print(f"OK: patch 51 present in {target} and {VENDOR_REL} installed")
            return 0
        print(f"FAIL: patch 51 incomplete (marker: {MARKER in content}, "
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

    # (b) rewire the top-k call sites (idempotent via marker).
    if MARKER in content:
        print(f"SKIP: patch 51 already applied to {target}")
        return 0

    content = ensure_import_os(content, target)
    content = replace_once(content, DISPATCHER_ANCHOR,
                           DISPATCHER_BLOCK + DISPATCHER_ANCHOR,
                           "radix top-k dispatcher insertion point", target)
    content = replace_once(content, PREFILL_OLD, PREFILL_NEW,
                           "prefill radix top-k dispatch", target)
    content = replace_once(content, DECODE_OLD, DECODE_NEW,
                           "decode radix top-k dispatch", target)

    target.write_text(content)
    print(f"OK: patch 51 applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
