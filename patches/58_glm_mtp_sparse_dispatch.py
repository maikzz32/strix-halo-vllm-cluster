#!/usr/bin/env python3
"""Patch 58: fix the GLM-5.3 MTP sparse-MLA dispatch on ROCm (vllm#53943 port).

In the GLM-5.3 tree (PR #53906 branch, ZJY0516/vllm glm-release), the BF16
Triton-lane selector is the module-level helper ``_use_rocm_sparse_triton``
in ``vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py``:

    plain_decode = num_decode_tokens == num_decodes
    return (
        not kv_cache_dtype.startswith("fp8")
        and head_size == kv_lora_rank
        and plain_decode
        and (num_prefills > 0 or (num_decodes > 0 and max_query_len == 1))
    )

i.e. the rope-free BF16 path is only taken for plain single-token decode
shapes. MTP draft steps present 2+ tokens per decode request, both shape
terms fall through, and execution lands on gfx950-only AITER asm kernels
that fault everywhere else. The helper is called from the metadata builder
(gates the persistent AITER work-metadata build) and from
``ROCMAiterMLASparseImpl._forward_mla`` — selector-true calls
``rocm_sparse_attn_prefill`` with ragged indices (polarity confirmed on the
real glm-release checkout) — so re-keying the helper fixes both call sites
consistently.

The fix (upstream: https://github.com/vllm-project/vllm/issues/53943, also
ZJY0516/vllm#5 and #7) keys the selector on geometry alone:

    not fp8 KV and head_size == kv_lora_rank

so MTP draft steps ride the ragged Triton kernel
(_rocm_sparse_attn_prefill_ragged_triton handles per-query indptr
natively). Measured on gfx950 TP=8: 2632 -> 3760 tok/s @ N=64.

Additionally this patch ORs the selector with the env knob
VLLM_GFX1X_FORCE_TRITON_SPARSE=1 (lazy, read per call): on gfx1151 every
asm/Gluon AITER lane is unreachable anyway, so the ragged Triton lane must
be force-selectable regardless of what shape/geometry heuristics upstream
adds later. The knob is a manual override, default off.

SKIP semantics (same as patch 57): if the tree has no glm5_next/Glm5Next
model package at all (e.g. a build off vLLM main without PR #53906), the
patch is not needed and reports SKIP with a note. If GLM-5.3 support IS
present but the anchors moved, exit 42 (re-audit).

STATUS: re-audited against the real glm-release checkout @ 36bb3795 (PR
#53906 head). The hypothesized inline selector in _forward_mla does not
exist there; the confirmed form is the module-level helper above (with the
``plain_decode`` indirection), and the anchors are now verbatim against
that tree. Upstream main/stable (incl. v0.28.0) has NO glm5_next model
package and no shape-keyed selector (the referenced upstream issue
vllm#53943 is still OPEN), so the patch SKIPs there — verified. Any
deviation from the anchored layout exits 42 for re-audit.

Usage:
    python3 58_glm_mtp_sparse_dispatch.py --src /opt/vllm          # apply
    python3 58_glm_mtp_sparse_dispatch.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / skipped (not needed) / check passed,
            1 = check failed / error,
            42 = GLM-5.3 present but target pattern not found (upstream
                 moved; re-audit needed).
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 58_glm_mtp_sparse_dispatch"
EXIT_REAUDIT = 42

# Primary location in the PR #53906 tree; falls back to a content search
# for the class name.
REL_PATH = "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
IMPL_CLASS = "ROCMAiterMLASparseImpl"
HELPER_FN = "def _use_rocm_sparse_triton(\n"

# The confirmed selector body on the PR #53906 glm-release branch
# (@ 36bb3795): two geometry terms ANDed with the two MTP-breaking shape
# terms (the first via the `plain_decode` indirection).
SELECTOR_OLD = """    plain_decode = num_decode_tokens == num_decodes
    return (
        not kv_cache_dtype.startswith("fp8")
        and head_size == kv_lora_rank
        and plain_decode
        and (num_prefills > 0 or (num_decodes > 0 and max_query_len == 1))
    )
"""

# Geometry-only re-key (upstream fix) ORed with the gfx1x force knob. The
# helper keeps its (now unused) shape parameters so both call sites stay
# untouched.
SELECTOR_NEW = """    # {marker}: re-keyed on geometry alone (upstream: vllm#53943).
    # The shape terms broke MTP draft steps (2+ tokens/decode request).
    return (
        not kv_cache_dtype.startswith("fp8")
        and head_size == kv_lora_rank
    ) or _gfx1x_force_triton_sparse()
""".replace("{marker}", MARKER)

HELPER_BLOCK = '''
# {marker} (upstream: vllm#53943)
def _gfx1x_force_triton_sparse():
    """VLLM_GFX1X_FORCE_TRITON_SPARSE=1 forces the ragged Triton lane.

    On gfx1151 every asm/Gluon AITER sparse-attention lane is unreachable
    (CDNA/gfx950-only), so the Triton lane must be force-selectable. Read
    lazily per call because Ray applies worker env after import.
    """
    import os
    return os.environ.get("VLLM_GFX1X_FORCE_TRITON_SPARSE", "0") == "1"


'''.replace("{marker}", MARKER)


def glm5_present(src: Path) -> bool:
    """True if the tree carries GLM-5.3 (glm5_next) support at all."""
    models_dir = src / "vllm" / "model_executor" / "models"
    if models_dir.is_dir():
        for p in models_dir.iterdir():
            if "glm5" in p.name.lower():
                return True
        registry = models_dir / "registry.py"
        if registry.is_file() and "glm5" in registry.read_text(
                errors="ignore").lower():
            return True
    configs_dir = src / "vllm" / "transformers_utils" / "configs"
    if configs_dir.is_dir():
        for p in configs_dir.iterdir():
            if "glm5" in p.name.lower():
                return True
    return False


def find_target(src: Path) -> Path | None:
    cand = src / REL_PATH
    if cand.is_file() and IMPL_CLASS in cand.read_text(errors="ignore"):
        return cand
    matches = sorted(
        p for p in src.rglob("*.py")
        if IMPL_CLASS in p.read_text(errors="ignore")
        and "_forward_mla" in p.read_text(errors="ignore"))
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    # Presence check FIRST (same convention as patch 57): upstream vLLM
    # main/stable carries ROCMAiterMLASparseImpl._forward_mla WITHOUT the
    # GLM-5.3 shape-keyed selector, so find_target alone cannot distinguish
    # "GLM tree" from "upstream tree". The shape selector only exists in the
    # PR #53906 glm-release branch this patch was written against.
    if not glm5_present(src):
        print(f"SKIP: no glm5_next model package under {src} — patch 58 "
              f"only applies to GLM-5.3 builds (PR #53906 branch).")
        return 0

    target = find_target(src)
    if target is None:
        print(f"ERROR: GLM-5.3 support present but {IMPL_CLASS}."
              f"_forward_mla not found under {src}. Upstream moved the "
              f"sparse-MLA backend; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    content = target.read_text()

    if args.check:
        if MARKER in content:
            print(f"OK: patch 58 present in {target}")
            return 0
        print(f"FAIL: patch 58 marker not found in {target}", file=sys.stderr)
        return 1

    if MARKER in content:
        print(f"SKIP: patch 58 already applied to {target}")
        return 0

    # Anchor 1: the shape-keyed selector body of _use_rocm_sparse_triton
    # (verbatim against the glm-release checkout; fail closed otherwise).
    # The verbatim anchor subsumes the old sanity probes: both shape terms
    # and both geometry terms must be present for it to match.
    count = content.count(SELECTOR_OLD)
    if count != 1:
        print(f"ERROR: selector anchor matched {count}x in {target} "
              f"(expected 1). The dispatch logic differs from the audited "
              f"glm-release layout (vllm#53943); re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT
    patched = content.replace(SELECTOR_OLD, SELECTOR_NEW, 1)

    # Anchor 2: inject the env-knob helper right before the selector helper
    # (module level; resolved at call time).
    count = patched.count(HELPER_FN)
    if count != 1:
        print(f"ERROR: '{HELPER_FN.strip()}' matched {count}x in {target} "
              f"(expected 1); re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    patched = patched.replace(HELPER_FN, HELPER_BLOCK + HELPER_FN, 1)

    # Fail-closed: never write a tree that does not parse.
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched {target} no longer parses ({e}). The "
              f"selector layout differs from vllm#53943; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT

    target.write_text(patched)
    print(f"OK: patch 58 applied to {target} "
          f"(selector re-keyed on geometry, VLLM_GFX1X_FORCE_TRITON_SPARSE "
          f"knob added)")

    # Best-effort: the audited tree has exactly one shape-keyed selector.
    # If more survived, MTP can still fall through on another path.
    for pat in ("num_decode_tokens == num_decodes", "max_query_len == 1"):
        leftover = patched.count(pat)
        if leftover:
            print(f"WARN: {leftover} further occurrence(s) of '{pat}' "
                  f"remain in {target}. The audited glm-release tree gates "
                  f"the Triton lane on a single selector; re-audit whether "
                  f"the others also gate it.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
