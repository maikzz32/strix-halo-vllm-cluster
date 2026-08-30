#!/usr/bin/env python3
"""Patch 58: fix the GLM-5.3 MTP sparse-MLA dispatch on ROCm (vllm#53943 port).

In the GLM-5.3 tree (PR #53906 branch, ZJY0516/vllm glm-release),
ROCMAiterMLASparseImpl._forward_mla keys its BF16 Triton-lane selector on

    num_decode_tokens == num_decodes and max_query_len == 1

i.e. "pure decode, exactly one token per request". MTP draft steps present
2+ tokens per decode request, the selector falls through, and execution
lands on gfx950-only AITER asm kernels that fault everywhere else.

The fix (upstream: https://github.com/vllm-project/vllm/pull/53943, also
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

STATUS: re-audited against vLLM v0.28.0. Upstream main/stable (incl.
v0.28.0) has NO glm5_next model package and no shape-keyed selector (the
referenced upstream issue vllm#53943 is still OPEN), so the patch SKIPs
there — verified. The exact selector expression (variable names for the
fp8-KV flag and kv_lora_rank) must still be confirmed on the first GLM
(PR #53906 glm-release) build; the anchors stay fail-closed for that tree.

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
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 58_glm_mtp_sparse_dispatch"
EXIT_REAUDIT = 42

# Primary location guess in the PR #53906 tree; falls back to a content
# search for the class name.
REL_PATH = "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
IMPL_CLASS = "ROCMAiterMLASparseImpl"

# The two shape terms that break MTP (2+ tokens per decode request).
SHAPE_PATTERNS = [
    r"num_decode_tokens\s*==\s*num_decodes",
    r"max_query_len\s*==\s*1",
]

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


def statement_bounds(lines: list[str], i: int) -> tuple[int, int]:
    """Expand line index i to the full logical statement around it.

    Walks up/down across parenthesized and backslash/keyword continuations.
    Defensive by design; the result is validated with ast.parse afterwards.
    """
    start = i
    while start > 0:
        prev = lines[start - 1].rstrip()
        block = "".join(lines[start:i + 1])
        if prev.endswith(("(", "[", "{", "\\", "and", "or", ",")):
            start -= 1
        elif block.count("(") > block.count(")"):
            start -= 1
        else:
            break
    end = i
    while end < len(lines) - 1:
        cur = lines[end].rstrip()
        block = "".join(lines[start:end + 1])
        if cur.endswith(("(", "[", "{", "\\", "and", "or", ",")):
            end += 1
        elif block.count("(") > block.count(")"):
            end += 1
        else:
            break
    return start, end


def remove_shape_terms(stmt: str) -> tuple[str, int]:
    """Remove the MTP-breaking shape terms (with adjacent 'and') from stmt.

    Returns (new_stmt, number_of_terms_removed).
    """
    removed = 0
    for pat in SHAPE_PATTERNS:
        for rx, repl in (
            # term followed by 'and' (next line, same line, or trailing)
            (re.compile(pat + r"[ \t]*\n[ \t]*and[ \t]+"), ""),
            (re.compile(pat + r"[ \t]+and[ \t]+"), ""),
            (re.compile(pat + r"[ \t]*and[ \t]*\n[ \t]*"), ""),
            # 'and' before the term (next line, same line, or leading)
            (re.compile(r"[ \t]*\n[ \t]*and[ \t]+" + pat), ""),
            (re.compile(r"[ \t]+and[ \t]+" + pat), ""),
            (re.compile(r"[ \t]*and[ \t]*\n[ \t]*" + pat), ""),
        ):
            stmt, n = rx.subn(repl, stmt)
            removed += n
        # bare term (sole condition) -> neutral element
        stmt, n = re.subn(pat, "True", stmt)
        removed += n
    return stmt, removed


def add_force_knob(stmt: str) -> str:
    """OR the selector with _gfx1x_force_triton_sparse()."""
    stripped = stmt.rstrip()
    suffix = ""
    if stripped.endswith(":"):  # if/elif/while header
        stripped, suffix = stripped[:-1], ":"
    indent = stmt[:len(stmt) - len(stmt.lstrip())]
    if re.match(r"^\s*(if|elif|while)\b", stmt) and "(" not in stmt.split(
            None, 1)[1][:1]:
        # unparenthesized header condition: wrap before appending
        head, cond = stripped.split(None, 1)
        return f"{indent}{head} ({cond}) or _gfx1x_force_triton_sparse(){suffix}\n"
    return f"{stripped} or _gfx1x_force_triton_sparse(){suffix}\n"


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

    lines = content.splitlines(keepends=True)

    # Anchor 1: the shape-keyed selector inside/near _forward_mla.
    anchor = next(
        (i for i, ln in enumerate(lines)
         if re.search(SHAPE_PATTERNS[0], ln)), None)
    if anchor is None:
        print(f"ERROR: anchor 'num_decode_tokens == num_decodes' not found "
              f"in {target}. The dispatch selector changed upstream; "
              f"re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    start, end = statement_bounds(lines, anchor)
    stmt = "".join(lines[start:end + 1])
    if not re.search(SHAPE_PATTERNS[1], stmt):
        print(f"ERROR: selector statement in {target} lacks "
              f"'max_query_len == 1'. The dispatch logic differs from "
              f"vllm#53943; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    # Sanity: the researched selector also carries the geometry terms; if
    # they are gone, the surrounding logic is not what we audited.
    if not re.search(r"kv_lora_rank|fp8|head_size", stmt):
        print(f"ERROR: selector statement in {target} has no geometry terms "
              f"(fp8 KV / head_size == kv_lora_rank). Layout differs from "
              f"vllm#53943; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    new_stmt, removed = remove_shape_terms(stmt)
    if removed < 2:
        print(f"ERROR: expected to remove 2 shape terms in {target}, "
              f"removed {removed}; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    new_stmt = add_force_knob(new_stmt)

    patched = "".join(lines[:start]) + new_stmt + "".join(lines[end + 1:])

    # Anchor 2: module-level injection point for the env-knob helper.
    m = re.search(rf"^class {IMPL_CLASS}\b", patched, flags=re.MULTILINE)
    if not m:
        print(f"ERROR: 'class {IMPL_CLASS}' not found in {target}; "
              f"re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    patched = patched[:m.start()] + HELPER_BLOCK + patched[m.start():]

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

    # Best-effort: the researched tree has exactly one shape-keyed selector.
    # If more survived, MTP can still fall through on another path.
    leftover = len(re.findall(SHAPE_PATTERNS[0], patched))
    if leftover:
        print(f"WARN: {leftover} further occurrence(s) of "
              f"'num_decode_tokens == num_decodes' remain in {target}. "
              f"vllm#53943 describes a single selector; re-audit whether the "
              f"others also gate the Triton lane.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
