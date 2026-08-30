#!/usr/bin/env python3
"""Patch 20: switchable rtld mode for librocm_smi64 in torch/_rocm_init.py.

torch loads librocm_smi64 via ctypes with the default (RTLD_LOCAL) mode,
which causes a symbol clash on ROCm/gfx1151 stacks (the exact symbol pair
varies by ROCm release). Our builds resolved it by re-opening the library
with RTLD_GLOBAL - but LucRoot's gfx1151 build record claims the clash is
fixed by the opposite (rtld_global=False, i.e. RTLD_LOCAL). The two fixes
are mutually exclusive and cannot be resolved without hardware, so the
mode is switchable at runtime: the rewritten code reads
VLLM_GFX1X_ROCM_SMI_RTLD (global|local) LAZILY at torch import time and
picks RTLD_GLOBAL vs RTLD_LOCAL accordingly. Default stays global. See
patches/manifest.d/20.md for the conflict and the A/B plan
(docs/RUNBOOK.md). NOTE: unlike the other patches, the target file
belongs to the installed torch package, not the vLLM checkout.

STATUS: expected-to-need-adjustment. The patched call shape
(`ctypes.CDLL(<path>)` on a single line containing "rocm_smi") matches
recent torch ROCm wheels; verify against the pinned TORCH_VERSION on the
first real build.

Usage:
    python3 20_rocm_smi_rtld.py            # apply
    python3 20_rocm_smi_rtld.py --check    # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 20_rocm_smi_rtld"
EXIT_REAUDIT = 42


def find_target() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    cand = Path(spec.submodule_search_locations[0]) / "_rocm_init.py"
    return cand if cand.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm",
                    help="unused; kept for a uniform CLI across patch scripts")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()

    target = find_target()
    if target is None:
        print("ERROR: torch/_rocm_init.py not found. Is the ROCm torch wheel "
              "installed in this environment?", file=sys.stderr)
        return EXIT_REAUDIT

    content = target.read_text()

    if args.check:
        if MARKER in content:
            print(f"OK: patch 20 present in {target}")
            return 0
        print(f"FAIL: patch 20 marker not found in {target}", file=sys.stderr)
        return 1

    if MARKER in content:
        print(f"SKIP: patch 20 already applied to {target}")
        return 0

    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "rocm_smi" not in line or "CDLL" not in line:
            continue
        stripped = line.rstrip("\n")
        if "RTLD_GLOBAL" in stripped:
            print(f"SKIP: {target}:{i+1} already loads librocm_smi64 with "
                  f"RTLD_GLOBAL (upstream fixed it?); marking as patched.")
            lines[i] = f"# {MARKER}: upstream already uses RTLD_GLOBAL\n" + line
            break
        if not stripped.endswith(")"):
            print(f"ERROR: CDLL call for librocm_smi64 at {target}:{i+1} spans "
                  f"multiple lines; re-audit this patch.", file=sys.stderr)
            return EXIT_REAUDIT
        indent = stripped[: len(stripped) - len(stripped.lstrip())]
        # Emit a switchable loader: the mode is resolved LAZILY at torch
        # import time from VLLM_GFX1X_ROCM_SMI_RTLD (global|local, default
        # global). Both branches carry the marker so --check passes no
        # matter which mode is active at runtime.
        lines[i] = (
            f"{indent}# {MARKER}: rtld mode switchable via "
            f"VLLM_GFX1X_ROCM_SMI_RTLD (global|local, default global);\n"
            f"{indent}# {MARKER}: LucRoot's gfx1151 record claims "
            f"RTLD_LOCAL fixes the rocm_smi symbol clash, our builds "
            f"needed RTLD_GLOBAL - unresolved without hardware, A/B at "
            f"first run (docs/RUNBOOK.md).\n"
            f"{indent}_gfx1x_rtld_mode = (ctypes.RTLD_LOCAL if "
            f"os.environ.get('VLLM_GFX1X_ROCM_SMI_RTLD', 'global') == "
            f"'local' else ctypes.RTLD_GLOBAL)  # {MARKER}\n"
            + stripped[:-1] + f", mode=_gfx1x_rtld_mode)  # {MARKER}\n"
        )
        break
    else:
        print(f"ERROR: no ctypes.CDLL call mentioning 'rocm_smi' found in "
              f"{target}. torch changed how it loads librocm_smi64; "
              f"re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    patched = "".join(lines)
    # Fail closed: never write a file that does not parse.
    try:
        ast.parse(patched, filename=str(target))
    except SyntaxError as e:
        print(f"ERROR: patched {target} would not parse ({e}); aborting "
              f"without writing. Re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    target.write_text(patched)
    print(f"OK: patch 20 applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
