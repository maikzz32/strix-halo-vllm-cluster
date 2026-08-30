#!/usr/bin/env python3
"""Patch 20: switchable rtld mode for the ROCm library preload in torch/_rocm_init.py.

torch 2.13+rocm7.14 (TheRock wheels) loads librocm_smi64 & friends via
`rocm_sdk.initialize_process(preload_shortnames=[...])`, whose rtld_global
parameter defaults to True (RTLD_GLOBAL). That default is exactly the fix
our builds needed for the rocm_smi symbol clash on ROCm/gfx1151 stacks —
but LucRoot's gfx1151 build record claims the clash is fixed by the
opposite (RTLD_LOCAL). The two fixes are mutually exclusive and cannot be
resolved without hardware, so the mode stays switchable at runtime: the
rewritten call passes rtld_global resolved LAZILY at torch import time
from VLLM_GFX1X_ROCM_SMI_RTLD (global|local). Default stays global. See
patches/manifest.d/20.md for the conflict and the A/B plan
(docs/RUNBOOK.md).

NOTE: with the TheRock loader the mode applies to ALL preloaded libraries
(rocm_smi64, amdhip64, comgr, rccl, MIOpen, ...), not just rocm_smi64 —
the per-library CDLL call this patch used to rewrite no longer exists.
The knob's blast radius is therefore wider; that IS the A/B experiment
(both records disagree on the whole-stack mode, not on one library).

NOTE: unlike the other patches, the target file belongs to the installed
torch package, not the vLLM checkout.

STATUS: re-audited against torch 2.13.0+rocm7.14.0 (TheRock loader;
verified against the real wheel). Pre-TheRock torch wheels (per-library
ctypes.CDLL(rocm_smi...) style) are detected and rejected with exit 42.

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

INIT_FN = "def initialize():"
INIT_CALL = "rocm_sdk.initialize_process("

HELPER_COMMENT = (
    "    # {marker}: rtld mode switchable via VLLM_GFX1X_ROCM_SMI_RTLD\n"
    "    # {marker}: (global|local, default global); LucRoot's gfx1151\n"
    "    # {marker}: record claims RTLD_LOCAL fixes the rocm_smi symbol\n"
    "    # {marker}: clash, our builds needed RTLD_GLOBAL - unresolved\n"
    "    # {marker}: without hardware, A/B at first run (docs/RUNBOOK.md).\n"
).replace("{marker}", MARKER)
HELPER_ASSIGN = (
    "    _gfx1x_rtld_global = os.environ.get("
    "'VLLM_GFX1X_ROCM_SMI_RTLD', 'global') != 'local'  # {marker}\n"
).replace("{marker}", MARKER)


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
    ap.add_argument("--check", action="store_true",
                    help="verify only, do not modify")
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

    # Fail-closed layout probes: exactly one TheRock-style loader call, no
    # leftover pre-TheRock CDLL loader, and a single initialize() function.
    if content.count(INIT_CALL) != 1:
        if "rocm_smi" in content and "CDLL" in content:
            print(f"ERROR: {target} uses the pre-TheRock per-library CDLL "
                  f"loader; this patch version targets TheRock wheels "
                  f"(rocm_sdk.initialize_process). Re-audit for this torch.",
                  file=sys.stderr)
        else:
            print(f"ERROR: expected exactly one '{INIT_CALL}' call in "
                  f"{target}, found {content.count(INIT_CALL)}. torch "
                  f"changed its ROCm preload; re-audit this patch.",
                  file=sys.stderr)
        return EXIT_REAUDIT
    if content.count(INIT_FN) != 1:
        print(f"ERROR: expected exactly one '{INIT_FN}' in {target}, found "
              f"{content.count(INIT_FN)}. Re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT

    patched = content
    # 1) `import os` inside initialize() (idempotent at text level: the
    #    helper line carries the marker, so re-runs SKIP earlier).
    if not re.search(r"^\s*import os(\s|$)", patched, flags=re.MULTILINE):
        patched = patched.replace(INIT_FN, INIT_FN + "\n    import os", 1)
    # 2) Resolve the mode lazily right before the call and pass it as the
    #    rtld_global kwarg (TheRock default is True = our known-good mode;
    #    VLLM_GFX1X_ROCM_SMI_RTLD=local flips it for the A/B).
    m = re.search(r"^(?P<indent>[ \t]*)" + re.escape(INIT_CALL) + r"\s*$",
                  patched, flags=re.MULTILINE)
    if not m:
        print(f"ERROR: '{INIT_CALL}' in {target} is not alone on its line; "
              f"call shape differs from torch 2.13.0+rocm7.14.0. Re-audit "
              f"this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    indent = m.group("indent")
    helper = "\n".join(
        (indent + ln[4:]) if ln.startswith("    ") else ln
        for ln in (HELPER_COMMENT + HELPER_ASSIGN).splitlines()
    ) + "\n"
    replacement = (
        helper
        + f"{indent}{INIT_CALL}\n"
        + f"{indent}    rtld_global=_gfx1x_rtld_global,  # {MARKER}\n"
    )
    patched = patched[:m.start()] + replacement + patched[m.end():]

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
