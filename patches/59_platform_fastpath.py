#!/usr/bin/env python3
"""Patch 59: ROCm platform fast path for Strix Halo (skip probing).

vLLM's platform detection (vllm/platforms/__init__.py, function
resolve_current_platform_cls_qualname) probes TPU (libtpu import), CUDA
(pynvml init/shutdown), ROCm (amdsmi init/shutdown), XPU and CPU in
sequence on EVERY process start; there is no upstream short-circuit for
ROCm (vllm#32730 confirmed the same hole for CPU). Each probe costs
tens-to-hundreds of ms and adds failure modes (pynvml/amdsmi import and
init on a box where only one stack is healthy) to every CLI invocation,
engine start and Ray worker spawn.

With VLLM_GFX1X_FAST_PLATFORM=1 (read lazily at detection time, like the
other VLLM_GFX1X_* knobs) the function returns
'vllm.platforms.rocm.RocmPlatform' immediately - BUT only if a ROCm
device is actually present (/dev/kfd exists). Otherwise it falls through
to stock detection, so GPU-less container builds and docker/smoke_test.sh
keep working: the UnspecifiedPlatform fallback survives. This build only
targets gfx1151, so no multi-platform portability is preserved beyond
that fallback.

Kill switch: unset VLLM_GFX1X_FAST_PLATFORM (or set it to 0).

STATUS: expected-to-need-adjustment. Anchor verified against vLLM dev
tag v0.28.1rc0 (79651d6). If upstream adds a native ROCm short-circuit
(or renames resolve_current_platform_cls_qualname), the anchor moves and
this patch exits 42 for re-audit.

Usage:
    python3 59_platform_fastpath.py --src /opt/vllm          # apply
    python3 59_platform_fastpath.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 59_platform_fastpath"
EXIT_REAUDIT = 42

PLATFORM_INIT_REL = "vllm/platforms/__init__.py"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise KeyError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def find_target(src: Path) -> Path | None:
    cand = src / PLATFORM_INIT_REL
    if cand.is_file():
        return cand
    matches = sorted(
        p for p in src.rglob("__init__.py")
        if p.parent.name == "platforms" and "site-packages" not in str(p)
    )
    return matches[0] if matches else None


FASTPATH = '''def resolve_current_platform_cls_qualname() -> str:
    # {marker}: Strix Halo fast path. Stock detection probes TPU (libtpu
    # import), CUDA (pynvml init/shutdown), ROCm (amdsmi init/shutdown),
    # XPU and CPU in sequence on every start; there is no upstream ROCm
    # short-circuit (vllm#32730). With VLLM_GFX1X_FAST_PLATFORM=1 (lazy
    # read) and a ROCm device actually present (/dev/kfd), select
    # RocmPlatform immediately; otherwise fall through to stock detection
    # so GPU-less container builds keep the UnspecifiedPlatform fallback.
    if os.environ.get("VLLM_GFX1X_FAST_PLATFORM") == "1" and os.path.exists(
        "/dev/kfd"
    ):
        logger.debug(
            "VLLM_GFX1X_FAST_PLATFORM=1 and /dev/kfd present: selecting "
            "RocmPlatform without probing (gfx1151 fast path)."
        )
        return "vllm.platforms.rocm.RocmPlatform"
'''.replace("{marker}", MARKER)


def patch_platform_init(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 59 already applied to {path}")
        return False
    source = replace_once(
        source,
        "def resolve_current_platform_cls_qualname() -> str:\n",
        FASTPATH,
        "resolve_current_platform_cls_qualname anchor",
    )
    # Fail closed: never write a file that does not parse.
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as e:
        print(f"ERROR: patched {path} would not parse ({e}); aborting "
              f"without writing. Re-audit this patch.", file=sys.stderr)
        sys.exit(EXIT_REAUDIT)
    path.write_text(source)
    print(f"OK: patch 59 applied to {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    target = find_target(src)
    if target is None:
        print(f"ERROR: {PLATFORM_INIT_REL} not found under {src}. Upstream "
              f"moved the platform detection code; re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT

    if args.check:
        if MARKER in target.read_text(errors="ignore"):
            print(f"OK: patch 59 present in {target}")
            return 0
        print(f"FAIL: patch 59 marker not found in {target}", file=sys.stderr)
        return 1

    try:
        patch_platform_init(target)
    except KeyError as exc:
        print(f"ERROR: {exc}. Upstream restructured the file; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
