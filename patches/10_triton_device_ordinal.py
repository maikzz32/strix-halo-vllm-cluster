#!/usr/bin/env python3
"""Patch 10: pin Triton's runtime driver to device 0 on gfx1151.

Workaround for Triton failing with "invalid device ordinal" inside the
forked vLLM EngineCore subprocess on gfx1151 (RDNA 3.5 iGPU).
Upstream reference: https://github.com/ROCm/TheRock/issues/4552

Every cluster node exposes exactly one gfx1151 iGPU, so forcing device
ordinal 0 is safe here. The patch wraps Triton's driver initialisation at
the point where vLLM's triton shim has just decided triton is usable
(vllm/triton_utils/importing.py, immediately before `if not HAS_TRITON:`;
that module runs before vllm/triton_utils/__init__.py performs the real
`import triton`).

STATUS: anchor re-audited for vLLM v0.28.0 (v0.28 removed the top-level
`import triton` from importing.py; previously the block was inserted after
that import). The wrapped triton internals
(triton.runtime.driver.active.utils.load_binary) differ between Triton
releases; the torch.cuda.set_device(0) part is version-independent.
Validate against the actual Triton version on the first real build.

Usage:
    python3 10_triton_device_ordinal.py --src /opt/vllm          # apply
    python3 10_triton_device_ordinal.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 10_triton_device_ordinal"
EXIT_REAUDIT = 42

# Primary location in the vLLM source tree; falls back to a name search.
REL_PATH = "vllm/triton_utils/importing.py"

PATCH_BLOCK = '''

# {marker} (upstream: ROCm/TheRock#4552)
def _gfx1151_pin_triton_device0():
    """Force Triton to use device 0 (gfx1151 nodes have exactly one iGPU)."""
    if not HAS_TRITON:
        return
    import torch
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(0)
    try:
        # vLLM v0.28+: importing.py only probes triton (find_spec), the real
        # import moved to vllm/triton_utils/__init__.py - import it here.
        import triton
        utils = triton.runtime.driver.active.utils
        _orig_load_binary = utils.load_binary

        def _load_binary_device0(name, kernel, shared, device, *args, **kwargs):
            # Ignore the requested ordinal: in forked EngineCore subprocesses
            # it can be invalid on gfx1151; device 0 is the only GPU anyway.
            return _orig_load_binary(name, kernel, shared, 0, *args, **kwargs)

        utils.load_binary = _load_binary_device0
    except (AttributeError, ImportError):
        # Triton internals moved; torch.cuda.set_device(0) above still applies.
        pass


_gfx1151_pin_triton_device0()
'''.format(marker=MARKER)


def find_target(src: Path) -> Path | None:
    cand = src / REL_PATH
    if cand.is_file():
        return cand
    matches = sorted(p for p in src.rglob("importing.py") if "triton_utils" in str(p))
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    target = find_target(src)
    if target is None:
        print(f"ERROR: {REL_PATH} not found under {src}. Upstream moved the "
              f"triton import shim; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    content = target.read_text()

    if args.check:
        if MARKER in content:
            print(f"OK: patch 10 present in {target}")
            return 0
        print(f"FAIL: patch 10 marker not found in {target}", file=sys.stderr)
        return 1

    if MARKER in content:
        print(f"SKIP: patch 10 already applied to {target}")
        return 0

    # Anchor: the `if not HAS_TRITON:` line. In vLLM <= v0.27 the file ended
    # its probe with a top-level `import triton`; vLLM v0.28 restructured
    # importing.py to only probe via find_spec (the real import moved to
    # vllm/triton_utils/__init__.py, which imports THIS module first). The pin
    # must run after the HAS_TRITON probing above is final and before the
    # placeholder classes are defined, so insert the block immediately before
    # `if not HAS_TRITON:`. Do not hardcode line numbers.
    m = re.search(r"^if not HAS_TRITON:\s*$", content, flags=re.MULTILINE)
    if not m:
        print(f"ERROR: anchor 'if not HAS_TRITON:' not found in {target}. "
              f"Upstream restructured the file; re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT

    patched = content[:m.start()] + PATCH_BLOCK.lstrip('\n') + '\n' + content[m.start():]
    target.write_text(patched)
    print(f"OK: patch 10 applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
