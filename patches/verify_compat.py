#!/usr/bin/env python3
"""verify_compat.py — fail-closed compatibility check for the patch layer.

Verifies:
  1. every patch actually landed (its marker string is present in the
     patched files),
  2. vllm imports,
  3. ray imports,
and prints the resolved versions of vllm, torch, triton and aiter.

Exit code 0 only if everything passes; 1 otherwise.

NOTE: `import vllm` must succeed in a GPU-less build stage. If a future
vLLM release hard-requires a GPU at import time, relax check 2 to
importlib.util.find_spec("vllm") and record the change in MANIFEST.md.
"""

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

MARKER_PREFIX = "gfx1151-patch: "

# patch id -> where its marker is expected ("vllm-src" = grep the checkout
# or the installed vllm package, "torch" = the installed torch package's
# _rocm_init.py)
EXPECTED = {
    "10_triton_device_ordinal": "vllm-src",
    "20_rocm_smi_rtld": "torch",
    "40_aiter_gfx1x_gating": "vllm-src",
    "51_radix_topk": "vllm-src",
    "52_tilelang_sparse_indexer": "vllm-src",
    "53_w8a8_bf16_skinny": "vllm-src",
    "54_aiter_rdna_enable": "vllm-src",
    "56_apu_memory_reporting": "vllm-src",
    "59_platform_fastpath": "vllm-src",
    "60_skinny_gemm_m1": "vllm-src",
}

# Conditional patches: marker required only when at least one of the listed
# relative paths exists under the scan root. Covers model-specific patches
# that legitimately SKIP on trees without the model package, and patch 30
# whose marker lives in requirements/rocm.txt: that file exists in the
# builder's source checkout (marker enforced there) but is not shipped in
# the installed wheel, so the final-stage scan (site-packages) legitimately
# SKIPs it.
CONDITIONAL = {
    "30_tensorizer_pin": ["requirements/rocm.txt"],
    "58_glm_mtp_sparse_dispatch": ["vllm/models/glm5next",
                                   "vllm/models/glm5_next"],
}

# Conditional on file CONTENT, not mere existence: patch id -> (relative
# path, substring that must be present for the patch to have anything to do).
# Patch 57 ports the PLE offload from vllm#53899. A tree with #53896 merged
# but #53899 still open HAS vllm/models/qwen4_exp yet nothing to port onto,
# so a plain path check would demand a marker the patch cannot produce.
# `pid in found` keeps it fail-closed the other way round: once the marker is
# there, it is still verified even though the patch rewrote the anchor.
CONDITIONAL_CONTENT = {
    "57_ple_offload_amd": ("vllm/models/qwen4_exp/amd/ple_layer.py",
                           "self.ngram_embedding = VocabParallelEmbedding("),
}

# Post-install-only patches (marker lands in installed third-party packages,
# not in the vllm tree): 50_conch_moe_mxfp4_tune (triton_kernels) and the
# aiter side of 54. They are exercised by the Dockerfile post-install steps
# and docker/smoke_test.sh, deliberately not enforced here.

SCAN_SUFFIXES = {".py", ".txt", ".toml", ".cfg"}


def find_markers(root: Path) -> set[str]:
    """Collect patch ids whose marker appears anywhere under root."""
    found = set()
    if not root.is_dir():
        return found
    for p in root.rglob("*"):
        if p.suffix not in SCAN_SUFFIXES or not p.is_file():
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if MARKER_PREFIX not in text:
            continue
        for line in text.splitlines():
            if MARKER_PREFIX in line:
                pid = line.split(MARKER_PREFIX, 1)[1].split()[0].rstrip(":,)")
                found.add(pid)
    return found


def torch_rocm_init() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    cand = Path(spec.submodule_search_locations[0]) / "_rocm_init.py"
    return cand if cand.is_file() else None


def version_of(module: str, dist: str | None = None) -> str:
    try:
        return importlib.import_module(module).__version__
    except Exception:
        pass
    # The wheel's dist name can differ from the import name (aiter is
    # published as amd-aiter since v0.1.19).
    for name in dict.fromkeys(n for n in (dist, module) if n):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not installed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--skip-imports", action="store_true",
                    help="skip the import checks (builder stage: vLLM is not "
                         "pip-installed yet, only marker/structure checks run)")
    args = ap.parse_args()
    src = Path(args.src)

    failures: list[str] = []

    # 1. patch markers
    found = find_markers(src)
    torch_file = torch_rocm_init()
    if torch_file is not None:
        text = torch_file.read_text(errors="ignore")
        if MARKER_PREFIX in text:
            found.add(text.split(MARKER_PREFIX, 1)[1].split()[0].rstrip(":,)"))
    for pid, where in EXPECTED.items():
        if pid in found:
            print(f"OK   marker for patch {pid}")
        else:
            failures.append(f"marker for patch {pid} not found ({where})")
            print(f"FAIL marker for patch {pid} not found ({where})")
    for pid, relpaths in CONDITIONAL.items():
        applies = any((src / rel).exists() for rel in relpaths)
        if not applies:
            print(f"SKIP conditional patch {pid} (no {relpaths[0]} in tree)")
        elif pid in found:
            print(f"OK   marker for conditional patch {pid}")
        else:
            failures.append(f"marker for conditional patch {pid} not found")
            print(f"FAIL marker for conditional patch {pid} not found")
    for pid, (rel, needle) in CONDITIONAL_CONTENT.items():
        target = src / rel
        text = target.read_text(errors="ignore") if target.is_file() else ""
        if not (needle in text or pid in found):
            print(f"SKIP conditional patch {pid} (nothing to patch in {rel})")
        elif pid in found:
            print(f"OK   marker for conditional patch {pid}")
        else:
            failures.append(f"marker for conditional patch {pid} not found")
            print(f"FAIL marker for conditional patch {pid} not found")

    # 2./3. imports
    if args.skip_imports:
        print("SKIP import checks (--skip-imports)")
    else:
        for module in ("vllm", "ray"):
            try:
                importlib.import_module(module)
                print(f"OK   import {module}")
            except Exception as exc:
                failures.append(f"import {module} failed: {exc!r}")
                print(f"FAIL import {module}: {exc!r}")

    # 4. versions (aiter may legitimately be absent: it is disabled on gfx1x)
    print("--- resolved versions ---")
    for module, dist in (("vllm", None), ("torch", None),
                         ("triton", None), ("aiter", "amd-aiter")):
        print(f"{module:8s} {version_of(module, dist)}")

    if failures:
        print(f"\nverify_compat FAILED ({len(failures)} problem(s))",
              file=sys.stderr)
        return 1
    print("\nverify_compat OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
