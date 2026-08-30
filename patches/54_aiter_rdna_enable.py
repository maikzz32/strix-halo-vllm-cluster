#!/usr/bin/env python3
"""Patch 54: enable AITER's TRITON kernels on gfx1151 (CK/ASM stay off).

AITER's master gate in vLLM requires CDNA (never passes on gfx1151), but
AITER's *Triton* kernels can work on RDNA. Blueprint: the gfx1100/RDNA
enablement approach (referenced as aiter#4604 in the research notes; in
current vLLM the same pattern landed as the RDNA4/gfx12 functions in
vllm/_aiter_ops.py). CK/ASM kernels are CDNA-only dead ends and stay off.

Four steps:

A. vllm/_aiter_ops.py (vLLM source checkout via --src, or the installed
   package in site-packages):
   1. add is_aiter_found_and_supported_on_gfx1151(), the gfx1151 analog of
      is_aiter_found_and_supported_on_rdna4(), gated on the new env
      VLLM_GFX1X_AITER_TRITON (default 0) so VLLM_ROCM_USE_AITER=0
      master-switch semantics stay intact (patch 40 force-disables the
      master toggle on gfx1x; this knob is deliberately independent);
   2. extend rocm_aiter_ops.is_rdna_aiter_enabled() so the Triton-only
      subpaths (is_rdna_linear_enabled, is_rdna_gdn_triton_kernels_available)
      become available on gfx1151 when the knob is on.
B. Installed aiter package, aiter/ops/triton/utils/_triton/arch_info.py
   (older layout: aiter/ops/triton/utils/arch_info.py): add gfx1151 (and
   gfx1150) to _LDS_CAP_BYTES so Triton kernels that budget LDS stop dying
   with KeyError; if an _ARCH_TO_DEVICE dict exists (research notes mention
   it; absent in aiter v0.1.19/main), add the gfx1151 entry too.
C. Installed aiter csrc/cpp_itfs/utils.py: ensure gfx1151 is in the
   validate_and_update_archs() allow-list (already present in aiter
   v0.1.19; this step is then a SKIP).
D. Install patches/vendor/vec_convert.h over
   aiter_meta/csrc/include/ck_tile/vec_convert.h (RDNA scalar fallbacks for
   CDNA-only packed conversion ISA).

CRITICAL correctness note: the vendored vec_convert.h makes FP4 conversions
return ZERO tensors off-gfx950. Any AITER FP4 path reached on gfx1151 is
silently wrong, so this patch keeps FP4 gated off: it never touches
is_fp4_avail() (stays gfx950/gfx1250-only) and asserts that property after
patching arch_info.py.

This patch partially targets INSTALLED packages (post-install). In the
Docker builder stage only step A applies (aiter is not installed yet); the
final stage must re-run it after wheel installation:

    python3 /opt/patches/54_aiter_rdna_enable.py \
        --src /opt/src/vllm \
        --site-packages "$(python3 -c 'import site; print(site.getsitepackages()[0])')"

(--site-packages is optional; without it the aiter/vllm installs are
located via importlib.)

Usage:
    python3 54_aiter_rdna_enable.py --src /opt/vllm          # apply
    python3 54_aiter_rdna_enable.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 54_aiter_rdna_enable"
EXIT_REAUDIT = 42

AITER_OPS_REL = "vllm/_aiter_ops.py"
ARCH_INFO_CANDIDATES = (
    "aiter/ops/triton/utils/_triton/arch_info.py",
    "aiter/ops/triton/utils/arch_info.py",
)
CPP_ITFS_REL = "aiter/csrc/cpp_itfs/utils.py"
VEC_CONVERT_REL = "aiter_meta/csrc/include/ck_tile/vec_convert.h"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise KeyError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def site_packages_root(override: str | None) -> Path | None:
    if override:
        return Path(override)
    for module in ("aiter", "vllm"):
        spec = importlib.util.find_spec(module)
        if spec and spec.submodule_search_locations:
            return Path(spec.submodule_search_locations[0]).parent
    return None


# --- Step A: vllm/_aiter_ops.py ---------------------------------------------

GFX1151_GATE_FN = '''

# {marker}
def is_aiter_found_and_supported_on_gfx1151() -> bool:
    """gfx1151 (RDNA 3.5) analog of is_aiter_found_and_supported_on_rdna4().

    Only AITER's Triton kernels are usable on gfx1151; CK/ASM kernels are
    CDNA-only dead ends. Gated on VLLM_GFX1X_AITER_TRITON (default 0) so the
    VLLM_ROCM_USE_AITER master-switch semantics stay intact (patch 40 forces
    the master toggle off on gfx1x). Resolved lazily: Ray applies worker env
    vars after import time.
    """
    import os

    if os.environ.get("VLLM_GFX1X_AITER_TRITON", "0") != "1":
        return False
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_gfx1151

        return on_gfx1151()
    return False
'''.replace("{marker}", MARKER)

IS_RDNA_ENABLED_EXTENSION = '''        # {marker}: gfx1151 Triton-only subpath, gated on
        # VLLM_GFX1X_AITER_TRITON and independent of the master toggle.
        if is_aiter_found_and_supported_on_gfx1151():
            return True
        return on_rdna4() and cls._AITER_ENABLED
'''.replace("{marker}", MARKER)


def patch_aiter_ops(path: Path) -> bool:
    """Returns True if the file was changed."""
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 54 already applied to {path}")
        return False
    # A1: insert the gfx1151 gate right after the rdna4 analog.
    source = replace_once(
        source,
        "        return on_rdna4()\n    return False\n",
        "        return on_rdna4()\n    return False\n" + GFX1151_GATE_FN,
        "is_aiter_found_and_supported_on_rdna4 tail",
    )
    # A2: extend is_rdna_aiter_enabled with the gfx1151 Triton subpath.
    source = replace_once(
        source,
        "        return on_rdna4() and cls._AITER_ENABLED\n",
        IS_RDNA_ENABLED_EXTENSION,
        "is_rdna_aiter_enabled extension",
    )
    path.write_text(source)
    print(f"OK: patch 54 applied to {path}")
    return True


def find_aiter_ops(src: Path, sp: Path | None) -> Path | None:
    cand = src / AITER_OPS_REL
    if cand.is_file():
        return cand
    if sp is not None:
        cand = sp / AITER_OPS_REL
        if cand.is_file():
            return cand
    return None


# --- Step B: aiter arch_info.py ---------------------------------------------

def patch_arch_info(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 54 already applied to {path}")
        return False
    if '"gfx1151"' in source:
        # Partially patched before (or upstream adopted gfx1151): only mark.
        source = f"# {MARKER}: gfx1151 already known to this arch_info\n" + source
        path.write_text(source)
        print(f"SKIP: gfx1151 entries already present in {path}; marked.")
        return True
    if "_ARCH_TO_DEVICE" in source:
        raise KeyError(
            "_ARCH_TO_DEVICE dict present; the research notes say gfx1151 must "
            "be added there, but this aiter version's layout was never "
            "validated - re-audit this patch"
        )
    # gfx1151/gfx1150 expose 128 KiB LDS per WGP; Triton budgets per CU, so
    # keep the conservative 64 KiB used for gfx942.
    source = replace_once(
        source,
        '_LDS_CAP_BYTES = {"gfx1250": 327680, "gfx950": 163840, "gfx942": 65536}',
        f"# {MARKER}: RDNA 3/3.5 entries; conservative 64 KiB per-CU budget\n"
        '_LDS_CAP_BYTES = {"gfx1250": 327680, "gfx950": 163840, "gfx942": 65536, '
        '"gfx1151": 65536, "gfx1150": 65536}',
        "_LDS_CAP_BYTES gfx1151 entry",
    )
    path.write_text(source)
    print(f"OK: patch 54 applied to {path}")
    return True


# --- Step C: aiter cpp_itfs allow-list --------------------------------------

def patch_cpp_itfs(path: Path) -> bool:
    source = path.read_text()
    if '"gfx1151"' in source:
        print(f"SKIP: gfx1151 already in the cpp_itfs allow-list ({path})")
        return False
    source = replace_once(
        source,
        '    allowed_archs = [\n        "native",\n',
        f'    # {MARKER}: allow JIT compiles for RDNA 3.5\n'
        '    allowed_archs = [\n        "native",\n        "gfx1151",\n',
        "cpp_itfs allowed_archs",
    )
    path.write_text(source)
    print(f"OK: patch 54 applied to {path}")
    return True


# --- Step D: install the vendored vec_convert.h ------------------------------

def install_vec_convert(sp: Path) -> bool:
    vendor = Path(__file__).resolve().parent / "vendor" / "vec_convert.h"
    if not vendor.is_file():
        raise KeyError(f"vendored vec_convert.h missing at {vendor}")
    dest = sp / VEC_CONVERT_REL
    if not dest.is_file():
        raise KeyError(
            f"{dest} not found; aiter_meta install layout changed (aiter not "
            f"installed with headers?)"
        )
    if MARKER in dest.read_text(errors="ignore"):
        print(f"SKIP: patch 54 already applied to {dest}")
        return False
    dest.write_text(vendor.read_text())
    print(f"OK: installed RDNA vec_convert.h to {dest}")
    return True


def find_in_sp(sp: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        cand = sp / rel
        if cand.is_file():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--site-packages", default=None,
                    help="site-packages of the target python install "
                         "(default: auto-detect via importlib)")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)
    sp = site_packages_root(args.site_packages)

    aiter_ops = find_aiter_ops(src, sp)
    if aiter_ops is None:
        print(f"ERROR: {AITER_OPS_REL} found neither under {src} nor in "
              f"site-packages. Upstream moved the AITER op registry; "
              f"re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    aiter_installed = sp is not None and (sp / "aiter").is_dir()

    if args.check:
        ok = MARKER in aiter_ops.read_text(errors="ignore")
        print(f"{'OK' if ok else 'FAIL'}: patch 54 in {aiter_ops}")
        if aiter_installed:
            for rel in ARCH_INFO_CANDIDATES + (VEC_CONVERT_REL,):
                p = sp / rel
                if p.is_file() and MARKER in p.read_text(errors="ignore"):
                    print(f"OK: patch 54 present in {p}")
                elif p.is_file():
                    # cpp_itfs needs no marker; arch_info/vec_convert do.
                    print(f"FAIL: patch 54 marker missing in {p}",
                          file=sys.stderr)
                    ok = False
        else:
            print("NOTE: aiter not installed; post-install steps B-D are "
                  "deferred to the final image stage (see manifest).")
        return 0 if ok else 1

    try:
        patch_aiter_ops(aiter_ops)
    except KeyError as exc:
        print(f"ERROR: {exc} in {aiter_ops}. Upstream restructured "
              f"_aiter_ops.py; re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    if not aiter_installed:
        print("WARN: aiter package not installed in this environment; steps "
              "B-D (arch_info, cpp_itfs, vec_convert.h) deferred. Re-run this "
              "patch post-install in the final image (see manifest 54.md).",
              file=sys.stderr)
        return 0

    try:
        arch_info = find_in_sp(sp, ARCH_INFO_CANDIDATES)
        if arch_info is None:
            raise KeyError(f"none of {ARCH_INFO_CANDIDATES} found under {sp}")
        patch_arch_info(arch_info)
        cpp_itfs = find_in_sp(sp, (CPP_ITFS_REL,
                                   "aiter_meta/csrc/cpp_itfs/utils.py",
                                   "csrc/cpp_itfs/utils.py"))
        if cpp_itfs is not None:
            patch_cpp_itfs(cpp_itfs)
        else:
            print(f"NOTE: cpp_itfs utils.py not found under {sp}; aiter "
                  f"version may not ship it - skipping step C.")
        install_vec_convert(sp)
    except KeyError as exc:
        print(f"ERROR: {exc}. Upstream aiter layout moved; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT

    # FP4 guard assertion on the patched arch_info: gfx1151 must never be an
    # FP4-capable arch (vendored vec_convert.h returns zeros for FP4 there).
    text = arch_info.read_text(errors="ignore")
    for line in text.splitlines():
        if "is_fp4_avail" in line and "gfx1151" in line:
            print(f"ERROR: is_fp4_avail() in {arch_info} covers gfx1151 - "
                  f"FP4 paths would silently return zeros. Refusing.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
