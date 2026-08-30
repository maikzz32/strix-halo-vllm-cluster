#!/usr/bin/env python3
"""Patch 50: tuned gfx1x MXFP4 MoE path in the INSTALLED triton_kernels.

Consumer for VLLM_GFX1X_MOE_TUNE (registered by patch 40): patches the
installed conch/ROCm ``triton_kernels`` package file
``matmul_ogs_details/opt_flags.py`` so that skinny (block_m < 128) scaled
BF16-activation x MXFP4-weight GEMMs on RDNA 3/4 (``get_rdna_version() in
(3, 4)``) use BM=16, BN=32, BK=256 (BK is the bandwidth lever vs stock 128),
num_warps=2, num_stages=2 and target_kernel_kwargs
{'waves_per_eu': 1, 'kpack': 1} (dropping the CDNA-only
matrix_instr_nonkdim). Every value is overridable via
VLLM_GFX1X_MOE_{BM,BN,BK,NW,NS,WPE}. Stock behavior is kept unless
VLLM_GFX1X_MOE_TUNE=1.

Measured on gfx1151 (kyuz0): 11.281 -> 21.766 tok/s decode on DeepSeek-V4-class
MXFP4 MoE. Source: kyuz0/amd-strix-halo-vllm-toolboxes
scripts/patch_conch_moe_gfx1x.py @ 614c91789e4609601e618a8d967390c04896acab
(https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes).

Unlike patches 10-40 this does NOT edit the vLLM source checkout: the
``triton_kernels`` package only exists after install (the ROCm vLLM wheel
vendors ROCm/triton's python/triton_kernels into
``vllm/third_party/triton_kernels/``; a standalone site-packages
``triton_kernels`` takes runtime priority if present). The patch therefore
locates the installed package via importlib and is executed POST-INSTALL by
docker/Dockerfile.fedora. When run earlier via apply_all.sh (no installed
vllm/triton_kernels yet) it defers with SKIP and exit 0; the Dockerfile run
after pip install is the one that must succeed.

Usage:
    python3 50_conch_moe_mxfp4_tune.py                     # patch installed pkg
    python3 50_conch_moe_mxfp4_tune.py --check             # verify only
    python3 50_conch_moe_mxfp4_tune.py --src /opt/vllm     # also search a tree

Exit codes: 0 = applied / check passed / deferred (pre-install),
            1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import ast
import sys
from importlib.util import find_spec
from pathlib import Path

MARKER = "gfx1151-patch: conch-moe-mxfp4-tune"
EXIT_REAUDIT = 42

REL_PATH = Path("matmul_ogs_details/opt_flags.py")

# Anchor strings verified against ROCm/triton @ 0f380657dbf3ee86eb57558ff71df24f03b5d4e7
# (the commit vLLM main pins in cmake/external_projects/triton_kernels.cmake):
# python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py.
ANCHOR_IMPORT = "from dataclasses import dataclass\n\nimport triton\n"
ANCHOR_IMPORT_NEW = "from dataclasses import dataclass\nimport os\n\nimport triton\n"

ANCHOR_HELPER = "from triton_kernels.tensor import bitwidth\n\n\n@dataclass\n"
HELPER_BLOCK = '''
# {marker}
# Opt-in tuned gfx1x (RDNA) MXFP4 MoE configuration; see manifest.d/50.md.
_GFX1X_MOE_CONFIG = None


def _gfx1x_moe_config():
    # Resolve lazily: Ray applies the final worker environment after
    # importing parts of vLLM, but before the first model execution.
    global _GFX1X_MOE_CONFIG
    if _GFX1X_MOE_CONFIG is None:
        if os.environ.get("VLLM_GFX1X_MOE_TUNE") != "1":
            _GFX1X_MOE_CONFIG = False
        else:
            _GFX1X_MOE_CONFIG = {{
                "block_m": int(os.environ.get("VLLM_GFX1X_MOE_BM", "16")),
                "block_n": int(os.environ.get("VLLM_GFX1X_MOE_BN", "32")),
                "block_k": int(os.environ.get("VLLM_GFX1X_MOE_BK", "256")),
                "num_warps": int(os.environ.get("VLLM_GFX1X_MOE_NW", "2")),
                "num_stages": int(os.environ.get("VLLM_GFX1X_MOE_NS", "2")),
                "waves_per_eu": int(os.environ.get("VLLM_GFX1X_MOE_WPE", "1")),
            }}
            print(
                f"[gfx1x_moe] enabled {{_GFX1X_MOE_CONFIG}}",
                flush=True,
            )
    return _GFX1X_MOE_CONFIG


@dataclass
'''.format(marker=MARKER)

ANCHOR_AMD = (
    "    # AMD-specific\n"
    '    target_kernel_kwargs = {"waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}\n'
    "    epilogue_subtile = constraints.get('epilogue_subtile', None)\n"
)
ANCHOR_AMD_NEW = (
    "    # AMD-specific\n"
    '    target_kernel_kwargs = {"waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}\n'
    "\n"
    "    # gfx1x (RDNA) decode MoE uses skinny BF16 x MXFP4 matmuls. Keep\n"
    "    # generic AMD behavior for every other dtype, architecture, large-M\n"
    "    # tile, and whenever the explicit opt-in is off. BK=256 is the\n"
    "    # bandwidth lever vs the stock 128; matrix_instr_nonkdim is CDNA-only\n"
    "    # and must not be passed on RDNA.\n"
    "    gfx1x_moe = _gfx1x_moe_config()\n"
    "    if (\n"
    "        gfx1x_moe\n"
    "        and get_rdna_version() in (3, 4)\n"
    "        and block_m < 128\n"
    "        and bitwidth(lhs_dtype) == 16\n"
    "        and bitwidth(rhs_dtype) == 4\n"
    "        and precision_config.weight_scale is not None\n"
    "    ):\n"
    '        block_m = gfx1x_moe["block_m"]\n'
    '        block_n = gfx1x_moe["block_n"]\n'
    '        block_k = gfx1x_moe["block_k"]\n'
    '        num_warps = gfx1x_moe["num_warps"]\n'
    '        num_stages = gfx1x_moe["num_stages"]\n'
    "        target_kernel_kwargs = {\n"
    '            "waves_per_eu": gfx1x_moe["waves_per_eu"],\n'
    '            "kpack": 1,\n'
    "        }\n"
    "\n"
    "    epilogue_subtile = constraints.get('epilogue_subtile', None)\n"
)


def _replace_once(source: str, old: str, new: str, description: str,
                  target: Path) -> str:
    count = source.count(old)
    if count != 1:
        print(f"ERROR: {description}: expected exactly one anchor in "
              f"{target}, found {count}. Upstream moved; re-audit this patch.",
              file=sys.stderr)
        sys.exit(EXIT_REAUDIT)
    return source.replace(old, new, 1)


def find_targets(src: Path) -> tuple[list[Path], bool]:
    """Locate installed opt_flags.py copies.

    Returns (targets, package_seen). package_seen=True means an importable
    vllm/triton_kernels exists but the expected file was missing under it
    (=> upstream moved, re-audit), as opposed to a pre-install stage where
    nothing is installed yet (=> defer).
    """
    targets: list[Path] = []
    package_seen = False

    if src.is_dir():
        for p in sorted(src.rglob("opt_flags.py")):
            if p.parent.name == "matmul_ogs_details" and p not in targets:
                targets.append(p)

    spec = find_spec("triton_kernels")
    if spec is not None and spec.submodule_search_locations:
        cand = Path(next(iter(spec.submodule_search_locations))) / REL_PATH
        package_seen = True
        if cand.is_file() and cand not in targets:
            targets.append(cand)

    spec = find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        cand = root / "third_party" / "triton_kernels" / REL_PATH
        package_seen = True
        if cand.is_file() and cand not in targets:
            targets.append(cand)

    return targets, package_seen


def patch_one(target: Path) -> None:
    """Patch one opt_flags.py in place (assumes marker absent)."""
    content = target.read_text()

    content = _replace_once(content, ANCHOR_IMPORT, ANCHOR_IMPORT_NEW,
                            "environment import", target)
    content = _replace_once(content, ANCHOR_HELPER, HELPER_BLOCK,
                            "gfx1x configuration helper", target)
    content = _replace_once(content, ANCHOR_AMD, ANCHOR_AMD_NEW,
                            "AMD kernel configuration", target)
    # Fail closed: never write a file that does not parse.
    try:
        ast.parse(content, filename=str(target))
    except SyntaxError as e:
        print(f"ERROR: patched {target} would not parse ({e}); aborting "
              f"without writing. Re-audit this patch.", file=sys.stderr)
        sys.exit(EXIT_REAUDIT)
    target.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm",
                    help="extra tree to search (vLLM checkout root); the "
                         "installed package is always located via importlib")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()

    targets, package_seen = find_targets(Path(args.src))

    if not targets:
        if package_seen:
            print(f"ERROR: vllm/triton_kernels is importable but "
                  f"{REL_PATH} was not found under it. Upstream restructured "
                  f"the package; re-audit this patch.", file=sys.stderr)
            return EXIT_REAUDIT
        print(f"SKIP: no installed vllm/triton_kernels found yet "
              f"(pre-install stage). Patch 50 is applied post-install by "
              f"docker/Dockerfile.fedora; see manifest.d/50.md.")
        return 0

    if args.check:
        unpatched = [t for t in targets if MARKER not in t.read_text()]
        if not unpatched:
            print(f"OK: patch 50 present in "
                  f"{', '.join(str(t) for t in targets)}")
            return 0
        print(f"FAIL: patch 50 marker missing in "
              f"{', '.join(str(t) for t in unpatched)}", file=sys.stderr)
        return 1

    for target in targets:
        if MARKER in target.read_text():
            print(f"SKIP: patch 50 already applied to {target}")
            continue
        patch_one(target)
        print(f"OK: patch 50 applied to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
