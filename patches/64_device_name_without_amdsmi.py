#!/usr/bin/env python3
"""Patch 64: resolve the ROCm device name without amdsmi.

amdsmi is deliberately absent from the image (that is why patch 59 exists:
"skip platform probing -- amdsmi is NOT in the image"). `vllm/platforms/rocm.py`
imports it in a try/except that only logs a warning:

    except ImportError as e:
        logger.warning("Failed to import from amdsmi with %r", e)

but `RocmPlatform.get_device_name` is decorated with `@with_amdsmi_context`,
whose wrapper calls `amdsmi_init()` unguarded. With the module missing, the
name was never bound, so the first caller dies with:

    NameError: name 'amdsmi_init' is not defined

Measured on 2026-09-02 serving Qwen3.8-Flash-Next (TP4). The caller is the MoE
tuning-config lookup, i.e. it happens on the first forward, after the weights
are already loaded:

    fused_moe/experts/triton_moe.py:676   apply
    fused_moe/fused_moe.py:1442           try_get_optimal_moe
    fused_moe/fused_moe.py:1134           get_moe_configs
    fused_moe/fused_moe.py:1098           get_config_file_name
    utils/platform_utils.py:72            get_device_name_as_file_name
    platforms/rocm.py:168                 wrapper -> amdsmi_init()

This patch splits the method: the amdsmi implementation is preserved verbatim
under a private name (still decorated, so an image WITH amdsmi keeps the exact
marketing name), and `get_device_name` gains a HIP fallback via
`torch.cuda.get_device_name` when the amdsmi symbols are absent.

The consequence of the fallback is benign and worth stating: the returned name
feeds the tuned-MoE-config FILENAME lookup. A different string means "no tuned
config found", so vLLM uses its defaults -- a possible throughput difference,
never a correctness one. FLASHNEXT.md already records that the tuned MoE
configs never took effect on gfx1151 anyway, because a mocked amdsmi returned
a MagicMock as the device name.

Deliberately narrow: the other `@with_amdsmi_context` users (get_device_uuid,
memory queries, topology) are NOT touched. If one of them is reached without
amdsmi it must fail loudly, so the next gap is diagnosed rather than papered
over. Patch 56 keeps its own amdsmi-based memory reporting untouched.

Usage:
    python3 64_device_name_without_amdsmi.py --src /opt/vllm            # build
    python3 64_device_name_without_amdsmi.py --src <site-packages>      # runtime
    python3 64_device_name_without_amdsmi.py --src /opt/vllm --check    # verify

Exit codes: 0 applied / check passed / nothing to do, 1 check failed,
42 = anchor moved -> re-audit, do not silently skip.

Re-audit trigger: exit 42, or amdsmi becoming part of the image (then this
patch is inert but harmless -- the amdsmi path is taken again).
Written against vLLM v0.29.0rc1 (33898f832c) on 2026-09-02.
"""

import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 64_device_name_without_amdsmi"
# Containers patched on 2026-09-02 carry this earlier marker text; accept it.
LEGACY_MARKER = "gfx1151-patch: device-name-without-amdsmi"
EXIT_REAUDIT = 42

ROCM_REL = "vllm/platforms/rocm.py"

OLD = '''    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        asic_info_device_id: str = asic_info["device_id"]
        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
        return asic_info["market_name"]
'''

NEW = f'''    @classmethod
    @lru_cache(maxsize=8)
    def get_device_name(cls, device_id: int = 0) -> str:
        # {MARKER}
        # amdsmi is not in this image; its import above fails with a warning
        # only, so with_amdsmi_context would raise NameError on amdsmi_init.
        # Fall back to the HIP device name. This string is used for the tuned
        # MoE config FILENAME lookup -- a miss means "use defaults", never a
        # wrong result.
        if "amdsmi_get_processor_handles" not in globals():
            import torch

            return torch.cuda.get_device_name(device_id)
        return cls._get_device_name_amdsmi(device_id)

    @classmethod
    @with_amdsmi_context
    @lru_cache(maxsize=8)
    def _get_device_name_amdsmi(cls, device_id: int = 0) -> str:
        # {MARKER}: unchanged upstream body, reached only when amdsmi exists.
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = amdsmi_get_processor_handles()[physical_device_id]
        asic_info = amdsmi_get_gpu_asic_info(handle)
        asic_info_device_id: str = asic_info["device_id"]
        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:
            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]
        return asic_info["market_name"]
'''

# The amdsmi import must still be the soft kind -- if upstream ever makes it
# hard (no try/except), the module cannot load at all without amdsmi and this
# patch addresses the wrong problem.
SOFT_IMPORT_GUARD = 'except ImportError as e:\n    logger.warning("Failed to import from amdsmi with %r", e)'

# with_amdsmi_context must still exist and still call amdsmi_init unguarded --
# that is the failure this patch routes around.
CONTEXT_GUARD = "def with_amdsmi_context(fn):"


def find_file(src: Path, rel: str) -> Path | None:
    cand = src / rel
    if cand.is_file():
        return cand
    name = rel.rsplit("/", 1)[-1]
    parent = rel.rsplit("/", 2)[-2]
    matches = sorted(p for p in src.rglob(name) if p.parent.name == parent)
    return matches[0] if matches else None


def die(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return EXIT_REAUDIT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source or site-packages root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    rocm = find_file(src, ROCM_REL)
    if rocm is None:
        return die(f"{ROCM_REL} not found under {src}.")

    content = rocm.read_text()

    if args.check:
        if MARKER in content or LEGACY_MARKER in content:
            print(f"OK: patch 64 present in {rocm}")
            return 0
        print(f"FAIL: patch 64 marker not found in {rocm}", file=sys.stderr)
        return 1

    if MARKER in content or LEGACY_MARKER in content:
        print(f"SKIP: patch 64 already applied to {rocm}")
        return 0

    if content.count(OLD) != 1:
        return die(
            f"RocmPlatform.get_device_name was not found in its expected form in "
            f"{rocm} (found {content.count(OLD)} matches). Upstream changed the "
            f"method or its decorators; re-audit this patch."
        )

    if SOFT_IMPORT_GUARD not in content:
        return die(
            f"{rocm} no longer imports amdsmi behind a warning-only try/except. "
            f"If the import became mandatory, this patch targets the wrong "
            f"problem; re-audit."
        )

    if CONTEXT_GUARD not in content:
        return die(f"with_amdsmi_context is gone from {rocm}; re-audit this patch.")

    rocm.write_text(content.replace(OLD, NEW, 1))
    print(f"OK: patch 64 applied to {rocm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
