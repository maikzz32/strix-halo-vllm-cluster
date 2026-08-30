#!/usr/bin/env python3
"""Patch 56: fix APU (unified memory) memory reporting on Strix Halo.

Prerequisite fix for correct KV-cache budgeting. On gfx1151 (AMD Ryzen AI
MAX+ "Strix Halo"), HIP memory APIs report only the ~15.5 GiB VRAM aperture
as total memory instead of the ~110+ GiB GTT/unified region (ROCm/hip#3892).
vLLM then budgets --gpu-memory-utilization and KV profiling against a wrong
total and either under-allocates KV massively or crashes with a spurious OOM.

Ported from vllm#40963 (detect AMD APU, read sysfs mem_info_gtt_*). Three
edits:

1. vllm/platforms/rocm.py: RocmPlatform.is_integrated_gpu() override.
   Detection: gfx1150/gfx1151/gfx1152 arch, or PCI device ID 0x1586
   (Strix Halo), or the vllm#40963 sysfs-vs-HIP size heuristic. This also
   activates the existing UMA free-memory fix in MemorySnapshot.measure()
   (vllm/utils/mem_utils.py), which needs is_integrated_gpu() to be True.
2. vllm/platforms/rocm.py: apu_gtt_total_memory() + mem_get_info()
   classmethods (mem_get_info mirrors vllm#40963 for forward compatibility),
   and get_device_total_memory() short-circuits to the sysfs GTT total on
   APUs (8 GiB reserved for system/driver overhead, as in the PR).
3. vllm/utils/mem_utils.py: MemorySnapshot.measure() additionally corrects
   total_memory via the platform's apu_gtt_total_memory() on ROCm; without
   this the snapshot total stays aperture-sized and the KV budget
   (init_snapshot.total_memory * gpu_memory_utilization) is still wrong.

Kill switch: VLLM_GFX1X_APU_MEM_REPORT=0 disables all of it. Env is
resolved lazily (Ray applies worker env after import).

Companion note: expandable segments stay important on UMA; the registry
default models/registry.yaml already sets
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True (ROCm's name for the
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True knob).

STATUS: expected-to-need-adjustment. Anchors verified against vLLM dev tag
v0.28.1rc0 (79651d6). If vllm#40963 (or a successor) lands upstream, the
is_navi/get_device_total_memory/MemorySnapshot anchors will move and this
patch exits 42 for re-audit - then likely most of it becomes obsolete.

Usage:
    python3 56_apu_memory_reporting.py --src /opt/vllm          # apply
    python3 56_apu_memory_reporting.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 56_apu_memory_reporting"
EXIT_REAUDIT = 42

ROCM_REL = "vllm/platforms/rocm.py"
MEM_UTILS_REL = "vllm/utils/mem_utils.py"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise KeyError(f"{description}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def find_target(src: Path, rel: str, name_hint: str) -> Path | None:
    cand = src / rel
    if cand.is_file():
        return cand
    matches = sorted(p for p in src.rglob(name_hint)
                     if "site-packages" not in str(p))
    return matches[0] if matches else None


APU_METHODS = '''

    # {marker}
    @classmethod
    def is_integrated_gpu(cls, device_id: int = 0) -> bool:
        """Detect AMD APU (integrated GPU sharing system memory).

        ROCm/hip#3892: on AMD APUs (e.g. Ryzen AI MAX+ 395, gfx1151), HIP
        reports only the small VRAM aperture (~15.5 GiB) as total memory
        even though the GPU can address ~110+ GiB via GTT/unified memory.
        Detection: gfx115x arch, PCI device ID 0x1586 (Strix Halo), or the
        vllm#40963 sysfs-vs-HIP size heuristic. Kill switch:
        VLLM_GFX1X_APU_MEM_REPORT=0. Resolved lazily: Ray applies worker
        env vars after import time.
        """
        import os

        if os.environ.get("VLLM_GFX1X_APU_MEM_REPORT", "1") == "0":
            return False
        if any(a in _GCN_ARCH for a in ("gfx1150", "gfx1151", "gfx1152")):
            return True
        try:
            import glob

            for dev in glob.glob("/sys/class/drm/card*/device/device"):
                with open(dev) as f:
                    if f.read().strip().lower() == "0x1586":
                        return True
        except Exception:
            pass
        # vllm#40963 heuristic: sysfs reports >4x the HIP-reported total,
        # meaning HIP only sees the local VRAM aperture.
        try:
            import glob

            cards = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
            if not cards:
                return False
            with open(cards[0]) as f:
                sysfs_vram = int(f.read().strip())
            _, hip_total = torch.cuda.mem_get_info(device_id)
            return sysfs_vram > hip_total * 4
        except Exception:
            return False

    @classmethod
    def apu_gtt_total_memory(cls, device_id: int = 0) -> "int | None":
        """Usable total memory (bytes) from sysfs GTT on AMD APUs.

        Reserves 8 GiB for system/driver overhead (as vllm#40963). Returns
        None when the device is not an APU or sysfs is unavailable.
        """
        if not cls.is_integrated_gpu(device_id):
            return None
        try:
            import glob

            cards = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")
            if not cards:
                return None
            with open(cards[0]) as f:
                gtt_total = int(f.read().strip())
            return gtt_total - (8 * 1024**3)
        except Exception:
            return None

    @classmethod
    def mem_get_info(cls, device=None) -> "tuple[int, int]":
        """Return (free, total) GPU memory in bytes.

        On AMD APUs with unified memory, hipMemGetInfo returns the VRAM
        aperture size (~15.5 GiB) as total while free tracks GTT/unified
        memory, which breaks KV-cache budgeting (ROCm/hip#3892, vllm#40963).
        On APUs, read the true sizes from sysfs instead.
        """
        free, total = torch.cuda.mem_get_info(device)
        safe_total = cls.apu_gtt_total_memory()
        if safe_total is not None:
            try:
                import glob

                cards = glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")
                gtt_used = 0
                if cards:
                    with open(cards[0]) as f:
                        gtt_used = int(f.read().strip())
                safe_free = max(0, safe_total - gtt_used)
                logger.info_once(
                    "AMD APU detected: using sysfs GTT memory "
                    "(total=%.1f GiB, free=%.1f GiB) instead of the "
                    "HIP-reported VRAM aperture (%.1f GiB).",
                    safe_total / (1024**3),
                    safe_free / (1024**3),
                    total / (1024**3),
                )
                return int(safe_free), int(safe_total)
            except Exception:
                pass
        return free, total
'''.replace("{marker}", MARKER)

TOTAL_MEMORY_SHORT_CIRCUIT = '''    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # {marker}: on AMD APUs the HIP/amdsmi VRAM figure is only the small
        # aperture (ROCm/hip#3892); budget against the sysfs GTT total.
        apu_total = cls.apu_gtt_total_memory(device_id)
        if apu_total is not None:
            return apu_total
        # Query total VRAM via amdsmi'''.replace("{marker}", MARKER)

MEM_UTILS_TOTAL_FIX = '''            self.free_memory = psutil.virtual_memory().available
            # {marker}: on AMD APUs HIP also underreports TOTAL memory
            # (VRAM aperture only, ROCm/hip#3892); correct it from sysfs GTT
            # so the KV budget (total_memory * gpu_memory_utilization) is
            # computed against the real unified-memory size.
            if current_platform.is_rocm():
                apu_total_fn = getattr(current_platform,
                                       "apu_gtt_total_memory", None)
                if apu_total_fn is not None:
                    apu_total = apu_total_fn(device.index)
                    if apu_total is not None:
                        self.total_memory = apu_total
'''.replace("{marker}", MARKER)


def patch_rocm_py(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 56 already applied to {path}")
        return False
    # 1. APU detection + memory classmethods after is_navi().
    source = replace_once(
        source,
        '    def is_navi(cls) -> bool:\n        return "gfx1" in _GCN_ARCH\n',
        '    def is_navi(cls) -> bool:\n        return "gfx1" in _GCN_ARCH\n'
        + APU_METHODS,
        "RocmPlatform.is_navi anchor",
    )
    # 2. get_device_total_memory short-circuit.
    source = replace_once(
        source,
        "    def get_device_total_memory(cls, device_id: int = 0) -> int:\n"
        "        # Query total VRAM via amdsmi",
        TOTAL_MEMORY_SHORT_CIRCUIT,
        "get_device_total_memory short-circuit",
    )
    path.write_text(source)
    print(f"OK: patch 56 applied to {path}")
    return True


def patch_mem_utils(path: Path) -> bool:
    source = path.read_text()
    if MARKER in source:
        print(f"SKIP: patch 56 already applied to {path}")
        return False
    source = replace_once(
        source,
        "            self.free_memory = psutil.virtual_memory().available\n",
        MEM_UTILS_TOTAL_FIX,
        "MemorySnapshot.measure total-memory fix",
    )
    path.write_text(source)
    print(f"OK: patch 56 applied to {path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    rocm_py = find_target(src, ROCM_REL, "rocm.py")
    mem_utils = find_target(src, MEM_UTILS_REL, "mem_utils.py")
    for path, rel in ((rocm_py, ROCM_REL), (mem_utils, MEM_UTILS_REL)):
        if path is None:
            print(f"ERROR: {rel} not found under {src}. Upstream moved the "
                  f"platform/memory code; re-audit this patch.",
                  file=sys.stderr)
            return EXIT_REAUDIT

    if args.check:
        ok = True
        for path in (rocm_py, mem_utils):
            if MARKER in path.read_text(errors="ignore"):
                print(f"OK: patch 56 present in {path}")
            else:
                print(f"FAIL: patch 56 marker missing in {path}",
                      file=sys.stderr)
                ok = False
        return 0 if ok else 1

    try:
        patch_rocm_py(rocm_py)
        patch_mem_utils(mem_utils)
    except KeyError as exc:
        print(f"ERROR: {exc}. Upstream restructured the file; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
