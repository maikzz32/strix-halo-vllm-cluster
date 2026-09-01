#!/usr/bin/env python3
"""Runtime env fix: make the ROCm-bundled amdsmi python package importable.

The dev-glm53-flash image carries the amdsmi sources under
/opt/rocm/share/amd_smi but never installs them into site-packages, so
vllm.platforms.rocm's `from amdsmi import ...` fails and every
@with_amdsmi_context call (e.g. get_device_name during fused-MoE tuning-config
lookup) dies with NameError. Dropping a .pth file with the bundled path into
site-packages is the minimal reversible fix (no pip install, no network).

Idempotent; refuses to write when the bundled package is absent.
"""

import os
import sys

SITE = "/usr/local/lib64/python3.12/site-packages"
PTH = os.path.join(SITE, "gfx1x_amdsmi_path.pth")
AMDSMI_PKG = "/opt/rocm/share/amd_smi/amdsmi/__init__.py"


def main():
    if not os.path.isfile(AMDSMI_PKG):
        print(f"   ERROR: bundled amdsmi package not found at {AMDSMI_PKG}")
        return 42
    if os.path.isfile(PTH):
        if open(PTH, encoding="utf-8").read().strip() == "/opt/rocm/share/amd_smi":
            print("   amdsmi .pth already present")
            return 0
        print(f"   ERROR: {PTH} exists with unexpected content")
        return 42
    with open(PTH, "w", encoding="utf-8", newline="\n") as f:
        f.write("/opt/rocm/share/amd_smi\n")
    print("   amdsmi .pth installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
