#!/usr/bin/env python3
"""Patch 40: keep AITER gated on gfx1x and add the VLLM_GFX1X_MOE_TUNE env.

Two edits to vllm/envs.py:

1. If VLLM_GFX1X_MOE_TUNE is not defined upstream yet, add it (default "0").
   Enabling it selects the tuned gfx1x MoE path (measured 11.3 -> 21.8 tok/s
   decode on gfx1151 MoE models).
2. Add a guard that force-disables the AITER master toggle
   (VLLM_ROCM_USE_AITER) when the device arch is gfx1*, because AITER
   sampler / MoE / FP8-linear kernels are unsupported on gfx1x. Attention
   backend selection is NOT touched (ROCM_AITER_UNIFIED_ATTN is used by
   qwen36-35b-a3b with the master toggle off; see models/registry.yaml).

STATUS: partially expected-to-need-adjustment.
- Verified assumption: envs.py defines VLLM_ROCM_USE_AITER and evaluates the
  env lazily through the `environment_variables` dict, so mutating
  os.environ at envs import time is honored.
- NOT verified: where VLLM_GFX1X_MOE_TUNE is consumed (fused-MoE layer) if
  upstream lacks it; this patch only guarantees the env exists. The apply
  step greps for a consumer and warns loudly if none is found.

Usage:
    python3 40_aiter_gfx1x_gating.py --src /opt/vllm          # apply
    python3 40_aiter_gfx1x_gating.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 40_aiter_gfx1x_gating"
EXIT_REAUDIT = 42

REL_PATH = "vllm/envs.py"

ENV_ENTRY = '''
    # {marker}
    # Tuned MoE path for gfx1x (RDNA) parts; see MANIFEST.md patch 40.
    "VLLM_GFX1X_MOE_TUNE":
    lambda: os.getenv("VLLM_GFX1X_MOE_TUNE", "0") == "1",
'''.replace("{marker}", MARKER)

GUARD_BLOCK = '''

# {marker}
# gfx1x (RDNA) iGPUs do not support AITER sampler / MoE / FP8-linear kernels;
# force the master toggle off on gfx1x even if the environment enables it.
def _gfx1x_disable_aiter():
    import sys
    try:
        import torch
        if not torch.cuda.is_available():
            return
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    except Exception:
        # torch/CUDA not ready at envs import time; the deployment env already
        # defaults VLLM_ROCM_USE_AITER=0, so failing soft here is acceptable.
        return
    if arch.split(":")[0].startswith("gfx1"):
        var = "VLLM_ROCM_USE_AITER"
        val = os.getenv(var)
        if val not in (None, "0"):
            print(f"WARN: {var}={val} overridden to 0 on {arch} "
                  f"(AITER unsupported on gfx1x)", file=sys.stderr)
        os.environ[var] = "0"


_gfx1x_disable_aiter()
'''.replace("{marker}", MARKER)


def find_target(src: Path) -> Path | None:
    cand = src / REL_PATH
    if cand.is_file():
        return cand
    matches = sorted(p for p in src.rglob("envs.py") if p.parent.name == "vllm")
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    target = find_target(src)
    if target is None:
        print(f"ERROR: {REL_PATH} not found under {src}; re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT
    content = target.read_text()

    env_ok = '"VLLM_GFX1X_MOE_TUNE"' in content
    guard_ok = "_gfx1x_disable_aiter" in content

    if args.check:
        if env_ok and guard_ok:
            print(f"OK: patch 40 present in {target}")
            return 0
        print(f"FAIL: patch 40 incomplete in {target} "
              f"(env entry: {env_ok}, gfx1x guard: {guard_ok})", file=sys.stderr)
        return 1

    changed = False

    if not env_ok:
        # Insert the new env entry right after the VLLM_ROCM_USE_AITER entry.
        anchor = None
        for i, line in enumerate(content.splitlines(keepends=True)):
            if '"VLLM_ROCM_USE_AITER"' in line:
                anchor = i
        if anchor is None:
            print(f"ERROR: VLLM_ROCM_USE_AITER not found in {target}. "
                  f"Upstream renamed the AITER toggle; re-audit.", file=sys.stderr)
            return EXIT_REAUDIT
        lines = content.splitlines(keepends=True)
        # Advance past the lambda continuation lines of that entry, i.e. until
        # the next `"KEY":` entry line, a blank line, or the dict-closing `}`.
        end = anchor + 1
        while (end < len(lines) and lines[end].strip()
               and not re.match(r'^    "[A-Z0-9_]+":', lines[end])
               and not lines[end].lstrip().startswith("}")):
            end += 1
        # If we stopped on a blank line directly before the dict's closing
        # brace, insert before the blank line so the entry stays in the dict.
        if end < len(lines) and not lines[end].strip():
            nxt = end + 1
            while nxt < len(lines) and not lines[nxt].strip():
                nxt += 1
            if nxt < len(lines) and lines[nxt].lstrip().startswith("}"):
                end = nxt
        lines.insert(end, ENV_ENTRY)
        content = "".join(lines)
        changed = True
        print(f"OK: added VLLM_GFX1X_MOE_TUNE env entry in {target}")
    else:
        print(f"SKIP: VLLM_GFX1X_MOE_TUNE already defined in {target} "
              f"(upstream adopted it?)")

    if not guard_ok:
        m = "\ndef __getattr__("
        idx = content.find(m)
        if idx == -1:
            print(f"ERROR: 'def __getattr__(' not found in {target}. envs.py "
                  f"layout changed; re-audit this patch.", file=sys.stderr)
            return EXIT_REAUDIT
        content = content[:idx] + GUARD_BLOCK + content[idx:]
        changed = True
        print(f"OK: added gfx1x AITER guard to {target}")
    else:
        print(f"SKIP: gfx1x AITER guard already present in {target}")

    if changed:
        target.write_text(content)

    # Best-effort: check whether anything consumes VLLM_GFX1X_MOE_TUNE outside
    # envs.py. Absence is a warning (re-audit trigger), not a hard failure.
    consumers = [p for p in src.rglob("*.py")
                 if p != target and "VLLM_GFX1X_MOE_TUNE" in p.read_text(errors="ignore")]
    if not consumers:
        print("WARN: no consumer of VLLM_GFX1X_MOE_TUNE found in the vLLM tree "
              "outside envs.py. The tuned gfx1x MoE path likely needs a "
              "follow-up patch in the fused-MoE layer; see MANIFEST.md.",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
