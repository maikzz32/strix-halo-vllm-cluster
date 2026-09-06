#!/usr/bin/env python3
"""Patch 68: let the misaligned shared expert be replicated for compressed-tensors too.

Qwen3.8-Flash-Next has shared_expert_intermediate_size=640. A W4A16 checkpoint
quantized with group_size=128 therefore cannot be tensor-parallelized: 640/2=320
and 640/4=160 are both indivisible by 128, and vLLM aborts in
verify_group_size_divides_partition with

    ValueError: Weight input_size_per_partition=320 for layer
    '...mlp.shared_expert.down_proj' is not divisible by group_size=128

vLLM already has the fix -- Qwen3NextSparseMoeBlock can replicate the shared
expert instead of splitting it (disable_tp=True) -- but it only triggers for
Quark OCP-MX checkpoints, because the group size comes from
get_quark_ocp_mx_group_size(), which returns None for compressed-tensors.

This patch does two things:
  1. falls back to the compressed-tensors group size (target_scheme_map)
  2. allows replication whenever the partition is misaligned, instead of
     requiring expert parallelism or sequence parallelism to be on

Cost of replicating: the shared expert is 640x2560x3 per layer, in 4 bit about
2.5 MiB, 118 MiB over 48 layers -- negligible next to the bandwidth saved by a
fully quantized checkpoint.

Gate: VLLM_GFX1X_SHARED_EXPERT_REPLICATE=1 (default) / 0 = upstream behaviour.

Usage: python3 68_shared_expert_replicate.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse, sys
from pathlib import Path

MARKER = "gfx1151-patch: 68_shared_expert_replicate"
REL = "vllm/model_executor/models/qwen3_next.py"

HELPER = f'''

# {MARKER}
def _gfx_ct_group_size(quant_config):
    """Weight group size of a compressed-tensors scheme, or None.

    get_quark_ocp_mx_group_size() only answers for Quark checkpoints; without
    this the shared expert of a group-quantized W4A16 checkpoint cannot be
    replicated and TP loading fails outright.
    """
    tsm = getattr(quant_config, "target_scheme_map", None)
    if not tsm:
        return None
    for entry in tsm.values():
        weights = entry.get("weights") if isinstance(entry, dict) else None
        gs = getattr(weights, "group_size", None)
        if gs:
            return int(gs)
    return None
'''

OLD_CALL = """        self.replicate_shared_expert = _should_replicate_misaligned_shared_expert("""
NEW_CALL = f"""        # {MARKER}: compressed-tensors reports its group size elsewhere
        import os as _gfx_os

        if shared_expert_group_size is None and _gfx_os.environ.get(
            "VLLM_GFX1X_SHARED_EXPERT_REPLICATE", "1"
        ) != "0":
            shared_expert_group_size = _gfx_ct_group_size(quant_config)
        self.replicate_shared_expert = _should_replicate_misaligned_shared_expert("""

OLD_RAISE = """    if enable_expert_parallel or is_sequence_parallel:
        return True
"""
NEW_RAISE = f"""    if enable_expert_parallel or is_sequence_parallel:
        return True

    # {MARKER}: replicating costs a few MiB per layer and is the only way to
    # serve a group-quantized shared expert whose partition is misaligned.
    import os as _gfx_os

    if _gfx_os.environ.get("VLLM_GFX1X_SHARED_EXPERT_REPLICATE", "1") != "0":
        return True
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/usr/local/lib64/python3.12/site-packages")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text(encoding="utf-8")
    if a.check:
        ok = MARKER in s
        print(("OK: patch 68 present in " if ok else "FAIL: patch 68 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 68 already applied to {f}")
        return 0
    if s.count(OLD_CALL) != 1 or s.count(OLD_RAISE) != 1:
        print(f"ERROR: anchors found {s.count(OLD_CALL)}/{s.count(OLD_RAISE)} times (want 1/1); re-audit patch 68", file=sys.stderr)
        return 42
    anchor = "def _should_replicate_misaligned_shared_expert("
    if anchor not in s:
        print("ERROR: _should_replicate_misaligned_shared_expert not found; re-audit patch 68", file=sys.stderr)
        return 42
    s = s.replace(anchor, HELPER.lstrip("\n") + "\n\n" + anchor, 1)
    s = s.replace(OLD_RAISE, NEW_RAISE, 1)
    s = s.replace(OLD_CALL, NEW_CALL, 1)
    f.write_text(s, encoding="utf-8")
    print(f"OK: patch 68 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
