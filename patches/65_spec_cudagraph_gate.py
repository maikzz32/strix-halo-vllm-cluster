#!/usr/bin/env python3
"""Patch 65: env gate for the MTP speculator's own CUDA/HIP graphs.

Measured on the gfx1151 cluster (TP4 over RoCE, ROCm 7.14, 2026-09-03):
FULL_DECODE_ONLY graphs capture fine and the first replay hangs -- but only
when MTP is on. Without --speculative-config the same server (graphs on)
serves normally (E2). The difference is `Speculator.capture()`
(spec_decode/autoregressive/speculator.py): two extra graph managers, each
capturing in its own graph_capture() stream into the shared global pool, and
the draft-prefill graph records compute_logits -> ncclAllGather.

This patch adds VLLM_GFX1X_SPEC_CUDAGRAPH, resolved lazily (Ray/mp workers
receive env after import):
  "1"       (default) unchanged upstream behaviour
  "0"       both speculator managers run eager (target graphs untouched)
  "prefill" keep the draft-prefill graph, draft-decode eager

Usage: python3 65_spec_cudagraph_gate.py --src <site-packages|/opt/vllm> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse, sys
from pathlib import Path

MARKER = "gfx1151-patch: 65_spec_cudagraph_gate"
REL = "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
OLD = """    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
"""
NEW = f"""    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # {MARKER}
        # On gfx1151 TP4/RoCE the speculator's graphs deadlock at first
        # replay while the target model's graphs are fine. Gate them.
        import os

        _spec_cg = os.environ.get("VLLM_GFX1X_SPEC_CUDAGRAPH", "1").strip().lower()
        _draft_decode_eager = _spec_cg in ("0", "prefill")
        if _spec_cg == "0":
            cudagraph_mode = CUDAGraphMode.NONE
        # Initialize cudagraph manager for draft prefill (draft position 0).
        self.prefill_cudagraph_manager = SpeculatorCudaGraphManager(
"""
OLD2 = """        # PIECEWISE cudagraphs are not supported for draft decodes.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
"""
NEW2 = f"""        # PIECEWISE cudagraphs are not supported for draft decodes.
        # {MARKER}: eager draft decode when gated.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and not _draft_decode_eager:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE
"""

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="/opt/vllm"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); f = Path(a.src) / REL
    if not f.is_file():
        print(f"SKIP: {REL} not found under {a.src}"); return 0
    s = f.read_text()
    if a.check:
        ok = MARKER in s; print(("OK: patch 65 present in " if ok else "FAIL: patch 65 marker not found in ") + str(f)); return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 65 already applied to {f}"); return 0
    if s.count(OLD) != 1 or s.count(OLD2) != 1:
        print(f"ERROR: anchors found {s.count(OLD)}/{s.count(OLD2)} times in {f}; re-audit patch 65", file=sys.stderr); return 42
    f.write_text(s.replace(OLD, NEW, 1).replace(OLD2, NEW2, 1)); print(f"OK: patch 65 applied to {f}"); return 0

if __name__ == "__main__":
    sys.exit(main())
