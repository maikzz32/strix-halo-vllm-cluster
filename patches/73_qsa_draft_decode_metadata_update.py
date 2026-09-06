#!/usr/bin/env python3
"""Patch 73: let the QSA metadata builder update draft-decode metadata in place.

vLLM's autoregressive speculator has two multi-step draft paths
(`v1/worker/gpu/spec_decode/autoregressive/speculator.py`):

  * `_fused_multi_step_decode` builds attention metadata ONCE (step=1) and then,
    between draft steps, only re-runs `compute_slot_mappings()` plus each attention
    group's `update_draft_decode_metadata()`. Both are plain kernel launches into
    persistent buffers, so the whole k-step draft fits into ONE full CUDA graph.
  * `_multi_step_decode` rebuilds the metadata in Python for every step.

`_configure_fused_multi_step_decode` picks the fused path only if every attention
backend sets `supports_draft_decode_metadata_update`. Ours does not, so the cluster
logs

    Fused multi-step draft decode is not supported by attention backend(s)
    QWEN4_EXP_EXP_QSA_STATE; falling back to rebuilding attention metadata
    between draft steps.

Measured cost of that fallback (2026-09-06, TP4, qwen38_rest, ShareGPT c=1):
iteration time is 31.8 ms + 7.25 ms per draft step -- k=0 31.8 ms, k=3 53.7 ms,
k=4 60.8 ms, a fit accurate to 0.2 ms. One draft step (a SINGLE layer) costs 11x
what a target-model layer costs, because each step re-runs
`_build_draft_attn_metadata()`: a CPU `torch.zeros`, CPU arithmetic and a full
Python metadata object rebuild, outside any graph.

Why QSA can support the flag (`models/qwen4_exp/common/qsa_cache.py`):
  * Its outputs are the persistent `token_to_req_buffer`, `logical_positions_buffer`,
    `slot_mapping_buffer` and `k_work_metadata_buffer` allocated in `__init__`.
  * Its step-dependent inputs -- `seq_lens`, `slot_mapping`, `query_start_loc` --
    are GPU buffers owned by the speculator, refreshed before each call.
  * It never reads `seq_lens_cpu_upper_bound`, the only value `step` actually
    changes in `_build_draft_attn_metadata`.
  * Every Python-side scalar (num_tokens, num_reqs, request_scan_size, max_num_work)
    is step-independent for a uniform draft decode batch.

So the update is exactly "run `build_qsa_metadata` again with the same arguments":
one Triton launch writing the same persistent buffers. That is capture-safe, which
is what the base-class contract demands -- replay does not re-run this Python method.
This mirrors `mla/sparse_swa.py`, which relaunches its own Triton kernel; the Triton
backend's implementation is empty only because its metadata needs no derived values.

Gate: VLLM_GFX1X_QSA_FUSED_DRAFT=1 (default) / 0 = upstream behaviour (rebuild per
step), so one env var reverts the change without reinstalling. The gate sits on the
class attribute, read at import time -- that is the value the speculator consults at
startup to choose a path. Gating the update method instead would keep the fused path
selected while skipping the refresh, i.e. draft on stale metadata.

Usage: python3 73_qsa_draft_decode_metadata_update.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 73_qsa_draft_decode_metadata_update"
REL = "vllm/models/qwen4_exp/common/qsa_cache.py"

# --- 1. advertise the capability on the builder class -------------------------
OLD_CLS = '''class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
'''

NEW_CLS = '''class QSAMetadataBuilder(AttentionMetadataBuilder[QSAForwardMetadata]):
    """Build QSA metadata from vLLM's cache-group-specific common metadata."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # ''' + MARKER + '''
    # All step-dependent inputs are persistent GPU buffers and all outputs are the
    # builder's own persistent buffers, so one relaunch of build_qsa_metadata()
    # refreshes the metadata in place. Lets the speculator use the fused
    # (single-graph) multi-step draft path.
    #
    # The gate sits HERE, on the class attribute, not inside the update method:
    # the speculator reads this flag once at startup to choose between the fused
    # and the per-step-rebuild path. A gate inside the method would leave the
    # fused path selected while silently skipping the refresh, i.e. draft steps
    # running on stale metadata. Read at import time, which is correct because
    # the environment is fixed for the life of the worker process.
    supports_draft_decode_metadata_update: bool = (
        __import__("os").environ.get("VLLM_GFX1X_QSA_FUSED_DRAFT", "1") != "0"
    )
'''

# --- 2. remember the exact build arguments for the in-place update ------------
OLD_BUILD = '''        token_to_req, logical_positions, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            k_work_metadata_buffer=k_work_metadata if build_k_work else None,
            request_capacity=request_capacity,
        )
'''

NEW_BUILD = '''        # ''' + MARKER + ''': keep the arguments so update_draft_decode_metadata()
        # can relaunch the identical kernel between draft steps. Every tensor below
        # is persistent (buffers owned by this builder, or speculator input buffers),
        # so holding the reference does not pin a stale allocation.
        self._draft_update_args = (
            common_attn_metadata,
            k_work_metadata if build_k_work else None,
            request_capacity,
        )
        token_to_req, logical_positions, slot_mapping = build_qsa_metadata(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            k_work_metadata_buffer=k_work_metadata if build_k_work else None,
            request_capacity=request_capacity,
        )
'''

# --- 3. the update method itself, appended after build() ---------------------
OLD_TAIL = '''            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )


class QSAStateBackend(AttentionBackend):
'''

NEW_TAIL = '''            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
        )

    def update_draft_decode_metadata(self, metadata: QSAForwardMetadata) -> None:
        """Refresh step-dependent QSA metadata in place between draft steps.

        ''' + MARKER + '''

        Called by the speculator's fused multi-step draft loop after it has
        advanced the slot mappings, and during full CUDA graph capture. Replay
        does not re-run this method, so it must only launch kernels that write
        the persistent buffers the captured graph already points at -- which is
        exactly what build_qsa_metadata() does.

        Not gated: when VLLM_GFX1X_QSA_FUSED_DRAFT=0 the class attribute above is
        False, the speculator never selects the fused path, and this method is
        never called. Skipping the refresh here instead would silently draft on
        stale metadata.
        """
        args = getattr(self, "_draft_update_args", None)
        if args is None:
            # No build() on this builder yet: nothing to refresh.
            return
        common_attn_metadata, k_work_metadata_buffer, request_capacity = args
        build_qsa_metadata(
            common_attn_metadata,
            self.token_to_req_buffer,
            self.logical_positions_buffer,
            self.slot_mapping_buffer,
            storage_block_size=self.storage_block_size,
            compress_ratio=self.compress_ratio,
            circular_buffer_size=(
                self.kv_cache_spec.block_size if self.is_circular_buffer else 0
            ),
            k_work_metadata_buffer=k_work_metadata_buffer,
            request_capacity=request_capacity,
        )
        del metadata  # updated in place through the shared persistent buffers


class QSAStateBackend(AttentionBackend):
'''

STEPS = (
    ("builder class", OLD_CLS, NEW_CLS),
    ("build() arguments", OLD_BUILD, NEW_BUILD),
    ("update method", OLD_TAIL, NEW_TAIL),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/usr/local/lib64/python3.12/site-packages")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text()
    if a.check:
        ok = MARKER in s
        print(("OK: patch 73 present in " if ok else "FAIL: patch 73 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 73 already applied to {f}")
        return 0
    for name, old, _ in STEPS:
        if s.count(old) != 1:
            print(
                f"ERROR: {name} anchor found {s.count(old)} times in {f}; re-audit patch 73",
                file=sys.stderr,
            )
            return 42
    for _, old, new in STEPS:
        s = s.replace(old, new, 1)
    f.write_text(s)
    print(f"OK: patch 73 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
