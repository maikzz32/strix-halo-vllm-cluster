#!/usr/bin/env python3
"""Runtime patch: let the ROCM_AITER_MLA_SPARSE backend init without aiter.

ROCMAiterMLASparseMetadataBuilder.__init__ unconditionally does
``from aiter import dtypes, get_mla_metadata_info_v1`` to size the persistent
work-splitting buffers. aiter is uninstalled on the gfx1151 nodes (it crashes
on gfx1x), so engine init dies in initialize_kv_cache. Those buffers are only
consumed by the aiter sparse decode kernel path, which is never selected on
gfx1151: patch 58 forces the ragged Triton lane
(VLLM_GFX1X_FORCE_TRITON_SPARSE=1), and with kv_cache_dtype != fp8 and
qk_rope_head_dim == 0 the geometry check alone already selects it.

The patch wraps the import in try/except and leaves all six buffers None when
aiter is missing. Idempotent, fail-closed (ast.parse before write).

Target: vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
"""

import ast
import io
import sys

P = "/usr/local/lib64/python3.12/site-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"

MARKER = "gfx1x-noaiter persistent-mla skip"

ANCHOR_IMPORT = "        from aiter import dtypes, get_mla_metadata_info_v1\n"
REPLACEMENT_IMPORT = (
    f"        # {MARKER}: buffers below are only consumed by the aiter sparse\n"
    "        # decode kernel; the forced Triton lane (patch 58) never selects\n"
    "        # it on gfx1151. aiter is absent on gfx1x -> leave them None.\n"
    "        try:\n"
    "            from aiter import dtypes, get_mla_metadata_info_v1\n"
    "        except ImportError:\n"
    "            get_mla_metadata_info_v1 = None\n"
)

ANCHOR_QDTYPE = "        q_dtype = self.model_dtype\n"
ANCHOR_END = "\n        self._prev_req_extent: int = 0"

IF_HEAD = (
    "        if get_mla_metadata_info_v1 is None:\n"
    "            self._mla_work_meta_data = None\n"
    "            self._mla_work_indptr = None\n"
    "            self._mla_work_info_set = None\n"
    "            self._mla_reduce_indptr = None\n"
    "            self._mla_reduce_final_map = None\n"
    "            self._mla_reduce_partial_map = None\n"
    "        else:\n"
)


def main():
    s = io.open(P, encoding="utf-8").read()

    if MARKER in s:
        print("   no-aiter backend patch already applied")
        return 0
    if s.count(ANCHOR_IMPORT) != 1:
        print(f"   ERROR: expected 1 aiter import anchor, found {s.count(ANCHOR_IMPORT)}")
        return 42
    if s.count(ANCHOR_QDTYPE) != 1:
        print(f"   ERROR: expected 1 q_dtype anchor, found {s.count(ANCHOR_QDTYPE)}")
        return 42

    s = s.replace(ANCHOR_IMPORT, REPLACEMENT_IMPORT)

    i = s.find(ANCHOR_QDTYPE)
    j = s.find(ANCHOR_END, i)
    if j < 0:
        print("   ERROR: end anchor (_prev_req_extent) not found after q_dtype")
        return 42
    block = s[i:j]
    indented = "".join(
        ("    " + ln if ln.strip() else ln) for ln in block.splitlines(True)
    )
    s = s[:i] + IF_HEAD + indented + s[j:]

    ast.parse(s)  # fail closed: never write an unparseable tree
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("   no-aiter persistent-MLA skip injected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
