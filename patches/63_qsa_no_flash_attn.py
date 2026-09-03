#!/usr/bin/env python3
"""Patch 63: let Qwen4Exp QSA (Qwen3.8-Flash-Next) start without flash_attn.

`Qwen4ExpQSAFlashAttentionImpl.__init__` refuses to build unless
`is_flash_attn_varlen_func_available()` is true:

    if not is_flash_attn_varlen_func_available():
        raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")

On gfx1151 that check is always false, because the only ROCm source of
`flash_attn_varlen_func` is `aiter`, whose JIT build of `module_aiter_core`
fails on import -- which is exactly why `scripts/harden_containers.sh`
uninstalls amd-aiter on every node. The result is a hard start failure:

    NotImplementedError: Qwen4Exp QSA requires FlashAttention

The gate is wrong for this class. QSA does not call flash attention at all:
`Qwen4ExpQSAAttention._run_qsa` calls only `impl.do_kv_cache_update` and
`impl.forward_qsa`, and `forward_qsa` ends in the Triton kernel
`qsa_sparse_paged_attention`. The gate is inherited boilerplate from
`FlashAttentionImpl`, and `super().__init__()` even runs before it, so it
guards no constructed state.

Exactly one FA-gated name is reached at runtime: `reshape_and_cache_flash`,
used by `FlashAttentionImpl.do_kv_cache_update`. On ROCm that is not a flash
attention symbol at all -- `fa_utils.py` binds it to
`vllm._custom_ops.reshape_and_cache_flash` OUTSIDE the try/except that guards
`flash_attn_varlen_func`. Verified in the running container on 2026-09-02:

    fa_available = False
    rcf          = <function reshape_and_cache_flash at 0x...>

The catch is that `flash_attn.py` re-imports that name only under
`if is_flash_attn_varlen_func_available():`, so its module global is absent
and the inherited `do_kv_cache_update` would die with NameError. This patch
therefore does two things, both confined to the AMD QSA class:

  1. replaces the gate with a direct bind of `reshape_and_cache_flash` from
     `fa_utils` onto the instance,
  2. overrides `do_kv_cache_update` to use that bound reference instead of
     `flash_attn.py`'s conditional module global. The body is a 1:1 copy of
     the upstream method.

`nvidia/qsa.py` carries the same anchor and is deliberately NOT touched:
on CUDA the gate is satisfied and meaningful.

Scope: this makes the model START. It contributes no throughput of its own.

Usage:
    python3 63_qsa_no_flash_attn.py --src /opt/vllm            # build-time
    python3 63_qsa_no_flash_attn.py --src <site-packages>      # runtime
    python3 63_qsa_no_flash_attn.py --src /opt/vllm --check    # verify only

Exit codes: 0 applied / check passed / nothing to do, 1 check failed,
42 = anchor moved -> re-audit, do not silently skip.

Re-audit trigger: exit 42, or upstream gaining a real RDNA flash-attention
source (then the gate becomes satisfiable and this patch can be dropped).
Written against vLLM v0.29.0rc1 (33898f832c) on 2026-09-02.
"""

import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 63_qsa_no_flash_attn"
# Containers patched on 2026-09-02 carry this earlier marker text; accept it.
LEGACY_MARKER = "gfx1151-patch: qsa-no-flash-attn"
EXIT_REAUDIT = 42

QSA_REL = "vllm/models/qwen4_exp/amd/qsa.py"
FA_UTILS_REL = "vllm/v1/attention/backends/fa_utils.py"
FLASH_ATTN_REL = "vllm/v1/attention/backends/flash_attn.py"

# --- anchor 1: the gate itself -------------------------------------------
GATE_OLD = """        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
"""
GATE_NEW = f"""        # {MARKER}
        # QSA never calls flash_attn_varlen_func -- forward_qsa runs the
        # Triton kernel qsa_sparse_paged_attention. The only FA-gated symbol
        # reached is reshape_and_cache_flash, which fa_utils binds on ROCm to
        # vllm._custom_ops OUTSIDE the flash_attn try/except. Bind it here so
        # do_kv_cache_update below does not depend on flash_attn.py's
        # conditional module global.
        from vllm.v1.attention.backends.fa_utils import (
            reshape_and_cache_flash as _qsa_reshape_and_cache_flash,
        )

        self._qsa_reshape_and_cache_flash = _qsa_reshape_and_cache_flash
"""

# --- anchor 2: insertion point for the override --------------------------
METHOD_ANCHOR = """        self.supports_quant_query_input = False

    def forward_qsa(
"""
METHOD_NEW = f"""        self.supports_quant_query_input = False

    # {MARKER}
    # 1:1 copy of FlashAttentionImpl.do_kv_cache_update, except that it calls
    # the instance-bound reshape_and_cache_flash. Upstream's version reads a
    # module global that only exists when is_flash_attn_varlen_func_available()
    # is true, which it never is on gfx1151.
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        self._qsa_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward_qsa(
"""

# --- fail-closed guards on files this patch does not edit ----------------
# fa_utils must still bind reshape_and_cache_flash at module level (4-space
# indent = inside the platform branch, NOT inside the try that guards
# flash_attn_varlen_func). If this moves into the try, the bind above becomes
# unsafe and the whole approach must be re-audited.
FA_UTILS_GUARD = "\n    reshape_and_cache_flash = ops.reshape_and_cache_flash\n"

# flash_attn.py must still import it conditionally -- that conditional import
# is the reason the override is needed. If upstream makes it unconditional,
# the override is dead weight and should be dropped.
FLASH_ATTN_GUARD = """if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import ("""

# The override copies this signature; if upstream changes it, the copy is stale.
DO_KV_SIGNATURE = """    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:"""


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

    qsa = find_file(src, QSA_REL)
    if qsa is None:
        print(
            f"SKIP: {QSA_REL} not found under {src}. This patch only applies to "
            f"trees carrying the merged qwen4_exp model (vllm#53896)."
        )
        return 0

    content = qsa.read_text()

    if args.check:
        if MARKER in content or LEGACY_MARKER in content:
            print(f"OK: patch 63 present in {qsa}")
            return 0
        print(f"FAIL: patch 63 marker not found in {qsa}", file=sys.stderr)
        return 1

    if MARKER in content or LEGACY_MARKER in content:
        print(f"SKIP: patch 63 already applied to {qsa}")
        return 0

    # --- guard 1: the gate is there, exactly once
    if content.count(GATE_OLD) != 1:
        return die(
            f"the FlashAttention gate was not found exactly once in {qsa} "
            f"(found {content.count(GATE_OLD)}). Upstream changed the QSA "
            f"constructor; re-audit this patch."
        )

    # --- guard 2: insertion point is there, exactly once
    if content.count(METHOD_ANCHOR) != 1:
        return die(
            f"the forward_qsa insertion anchor was not found exactly once in "
            f"{qsa} (found {content.count(METHOD_ANCHOR)}). Re-audit."
        )

    # --- guard 3: qsa.py must already import torch and AttentionType
    for needed in ("import torch", "AttentionType"):
        if needed not in content:
            return die(f"{qsa} no longer imports {needed!r}; the override would not compile.")

    # --- guard 4: fa_utils still binds reshape_and_cache_flash outside the try
    fa_utils = find_file(src, FA_UTILS_REL)
    if fa_utils is None:
        return die(f"{FA_UTILS_REL} not found under {src}; cannot verify the bind.")
    if FA_UTILS_GUARD not in fa_utils.read_text():
        return die(
            f"{fa_utils} no longer binds reshape_and_cache_flash at platform-branch "
            f"level. It may have moved inside the flash_attn try/except, which would "
            f"make this patch unsafe. Re-audit."
        )

    # --- guard 5: flash_attn.py still imports it conditionally, and the
    #     copied signature is unchanged
    flash_attn = find_file(src, FLASH_ATTN_REL)
    if flash_attn is None:
        return die(f"{FLASH_ATTN_REL} not found under {src}.")
    fa_text = flash_attn.read_text()
    if FLASH_ATTN_GUARD not in fa_text:
        return die(
            f"{flash_attn} no longer imports reshape_and_cache_flash under "
            f"'if is_flash_attn_varlen_func_available():'. If that import became "
            f"unconditional, the do_kv_cache_update override is unnecessary and "
            f"this patch should be reduced to removing the gate. Re-audit."
        )
    if DO_KV_SIGNATURE not in fa_text:
        return die(
            f"FlashAttentionImpl.do_kv_cache_update changed its signature in "
            f"{flash_attn}; the copy in this patch is stale. Re-audit."
        )

    patched = content.replace(GATE_OLD, GATE_NEW, 1).replace(METHOD_ANCHOR, METHOD_NEW, 1)
    qsa.write_text(patched)
    print(f"OK: patch 63 applied to {qsa}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
