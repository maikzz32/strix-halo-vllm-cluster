#!/usr/bin/env python3
"""Runtime patch v2 for the local GLM-5.3-Flash int4 re-quant checkpoint.

Two adaptations vs. the upstream PR #53906 glm5next loader:

1. Name remap (from v1): AWQ-era tensor names -> upstream names
   (attn_hc.* -> hc_attn_*, ffn_hc.* -> hc_ffn_*, forget_gate.* flattened,
   fused conv1d [3*8192] split into q/k/v_conv1d, order assumed q,k,v).

2. NEW: load-time dequant of the packed int4 MLA projections. The model
   class builds MLA attention with quant_config=None ("MLA projections are
   BF16 in checkpoint"), but the local re-quant packed kv_b_proj / q_b_proj /
   o_proj in the sparse layers (3, 7, 11, ..., 43). Format (verified against
   the safetensors headers): llm-compressor pack-quantized int4, group 32,
   asymmetric; weight_packed [out, in/8] int32 (8 nibbles along input dim),
   weight_scale [out, in/32] BF16, weight_zero_point [out/8, in/32] int32
   (8 nibbles packed along the OUTPUT dim). KDA layers keep their packed
   tensors - the model builds those quantized.

Idempotent (v1 -> v2 upgrade in place), fail-closed, ast.parse before write.
"""

import ast
import io
import sys

P = "/usr/local/lib64/python3.12/site-packages/vllm/models/glm5next/nvidia/model.py"

HELPER_V2 = '''

_GLM53_SPARSE_LAYERS = frozenset(3 + 4 * i for i in range(11))  # 3..43 step 4
# remap version: v2.2 (KDA q/k/v/o + MLA kv_b/q_b/o dequant, no layer gate)


def _glm53_dequant_int4_g32(packed, scale, zp_packed, group=32):
    """llm-compressor pack-quantized int4 (group 32, asymmetric) -> BF16."""
    import torch
    out_features = scale.shape[0]
    in_features = scale.shape[1] * group
    shifts = 4 * torch.arange(8, dtype=torch.int64)
    p32 = packed.to(torch.int64) & 0xFFFFFFFF
    q = (p32.unsqueeze(-1) >> shifts) & 0xF          # [out, in/8, 8]
    q = q.reshape(out_features, in_features)
    z32 = zp_packed.to(torch.int64) & 0xFFFFFFFF
    zp = (z32.unsqueeze(-1) >> shifts) & 0xF         # [out/8, groups, 8]
    # nibbles run along the OUTPUT dim: out = block*8 + lane
    zp = zp.permute(0, 2, 1).reshape(out_features, -1)  # [out, groups]
    w = (q.float() - zp.repeat_interleave(group, dim=1).float())
    w = w * scale.repeat_interleave(group, dim=1).float()
    return w.to(torch.bfloat16)


class _Glm53PackedMlaDequant:
    """Buffers the packed triplet per BF16-built attention projection and
    emits BF16 weight. The glm5next class hard-codes quant_config=None for
    KDA projections AND MLA projections ("BF16 in checkpoint"), but the
    local re-quant packed them anyway - so we dequantize at load time.
    MoE experts stay packed (the model builds those quantized)."""

    _PROJS = (".self_attn.kv_b_proj.", ".self_attn.q_b_proj.",
              ".self_attn.o_proj.", ".self_attn.q_proj.",
              ".self_attn.k_proj.", ".self_attn.v_proj.")
    _SUFFIXES = (".weight_packed", ".weight_scale", ".weight_zero_point")

    def __init__(self):
        self.buf = {}

    def __call__(self, name, w):
        if ".self_attn." not in name:
            return [(name, w)]
        if not any(p in name for p in self._PROJS):
            return [(name, w)]
        if name.endswith(".weight_shape"):
            return []  # redundant: original shape derivable from scale
        for suffix in self._SUFFIXES:
            if name.endswith(suffix):
                stem = name[: -len(suffix)]
                entry = self.buf.setdefault(stem, {})
                entry[suffix] = w
                if all(s in entry for s in self._SUFFIXES):
                    del self.buf[stem]
                    deq = _glm53_dequant_int4_g32(
                        entry[".weight_packed"], entry[".weight_scale"],
                        entry[".weight_zero_point"])
                    return [(stem + ".weight", deq)]
                return []
        return [(name, w)]


def _glm53_awq_name_remap(weights):
    """Map AWQ-era tensor names of the local GLM-5.3-Flash int4 checkpoint
    onto the names this model class expects (PR #53906), and dequantize the
    packed MLA projections the model class keeps in BF16."""
    dequant = _Glm53PackedMlaDequant()
    renames = (
        (".attn_hc.base", ".hc_attn_base"),
        (".attn_hc.fn", ".hc_attn_fn"),
        (".attn_hc.scale", ".hc_attn_scale"),
        (".ffn_hc.base", ".hc_ffn_base"),
        (".ffn_hc.fn", ".hc_ffn_fn"),
        (".ffn_hc.scale", ".hc_ffn_scale"),
        (".self_attn.forget_gate.", ".self_attn."),
    )
    for name, w in weights:
        # fused conv1d [3*8192] -> separate q/k/v conv1d [8192] each
        if name.endswith(".self_attn.conv1d.weight"):
            part = w.shape[0] // 3
            stem = name[: -len("conv1d.weight")]
            for i, head in enumerate(("q", "k", "v")):
                for item in dequant(stem + head + "_conv1d.weight",
                                    w[i * part : (i + 1) * part]):
                    yield item
            continue
        for old, new in renames:
            if old in name:
                name = name.replace(old, new)
                break
        for item in dequant(name, w):
            yield item

'''

ANCHOR = ("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
          "        stacked_params_mapping = [")
REPLACEMENT = ("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
               "        weights = _glm53_awq_name_remap(weights)\n"
               "        stacked_params_mapping = [")


def find_helper_span(s):
    """Locate the injected helper block (any version): from the first
    helper marker up to the first REAL model class (never the helper's own
    class - a plain '\\nclass ' search would truncate the block and leave a
    shadowing duplicate behind)."""
    start = s.find("\n_GLM53_SPARSE_LAYERS")
    if start < 0:
        start = s.find("\ndef _glm53_awq_name_remap")
    if start < 0:
        return None
    end = s.find("\nclass Glm5Next", start)
    if end < 0:
        return None
    return start, end


def main():
    s = io.open(P, encoding="utf-8").read()

    if "remap version: v2.2" in s:
        print("   v2.2 already applied")
        return 0
    if ANCHOR not in s and "weights = _glm53_awq_name_remap(weights)" not in s:
        print(f"   ERROR: no load_weights anchor in {P}")
        return 42

    span = find_helper_span(s)
    if span is not None:
        # v1 present: replace the helper block, keep the call site
        s = s[: span[0]] + HELPER_V2 + s[span[1]:]
    else:
        i = s.find("\nclass ")
        if i < 0:
            print("   ERROR: no class definition found")
            return 42
        s = s[:i] + HELPER_V2 + s[i:]
    if "weights = _glm53_awq_name_remap(weights)" not in s:
        if s.count(ANCHOR) != 1:
            print(f"   ERROR: expected 1 anchor, found {s.count(ANCHOR)}")
            return 42
        s = s.replace(ANCHOR, REPLACEMENT)

    ast.parse(s)  # fail closed: never write an unparseable tree
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("   v2 injected (name remap + MLA int4 dequant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
