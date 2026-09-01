#!/usr/bin/env python3
"""Patch 61: load the local GLM-5.3-Flash int4 re-quant on the glm5next loader.

Two adaptations in ``vllm/models/glm5next/nvidia/model.py`` (PR #53906
branch), needed because the local checkpoint (/home/maik/glm53_flash, based
on cyankiwi/GLM-5.3-Flash-AWQ-INT4) does not match the loader's assumptions:

1. **Name remap** — the checkpoint carries AWQ-era tensor names; the loader
   expects the upstream zai-org names (evidence: glm_remap.py analysis,
   shapes compared tensor-by-tensor against the original header):
     attn_hc.{base,fn,scale}  -> hc_attn_{base,fn,scale}
     ffn_hc.{base,fn,scale}   -> hc_ffn_{base,fn,scale}
     self_attn.forget_gate.X  -> self_attn.X
     self_attn.conv1d [3*8192] -> self_attn.{q,k,v}_conv1d [8192] each
     (fused-tensor order ASSUMED q,k,v — unprovable from the checkpoint, but
     a wrong order breaks attention loudly on the first prompt, not silently)

2. **Load-time int4 dequant of BF16-built projections** — the model class
   hard-codes ``quant_config=None`` for KDA projections (kda.py: "KDA
   projections remain BF16 because fp8 checkpoints omit their scales") and
   for MLA projections (model.py: "MLA projections are BF16 in checkpoint"),
   but the re-quant packed them anyway (llm-compressor pack-quantized int4,
   group 32, asymmetric). Without this, weight loading dies with
   ``KeyError: 'layers.N.self_attn.<proj>.weight_packed'``.
   Dequantized at load time (verified by a synthetic roundtrip test, every
   element within its group quantization bound). MoE experts stay packed —
   the model builds those quantized (TRITON WNA16).

Layout (verified against the safetensors headers): weight_packed [out, in/8]
int32 (8 nibbles along the input dim), weight_scale [out, in/32] BF16,
weight_zero_point [out/8, in/32] int32 (8 nibbles packed along the OUTPUT
dim; unpack needs a permute — plain reshape mixes block and group axes).

SKIP semantics: no glm5next package in the tree (stable/main builds without
PR #53906) -> SKIP, exit 0. glm5next present but the load_weights anchor
missing -> exit 42 (re-audit). The patched file is ast.parse'd before
writing (fail closed).

Usage:
    python3 61_glm53_awq_weight_remap.py --src /opt/vllm          # apply
    python3 61_glm53_awq_weight_remap.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / skipped / check passed, 1 = check failed,
            42 = glm5next present but anchors moved (re-audit needed).
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 61_glm53_awq_weight_remap"
EXIT_REAUDIT = 42

REL_PATH = "vllm/models/glm5next/nvidia/model.py"

ANCHOR = ("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
          "        stacked_params_mapping = [")
REPLACEMENT = ("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
               "        # {marker}: remap + dequant the local int4 re-quant\n"
               "        weights = _glm53_awq_name_remap(weights)\n"
               "        stacked_params_mapping = [").replace("{marker}", MARKER)

HELPER_BLOCK = '''

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


class _Glm53PackedAttnDequant:
    """Buffers the packed triplet per BF16-built attention projection and
    emits BF16 weight. The glm5next class hard-codes quant_config=None for
    KDA projections AND MLA projections ("BF16 in checkpoint"), but the
    local re-quant packed them anyway - so dequantize at load time.
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
    """{marker}: map AWQ-era tensor names of the local GLM-5.3-Flash int4
    checkpoint onto the names this model class expects, and dequantize the
    packed projections the model class keeps in BF16."""
    dequant = _Glm53PackedAttnDequant()
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

'''.replace("{marker}", MARKER)


def glm5_present(src: Path) -> bool:
    cand = src / REL_PATH
    if cand.is_file():
        return True
    models_dir = src / "vllm" / "model_executor" / "models"
    if models_dir.is_dir():
        registry = models_dir / "registry.py"
        if registry.is_file() and "glm5" in registry.read_text(
                errors="ignore").lower():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    if not glm5_present(src):
        print(f"SKIP: no glm5next package under {src} — patch 61 only "
              f"applies to GLM-5.3 builds (PR #53906 branch).")
        return 0

    target = src / REL_PATH
    if not target.is_file():
        matches = sorted(p for p in src.rglob("model.py")
                         if "glm5next" in str(p).lower())
        if matches:
            target = matches[0]
        else:
            print(f"ERROR: glm5next present but {REL_PATH} not found under "
                  f"{src}. Upstream moved the model file; re-audit.",
                  file=sys.stderr)
            return EXIT_REAUDIT

    text = target.read_text(encoding="utf-8")

    if args.check:
        ok = MARKER in text and "_glm53_awq_name_remap(weights)" in text
        print(("OK" if ok else "MISSING") + f": patch 61 in {target}")
        return 0 if ok else 1

    if MARKER in text:
        print(f"SKIP: patch 61 already applied to {target}")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"ERROR: expected exactly 1 load_weights anchor in {target}, "
              f"found {text.count(ANCHOR)}. Re-audit needed.", file=sys.stderr)
        return EXIT_REAUDIT

    i = text.find("\nclass ")
    if i < 0:
        print(f"ERROR: no class definition in {target}", file=sys.stderr)
        return EXIT_REAUDIT
    text = text[:i] + HELPER_BLOCK + text[i:]
    text = text.replace(ANCHOR, REPLACEMENT)

    ast.parse(text)  # fail closed: never write an unparseable tree
    target.write_text(text, encoding="utf-8", newline="\n")
    print(f"OK: patch 61 applied to {target} (name remap + int4 dequant)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
