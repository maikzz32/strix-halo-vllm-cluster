#!/usr/bin/env python3
"""Quantisiert den LM-Head eines bereits requantisierten Checkpoints auf W4A16.

Nur die eine Gewichtsdatei, die lm_head.weight enthaelt, wird neu geschrieben;
alle anderen bleiben unberuehrt. Format wie im Rest des Checkpoints:
pack-quantized, 4 bit, group_size 32, asymmetrisch.
"""
import json, os, sys, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized import pack_to_int32

GROUP, BITS = 32, 4
src, dst = sys.argv[1], sys.argv[2]

idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]
target = "lm_head.weight"
fname = wm[target]
print(f"{target} liegt in {fname}")

def quant(w):
    wf = w.to(torch.float32)
    out, inp = wf.shape
    g = inp // GROUP
    b = wf.reshape(out, g, GROUP)
    qmin, qmax = -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1
    scale = (b.amax(2) - b.amin(2)).clamp(min=1e-8) / (qmax - qmin)
    zp = torch.round(qmin - b.amin(2) / scale).clamp(qmin, qmax)
    q = torch.round(b / scale.unsqueeze(2) + zp.unsqueeze(2)).clamp(qmin, qmax)
    deq = ((q - zp.unsqueeze(2)) * scale.unsqueeze(2)).reshape(out, inp)
    err = ((deq - wf).abs().mean() / wf.abs().mean().clamp(min=1e-8)).item()
    packed = pack_to_int32(q.reshape(out, inp).to(torch.int8), BITS)
    zpp = pack_to_int32(zp.to(torch.int8).t().contiguous(), BITS).t().contiguous()
    return packed, scale.to(torch.bfloat16), zpp, torch.tensor([out, inp], dtype=torch.int64), err

names = [k for k, v in wm.items() if v == fname]
out_t = {}
with safe_open(os.path.join(src, fname), framework="pt") as fh:
    for n in names:
        t = fh.get_tensor(n)
        if n == target:
            packed, scale, zp, shape, err = quant(t)
            out_t["lm_head.weight_packed"] = packed
            out_t["lm_head.weight_scale"] = scale
            out_t["lm_head.weight_zero_point"] = zp
            out_t["lm_head.weight_shape"] = shape
            before = t.numel() * 2
            after = packed.numel() * 4 + scale.numel() * 2 + zp.numel() * 4
            print(f"  {tuple(t.shape)} bf16: {before/2**30:.2f} GiB -> {after/2**30:.2f} GiB, "
                  f"relativer Fehler {err:.4%}")
        else:
            out_t[n] = t
save_file(out_t, os.path.join(dst, fname), metadata={"format": "pt"})

# Index aktualisieren
del wm[target]
for suf in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
    wm["lm_head." + suf] = fname
json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

# lm_head aus der ignore-Liste nehmen
cfg = os.path.join(dst, "config.json")
c = json.load(open(cfg, encoding="utf-8"))
q = c["quantization_config"]
n0 = len(q["ignore"])
q["ignore"] = [x for x in q["ignore"] if x != "lm_head"]
json.dump(c, open(cfg, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"ignore-Liste: {n0} -> {len(q['ignore'])}")
print("FERTIG")
