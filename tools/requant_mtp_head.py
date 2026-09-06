#!/usr/bin/env python3
"""Quantisiert die dichten Gewichte des MTP-Entwurfskopfs -- LOHNT NICHT, siehe unten.

Gemessen: 38,79 statt 39,78 tok/s. Die Akzeptanzrate faellt von 53,3 auf 50,6 Prozent
(Laenge 2,60 -> 2,52), weil der Entwurfskopf ungenauer raet. Der Verlust an akzeptierten
Tokens ist groesser als der Gewinn an Bandbreite. Das Skript bleibt als Beleg hier.


Der Entwurfskopf wird bei jedem Entwurfsschritt gerechnet, bei k=3 also dreimal
je Iteration. Seine dichten Projektionen sind zusammen rund 0,15 GiB in BF16.
Die fusionierten Experten (3D, 4,69 GiB) bleiben unangetastet -- sie haben ein
anderes Format als compressed-tensors erwartet.
"""
import json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized import pack_to_int32

GROUP, BITS = 32, 4
QMIN, QMAX = -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1
TARGETS = (
    r"^mtp\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
    r"^mtp\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$",
    r"^mtp\.(fc_embedding|fc_hidden)\.weight$",
)
TARGET_RE = re.compile("|".join(TARGETS))

def quant_one(w):
    wf = w.to(torch.float32); out, inp = wf.shape; g = inp // GROUP
    b = wf.reshape(out, g, GROUP)
    s = (b.amax(2) - b.amin(2)).clamp(min=1e-8) / (QMAX - QMIN)
    z = torch.round(QMIN - b.amin(2) / s).clamp(QMIN, QMAX)
    q = torch.round(b / s.unsqueeze(2) + z.unsqueeze(2)).clamp(QMIN, QMAX)
    deq = ((q - z.unsqueeze(2)) * s.unsqueeze(2)).reshape(out, inp)
    rel = ((deq - wf).abs().mean() / wf.abs().mean().clamp(min=1e-8)).item()
    return (pack_to_int32(q.reshape(out, inp).to(torch.int8), BITS), s.to(torch.bfloat16),
            pack_to_int32(z.to(torch.int8).t().contiguous(), BITS).t().contiguous(),
            torch.tensor([out, inp], dtype=torch.int64), rel)

src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]
tg = [k for k in wm if TARGET_RE.match(k)]
files = sorted(set(wm[k] for k in tg))
print(f"{len(tg)} Gewichte in {len(files)} Dateien: {files}", flush=True)
os.makedirs(dst, exist_ok=True)
errs = []
for f in files:
    names = [k for k, v in wm.items() if v == f]
    out_t = {}
    with safe_open(os.path.join(src, f), framework="pt") as fh:
        for n in names:
            t = fh.get_tensor(n)
            if TARGET_RE.match(n):
                packed, scale, zp, shape, rel = quant_one(t)
                base = n[: -len(".weight")]
                for suf, val in (("weight_packed", packed), ("weight_scale", scale),
                                 ("weight_zero_point", zp), ("weight_shape", shape)):
                    out_t[f"{base}.{suf}"] = val
                    wm[f"{base}.{suf}"] = f
                del wm[n]
                errs.append(rel)
            else:
                out_t[n] = t
    save_file(out_t, os.path.join(dst, f), metadata={"format": "pt"})
    print(f"geschrieben: {f}", flush=True)
idx["weight_map"] = wm
json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
shutil.copy2(os.path.join(src, "config.json"), os.path.join(dst, "config.json"))
cfg = os.path.join(dst, "config.json")
c = json.load(open(cfg, encoding="utf-8"))
q = c["quantization_config"]
PAT = re.compile(r"^mtp\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
                 r"|^mtp\.layers\.\d+\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$"
                 r"|^mtp\.(fc_embedding|fc_hidden)$")
q["ignore"] = [n for n in q["ignore"] if not PAT.match(n)]
# der Sammel-Eintrag re:mtp\..* wuerde alles wieder ausnehmen -> praezisieren
q["ignore"] = [n for n in q["ignore"] if n != "re:mtp\..*"]
q["ignore"] += ["re:mtp\..*experts.*", "re:mtp\..*hyper_connection.*",
                "re:mtp\..*norm.*", "re:mtp\..*\.gate$", "re:mtp\..*indexer.*"]
json.dump(c, open(cfg, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"{len(errs)} Gewichte, mittlerer relativer Fehler {sum(errs)/len(errs):.4%}")
print("FERTIG")
