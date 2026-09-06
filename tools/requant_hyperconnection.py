#!/usr/bin/env python3
"""Quantisiert die Hyper-Connection-Projektionen -- ZERSTOERT DAS MODELL, nicht benutzen.

Die Hyper-Connection-Mischer sind mit 1,23 GiB der letzte groessere Posten in BF16, der bei
jedem Token vollstaendig gelesen wird. Rechnerisch braechten sie 1,15 ms je Iteration.

Technisch laedt es, wenn man BEIDE Teile des Merges quantisiert (input_mix_weight_down und
block_inject_weight) und keinen der drei Namen -- inklusive des synthetischen
_input_mix_padding -- auf der Ignore-Liste laesst. Dann bekommen alle dasselbe Schema.

ABER gemessen (zwei Nodes, sonst identische Konfiguration):

    ohne                40,34 tok/s   Akzeptanz 54,0 %   Laenge 2,62
    mit quantisierten    6,58 tok/s   Akzeptanz 19,8 %   Laenge 1,59

Die Hyper-Connections mischen den Residual-Strom ueber alle 48 Layer; Quantisierungsfehler
akkumulieren dort, statt sich lokal auszumitteln. Der Entwurfskopf trifft kaum noch zu.
Deshalb erzeugt vLLM sie ausdruecklich mit quant_config=None. Patch 70 bleibt per
VLLM_GFX1X_HC_QUANT=0 aus.

Quantisiert die Hyper-Connection-Projektionen -- FUNKTIONIERT NICHT, siehe unten.

Die Hyper-Connection-Mischer sind mit 1,23 GiB der letzte groessere Posten in BF16,
der bei jedem Token vollstaendig gelesen wird. Ihre Formen waeren sauber
quantisierbar (Eingangsdimension 10240 bzw. 320, beide durch 32 teilbar), und
rechnerisch braechten sie rund 2 ms je Token.

ABER: vLLM fuehrt input_mix_weight_down, block_inject_weight und ein synthetisch
erzeugtes _input_mix_padding zu einem MergedColumnParallelLinear zusammen. Dieses
Modul verlangt fuer alle Teilgewichte dasselbe Quantisierungsschema. Das Padding
existiert im Checkpoint nicht und bekommt deshalb keines:

    ValueError: Found a different quantization schemes for
    ['input_mix_weight_down', 'block_inject_weight', '_input_mix_padding']

Das passiert, sobald die Hyper-Connections ueberhaupt eine quant_config sehen --
unabhaengig davon, welche der Gewichte man quantisiert. Der Weg ist damit ohne
Aenderung an der Merge-Logik versperrt. Das Skript bleibt als Ausgangspunkt hier,
falls das Padding spaeter eine eigene Behandlung bekommt.

import argparse, json, os, re, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized import pack_to_int32

GROUP, BITS = 32, 4
QMIN, QMAX = -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1
TARGETS = (
    r"_hyper_connection\.input_mix_weight_up\.weight$",
    r"^model\.language_model\.hyper_connection_mixer\.input_mix_weight_up\.weight$",
)
SKIP = (r"^mtp\.", r"visual", r"\.ple\.")
TARGET_RE = re.compile("|".join(TARGETS))
SKIP_RE = re.compile("|".join(SKIP))
is_target = lambda n: bool(TARGET_RE.search(n)) and not SKIP_RE.search(n)


def quant_one(w):
    wf = w.to(torch.float32)
    out, inp = wf.shape
    g = inp // GROUP
    b = wf.reshape(out, g, GROUP)
    s = (b.amax(2) - b.amin(2)).clamp(min=1e-8) / (QMAX - QMIN)
    z = torch.round(QMIN - b.amin(2) / s).clamp(QMIN, QMAX)
    q = torch.round(b / s.unsqueeze(2) + z.unsqueeze(2)).clamp(QMIN, QMAX)
    deq = ((q - z.unsqueeze(2)) * s.unsqueeze(2)).reshape(out, inp)
    rel = ((deq - wf).abs().mean() / wf.abs().mean().clamp(min=1e-8)).item()
    packed = pack_to_int32(q.reshape(out, inp).to(torch.int8), BITS)
    zpp = pack_to_int32(z.to(torch.int8).t().contiguous(), BITS).t().contiguous()
    return packed, s.to(torch.bfloat16), zpp, torch.tensor([out, inp], dtype=torch.int64), rel


ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()
idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
wm = idx["weight_map"]
files = sorted(set(wm.values()))
tg = [k for k in wm if is_target(k)]
print(f"{len(tg)} Hyper-Connection-Gewichte in {len(files)} Dateien", flush=True)
os.makedirs(a.dst, exist_ok=True)
new_map, errs = {}, []
for i, f in enumerate(files):
    names = [k for k, v in wm.items() if v == f]
    out_t = {}
    with safe_open(os.path.join(a.src, f), framework="pt") as fh:
        for n in names:
            t = fh.get_tensor(n)
            if is_target(n):
                packed, scale, zp, shape, rel = quant_one(t)
                base = n[: -len(".weight")]
                for suf, val in (("weight_packed", packed), ("weight_scale", scale),
                                 ("weight_zero_point", zp), ("weight_shape", shape)):
                    out_t[f"{base}.{suf}"] = val
                    new_map[f"{base}.{suf}"] = f
                errs.append(rel)
            else:
                out_t[n] = t
                new_map[n] = f
    save_file(out_t, os.path.join(a.dst, f), metadata={"format": "pt"})
    print(f"[{i+1}/{len(files)}] {f}", flush=True)
idx["weight_map"] = new_map
json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
for e in os.listdir(a.src):
    if e.endswith(".safetensors") or e == "model.safetensors.index.json":
        continue
    s = os.path.join(a.src, e)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(a.dst, e))
cfg = os.path.join(a.dst, "config.json")
c = json.load(open(cfg, encoding="utf-8"))
q = c["quantization_config"]
PAT = re.compile(r"_hyper_connection\.input_mix_weight_up$|^model\.language_model\.hyper_connection_mixer\.input_mix_weight_up$")
q["ignore"] = [n for n in q["ignore"] if n.startswith("mtp.") or "visual" in n or ".ple." in n or not PAT.search(n)]
json.dump(c, open(cfg, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"{len(errs)} Gewichte, mittlerer relativer Fehler {sum(errs)/len(errs):.4%}")
print("FERTIG")
