#!/usr/bin/env python3
"""Quantisiert Experten-Router und QSA-Indexer (0,154 GiB je Token).

Beide treffen diskrete Entscheidungen -- der Router waehlt die Experten, der
Indexer die Attention-Bloecke. Quantisierungsrauschen schlaegt dort direkter
durch als bei einer gewoehnlichen Projektion, deshalb ist die Ausgabe nach dem
Umbau gegen das Original zu pruefen.
"""
import json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized import pack_to_int32

GROUP, BITS = 32, 4
QMIN, QMAX = -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS = (r"\.mlp\.gate\.weight$", r"\.self_attn\.indexer\.index_qk_proj\.weight$")
SKIP = (r"^mtp\.", r"visual", r"\.ple\.")
TR = re.compile("|".join(TARGETS)); SK = re.compile("|".join(SKIP))
is_target = lambda n: bool(TR.search(n)) and not SK.search(n)

def quant_one(w):
    wf = w.to(DEV, torch.float32); out, inp = wf.shape; g = inp // GROUP
    b = wf.reshape(out, g, GROUP)
    s = (b.amax(2) - b.amin(2)).clamp(min=1e-8) / (QMAX - QMIN)
    z = torch.round(QMIN - b.amin(2) / s).clamp(QMIN, QMAX)
    q = torch.round(b / s.unsqueeze(2) + z.unsqueeze(2)).clamp(QMIN, QMAX)
    deq = ((q - z.unsqueeze(2)) * s.unsqueeze(2)).reshape(out, inp)
    rel = ((deq - wf).abs().mean() / wf.abs().mean().clamp(min=1e-8)).item()
    return (pack_to_int32(q.reshape(out, inp).to(torch.int8).cpu(), BITS),
            s.to(torch.bfloat16).cpu(),
            pack_to_int32(z.to(torch.int8).t().contiguous().cpu(), BITS).t().contiguous(),
            torch.tensor([out, inp], dtype=torch.int64), rel)

src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]; files = sorted(set(wm.values()))
tg = [k for k in wm if is_target(k)]
print(f"{len(tg)} Gewichte (Router + Indexer), Geraet {DEV}", flush=True)
os.makedirs(dst, exist_ok=True)
new_map, errs = {}, []; t0 = time.time()
for i, f in enumerate(files):
    names = [k for k, v in wm.items() if v == f]
    out_t = {}
    with safe_open(os.path.join(src, f), framework="pt") as fh:
        for n in names:
            t = fh.get_tensor(n)
            if is_target(n):
                packed, scale, zp, shape, rel = quant_one(t)
                base = n[: -len(".weight")]
                for suf, val in (("weight_packed", packed), ("weight_scale", scale),
                                 ("weight_zero_point", zp), ("weight_shape", shape)):
                    out_t[f"{base}.{suf}"] = val; new_map[f"{base}.{suf}"] = f
                errs.append(rel)
            else:
                out_t[n] = t; new_map[n] = f
    save_file(out_t, os.path.join(dst, f), metadata={"format": "pt"})
    print(f"[{i+1}/{len(files)}] ({int(time.time()-t0)}s)", flush=True)
idx["weight_map"] = new_map
json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
for e in os.listdir(src):
    if e.endswith(".safetensors") or e == "model.safetensors.index.json": continue
    p = os.path.join(src, e)
    if os.path.isfile(p): shutil.copy2(p, os.path.join(dst, e))
cfg = os.path.join(dst, "config.json")
c = json.load(open(cfg, encoding="utf-8")); q = c["quantization_config"]
PAT = re.compile(r"\.mlp\.gate$|\.self_attn\.indexer\.index_qk_proj$")
q["ignore"] = [n for n in q["ignore"]
               if n.startswith("mtp.") or "visual" in n or ".ple." in n
               or not (PAT.search(n) or n in ("re:.*\.gate$", "re:.*indexer.*"))]
q["ignore"] += ["re:.*indexer\.(k_layernorm|q_layernorm)"]
q["config_groups"]["group_0"]["targets"] = ["Linear", "re:.*lm_head$"]
json.dump(c, open(cfg, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"{len(errs)} Gewichte, mittlerer relativer Fehler {sum(errs)/len(errs):.4%}")
print("FERTIG")
