#!/usr/bin/env python3
"""Quantisiert die dichten BF16-Teile von Qwen3.8-Flash-Next auf W4A16.

Der Checkpoint laesst genau die Teile unquantisiert, die pro Token vollstaendig
gelesen werden: die Gated-DeltaNet-Projektionen, die Attention-Projektionen und
die Shared Experts, zusammen rund 5,4 GiB. Sie auf INT4 zu bringen halbiert bis
viertelt den Byteverkehr je Token.

Format wie im vorhandenen Checkpoint (damit vLLM denselben Pfad nimmt):
  pack-quantized, 4 bit, group_size 32, asymmetrisch (zero points),
  Gruppierung entlang der Eingangsdimension.

Bewusst NICHT angefasst, weil vLLMs qwen4_exp sie dort unquantisiert erwartet
oder sie zu klein/qualitaetskritisch sind:
  lm_head, embed_tokens, mlp.gate, *hyper_connection*, *indexer*, *norm*,
  conv1d, ple.*, mtp.*, visual.*, linear_attn.in_proj_a/b (24 Zeilen breit).

Aufruf:
  python3 requant_dense.py --src <original> --dst <ziel> [--limit N] [--check-only]
"""
import argparse, json, os, re, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized import pack_to_int32, unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize

GROUP = 32
BITS = 4

# Welche Gewichte requantisiert werden (Suffix .weight, dtype bf16)
TARGETS = (
    r"\.linear_attn\.in_proj_qkv\.weight$",
    r"\.linear_attn\.in_proj_z\.weight$",
    r"\.linear_attn\.out_proj\.weight$",
    r"\.self_attn\.q_proj\.weight$",
    r"\.self_attn\.k_proj\.weight$",
    r"\.self_attn\.v_proj\.weight$",
    r"\.self_attn\.o_proj\.weight$",
    r"\.mlp\.shared_expert\.gate_proj\.weight$",
    r"\.mlp\.shared_expert\.up_proj\.weight$",
    r"\.mlp\.shared_expert\.down_proj\.weight$",
)
SKIP = (r"^mtp\.", r"visual", r"\.ple\.")
TARGET_RE = re.compile("|".join(TARGETS))
SKIP_RE = re.compile("|".join(SKIP))


def is_target(name: str) -> bool:
    return bool(TARGET_RE.search(name)) and not SKIP_RE.search(name)


def quant_one(w: torch.Tensor):
    """bf16 [out, in] -> (packed int32, scale bf16, zero_point int32, shape int64)."""
    args = QuantizationArgs(
        num_bits=BITS, type="int", symmetric=False,
        strategy=QuantizationStrategy.GROUP, group_size=GROUP, observer="minmax",
    )
    wf = w.to(torch.float32)
    out, inp = wf.shape
    g = inp // GROUP
    blocks = wf.reshape(out, g, GROUP)
    mx = blocks.amax(dim=2)
    mn = blocks.amin(dim=2)
    qmin, qmax = -(2 ** (BITS - 1)), 2 ** (BITS - 1) - 1        # -8 .. 7
    scale = (mx - mn).clamp(min=1e-8) / (qmax - qmin)
    zp = torch.round(qmin - mn / scale).clamp(qmin, qmax)
    q = torch.round(blocks / scale.unsqueeze(2) + zp.unsqueeze(2))
    q = q.clamp(qmin, qmax).reshape(out, inp).to(torch.int8)
    err = None
    deq = ((q.reshape(out, g, GROUP).to(torch.float32) - zp.unsqueeze(2)) * scale.unsqueeze(2)).reshape(out, inp)
    denom = wf.abs().mean().clamp(min=1e-8)
    err = ((deq - wf).abs().mean() / denom).item()
    packed = pack_to_int32(q, BITS)
    zp_packed = pack_to_int32(zp.to(torch.int8).t().contiguous(), BITS).t().contiguous()
    return (packed, scale.to(torch.bfloat16), zp_packed,
            torch.tensor([out, inp], dtype=torch.int64), err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--limit", type=int, default=0, help="nur N Dateien (Test)")
    ap.add_argument("--check-only", action="store_true", help="nur einen Tensor pruefen")
    a = ap.parse_args()

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    files = sorted(set(wm.values()))
    targets = [k for k in wm if is_target(k)]
    print(f"{len(targets)} Gewichte zu requantisieren, {len(files)} Dateien", flush=True)

    if a.check_only:
        name = targets[0]
        with safe_open(os.path.join(a.src, wm[name]), framework="pt") as fh:
            w = fh.get_tensor(name)
        t0 = time.time()
        packed, scale, zp, shape, err = quant_one(w)
        print(f"{name}\n  {tuple(w.shape)} bf16 -> packed {tuple(packed.shape)} {packed.dtype}, "
              f"scale {tuple(scale.shape)}, zp {tuple(zp.shape)}")
        print(f"  relativer mittlerer Fehler {err:.4%}, {time.time()-t0:.1f}s")
        print(f"  Bytes: {w.numel()*2/2**20:.1f} MiB -> "
              f"{(packed.numel()*4 + scale.numel()*2 + zp.numel()*4)/2**20:.1f} MiB")
        return 0

    os.makedirs(a.dst, exist_ok=True)
    new_map, errs, saved = {}, [], 0
    for i, f in enumerate(files if not a.limit else files[: a.limit]):
        names = [k for k, v in wm.items() if v == f]
        out_t = {}
        with safe_open(os.path.join(a.src, f), framework="pt") as fh:
            for n in names:
                t = fh.get_tensor(n)
                if is_target(n):
                    packed, scale, zp, shape, err = quant_one(t)
                    base = n[: -len(".weight")]
                    out_t[base + ".weight_packed"] = packed
                    out_t[base + ".weight_scale"] = scale
                    out_t[base + ".weight_zero_point"] = zp
                    out_t[base + ".weight_shape"] = shape
                    for suf in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
                        new_map[base + "." + suf] = f
                    errs.append(err)
                    saved += t.numel() * 2 - (packed.numel() * 4 + scale.numel() * 2 + zp.numel() * 4)
                else:
                    out_t[n] = t
                    new_map[n] = f
        save_file(out_t, os.path.join(a.dst, f), metadata={"format": "pt"})
        print(f"[{i+1}/{len(files)}] {f}  ({len(out_t)} Tensoren)", flush=True)

    idx["weight_map"] = new_map
    json.dump(idx, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    for extra in os.listdir(a.src):
        if extra.endswith(".safetensors") or extra == "model.safetensors.index.json":
            continue
        s = os.path.join(a.src, extra)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(a.dst, extra))
    if errs:
        print(f"\n{len(errs)} Gewichte quantisiert, mittlerer relativer Fehler "
              f"{sum(errs)/len(errs):.4%}, max {max(errs):.4%}")
    print(f"eingespart: {saved/2**30:.2f} GiB je vollstaendigem Durchlauf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
