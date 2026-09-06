#!/usr/bin/env python3
"""Patch 69: reicht die Quantisierungskonfiguration an den LM-Head durch.

Qwen3.8-Flash-Next hat ein Vokabular von 248320 Eintraegen, der LM-Head ist damit
1,27 GiB in BF16. Mit MTP wird er **viermal je Iteration** gelesen: einmal je
Entwurfsschritt und einmal beim Verifizieren. Bei zwei Raengen sind das rund
2,5 GiB je Iteration -- nach den dichten Projektionen der groesste verbliebene
Bandbreitenposten.

vLLM kann einen quantisierten LM-Head laden: CompressedTensorsConfig.get_quant_method()
behandelt ParallelLMHead ausdruecklich wie ein Linear und liefert
CompressedTensorsLinearMethod. Nur erzeugt qwen3_next.py den LM-Head ohne
quant_config, sodass die Quantisierung nie greift.

Der Patch reicht die vorhandene self.quant_config durch. Fuer Checkpoints, die
lm_head auf ihrer ignore-Liste haben (also alle bisherigen), aendert das nichts:
get_scheme() liefert dann None und der LM-Head bleibt unquantisiert.

Gate: VLLM_GFX1X_LM_HEAD_QUANT=1 (default) / 0 = upstream behaviour.

Usage: python3 69_lm_head_quant.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).

WICHTIG (gemessen 06.09.2026):
  1. Der Layer heisst intern "language_model.lm_head", nicht "lm_head". Im
     Checkpoint muss das Ziel deshalb als Regex stehen:
         "targets": ["Linear", "re:.*lm_head$"]
     Ein Ziel "lm_head" matcht NICHT.
  2. Der MTP-Kopf (qwen4_exp/amd/mtp.py) hat einen zweiten ParallelLMHead und
     braucht dieselbe Aenderung, sonst scheitert das Laden dort.

Ergebnis auf zwei Nodes (TP2, greedy, ShareGPT):
  dichte Teile in 4 Bit               35,52 tok/s   TPOT 25,05 ms
  zusaetzlich LM-Head in 4 Bit        39,51 tok/s   TPOT 22,56 ms
Akzeptanzrate steigt dabei leicht (52,1 -> 53,3 %), Ausgabe bleibt korrekt.
"""
import argparse, sys
from pathlib import Path

MARKER = "gfx1151-patch: 69_lm_head_quant"
REL = "vllm/models/qwen4_exp/amd/model.py"

OLD = """        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""
NEW = f"""        # {MARKER}: ohne quant_config bleibt der LM-Head immer BF16, also
        # 1,27 GiB je Lesevorgang und mit MTP viermal je Iteration.
        import os as _gfx_os

        _gfx_lm_quant = (
            self.quant_config
            if _gfx_os.environ.get("VLLM_GFX1X_LM_HEAD_QUANT", "1") != "0"
            else None
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=_gfx_lm_quant,
            prefix=maybe_prefix(prefix, "lm_head"),
        )"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/usr/local/lib64/python3.12/site-packages")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text(encoding="utf-8")
    if a.check:
        ok = MARKER in s
        print(("OK: patch 69 present in " if ok else "FAIL: patch 69 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 69 already applied to {f}")
        return 0
    n = s.count(OLD)
    if n != 1:
        print(f"ERROR: lm_head anchor found {n} times (want 1); re-audit patch 69", file=sys.stderr)
        return 42
    if "self.quant_config = vllm_config.quant_config" not in s:
        print("ERROR: self.quant_config not set in this file; re-audit patch 69", file=sys.stderr)
        return 42
    f.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"OK: patch 69 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
