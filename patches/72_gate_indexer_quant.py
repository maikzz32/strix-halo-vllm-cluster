from pathlib import Path
MARK = "gfx1151-patch: 72_gate_indexer_quant"
GATE = Path("/usr/local/lib64/python3.12/site-packages/vllm/model_executor/models/qwen3_next.py")
IDX = Path("/usr/local/lib64/python3.12/site-packages/vllm/models/qwen4_exp/amd/indexer_qsa.py")

helper = f'''
# {MARK}
def _gfx_gate_qc(quant_config):
    """Quantisierungskonfiguration fuer Router bzw. QSA-Indexer.

    vLLM erzeugt beide fest mit quant_config=None. Zusammen sind sie 0,154 GiB,
    die je Token vollstaendig gelesen werden. Nur aktiv, wenn der Checkpoint sie
    quantisiert enthaelt -- sonst liefert get_scheme() ohnehin None.
    """
    import os

    if os.environ.get("VLLM_GFX1X_GATE_QUANT", "0") != "1":
        return None
    return quant_config

'''

s = GATE.read_text(encoding="utf-8")
if MARK not in s:
    old = """        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )"""
    new = """        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=_gfx_gate_qc(quant_config),
            prefix=f"{prefix}.gate",
        )"""
    assert s.count(old) == 1, f"gate-Anker {s.count(old)}x"
    i = s.index("\nclass ")
    s = s[:i] + "\n" + helper + s[i:]
    s = s.replace(old, new, 1)
    import ast; ast.parse(s)
    GATE.write_text(s, encoding="utf-8")
    print("qwen3_next.py: Router gepatcht")
else:
    print("qwen3_next.py schon gepatcht")

t = IDX.read_text(encoding="utf-8")
if MARK not in t:
    import re
    m = re.search(r"(        self\.index_qk_proj = ReplicatedLinear\((?:[^)]*?))quant_config=None,", t, re.S)
    if not m:
        print("indexer: Anker nicht gefunden -- uebersprungen")
    else:
        t = t[:m.start()] + m.group(1) + "quant_config=_gfx_gate_qc_idx(quant_config)," + t[m.end():]
        hidx = helper.replace("_gfx_gate_qc", "_gfx_gate_qc_idx")
        j = t.index("\nclass ")
        t = t[:j] + "\n" + hidx + t[j:]
        import ast; ast.parse(t)
        IDX.write_text(t, encoding="utf-8")
        print("indexer_qsa.py: Indexer gepatcht")
else:
    print("indexer_qsa.py schon gepatcht")
