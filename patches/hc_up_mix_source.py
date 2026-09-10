"""Hash-guarded, default-off integration of the adjacent HC-up/gate pair."""
import hashlib
BASE_HASHES = {
    "94da5c6e73972a3194307c70472ef039eab47ee6e5e255c0a2d5cfe99cbf2f39",
    "5feb03eb0b6b191e1e5c5a80855b34c1d3cf684722608668608938a3b84446ff",
}


def generate(source: bytes) -> str:
    assert hashlib.sha256(source).hexdigest() in BASE_HASHES, "Unexpected HC source"
    text = source.decode()
    old = "import torch\n"
    assert text.count(old) == 1
    text = text.replace(old, "import os\nimport torch\n\n_strix_up_mix = None\nif os.environ.get('STRIX_HC_UP_MIX', '0') == '1':\n    import hc_up_mix_runtime as _strix_up_mix\n", 1)
    old = "    def mix(\n"
    assert text.count(old) == 1
    text = text.replace(old, "        if _strix_up_mix is not None:\n            _strix_up_mix.prepare(self)\n\n" + old)
    old = "        gate = self.input_mix_weight_up(lora)  # [M, D]\n        block_input = hc_gate_mix(xn, gate, self.hc_count)"
    assert text.count(old) == 2
    text = text.replace(old, "        if _strix_up_mix is not None:\n            block_input = _strix_up_mix.mix(self, lora, xn, hc_gate_mix)\n        else:\n            gate = self.input_mix_weight_up(lora)  # [M, D]\n            block_input = hc_gate_mix(xn, gate, self.hc_count)")
    compile(text, '<hc-up-mix>', 'exec')
    return text
