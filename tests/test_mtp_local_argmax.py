#!/usr/bin/env python3
"""CPU correctness check using the installed vLLM LogitsProcessor source.

No model load, GPU allocation, distributed job, or serving request. TP collectives
are simulated in one process. This proves the source-level reduction/dispatch,
not RCCL execution or the numerical behavior of the actual GPU INT4 kernel.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import torch
from torch import nn

DEFAULT_ROOT = Path('/usr/local/lib64/python3.12/site-packages/vllm')


def extract_class(source: str, original_name: str, name: str, methods: set[str], namespace: dict):
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == original_name)
    nodes = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in nodes} != methods:
        raise AssertionError(f'Method set changed in {original_name}')
    cls = ast.ClassDef(name=name, bases=[ast.Attribute(value=ast.Name(id='nn', ctx=ast.Load()), attr='Module', ctx=ast.Load())], keywords=[], body=nodes, decorator_list=[])
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    ast.fix_missing_locations(tree)
    exec(compile(tree, '<installed-vllm-methods>', 'exec'), namespace)
    return namespace[name]


class Captured(Exception):
    def __init__(self, value):
        self.value = value


class CpuQuantMethod:
    """Quant-method adapter deliberately hides .weight to catch bypasses."""
    calls = 0

    def apply(self, layer, hidden, bias=None):
        self.calls += 1
        return torch.nn.functional.linear(hidden, layer.decoded_weight, bias)


def check_case(processor_cls, model_cls, namespace, *, tp, vocab, padded, dtype, ties=False, scale=1.0, cap=None):
    assert padded % tp == 0 and vocab <= padded
    torch.manual_seed(314159)
    hidden = torch.randn(5, 8, dtype=dtype)
    # Int4-representable values with exact binary scales. No GPU kernel needed.
    weights = (torch.randint(-8, 8, (padded, 8)).to(dtype) * 0.125)
    if ties:
        hidden.fill_(1)
        weights.fill_(-1)
        # Equal global winners on separate ranks: smallest token id must win.
        weights[1].fill_(1)
        weights[min(padded // tp + 1, vocab - 1)].fill_(1)
    if vocab < padded:
        # Padded rows would beat every real row if not masked correctly.
        hidden.fill_(1)
        weights[vocab:].fill_(1000)
    width = padded // tp
    models, raw_logits, quantizers = [], [], []
    for rank in range(tp):
        start = rank * width
        quant = CpuQuantMethod()
        head = SimpleNamespace(
            decoded_weight=weights[start:start + width], quant_method=quant,
            tp_size=tp,
            shard_indices=SimpleNamespace(org_vocab_start_index=start, num_org_vocab_padding=max(0, start + width - max(start, vocab))),
        )
        p = processor_cls()
        p.logits_as_input = False
        p.use_all_gather = True
        p.org_vocab_size = vocab
        p.head_dtype = None
        p.scale = scale
        p.soft_cap = cap
        m = model_cls()
        m.lm_head = head
        m.logits_processor = p
        models.append(m)
        quantizers.append(quant)
        raw_logits.append(quant.apply(head, hidden))
    full_raw = torch.cat(raw_logits, dim=-1)
    namespace['tensor_model_parallel_all_gather'] = lambda value, dim=-1: full_raw
    references = [m.compute_logits(hidden).argmax(dim=-1) for m in models]
    assert all(torch.equal(references[0], r) for r in references)
    if tp > 1:
        bids = []
        def capture(value, dim=-1):
            raise Captured(value.clone())
        namespace['tensor_model_parallel_all_gather'] = capture
        for m in models:
            try:
                m.get_top_tokens(hidden)
            except Captured as exc:
                bids.append(exc.value)
            else:
                raise AssertionError('Expected the TP argmax collective')
        assert all(v.shape == (hidden.shape[0], 2) and v.dtype == torch.float32 for v in bids)
        namespace['tensor_model_parallel_all_gather'] = lambda value, dim=-1: torch.cat(bids, dim=dim)
    outputs = [m.get_top_tokens(hidden) for m in models]
    assert all(torch.equal(references[0], result) for result in outputs), (tp, vocab, dtype, ties, scale, cap, references, outputs)
    assert all(result.dtype == torch.int64 for result in outputs)
    assert all(bool(((result >= 0) & (result < vocab)).all()) for result in outputs)
    assert all(q.calls >= 3 for q in quantizers), 'Quantization method was bypassed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--patch-script', type=Path, default=Path(__file__).resolve().parents[1] / 'patches/mtp_local_argmax.py')
    args = parser.parse_args()
    torch.set_num_threads(1)
    transform = runpy.run_path(str(args.patch_script))['transform']
    mtp = args.vllm_root / 'models/qwen4_exp/amd/mtp.py'
    original = mtp.read_bytes()
    proposed = transform(original).decode()
    namespace = {'torch': torch, 'nn': nn, 'F': torch.nn.functional}
    processor = extract_class((args.vllm_root / 'model_executor/layers/logits_processor.py').read_text(), 'LogitsProcessor', 'ProcessorUnderTest', {'forward', '_gather_logits', '_get_logits', '_apply_head', 'get_top_tokens'}, namespace)
    model = extract_class(proposed, 'Qwen4ExpMTP', 'ModelUnderTest', {'compute_logits', 'get_top_tokens'}, namespace)
    count = 0
    for tp in [1, 2, 4]:
        for dtype in [torch.float32, torch.bfloat16]:
            for vocab, padded in [(19, 24), (32, 32), (248320, 248320)]:
                check_case(processor, model, namespace, tp=tp, vocab=vocab, padded=padded, dtype=dtype)
                count += 1
            for scale, cap in [(1.0, None), (0.7, 2.0)]:
                check_case(processor, model, namespace, tp=tp, vocab=19, padded=24, dtype=dtype, ties=True, scale=scale, cap=cap)
                count += 1
    assert mtp.read_bytes() == original, 'The test must not modify installed code'
    print(f'PASS: {count} CPU cases; TP1/2/4, BF16/FP32, real vocab offsets, padding, ties, softcap, quant dispatch; source unchanged')


if __name__ == '__main__':
    main()
