"""Compare two observed launch geometries of one emitted norm/gate kernel.

Diagnostic only: a difference is recorded, not treated as a test-harness failure.
Run while the serving engine is idle. No model source or weights are modified.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--sha256', required=True)
args = parser.parse_args()
source = Path(args.source).read_bytes()
assert hashlib.sha256(source).hexdigest() == args.sha256
tree = ast.parse(source)
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and any(ast.unparse(d) == 'triton.jit' for d in n.decorator_list)]
assert len(functions) == 1
fn = functions[0]
fn.decorator_list = [ast.parse('triton.jit', mode='eval').body]
fn.name = 'norm_gate'
body = 'import triton\nimport triton.language as tl\n' + ast.unparse(fn)
path = Path(tempfile.mkdtemp(prefix='norm-gate-geometry-')) / 'kernel.py'
path.write_text(body)
scope = {'__file__': str(path), '__name__': '_norm_gate_geometry'}
exec(compile(body, str(path), 'exec'), scope)
kernel = scope['norm_gate']

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)
records = []
for seed in range(9580, 9588):
    torch.manual_seed(seed)
    for rows in (4, 128, 1024):
        x = torch.randn((rows * 12, 128), device='cuda', dtype=torch.bfloat16)
        weight = torch.randn((128,), device='cuda', dtype=torch.bfloat16)
        gate = torch.randn((rows, 4096), device='cuda', dtype=torch.bfloat16)
        outputs = {}
        for block in (4, 1):
            out = torch.empty_like(x)
            kernel[((rows * 12 + block - 1) // block,)](
                x, weight, gate, out, rows * 12, 128,
                XBLOCK=block, num_warps=2, num_stages=1)
            outputs[block] = out
        different = outputs[4].view(torch.int16) != outputs[1].view(torch.int16)
        records.append(dict(seed=seed, rows=rows, elements=x.numel(),
                            differing_bf16_bits=int(different.sum().item()),
                            max_abs=float((outputs[4].float() - outputs[1].float()).abs().max().item())))
torch.cuda.synchronize()
print('RESULT=' + json.dumps(dict(status='completed', diagnostic_only=True,
      source_sha256=args.sha256, xblocks=[4, 1], num_warps=2,
      total_differing_elements=sum(r['differing_bf16_bits'] for r in records),
      comparisons=records)), flush=True)
