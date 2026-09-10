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
import pin_norm_gate_aot as pin
import os
log_dir = Path(tempfile.mkdtemp(prefix='norm-pin-proof-'))
os.environ['STRIX_NORM_GEOMETRY_LOG_DIR'] = str(log_dir)
pin.install()
from torch._inductor.async_compile import AsyncCompile, CompiledTritonKernels
from torch._inductor.runtime import triton_heuristics
from triton import Config
class Poison:
    def result(self):
        raise AssertionError('Old bundled launcher was reused')
text = source.decode()
CompiledTritonKernels.save(text, Poison())
CompiledTritonKernels.save(text + pin._SOURCE_TAG, Poison())
lookup = triton_heuristics.lookup_autotune_config
triton_heuristics.lookup_autotune_config = lambda *a, **k: Config({'XBLOCK': 1}, num_warps=2, num_stages=1)
try:
    original_name = next(n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef))
    pinned = AsyncCompile().triton(original_name, text)
finally:
    triton_heuristics.lookup_autotune_config = lookup
assert getattr(pinned, '_strix_norm_geometry_pinned', False)
# Unrelated cache entries must still take the original cached path.
class Sentinel:
    def result(self):return 'unrelated-cache-preserved'
other = 'def unrelated(): pass'
CompiledTritonKernels.save(other, Sentinel())
assert AsyncCompile().triton('unrelated', other) == 'unrelated-cache-preserved'
assert pin.body_hash(pinned.fn.src.replace('1e-06', '2e-06')) != pin.BODY_SHA256
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
        pinned_out = torch.empty_like(x)
        pinned.run(x, weight, gate, pinned_out, rows * 12, 128,
                   stream=torch.cuda.current_stream().cuda_stream)
        assert torch.equal(pinned_out, outputs[4]), ('pinned parity', seed, rows)
        different = outputs[4].view(torch.int16) != outputs[1].view(torch.int16)
        records.append(dict(seed=seed, rows=rows, elements=x.numel(),
                            differing_bf16_bits=int(different.sum().item()),
                            max_abs=float((outputs[4].float() - outputs[1].float()).abs().max().item())))
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pinned.run(x, weight, gate, pinned_out, rows * 12, 128, stream=stream.cuda_stream)
    x.mul_(0.75)
    gate.add_(0.0625)
    graph.replay()
    reference = torch.empty_like(x)
    kernel[((rows * 12 + 3) // 4,)](x, weight, gate, reference, rows * 12, 128,
                                  XBLOCK=4, num_warps=2, num_stages=1)
stream.synchronize()
assert torch.equal(reference, pinned_out), 'Changed-input graph replay mismatch'
torch.cuda.current_stream().wait_stream(stream)
proof = [json.loads(line) for line in (log_dir / f'{os.getpid()}.jsonl').read_text().splitlines()]
assert {'selected', 'precompile'} <= {r['stage'] for r in proof}, proof
assert all(r['kwargs'] == {'XBLOCK': 4} and r['num_warps'] == 2 and r['num_stages'] == 1 for r in proof)
torch.cuda.synchronize()
print('RESULT=' + json.dumps(dict(status='completed', diagnostic_only=True, pinned_matches_original=True, overrides_cached_xblock1=True, bypasses_old_bundle=True, unrelated_cache_preserved=True, changed_graph_exact=True, geometry_proof=proof,
      source_sha256=args.sha256, xblocks=[4, 1], num_warps=2,
      total_differing_elements=sum(r['differing_bf16_bits'] for r in records),
      comparisons=records)), flush=True)
