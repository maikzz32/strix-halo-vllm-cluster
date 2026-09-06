#!/usr/bin/env python3
"""Isolated QSA scoring parity and CUDA-graph timing; --gpu explicit.

Runs original and opt-in kernels from immutable sources. A temporary module
contains the candidate for Triton's source inspection; serving files untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile

DEFAULT_SOURCE = Path('/usr/local/lib64/python3.12/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py')
GATE = 'VLLM_QSA_SKIP_INVISIBLE_TILES'


def gpu_study(original, patched, page_size, capacities):
    import importlib
    import importlib.util
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.version.hip
    prop = torch.cuda.get_device_properties(0)
    assert 'gfx1151' in prop.gcnArchName
    baseline = importlib.import_module('vllm.models.qwen4_exp.amd.ops.qsa')
    assert Path(baseline.__file__).read_text(encoding='utf-8') == original
    with tempfile.TemporaryDirectory(prefix='qsa-kernel-study-') as folder:
        candidate_path = Path(folder) / 'qsa_candidate.py'
        candidate_path.write_text(patched, encoding='utf-8')
        spec = importlib.util.spec_from_file_location('qsa_invisible_candidate', candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        torch.manual_seed(20260906)
        torch.cuda.reset_peak_memory_stats()
        max_capacity = max(capacities)
        cache = torch.randn((max_capacity // page_size, page_size, 1, 128), device='cuda', dtype=torch.bfloat16)
        query = torch.randn((4, 4, 128), device='cuda', dtype=torch.bfloat16)
        table_full = torch.arange(cache.size(0), device='cuda', dtype=torch.int32).reshape(1, -1)
        reqs = torch.zeros(4, device='cuda', dtype=torch.int32)
        print(json.dumps({'device': prop.name, 'arch': prop.gcnArchName,
                          'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                          'query_shape': list(query.shape), 'cache_shape': list(cache.shape),
                          'page_size': page_size, 'compressed_capacity': max_capacity}), flush=True)

        def params(capacity, visible):
            return dict(q=query, k_cache=cache, page_table=table_full[:, :capacity // page_size],
                        token_to_req=reqs,
                        query_positions=torch.tensor([visible * 4 - 1] * 4, device='cuda', dtype=torch.int64),
                        sequence_lengths=torch.tensor([visible * 4], device='cuda', dtype=torch.int32),
                        compress_ratio=4)

        def time_graph(call):
            for _ in range(3):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):
                    call()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 80)
            return sorted(samples)[1]

        tests = 0
        for capacity in capacities:
            for visible in (0, 31, 32, 33, 128, 512, 2048):
                kwargs = params(capacity, visible)
                old, old_visible = baseline.qsa_mqa_paged(**kwargs)
                for enabled in ('0', '1'):
                    os.environ[GATE] = enabled
                    new, new_visible = candidate.qsa_mqa_paged(**kwargs)
                    torch.cuda.synchronize()
                    assert torch.equal(old_visible, new_visible)
                    assert torch.equal(new_visible, torch.full_like(new_visible, visible))
                    assert torch.equal(torch.isneginf(old), torch.isneginf(new)), 'Infinity masks differ'
                    # Exact score bits, including valid tiles that cross the
                    # visible boundary; no relaxed numerical tolerance here.
                    assert torch.equal(old, new), 'Logits changed'
                    assert torch.isneginf(new[:, visible:]).all()
                    tests += 1
                record = dict(capacity=capacity, visible=visible, rows=4,
                              workgroups=4 * ((capacity + 31) // 32),
                              candidate_active_workgroups=4 * ((visible + 31) // 32), status='PASS')
                if visible in (128, 512, 2048):
                    stock_ms = time_graph(lambda: baseline.qsa_mqa_paged(**kwargs))
                    patched_ms = time_graph(lambda: candidate.qsa_mqa_paged(**kwargs))
                    record.update(stock_ms=stock_ms, invisible_skip_ms=patched_ms, speedup=stock_ms / patched_ms)
                print(json.dumps(record), flush=True)
                del kwargs, old, old_visible, new, new_visible

        # Explicitly cover invalid request rows and invalid physical pages,
        # while checking visible-count writes even for tile0 at visible0.
        for label in ('invalid_requests', 'invalid_pages', 'row_mixed_visibility', 'ragged_columns'):
            kwargs = params(max_capacity, 128)
            if label == 'invalid_requests':
                kwargs['token_to_req'] = torch.tensor([-1, 1, 0, 0], device='cuda', dtype=torch.int32)
            elif label == 'invalid_pages':
                table = table_full.clone()
                table[0, :2] = -1
                table[0, 2] = cache.size(0)
                kwargs['page_table'] = table
            elif label == 'row_mixed_visibility':
                kwargs['query_positions'] = torch.tensor([-1, 0, 131, 511], device='cuda', dtype=torch.int64)
            else:
                kwargs['num_columns'] = 127
            old, old_visible = baseline.qsa_mqa_paged(**kwargs)
            new, new_visible = candidate.qsa_mqa_paged(**kwargs)
            torch.cuda.synchronize()
            assert torch.equal(old_visible, new_visible) and torch.equal(old, new)
            tests += 1
            print(json.dumps({'case': label, 'visible_blocks': new_visible.tolist(), 'status': 'PASS'}), flush=True)
        peak = torch.cuda.max_memory_allocated() / 2**20
        assert peak < 256, peak
        print(json.dumps({'gpu_cases': tests, 'logits_and_visible_bitwise_equal': True,
                          'peak_allocated_mib': peak, 'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                          'status': 'PASS'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--patch-script', type=Path, default=Path(__file__).resolve().parents[1] / 'patches/qsa_invisible_tiles.py')
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--capacities', type=int, nargs=2, default=[8192, 65536])
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    assert args.page_size > 0
    assert all(2048 <= c <= 131072 and c % args.page_size == 0 for c in args.capacities)
    patch = runpy.run_path(str(args.patch_script))
    raw = args.source.read_bytes()
    patched = patch['transform'](raw)
    assert patched != raw, 'Use unpatched installed source'
    assert patch['transform'](patched) == patched
    print(json.dumps({'source_and_idempotence': 'PASS'}), flush=True)
    if args.gpu:
        gpu_study(raw.decode('utf-8'), patched.decode('utf-8'), args.page_size, args.capacities)


if __name__ == '__main__':
    main()
