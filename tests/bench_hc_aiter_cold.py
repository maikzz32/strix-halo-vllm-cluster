#!/usr/bin/env python3
"""Bounded AITER gfx1151 small-M tile comparison; same synthetic BF16 weights."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VLLM_GFX1X_HC_W8'] = '0'


def run(args):
    import torch
    import hashlib
    import importlib
    from hc_aiter_pr4814 import gemm_a16w16
    from hc_aiter_pr4814 import compute_splitk_params
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    assert torch.version.hip
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    stream = torch.cuda.Stream()
    source_hashes = {}
    for name in ('hc_aiter_pr4814',):
        file = Path(importlib.import_module(name).__file__)
        source_hashes[name] = {'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    report = {'source_hashes':source_hashes, 'status': 'running', 'cases': [], 'device': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'hip': torch.version.hip,
              'notes': ['24 distinct matrices per shape, cyclic CUDA/HIP graphs.',
                        'A single BF16 weight pool per shape; at least 150 MiB, cache misses unprofiled.',
                        'Same BF16 values and inputs for both paths; no quantization.',
                        'Isolated AITER PR4814 kernel excerpt and tile; source identities recorded. No AITER installation.',
                        'Peak tensor allocation excludes driver/context and serving processes.']}

    def metrics(actual, ref):
        a, r = actual.float(), ref.float()
        delta = a - r
        return {'exact': bool(torch.equal(actual, ref)),
                'bf16_bit_mismatches': int((actual.view(torch.uint16) != ref.view(torch.uint16)).sum())
                    if actual.dtype == ref.dtype == torch.bfloat16 else None,
                'max_abs': float(delta.abs().max()),
                'relative_l2': float(torch.linalg.vector_norm(delta) /
                                     torch.linalg.vector_norm(r).clamp_min(1e-12)),
                'finite': bool(torch.isfinite(a).all())}

    for shape_id, (label, n, k, padding) in enumerate([
        ('merged_down', 336, 10240, 12), ('final_down', 320, 10240, 0), ('up', 10240, 320, 0)
    ]):
        torch.manual_seed(9001 + shape_id)
        pool = []
        for i in range(24):
            w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * 0.02
            if padding:
                w[-padding:] = 0
            pool.append(w)
        del w
        x = torch.randn((4, k), device='cuda', dtype=torch.bfloat16)
        pool_mib = sum(w.numel() * w.element_size() for w in pool) / 1024**2
        assert pool_mib >= 150
        cfg = compute_splitk_params(dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=1,
            matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1), k)
        ys = [torch.empty((4,n),device='cuda',dtype=torch.bfloat16) for _ in pool]
        def candidate(index):
            return gemm_a16w16(x,pool[index],y=ys[index],config=cfg,
                              bias=None,activation=None,backend='triton')
        numerical = []
        for i, w in enumerate(pool):
            reference = x.float() @ w.float().t()
            baseline = rocm_unquantized_gemm_impl(x, w)
            aiter_output = candidate(i)
            check = {'pool_index': i,
                     'baseline_vs_fp32': metrics(baseline, reference),
                     'aiter_vs_fp32': metrics(aiter_output, reference),
                     'aiter_vs_baseline': metrics(aiter_output, baseline)}
            for key in ('baseline_vs_fp32', 'aiter_vs_fp32'):
                assert check[key]['finite'] and check[key]['relative_l2'] < 0.01, check
            if padding:
                assert not bool(baseline[:, -padding:].any())
                assert not bool(aiter_output[:, -padding:].any())
            numerical.append(check)
        del w, reference, baseline, aiter_output
        # Numerical validation precedes all timing and touches every matrix.
        for method in ('existing_wvSplitK', 'aiter_pr4814_tile'):
            def sequence():
                if method == 'existing_wvSplitK':
                    return [rocm_unquantized_gemm_impl(x, w) for w in pool]
                return [candidate(i) for i in range(len(pool))]

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    sequence()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = sequence()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert all(bool(torch.isfinite(o).all()) for o in outputs)
            if method == 'aiter_pr4814_tile':
                expected = [output.clone() for output in outputs]
                x.neg_()
                # Compare replay with the same kernel on changed inputs. Negating
                # an earlier floating-point answer is not an independent oracle.
                eager_changed = [gemm_a16w16(x,w,y=torch.empty_like(y),config=cfg,
                    bias=None,activation=None,backend='triton') for w,y in zip(pool,ys)]
                graph.replay(); torch.cuda.synchronize()
                changed_checks = [metrics(output, eager) for output,eager in zip(outputs,eager_changed)]
                negation_checks = [metrics(output, -old) for output,old in zip(outputs,expected)]
                assert all(check['bf16_bit_mismatches'] == 0 for check in changed_checks), changed_checks
                x.neg_()
                graph.replay(); torch.cuda.synchronize()
                restored_checks = [metrics(output, old) for output,old in zip(outputs,expected)]
                assert all(check['bf16_bit_mismatches'] == 0 for check in restored_checks), restored_checks
                del expected, eager_changed
            samples = []
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeats):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / (args.repeats * 24))
            case = {'shape': label, 'M': 4, 'N': n, 'K': k, 'method': method,
                    'config':cfg, 'pool_count': 24, 'weight_pool_per_layout_MiB': pool_mib,
                    'graph_us_per_matrix': statistics.median(samples), 'samples_us': samples,
                    'peak_tensor_MiB': torch.cuda.max_memory_allocated() / 1024**2}
            assert case['peak_tensor_MiB'] < 1024, case
            if method == 'existing_wvSplitK':
                case['numerical_all_matrices'] = numerical
            else:
                case['changed_input_graph_vs_eager'] = changed_checks
                case['restored_input_graph_vs_original'] = restored_checks
                case['changed_input_vs_negated_original'] = negation_checks
            report['cases'].append(case)
            Path(args.output).write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in case.items() if k != 'numerical_all_matrices'}), flush=True)
            graph.reset()
            del graph, outputs, sequence
        del candidate, ys, pool, x, numerical
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'passed'
    report['wall_seconds_after_import'] = time.monotonic() - started
    report['peak_tensor_MiB'] = torch.cuda.max_memory_allocated() / 1024**2
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'passed', 'wall_seconds_after_import': report['wall_seconds_after_import']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    assert 5 <= args.repeats <= 20
    run(args)
