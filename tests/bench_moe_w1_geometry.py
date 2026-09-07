#!/usr/bin/env python3
"""Isolated TP4 routed W1 (not full MoE) geometry study; --gpu is explicit.

Reuses the installed immutable Triton kernels with six private driver variants.
Full E512 packed weights occupy 231.25 MiB including scales/zero-points. CPU
references unpack one expert at a time; no serving runtime file is changed.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

DEFAULT_SOURCE = Path('/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py')
FUNCTION = '_glm53_moe_int4_gemv'
# SPLITK, BK, BN, num_warps, num_stages. Baseline is first and unchanged.
GEOMETRIES = [(4, 128, 64, 4, 2), (5, 256, 64, 4, 1),
              (10, 128, 64, 4, 1), (10, 256, 64, 4, 1),
              (20, 128, 64, 4, 1), (10, 128, 32, 4, 1)]
ANCHOR = 'SPLITK, BN, BK, num_warps, num_stages = 4, 64, 128, 4, 2'
E, N, K, GS, M, TOPK, BM = 512, 320, 2560, 32, 4, 10, 16


def make_variant(source, geometry, namespace):
    matches = [n for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name == FUNCTION]
    assert len(matches) == 1, FUNCTION
    body = ast.get_source_segment(source, matches[0])
    assert body.count(ANCHOR) == 1, 'Baseline geometry anchor changed'
    assert 'if K >= 2048:' in body and 'BM = config["BLOCK_SIZE_M"]' in body
    assert '_glm53_moe_int4_gemv_partial[grid]' in body
    assert 'MUL_ROUTED=mul_routed_weight' in body
    split, bk, bn, warps, stages = geometry
    assert K % (bk * split) == 0 and bk % GS == 0 and N % bn == 0
    assert bk & (bk - 1) == 0 and bn & (bn - 1) == 0
    assert bk >= GS and bn >= 32 and len(GEOMETRIES) <= 6
    body = body.replace(ANCHOR, f'SPLITK, BN, BK, num_warps, num_stages = {split}, {bn}, {bk}, {warps}, {stages}')
    ns = namespace.copy()
    exec(compile('from __future__ import annotations\n' + body,
                 '<isolated-w1-geometry>', 'exec'), ns)
    return ns[FUNCTION]


def gpu_study(source, wall_seconds):
    import importlib
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.version.hip and torch.cuda.is_available(), 'ROCm GPU required'
    prop = torch.cuda.get_device_properties(0)
    assert 'gfx1151' in prop.gcnArchName, prop.gcnArchName
    fm = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    assert Path(fm.__file__).read_text(encoding='utf-8') == source
    variants = {g: make_variant(source, g, fm.__dict__) for g in GEOMETRIES}
    os.environ['VLLM_GFX1X_MOE_INT4_GEMV'] = '1'
    torch.manual_seed(20260906)
    torch.cuda.reset_peak_memory_stats()
    # In-place fills avoid transient full-sized weight/scale copies.
    weights = torch.empty((E, N, K // 2), device='cuda', dtype=torch.uint8).random_(0, 256)
    scales = torch.empty((E, N, K // GS), device='cuda', dtype=torch.bfloat16).uniform_(0.002, 0.01)
    zp = torch.empty((E, N // 2, K // GS), device='cuda', dtype=torch.uint8).random_(0, 256)
    a = torch.empty((M, K), device='cuda', dtype=torch.bfloat16).normal_().mul_(0.25)
    out_ref = torch.empty((M, TOPK, N), device='cuda', dtype=torch.bfloat16)
    out_new = torch.empty_like(out_ref)
    packed_mib = sum(x.numel() * x.element_size() for x in (weights, scales, zp)) / 2**20
    print(json.dumps({'study': 'routed_W1_partial_plus_split_reduce_only',
                      'device': prop.name, 'arch': prop.gcnArchName,
                      'shape': {'E': E, 'N': N, 'K': K, 'M': M, 'topk': TOPK, 'GS': GS, 'BM': BM},
                      'packed_weights_scales_zp_mib': packed_mib,
                      'source_sha256': hashlib.sha256(source.encode()).hexdigest()}), flush=True)

    common = dict(A=a, B=weights, B_scale=scales, B_zp=zp,
                  topk_weights=None, mul_routed_weight=False, top_k=TOPK,
                  compute_type=fm.tl.bfloat16, use_int4_w4a16=True,
                  use_int8_w8a16=False, block_shape=[0, GS],
                  config={'BLOCK_SIZE_M': BM})

    disjoint = [[(i * 13 + 7) % E for i in range(r * TOPK, (r + 1) * TOPK)] for r in range(M)]
    reuse = [[0, 1, 31, 32, 127, 128, 255, 256, 510, 511] for _ in range(M)]
    overlap = [[(start + j) % E for j in range(TOPK)] for start in (0, 5, 10, 0)]
    distributions = [('reuse10', reuse, None), ('overlap20', overlap, None),
                     ('disjoint40', disjoint, None),
                     ('mixed_ep', disjoint, [(gid * 97 + 23) % E if gid % 3 else -1 for gid in range(E)]),
                     ('all_miss', disjoint, [-1] * E)]

    def routing(rows, map_cpu):
        ids = torch.tensor(rows, device='cuda', dtype=torch.int32)
        expert_map = None if map_cpu is None else torch.tensor(map_cpu, device='cuda', dtype=torch.int32)
        sorted_ids, experts, ntp = moe_align_block_size(ids, BM, E, expert_map)
        count = int(ntp.item())
        assert count == BM * len(set(sum(rows, [])))
        assert sorted_ids.numel() == M * TOPK * BM
        valid = sorted_ids[:count].cpu().tolist()
        assert sorted(v for v in valid if v < M * TOPK) == list(range(M * TOPK))
        assert sum(v >= M * TOPK for v in valid) == count - M * TOPK
        return dict(sorted_token_ids=sorted_ids, expert_ids=experts,
                    num_tokens_post_padded=ntp), count

    def cpu_reference(rows, map_cpu):
        # Decode K-packed bytes and output-channel-packed zero-points
        # independently. Work on CPU to keep GPU allocations under 256 MiB.
        flat = sum(rows, [])
        mapped = flat if map_cpu is None else [map_cpu[e] for e in flat]
        result = torch.zeros((M * TOPK, N), dtype=torch.float32)
        x = a.cpu().float()
        for expert in sorted(set(mapped)):
            if expert < 0:
                continue
            packed = weights[expert].cpu()
            zpacked = zp[expert].cpu()
            sc = scales[expert].cpu().float()
            w = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(N, K)
            z = torch.stack((zpacked & 15, zpacked >> 4), dim=1).reshape(N, K // GS)
            dequant = ((w.float() - z.float().repeat_interleave(GS, dim=1))
                       * sc.repeat_interleave(GS, dim=1)).to(torch.bfloat16).float()
            assignments = [i for i, e in enumerate(mapped) if e == expert]
            token_indices = [i // TOPK for i in assignments]
            result[assignments] = x[token_indices] @ dequant.T
        return result.to(torch.bfloat16).reshape(M, TOPK, N), mapped

    def run(fn, out, route):
        assert fn(C=out, **common, **route), 'Custom W1 guard unexpectedly rejected'

    # Compile each of the six geometries once before bounded measurement time.
    first_route, _ = routing(reuse, None)
    compile_started = time.monotonic()
    for geometry, fn in variants.items():
        run(fn, out_new, first_route)
        torch.cuda.synchronize()
        print(json.dumps({'compiled_geometry': geometry}), flush=True)
    print(json.dumps({'compile_wall_seconds': time.monotonic() - compile_started}), flush=True)
    del first_route
    measurement_started = time.monotonic()

    def check_deadline():
        assert time.monotonic() - measurement_started < wall_seconds, 'Bounded measurement deadline exceeded'

    checks = 0
    # Every geometry must pass independently before any timing is reported.
    for name, rows, map_cpu in distributions:
        check_deadline()
        route, count = routing(rows, map_cpu)
        independent_cpu, mapped = cpu_reference(rows, map_cpu)
        missing = torch.tensor([e < 0 for e in mapped])
        for layout in ('native_capacity', 'trimmed_capacity'):
            active_route = route.copy()
            if layout == 'trimmed_capacity':
                active_route['sorted_token_ids'] = route['sorted_token_ids'][:count]
                active_route['expert_ids'] = route['expert_ids'][:count // BM]
            out_ref.fill_(float('nan'))
            run(variants[GEOMETRIES[0]], out_ref, active_route)
            baseline_cpu = out_ref.cpu()
            assert torch.isfinite(baseline_cpu).all()
            torch.testing.assert_close(baseline_cpu, independent_cpu, rtol=0.02, atol=0.015625)
            for geometry, fn in variants.items():
                out_new.fill_(float('nan'))
                run(fn, out_new, active_route)
                actual_cpu = out_new.cpu()
                assert torch.isfinite(actual_cpu).all()
                torch.testing.assert_close(actual_cpu, baseline_cpu, rtol=0.02, atol=0.015625)
                torch.testing.assert_close(actual_cpu, independent_cpu, rtol=0.02, atol=0.015625)
                assert torch.count_nonzero(actual_cpu.reshape(M * TOPK, N)[missing]) == 0
                checks += 1
                print(json.dumps({'check': name, 'layout': layout, 'geometry': geometry,
                                  'max_abs_vs_baseline': (actual_cpu.float() - baseline_cpu.float()).abs().max().item(),
                                  'max_abs_vs_cpu_reference': (actual_cpu.float() - independent_cpu.float()).abs().max().item(),
                                  'bitwise_baseline': torch.equal(actual_cpu, baseline_cpu), 'status': 'PASS'}), flush=True)
        del route, active_route

    def elapsed(fn, route):
        for _ in range(4):
            run(fn, out_new, route)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        calls = 8
        with torch.cuda.graph(graph):
            for _ in range(calls):
                run(fn, out_new, route)
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(3):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(30):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / (calls * 30))
        del graph
        gc.collect()
        torch.cuda.empty_cache()
        return statistics.median(samples), samples

    for name, rows, map_cpu in distributions[:4]:
        check_deadline()
        route, count = routing(rows, map_cpu)
        measurements = {}
        # Alternate direction between distributions to expose obvious order drift.
        ordered = GEOMETRIES if name in ('reuse10', 'disjoint40') else list(reversed(GEOMETRIES))
        for geometry in ordered:
            check_deadline()
            median_us, samples = elapsed(variants[geometry], route)
            measurements[geometry] = median_us
            print(json.dumps({'timing': 'warm_W1_partial_plus_reduce_graph8_median3x30',
                              'routing': name, 'distinct_global_experts': count // BM,
                              'geometry': geometry, 'median_us': median_us,
                              'sample_us': samples}), flush=True)
        baseline_us = measurements[GEOMETRIES[0]]
        print(json.dumps({'routing_summary': name, 'baseline_us': baseline_us,
                          'relative_speedups': [{'geometry': g, 'speedup': baseline_us / measurements[g]}
                                                for g in GEOMETRIES]}), flush=True)
        del route
    peak = torch.cuda.max_memory_allocated() / 2**20
    assert peak < 256, f'GPU live allocation limit exceeded: {peak:.3f} MiB'
    print(json.dumps({'correctness_checks': checks, 'timing_configurations': 4 * len(GEOMETRIES),
                      'peak_allocated_mib': peak, 'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                      'measurement_wall_seconds': time.monotonic() - measurement_started,
                      'status': 'PASS', 'gpu_released_on_process_exit': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=120)
    args = parser.parse_args()
    assert 1 <= args.wall_seconds <= 120
    source = args.source.read_text(encoding='utf-8')
    for geometry in GEOMETRIES:
        make_variant(source, geometry, {})
    assert GEOMETRIES[0] == (4, 128, 64, 4, 2)
    print(json.dumps({'source_variants': len(GEOMETRIES), 'shape_divisibility': 'PASS',
                      'gpu_requested': args.gpu, 'status': 'PASS'}), flush=True)
    if args.gpu:
        gpu_study(source, args.wall_seconds)


if __name__ == '__main__':
    main()
