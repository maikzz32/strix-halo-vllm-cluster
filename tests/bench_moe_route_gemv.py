#!/usr/bin/env python3
"""Isolated routed W2 SIMD GEMV experiment; GPU execution requires --gpu.

One fixed alternative to the current BM16 WMMA/SplitK kernel, not a sweep.
Exact layout: uint8 [512,2560,80] K-packed low nibble first, BF16 scales
[512,2560,5], uint8 zero points [512,1280,5] packed along output channels,
GS32. Dequantization rounds to BF16 before FP32 products/reduction. W2 has
40 independent route activations, unlike W1's four shared token activations.

Routing cases and independent one-expert-at-a-time CPU reference derive from
bench_moe_w1_geometry.py. This benchmark uses original topk_ids + expert_map;
integrating it would require passing those IDs to the dispatch site. Baseline
and candidate timings both exclude already-prepared routing. No runtime patch.

Potential gain: remove padded BM16 arithmetic, SplitK buffer and reduce launch.
Limits: SIMD replaces faster matrix instructions; repeated experts reread
weights per route; 6400 programs can remain launch/occupancy limited. K160 is
masked within BK256, so SIMD still has 1.6x arithmetic lane padding. This is
not a model TPS forecast or a substitute for the pending real GPU profile.

Reports hot fixed weights separately from rotating 4 banks x 8 expert sets.
The latter touches at least 72.265625 MiB of packed expert data per cycle
for reuse10 (more for other cases), rather than repeatedly timing ~2-9 MiB.
It stresses cache capacity but does not prove a particular cache-miss rate.
Four complete banks consume 462.5 MiB; live-allocation limit is 768 MiB.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

DEFAULT_SOURCE = Path('/usr/local/lib64/python3.12/site-packages/vllm/model_executor/layers/fused_moe/fused_moe.py')
E, N, K, GS, M, TOPK, BM = 512, 2560, 160, 32, 4, 10, 16
ROUTES, BN, BK, WARPS = M * TOPK, 16, 256, 4
BANKS, ROTATIONS = 4, 8


# Deferred triton.jit decoration keeps the default path pure stdlib/CPU.
# tl is installed in this module's globals only inside gpu_study().
def _route_gemv_kernel(
    x_ptr, b_ptr, scale_ptr, zp_ptr, ids_ptr, map_ptr, route_weight_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr, GS: tl.constexpr,
    BN: tl.constexpr, BK: tl.constexpr,
    HAS_MAP: tl.constexpr, MUL_ROUTED: tl.constexpr,
):
    route = tl.program_id(0)
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(ids_ptr + route).to(tl.int64)
    if HAS_MAP:
        expert = tl.load(map_ptr + expert).to(tl.int64)
    if expert < 0:
        tl.store(out_ptr + route * N + offs_n, tl.zeros((BN,), tl.float32), offs_n < N)
        return

    # Full K160 in one block. Loads, including group scale/ZP loads, mask
    # the 96 extra K lanes; BF16 dequant rounds before FP32 products.
    offs_kw = tl.arange(0, BK // 2)
    b8 = tl.load(
        b_ptr + expert * (N * (K // 2)) + offs_n[:, None] * (K // 2) + offs_kw[None, :],
        mask=(offs_n[:, None] < N) & (offs_kw[None, :] < K // 2), other=0,
    )
    nib = tl.interleave((b8 & 15).to(tl.float32), (b8 >> 4).to(tl.float32))
    offs_g = tl.arange(0, BK // GS)
    group_mask = (offs_n[:, None] < N) & (offs_g[None, :] < K // GS)
    scales = tl.load(
        scale_ptr + expert * (N * (K // GS)) + offs_n[:, None] * (K // GS) + offs_g[None, :],
        mask=group_mask, other=0,
    ).to(tl.float32)
    zp8 = tl.load(
        zp_ptr + expert * ((N // 2) * (K // GS))
        + (offs_n[:, None] // 2) * (K // GS) + offs_g[None, :],
        mask=group_mask, other=0,
    )
    zero_points = ((zp8 >> ((offs_n[:, None] % 2) * 4)) & 15).to(tl.float32)
    scales = tl.reshape(tl.broadcast_to(scales[:, :, None], (BN, BK // GS, GS)), (BN, BK))
    zero_points = tl.reshape(tl.broadcast_to(zero_points[:, :, None], (BN, BK // GS, GS)), (BN, BK))
    dequant = ((nib - zero_points) * scales).to(tl.bfloat16).to(tl.float32)
    offs_k = tl.arange(0, BK)
    # W2 inputs are already expanded: each routed expert has its own row.
    x = tl.load(x_ptr + route * K + offs_k, mask=offs_k < K, other=0).to(tl.float32)
    accum = tl.sum(dequant * x[None, :], axis=1)
    if MUL_ROUTED:
        accum *= tl.load(route_weight_ptr + route).to(tl.float32)
    tl.store(out_ptr + route * N + offs_n, accum.to(out_ptr.dtype.element_ty), offs_n < N)


def routing_cases():
    disjoint = [[(i * 13 + 7) % E for i in range(r * TOPK, (r + 1) * TOPK)] for r in range(M)]
    reuse = [[0, 1, 31, 32, 127, 128, 255, 256, 510, 511] for _ in range(M)]
    overlap = [[(start + j) % E for j in range(TOPK)] for start in (0, 5, 10, 0)]
    return [('reuse10', reuse, None), ('overlap20', overlap, None),
            ('disjoint40', disjoint, None),
            ('mixed_ep', disjoint, [(gid * 97 + 23) % E if gid % 3 else -1 for gid in range(E)]),
            ('all_miss', disjoint, [-1] * E)]


def rotating_rows(rows):
    """Disjoint expert sets across steps; preserve per-step reuse topology."""
    unique = sorted(set(sum(rows, [])))
    permutation = random.Random(20260906).sample(range(E), E)
    assert len(unique) * ROTATIONS <= E
    result = []
    for step in range(ROTATIONS):
        remap = dict(zip(unique, permutation[step * len(unique):(step + 1) * len(unique)]))
        result.append([[remap[gid] for gid in row] for row in rows])
    return result


def rotating_working_set(rows, mapping):
    gids = set(gid for step in rotating_rows(rows) for row in step for gid in row)
    physical = gids if mapping is None else {mapping[gid] for gid in gids if mapping[gid] >= 0}
    per_expert_bytes = N * (K // 2) + N * (K // GS) * 2 + (N // 2) * (K // GS)
    return {'distinct_global_experts_per_bank': len(gids),
            'distinct_physical_experts_per_bank': len(physical), 'banks': BANKS,
            'packed_working_set_mib': len(physical) * BANKS * per_expert_bytes / 2**20}


def cpu_contract_checks(source):
    tree = ast.parse(source)
    names = {item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)}
    required = ['_glm53_moe_int4_gemv', '_glm53_moe_int4_gemv_partial', '_glm53_moe_int4_gemv_reduce']
    for name in required:
        assert name in names, f'Installed baseline function missing: {name}'
    body = ast.get_source_segment(source, names[required[0]])
    partial = ast.get_source_segment(source, names[required[1]])
    reduce = ast.get_source_segment(source, names[required[2]])
    assert 'SPLITK, BN, BK, num_warps, num_stages = K // 32, 64, 32, 4, 1' in body
    assert 'BM = config["BLOCK_SIZE_M"]' in body
    assert 'x_row = offs_token // TOPK' in partial
    assert 'b = _glm53_tl.trans(((nib - zp) * sc).to(_glm53_tl.bfloat16))' in partial
    assert 'acc = acc * _glm53_tl.load(w_ptr + row)' in reduce
    assert K <= BK and BK % GS == K % GS == 0
    assert N % BN == N % 2 == 0 and ROUTES == 40
    assert BN * BK // (WARPS * 32) == 32, 'Unexpected per-thread element count'
    cases = routing_cases()
    assert [len(set(sum(rows, []))) for _, rows, _ in cases[:3]] == [10, 20, 40]
    for _, rows, mapping in cases:
        assert len(rows) == M and all(len(row) == TOPK for row in rows)
        assert all(len(set(row)) == TOPK for row in rows)
        assert all(0 <= gid < E for gid in sum(rows, []))
        if mapping is not None:
            assert len(mapping) == E and all(-1 <= lid < E for lid in mapping)
        rotated = rotating_rows(rows)
        assert len(rotated) == ROTATIONS
        assert all(len(set(sum(step, []))) == len(set(sum(rows, []))) for step in rotated)
        assert len(set(gid for step in rotated for row in step for gid in row)) == ROTATIONS * len(set(sum(rows, [])))
    assert rotating_working_set(cases[0][1], None)['packed_working_set_mib'] == 72.265625
    # Explicit nibble/channel/group boundary checks independent of Triton.
    assert [(0xA3 >> ((k % 2) * 4)) & 15 for k in (0, 1)] == [3, 10]
    assert [(0xD2 >> ((n % 2) * 4)) & 15 for n in (0, 1, 2, 3)] == [2, 13, 2, 13]
    assert [k // GS for k in (0, 31, 32, 63, 64, 127, 128, 159)] == [0, 0, 1, 1, 2, 3, 4, 4]
    assert sum(k < K for k in range(BK)) == 160
    print(json.dumps({'cpu_contract_checks': 'PASS', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'shape': {'E': E, 'N': N, 'K': K, 'GS': GS, 'M': M, 'topk': TOPK},
                      'candidate': {'BN': BN, 'BK': BK, 'warps': WARPS, 'programs': ROUTES * (N // BN)},
                      'rotating_working_sets': {name: rotating_working_set(rows, mapping) for name, rows, mapping in cases[:4]},
                      'gpu_imported': False}), flush=True)


def gpu_study(source, wall_seconds):
    import importlib
    import torch
    import triton
    import triton.language as tl
    globals()['tl'] = tl
    candidate_kernel = triton.jit(_route_gemv_kernel)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.version.hip and torch.cuda.is_available(), 'ROCm GPU required'
    prop = torch.cuda.get_device_properties(0)
    assert 'gfx1151' in prop.gcnArchName, prop.gcnArchName
    fm = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    assert Path(fm.__file__).read_text(encoding='utf-8') == source, 'Installed baseline differs from audited source'
    os.environ['VLLM_GFX1X_MOE_INT4_GEMV'] = '1'
    torch.manual_seed(20260906)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started >= wall_seconds:
            raise TimeoutError('GPU experiment deadline exceeded')

    banks = []
    for _ in range(BANKS):
        weights = torch.empty((E, N, K // 2), device='cuda', dtype=torch.uint8).random_(0, 256)
        scales = torch.empty((E, N, K // GS), device='cuda', dtype=torch.bfloat16).uniform_(0.002, 0.01)
        zp = torch.empty((E, N // 2, K // GS), device='cuda', dtype=torch.uint8).random_(0, 256)
        banks.append((weights, scales, zp))
    a = torch.empty((ROUTES, K), device='cuda', dtype=torch.bfloat16).normal_().mul_(0.25)
    route_weights = torch.empty((ROUTES,), device='cuda', dtype=torch.float32).uniform_(0.01, 0.2)
    route_weights[0] = 0.0
    route_weights[-1] = 1.0
    output = torch.empty((M, TOPK, N), device='cuda', dtype=torch.bfloat16)
    print(json.dumps({'study': 'routed_W2_SIMD_no_splitK_vs_installed_BM16_WMMA',
                      'device': prop.name, 'arch': prop.gcnArchName,
                      'torch': torch.__version__, 'hip': torch.version.hip, 'triton': triton.__version__,
                      'packed_weights_scales_zp_mib': sum(t.numel() * t.element_size() for bank in banks for t in bank) / 2**20,
                      'banks': BANKS, 'rotations_per_bank': ROTATIONS,
                      'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'note': 'Synthetic weights in exact runtime layout; route activations vary independently. Not model throughput.'}), flush=True)

    def routing(rows, mapping):
        ids = torch.tensor(rows, device='cuda', dtype=torch.int32)
        expert_map = None if mapping is None else torch.tensor(mapping, device='cuda', dtype=torch.int32)
        sorted_ids, experts, ntp = moe_align_block_size(ids, BM, E, expert_map)
        count = int(ntp.item())
        assert count == BM * len(set(sum(rows, [])))
        valid = sorted_ids[:count].cpu().tolist()
        assert sorted(v for v in valid if v < ROUTES) == list(range(ROUTES))
        assert sum(v >= ROUTES for v in valid) == count - ROUTES
        assert sorted_ids.numel() == ROUTES * BM
        return {'ids': ids, 'map': expert_map, 'sorted': sorted_ids, 'experts': experts, 'ntp': ntp}, count

    def cpu_reference(rows, mapping, mul_routed, bank=0):
        # Same independent unpacking as bench_moe_w1_geometry.py. W2 uses
        # x[assignment], not x[assignment // TOPK], and weights before BF16 cast.
        flat = sum(rows, [])
        weights, scales, zp = banks[bank]
        mapped = flat if mapping is None else [mapping[e] for e in flat]
        result = torch.zeros((ROUTES, N), dtype=torch.float32)
        x = a.cpu().float()
        for expert in sorted(set(mapped)):
            if expert < 0:
                continue
            packed, zpacked, sc = weights[expert].cpu(), zp[expert].cpu(), scales[expert].cpu().float()
            w = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(N, K)
            z = torch.stack((zpacked & 15, zpacked >> 4), dim=1).reshape(N, K // GS)
            dequant = ((w.float() - z.float().repeat_interleave(GS, dim=1))
                       * sc.repeat_interleave(GS, dim=1)).to(torch.bfloat16).float()
            assignments = [i for i, e in enumerate(mapped) if e == expert]
            result[assignments] = x[assignments] @ dequant.T
        if mul_routed:
            result *= route_weights.cpu().reshape(-1, 1)
        return result.to(torch.bfloat16).reshape(M, TOPK, N), mapped

    def baseline(route, mul_routed, bank=0):
        weights, scales, zp = banks[bank]
        assert fm._glm53_moe_int4_gemv(
            A=a, B=weights, C=output, B_scale=scales, B_zp=zp,
            topk_weights=route_weights, sorted_token_ids=route['sorted'],
            expert_ids=route['experts'], num_tokens_post_padded=route['ntp'],
            mul_routed_weight=mul_routed, top_k=1, compute_type=fm.tl.bfloat16,
            use_int4_w4a16=True, use_int8_w8a16=False, block_shape=[0, GS],
            config={'BLOCK_SIZE_M': BM}), 'Baseline guard rejected actual W2 contract'

    def candidate(route, mul_routed, bank=0):
        weights, scales, zp = banks[bank]
        candidate_kernel[(ROUTES, triton.cdiv(N, BN))](
            a, weights, scales, zp, route['ids'], route['map'] if route['map'] is not None else route['ids'],
            route_weights, output, N=N, K=K, GS=GS, BN=BN, BK=BK,
            HAS_MAP=route['map'] is not None, MUL_ROUTED=mul_routed,
            num_warps=WARPS, num_stages=1, enable_fp_fusion=False)

    cases = routing_cases()
    checks = 0
    # Complete correctness across reuse, disjoint, remapped and all-missing
    # routing before reporting any performance result. NaN poisoning exposes
    # missing stores; zeros from missing EP experts must be exact.
    for name, rows, mapping in cases:
        deadline()
        route, count = routing(rows, mapping)
        for mul_routed in (False, True):
            expected, mapped = cpu_reference(rows, mapping, mul_routed)
            missing = torch.tensor([e < 0 for e in mapped], dtype=torch.bool)
            output.fill_(float('nan'))
            baseline(route, mul_routed)
            reference = output.cpu()
            assert torch.isfinite(reference).all()
            torch.testing.assert_close(reference, expected, rtol=0.02, atol=0.015625)
            # The SIMD kernel consumes original IDs; it must match the existing
            # aligned-route baseline regardless of native or trimmed capacity.
            for layout in ('native_capacity', 'trimmed_capacity'):
                baseline_route = route.copy()
                if layout == 'trimmed_capacity':
                    baseline_route['sorted'] = route['sorted'][:count]
                    baseline_route['experts'] = route['experts'][:count // BM]
                output.fill_(float('nan'))
                baseline(baseline_route, mul_routed)
                torch.testing.assert_close(output.cpu(), reference, rtol=0.02, atol=0.015625)
            output.fill_(float('nan'))
            candidate(route, mul_routed)
            actual = output.cpu()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015625)
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.015625)
            assert torch.count_nonzero(actual.reshape(ROUTES, N)[missing]) == 0
            checks += 1
            print(json.dumps({'check': name, 'mul_routed': mul_routed,
                              'max_abs_vs_baseline': (actual.float() - reference.float()).abs().max().item(),
                              'max_abs_vs_cpu': (actual.float() - expected.float()).abs().max().item(),
                              'bitwise_baseline': torch.equal(actual, reference), 'status': 'PASS'}), flush=True)
        del route
    print(json.dumps({'all_correctness_checks': checks, 'status': 'PASS'}), flush=True)

    # Confirm independently generated additional banks against the same CPU
    # unpacking before they can contribute to any timing result.
    for bank in range(1, BANKS):
        deadline()
        rows = rotating_rows(cases[0][1])[0]
        route, _ = routing(rows, None)
        expected, _ = cpu_reference(rows, None, True, bank)
        for implementation, fn in [('baseline', baseline), ('route_simd', candidate)]:
            output.fill_(float('nan'))
            fn(route, True, bank)
            actual = output.cpu()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015625)
            print(json.dumps({'bank_check': bank, 'implementation': implementation, 'status': 'PASS'}), flush=True)
        del route

    def elapsed(fn, schedule):
        deadline()
        graph = None
        try:
            for route, bank in schedule:
                fn(route, True, bank)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            calls = len(schedule)
            replays = 160 // calls
            assert calls * replays == 160
            with torch.cuda.graph(graph):
                for route, bank in schedule:
                    fn(route, True, bank)
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(3):
                deadline()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(replays):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / (calls * replays))
            return statistics.median(samples), samples
        finally:
            if graph is not None:
                torch.cuda.synchronize()
                graph.reset()
                del graph
            gc.collect()
            torch.cuda.empty_cache()

    rotating_checks = 0
    for index, (name, rows, mapping) in enumerate(cases[:4]):
        route, count = routing(rows, mapping)
        rotated = [routing(step, mapping)[0] for step in rotating_rows(rows)]
        cold_schedule = [(step, bank) for step in rotated for bank in range(BANKS)]
        assert len(cold_schedule) == 32
        # Every rotating bank/ID combination is compared to the original
        # baseline before its timings; both kernels use the identical schedule.
        for active_route, bank in cold_schedule:
            deadline()
            output.fill_(float('nan'))
            baseline(active_route, True, bank)
            expected = output.cpu()
            output.fill_(float('nan'))
            candidate(active_route, True, bank)
            actual = output.cpu()
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015625)
            rotating_checks += 1
        print(json.dumps({'rotating_check': name, 'bank_id_combinations': len(cold_schedule),
                          'working_set': rotating_working_set(rows, mapping), 'status': 'PASS'}), flush=True)
        order = [('baseline', baseline), ('route_simd', candidate)]
        if index % 2:
            order.reverse()
        for mode, schedule in [('hot_fixed', [(route, 0)] * 8), ('rotating_4banks_8sets', cold_schedule)]:
            times = {}
            for implementation, fn in order:
                median_us, samples = elapsed(fn, schedule)
                times[implementation] = median_us
                print(json.dumps({'timing': 'W2_graph_median3x160calls', 'cache_mode': mode, 'routing': name,
                                  'implementation': implementation, 'median_us': median_us,
                                  'samples_us': samples, 'calls_per_graph': len(schedule),
                                  'distinct_global_experts_per_step': count // BM,
                                  'mul_routed': True, 'routing_preprocessing_included': False,
                                  'cache_miss_rate_measured': False}), flush=True)
            print(json.dumps({'routing_summary': name, 'cache_mode': mode,
                              'speedup': times['baseline'] / times['route_simd'],
                              'baseline_us': times['baseline'], 'route_simd_us': times['route_simd']}), flush=True)
        del route, rotated, cold_schedule, schedule
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    assert peak < 768, f'GPU live allocation limit exceeded: {peak:.3f} MiB'
    print(json.dumps({'status': 'PASS', 'correctness_checks': checks,
                      'rotating_bank_id_checks': rotating_checks, 'timing_configurations': 16,
                      'peak_allocated_mib': peak, 'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                      'gpu_phase_wall_seconds': time.monotonic() - started,
                      'graphs_reset': True, 'gpu_released_on_process_exit': True}), flush=True)


def bounded_gpu_child(args):
    """Independent process group; timeout never signals serving processes."""
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--gpu', '--worker',
               '--source', str(args.source.resolve()), '--wall-seconds', str(args.wall_seconds)]
    process = subprocess.Popen(command, start_new_session=True)

    def stop_owned_group():
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    try:
        code = process.wait(timeout=args.wall_seconds + 20)
    except subprocess.TimeoutExpired:
        stop_owned_group()
        code = 124
    except BaseException:
        stop_owned_group()
        raise
    print(json.dumps({'gpu_child_exit_code': code, 'status': 'PASS' if code == 0 else 'FAIL'}), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--gpu', action='store_true', help='Explicit opt-in; requires an isolated GPU lease')
    parser.add_argument('--wall-seconds', type=int, default=120)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 10 <= args.wall_seconds <= 120:
        parser.error('wall-seconds must be between 10 and 120')
    if args.worker and not args.gpu:
        parser.error('--worker requires --gpu')
    source = args.source.read_text(encoding='utf-8')
    cpu_contract_checks(source)
    if not args.gpu:
        return 0
    if os.name != 'posix':
        parser.error('GPU execution requires Linux')
    if args.worker:
        gpu_study(source, args.wall_seconds)
        return 0
    return bounded_gpu_child(args)


if __name__ == '__main__':
    raise SystemExit(main())
