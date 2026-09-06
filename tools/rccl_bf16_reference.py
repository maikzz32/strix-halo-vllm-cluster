#!/usr/bin/env python3
"""Collect actual RCCL BF16 bits for the fixed CPU-transport input banks.

Use run_rccl_microbench.py --bf16-reference only after an explicit GPU/network
lease. No timing or model-quality claim. Importing this module is CPU-only.
"""
import argparse
import array
import base64
import datetime
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import sys

_REFERENCE_SOURCE_B64 = None  # Injected by the existing owned RCCL launcher.
COUNT = 10240
BANKS = 32
BYTES = COUNT * 2


def reference_source():
    if _REFERENCE_SOURCE_B64 is not None:
        return base64.b64decode(_REFERENCE_SOURCE_B64, validate=True)
    return pathlib.Path(__file__).with_name('cpu_rdma_reference.py').read_bytes()


def input_banks(rank):
    source = reference_source()
    scope = {'__name__': 'cpu_rdma_reference_embedded'}
    exec(compile(source, 'cpu_rdma_reference_embedded.py', 'exec'), scope)
    if scope['COUNT'] != COUNT or scope['PERIOD'] != BANKS or rank not in range(4):
        raise ValueError('reference shape/rank mismatch')
    banks = []
    for bank in range(BANKS):
        words = array.array('H', (scope['input_bits'](rank, bank, i) for i in range(COUNT)))
        if sys.byteorder != 'little':
            words.byteswap()
        banks.append(words.tobytes())
    return banks, hashlib.sha256(source).hexdigest()


def little_endian_bytes(words):
    result = array.array('H', words)
    if sys.byteorder != 'little':
        result.byteswap()
    return result.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='20480')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--deadline', type=int, default=45)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.sizes != '20480' or not args.graph or not 10 <= args.deadline <= 45 or not re.fullmatch('[0-9a-f]{32}', args.run_id):
        parser.error('requires graph, 20480 bytes, deadline 10..45s, and owned 32-hex run ID')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4 or rank not in range(4):
        parser.error('requires exactly four ranks')

    def expired(signum, frame):
        print(json.dumps({'kind':'reference_error', 'rank':rank, 'run_id':args.run_id,
                          'error':'hard deadline exceeded'}), file=sys.stderr, flush=True)
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.deadline)
    banks, generator_sha = input_banks(rank)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=min(25, args.deadline)))
    graph = None
    output_banks = []
    repeat_equal = False
    metadata = {'kind':'reference_metadata', 'run_id':args.run_id, 'rank':rank, 'pid':os.getpid(),
                'world_size':world, 'host':socket.gethostname(), 'torch':torch.__version__, 'hip':torch.version.hip,
                'values':COUNT, 'banks':BANKS, 'dtype':'bf16', 'byte_order':'little', 'bytes_per_bank':BYTES,
                'generator_sha256':generator_sha, 'input_bank_sha256':[hashlib.sha256(b).hexdigest() for b in banks],
                'environment':{key:os.environ[key] for key in ('NCCL_IB_GID_INDEX','NCCL_NET_GDR_LEVEL',
                    'NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS','NCCL_LAUNCH_MODE','NCCL_GRAPH_MIXING_SUPPORT',
                    'NCCL_SOCKET_IFNAME','MASTER_ADDR','MASTER_PORT') if key in os.environ}}
    print(json.dumps(metadata), flush=True)
    try:
        cpu = [torch.frombuffer(bytearray(data), dtype=torch.uint16).clone() for data in banks]
        work = torch.empty(COUNT, dtype=torch.bfloat16, device='cuda')
        # Warm the communicator using freshly initialized inputs every time.
        for _ in range(3):
            work.fill_(rank + 1)
            dist.all_reduce(work)
            torch.cuda.synchronize()
            if not bool(torch.all(work == 10).item()):
                raise RuntimeError('eager all-reduce sanity check failed')
        work.zero_()
        torch.cuda.synchronize()
        dist.barrier(device_ids=[0])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            # Capture only reduction on a stable pointer; reset stays outside.
            dist.all_reduce(work)
        torch.cuda.synchronize()
        for bank in list(range(BANKS)) + [0]:
            work.copy_(cpu[bank].view(torch.bfloat16))
            torch.cuda.synchronize()
            if not torch.equal(work.view(torch.uint16).cpu(), cpu[bank]):
                raise RuntimeError(f'fresh GPU input differs from immutable CPU bank {bank}')
            graph.replay()
            torch.cuda.synchronize()
            if not bool(torch.isfinite(work).all().item()):
                raise RuntimeError(f'nonfinite RCCL output bank {bank}')
            actual = little_endian_bytes(work.view(torch.uint16).cpu().tolist())
            if len(output_banks) < BANKS:
                output_banks.append(actual)
            else:
                repeat_equal = actual == output_banks[0]
                if not repeat_equal:
                    raise RuntimeError('bank0 repeat differs after resetting all 32 banks')
        graph.reset()
        graph = None
        dist.barrier(device_ids=[0])
    finally:
        if graph is not None:
            graph.reset()
            graph = None
        dist.destroy_process_group()
    payload = b''.join(output_banks)
    if len(payload) != BANKS * BYTES:
        raise RuntimeError('incomplete output bank payload')
    # Emit after communicator teardown, avoiding asynchronous NCCL log writes
    # interleaved with this large, single-line archival JSON record.
    print(json.dumps({'kind':'reference_payload', 'rank':rank, 'run_id':args.run_id,
        'values':COUNT, 'banks':BANKS, 'byte_order':'little', 'finite':True,
        'all_gpu_inputs_checked':True, 'input_resets':BANKS + 1, 'bank0_repeat_equal':repeat_equal,
        'generator_sha256':generator_sha, 'output_sha256':hashlib.sha256(payload).hexdigest(),
        'output_bank_sha256':[hashlib.sha256(b).hexdigest() for b in output_banks],
        'payload_b64':base64.b64encode(payload).decode('ascii')}), flush=True)
    print(json.dumps({'kind':'complete', 'rank':rank, 'run_id':args.run_id,
                      'graph_reset':True, 'communicator_destroyed':True}), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
