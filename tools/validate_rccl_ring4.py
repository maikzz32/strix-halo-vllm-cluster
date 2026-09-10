"""Startup-only comparison against the actual PyNccl communicator.

Import has no GPU/NumPy/Torch initialization. No serving hook is installed.
"""
import hashlib


def probe_bank(bank, seed=202609071027):
    """Fresh independent finite BF16 values, including rotated cancellation."""
    import numpy as np
    rng = np.random.default_rng(seed + bank)
    bits = rng.integers(0x3000, 0x4700, size=(4, 10240), dtype=np.uint16)
    bits |= rng.integers(0, 2, size=bits.shape, dtype=np.uint16) << 15
    # Vary the operand/rank assignment, unlike the older periodic bank generator.
    for offset, values in enumerate(((0x4380,0x3f80,0xc380,0x3f80),
                                     (0x3f80,0x3b80,0xbf80,0x3b80))):
        values = np.roll(np.array(values, dtype=np.uint16), bank % 4)
        bits[:, offset::32] = values[:, None]
    return bits


def validate(backend, reference, cpu_group, banks=32):
    import numpy as np
    import torch
    import torch.distributed as dist
    from cpu_rdma_ring4_reference import PROFILE_SHA256, reduce_rccl_ring4_bits
    if banks != 32 or backend.settings['mode'] != 'ring4':
        raise ValueError('This validator requires ring4 and 32 independent banks')
    rank, device = backend.rank, backend.device
    source = torch.empty(10240, dtype=torch.bfloat16, device=device)
    errors = []
    hashes = []
    def check(kind, output, expected, bank):
        if output is None or output.data_ptr() == source.data_ptr():
            errors.append({'kind':kind, 'bank':bank, 'error':'missing/out-of-place output'})
            return
        bits = output.view(torch.uint16).cpu().numpy()
        if not np.array_equal(bits, expected):
            different = np.flatnonzero(bits != expected)
            errors.append({'kind':kind, 'bank':bank, 'mismatches':int(different.size),
                           'first':int(different[0]), 'actual':int(bits[different[0]]),
                           'expected':int(expected[different[0]])})
    graph = None
    try:
        for phase in ('eager', 'graph'):
            if phase == 'graph':
                source.fill_(rank + 1)
                torch.cuda.synchronize(device)
                dist.barrier(group=cpu_group)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    rccl_out = reference.all_reduce(source, stream=torch.cuda.current_stream(device))
                    candidate_out = backend.try_all_reduce(source)
                torch.cuda.synchronize(device)
            for bank in list(range(banks)) + [0]:
                inputs = probe_bank(bank)
                expected = reduce_rccl_ring4_bits(inputs)
                cpu_input = torch.from_numpy(inputs[rank].copy())
                source.copy_(cpu_input.view(torch.bfloat16))
                if phase == 'eager':
                    rccl_out = reference.all_reduce(source, stream=torch.cuda.current_stream(device))
                    candidate_out = backend.try_all_reduce(source)
                else:
                    graph.replay()
                torch.cuda.synchronize(device)
                if not torch.equal(source.view(torch.uint16).cpu(), cpu_input):
                    errors.append({'kind':'input_mutation', 'bank':bank, 'phase':phase})
                check(phase + '_rccl', rccl_out, expected, bank)
                check(phase + '_candidate', candidate_out, expected, bank)
                if phase == 'eager' and bank < banks and len(hashes) < banks:
                    hashes.append(hashlib.sha256(inputs.tobytes()).hexdigest())
            if graph is not None:
                graph.reset()
                graph = None
    finally:
        if graph is not None:
            torch.cuda.synchronize(device)
            graph.reset()
    # Numerical failures do not change the collective call sequence on any rank.
    results = [None] * 4
    dist.all_gather_object(results, {'rank':rank, 'errors':errors}, group=cpu_group)
    passed = all(not r['errors'] for r in results)
    proof = {'banks':banks, 'elements_per_rank':10240, 'exact_eager':passed,
             'exact_graph':passed, 'validated_against_active_pynccl':passed,
             'profile_sha256':PROFILE_SHA256, 'input_bank_sha256':hashes,
             'repeat_bank0':True, 'graph_destroyed':True, 'rank_results':results}
    # Kept even if create_if_enabled subsequently terminates a mismatching worker.
    import json
    print(json.dumps({'event':'parity_proof', 'rank':rank, 'proof':proof}), flush=True)
    backend.parity_proof = proof
    return proof
