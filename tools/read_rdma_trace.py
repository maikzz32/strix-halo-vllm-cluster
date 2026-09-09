"""Read bounded RDMA timing records, or arm an unused live diagnostic map."""
import argparse
import json
import mmap
from pathlib import Path
import struct

HEADER = struct.Struct('<8Q')
ROW = struct.Struct('<16Q')
MAGIC = 0x31524354414D4452


def decode(data):
    magic, version, rank, capacity, requested, published, first, reserved = HEADER.unpack_from(data)
    assert magic == MAGIC and version == 1 and rank < 4 and capacity == 2048
    assert len(data) == HEADER.size + capacity * ROW.size
    assert 0 <= published <= requested <= capacity and reserved == 0
    rows = []
    for i in range(published):
        values = ROW.unpack_from(data, HEADER.size + i * ROW.size)
        seq, start, posted, ready, reduced, entry = values[:6]
        recv, send = list(values[6:10]), list(values[10:14])
        assert seq == first + i and 0 < start <= posted <= ready <= reduced
        peers = [r for r in range(4) if r != rank]
        assert not (entry & (1 << rank)) and entry < 16
        assert all(start <= send[r] <= ready for r in peers)
        assert send[rank] == recv[rank] == 0
        assert all((0 < recv[r] <= ready) or (entry & (1 << r)) for r in peers)
        assert values[15] == 0
        latest = max(recv[r] for r in peers)
        # Equal timestamps mean the CQ batch cannot order those peers.
        last_peers = [r for r in peers if recv[r] == latest] if all(recv[r] for r in peers) else []
        rows.append({'sequence': seq, 'post_us': (posted-start)/1000,
                     'completion_wait_us': (ready-posted)/1000,
                     'reduction_and_fence_us': (reduced-ready)/1000,
                     'total_us': (reduced-start)/1000, 'entry_recv_mask': entry,
                     'recv_relative_us': [(t-start)/1000 if t else None for t in recv],
                     'send_relative_us': [(t-start)/1000 if t else None for t in send],
                     'last_observed_recv_peers': last_peers, 'polls': values[14]})
    return {'rank': rank, 'requested': requested, 'published': published, 'rows': rows,
            'note': 'Local CQ observation times, not NIC arrival timestamps. Equal batch '
            'timestamps are ties. Negative receive offsets represent observations in a '
            'previous collective. Missing pre-arm observations remain unknown. '
            'No cross-host clock alignment or serving latency inferred.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path)
    p.add_argument('--arm', type=int)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.arm is not None:
        if not 1 <= a.arm <= 2048:
            p.error('--arm must be 1..2048')
        with a.path.open('r+b') as f, mmap.mmap(f.fileno(), 0) as memory:
            state = decode(memory)
            assert state['requested'] == state['published'] == 0, 'Only arm an unused map'
            struct.pack_into('<Q', memory, 32, a.arm)
        print('Armed', a.arm, 'records; existing samples are never reset or overwritten')
    else:
        result = decode(a.path.read_bytes())
        text = json.dumps(result, indent=2) + '\n'
        if a.output:
            a.output.write_text(text, encoding='utf-8')
        else:
            print(text)
