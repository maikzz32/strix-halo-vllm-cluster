"""Validate bounded probe trace artifacts and summarize their local timings."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from read_rdma_trace import decode


def collect(directory):
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    output = []
    for config in manifest['configs']:
        path = directory / f"rank{config['rank']}-node{config['node']}.log"
        records = []
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict) and item.get('run_id') == manifest['run_id'] and item.get('rank') == config['rank']:
                records.append(item)
        traces = [r for r in records if r.get('event') == 'transport_trace']
        artifacts = [r['artifact'] for r in records if r.get('event') == 'rank_artifact']
        assert len(traces) == len(artifacts) == 1
        assert any(r.get('event') == 'launcher_complete' and r['owned_processes_gone'] for r in records)
        a, t = artifacts[0], traces[0]
        assert a['status'] == 'passed' and a['exact'] and a['backend_closed']
        data = base64.b64decode(t['base64'], validate=True)
        assert hashlib.sha256(data).hexdigest() == t['sha256']
        assert Path(t['name']).name == t['name']
        (directory / t['name']).write_bytes(data)
        decoded = decode(data)
        assert decoded['rank'] == config['rank'] and decoded['requested'] == config['trace_count']
        if config['trace_count']:
            assert decoded['published'] > 0
        (directory / f"trace-rank{config['rank']}.json").write_text(json.dumps(decoded, indent=2)+'\n', encoding='utf-8')
        rows = decoded['rows']
        output.append({'rank': config['rank'], 'trace_sha256': t['sha256'],
                       'published': decoded['published'], 'parity_proof': a['parity_proof'],
                       'median_us': {k: statistics.median(r[k] for r in rows) for k in
                                     ('post_us', 'completion_wait_us', 'reduction_and_fence_us', 'total_us')} if rows else {},
                       'last_peer_sets': dict(Counter(','.join(map(str, r['last_observed_recv_peers'])) or 'unknown' for r in rows)),
                       'hip_event_samples_us': a['hip_event_samples_us'],
                       'wall_samples_us': a['wall_samples_us'],
                       'iterations': a['iterations'], 'operations_per_graph': a['operations_per_graph']})
    assert len(output) == 4
    return {'run_id': manifest['run_id'], 'source_sha256': manifest['source_sha256'], 'ranks': output,
            'note': 'Isolated probe, not model execution. CQ observations include peer waiting. '
            'A full buffer stops tracing; exclude mixed enabled/disabled timing samples '
            'from instrumentation-overhead comparisons.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = collect(a.directory)
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'parity_proof'} for r in result['ranks']], indent=2))
