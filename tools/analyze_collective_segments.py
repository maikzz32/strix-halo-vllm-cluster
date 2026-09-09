"""Attribute local intervals preceding UMA exchanges without host-clock alignment."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def analyze(path):
    events = json.loads(gzip.decompress(path.read_bytes()))['traceEvents']
    events = [e for e in events if e.get('ph') == 'X']
    scopes = [e for e in events if e.get('cat') == 'user_annotation'
              and e.get('name') == 'execute_context_0(0)_generation_1(4)']
    kernels = defaultdict(list)
    for event in events:
        if event.get('cat') in ('kernel', 'gpu_kernel'):
            kernels[event.get('args', {}).get('correlation')].append(event)
    graphs = []
    for launch in events:
        if launch.get('name') != 'hipGraphLaunch':
            continue
        parents = [s for s in scopes if (s['pid'], s['tid']) == (launch['pid'], launch['tid'])
                   and s['ts'] <= launch['ts'] < s['ts'] + s['dur']]
        if not parents:
            continue
        assert len(parents) == 1
        ks = sorted(kernels[launch['args']['correlation']], key=lambda e: e['ts'])
        assert len(ks) == 2646
        indices = [i for i, k in enumerate(ks) if '(anonymous namespace)::exchange(' in k['name']]
        assert len(indices) == 98
        rows = []
        previous = -1
        for index in indices:
            segment = ks[previous + 1:index]
            start = ks[previous]['ts'] + ks[previous]['dur'] if previous >= 0 else ks[0]['ts']
            elapsed = ks[index]['ts'] - start
            # Profiler timestamps can overlap. Measure the union clipped to
            # the local interval instead of assuming summed events are disjoint.
            assert elapsed >= 0
            cursor = start
            compute = 0.0
            for k in segment:
                left = max(start, k['ts'])
                right = min(ks[index]['ts'], k['ts'] + k['dur'])
                compute += max(0.0, right - max(left, cursor))
                cursor = max(cursor, right)
            rows.append({'preceding_interval_us': elapsed, 'traced_work_us': compute,
                         'untraced_interval_us': elapsed - compute,
                         'exchange_us': ks[index]['dur'], 'preceding_events': len(segment)})
            previous = index
        graphs.append(rows)
    assert len(graphs) == 16
    assert all([r['preceding_events'] for r in g] == [r['preceding_events'] for r in graphs[0]] for g in graphs)
    metrics = ('preceding_interval_us', 'traced_work_us', 'untraced_interval_us', 'exchange_us')
    positions = [{key: statistics.mean(g[i][key] for g in graphs) for key in metrics}
                 for i in range(98)]
    return {'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'trace_name': path.name,
            'graph_count': len(graphs), 'exchanges_per_graph': 98,
            'mean_totals_ms': {key: sum(p[key] for p in positions) / 1000 for key in metrics},
            'positions': positions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = [analyze(next(d.glob('*.gz'))) for d in args.directories]
    result = {'note': 'Local durations only. Position means are not synchronized arrivals. '
              'Untraced intervals are not proven GPU idle time. Exchange includes transport, '
              'reduction, GPU handshake and peer waiting. No causal late-rank attribution.',
              'ranks': records}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for record in records:
        print(json.dumps({k: v for k, v in record.items() if k != 'positions'}))
