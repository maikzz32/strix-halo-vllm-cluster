"""Compare matching context probes without conflating cache states."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(before_path, after_path):
    before, after = [json.loads(p.read_text(encoding='utf-8')) for p in (before_path, after_path)]
    for key in ('model', 'prefix_id'):
        assert before[key] == after[key], key
    for key in ('lengths', 'tokens', 'repeats', 'phase_metrics'):
        assert before['arguments'][key] == after['arguments'][key], key
    assert len(before['cases']) == len(after['cases']) == len(before['arguments']['lengths'])
    rows = []
    for a, b in zip(before['cases'], after['cases']):
        for key in ('prompt_tokens', 'prompt_ids_sha256'):
            assert a[key] == b[key], key
        assert len(a['requests']) == len(b['requests']) == before['arguments']['repeats']
        for x, y in zip(a['requests'], b['requests']):
            assert x['repeat'] == y['repeat']
            def metric(request, name):
                values = [v for k, v in request['counter_delta'].items() if k.split('{')[0] == 'vllm:' + name]
                assert len(values) == 1, (name, values)
                return values[0]
            cache = {name: [metric(r, name) for r in (x, y)] for name in (
                'prefix_cache_hits_total', 'prefix_cache_queries_total',
                'request_prefill_kv_computed_tokens_sum')}
            rows.append(dict(prompt_tokens=a['prompt_tokens'], repeat=x['repeat'],
                identical_text=x['text_sha256'] == y['text_sha256'],
                identical_usage=x['usage'] == y['usage'], cache=cache,
                matched_cache_work=all(v[0] == v[1] for v in cache.values()),
                ttft_ms=[x['ttft_ms'], y['ttft_ms']],
                prefill_ms=[1000 * metric(r, 'request_prefill_time_seconds_sum') for r in (x, y)],
                wall_s=[x['wall_s'], y['wall_s']]))
    return dict(before_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest(),
        after_sha256=hashlib.sha256(after_path.read_bytes()).hexdigest(), rows=rows,
        all_outputs_match=all(r['identical_text'] and r['identical_usage'] for r in rows),
        all_cache_work_matches=all(r['matched_cache_work'] for r in rows),
        note='Individual paired timings; no aggregate speedup across unmatched cache work. Phase metrics are not pure GPU time.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
