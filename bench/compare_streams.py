#!/usr/bin/env python3
"""Compare saved C1 streams only when the request bodies and model agree."""
import argparse
import json
from pathlib import Path
import statistics


def compare(before, after):
    if before['arguments']['concurrency'] != 1 or after['arguments']['concurrency'] != 1:
        raise ValueError('This comparison is for one concurrent request only')
    old = {(row['prompt'], row['repeat']): row for row in before['results']}
    new = {(row['prompt'], row['repeat']): row for row in after['results']}
    if old.keys() != new.keys():
        raise ValueError('Prompt/repeat sets differ')
    rows = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if a['request'] != b['request']:
            raise ValueError(f'Request changed: {key}')
        same = all(a.get(k) == b.get(k) for k in ('content', 'reasoning', 'finish_reason'))
        same_tokens = a['usage'].get('completion_tokens') == b['usage'].get('completion_tokens')
        rows.append({'prompt': key[0], 'repeat': key[1], 'identical_output': same,
                     'identical_token_count': same_tokens,
                     'before_decode_tps': a['decode_tps'], 'after_decode_tps': b['decode_tps'],
                     'decode_change_percent': 100 * (b['decode_tps'] / a['decode_tps'] - 1),
                     'before_ttft_ms': 1000 * a['ttft_s'], 'after_ttft_ms': 1000 * b['ttft_s']})
    return {'matched_streams': len(rows), 'identical_outputs': sum(r['identical_output'] for r in rows),
            'median_paired_decode_change_percent': statistics.median(r['decode_change_percent'] for r in rows),
            'note': 'Changed outputs can change speculative acceptance. Repeated paired runs are needed to distinguish small gains from noise.',
            'results': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text(encoding='utf-8')),
                     json.loads(args.after.read_text(encoding='utf-8')))
    report = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
