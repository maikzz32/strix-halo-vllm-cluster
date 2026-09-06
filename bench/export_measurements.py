"""Export measured scalars and output digests without copying generated text."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--additional-run', action='append', default=[],
                        help='Additional result directory name to include after the original experiments')
    args = parser.parse_args()
    baseline = args.results / 'baseline-sharegpt-tp4/benchmark.log'
    text = baseline.read_text(encoding='utf-8')
    fields = {
        'completed': 'Successful requests', 'failed': 'Failed requests',
        'max_concurrency': 'Maximum request concurrency', 'duration': 'Benchmark duration (s)',
        'total_input_tokens': 'Total input tokens', 'total_output_tokens': 'Total generated tokens',
        'output_throughput': 'Output token throughput (tok/s)',
        'mean_ttft_ms': 'Mean TTFT (ms)', 'mean_tpot_ms': 'Mean TPOT (ms)',
        'spec_decode_acceptance_rate': 'Acceptance rate (%)',
    }
    first = {'name': 'baseline-sharegpt-tp4', 'source_sha256': sha(baseline.read_bytes()),
             'source_kind': 'rounded benchmark log; detailed output text not retained',
             'generated_texts_sha256': None}
    for key, label in fields.items():
        match = re.search(r'^' + re.escape(label) + r':\s*([\d.]+)', text, re.M)
        if not match:
            raise ValueError(f'Missing baseline field: {label}')
        first[key] = float(match.group(1))
    runs = [first]
    names = ['argmax-sharegpt-tp4', 'qsa-invisible-sharegpt-tp4',
             'qsa-warm-sharegpt-tp4', 'ep-k640-qsa-sharegpt-tp4', *args.additional_run]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate result directory')
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError(f'Invalid result directory name: {name}')
        path = args.results / name / 'result.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        row = {'name': name, 'source_sha256': sha(path.read_bytes()),
               'source_kind': 'vLLM result.json',
               'generated_texts_sha256': [sha(value.encode('utf-8')) for value in data['generated_texts']]}
        for key in (*fields, 'spec_decode_acceptance_length', 'spec_decode_num_drafts',
                    'spec_decode_draft_tokens', 'spec_decode_accepted_tokens',
                    'spec_decode_per_position_acceptance_rates'):
            row[key] = data[key]
        row['texts_identical_to_argmax'] = None if len(runs) == 1 else (
            row['generated_texts_sha256'] == runs[1]['generated_texts_sha256'])
        runs.append(row)
    result = {'schema': 1, 'date': '2026-09-06', 'workload': 'ShareGPT48, seed42, temperature0, C1',
              'model': '/home/maik/qwen38_rest', 'quantization': 'existing INT4 compressed-tensors GS32',
              'context_capacity': 262144,
              'metric': 'output_throughput includes prompt processing and request overhead',
              'caveat': 'QSA gain about2%; substantially greater single-answer speed not yet achieved',
              'runs': runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'saved': str(args.output), 'runs': len(runs)}))


if __name__ == '__main__':
    main()
