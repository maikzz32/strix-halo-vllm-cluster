"""One C1 answer; trigger bounded vLLM profiling after its first output chunk."""
import argparse
import concurrent.futures
import datetime
import json
from pathlib import Path
import sys
import time
import urllib.request

from bench_stream import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(args.url + '/v1/models', timeout=10) as response:
        model = json.load(response)['data'][0]['id']
    request_body = {'model': model, 'messages': [{'role': 'user', 'content': PROMPTS['code']}],
                    'temperature': 0, 'seed': 42, 'max_tokens': 2048, 'ignore_eos': True,
                    'stream': True, 'stream_options': {'include_usage': True, 'continuous_usage_stats': True},
                    'chat_template_kwargs': {'enable_thinking': False}}
    record = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'request': request_body, 'profile_calls': [], 'events': [],
              'note': 'Timings are perturbed by profiling and are not a performance benchmark.'}
    started = time.perf_counter()
    fragments, thoughts, usage = [], [], {}
    stream_done = False

    def profile_call(endpoint):
        stamp = time.perf_counter() - started
        req = urllib.request.Request(args.url + endpoint, data=b'', method='POST')
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            response.read()
        return {'endpoint': endpoint, 'started_s': stamp,
                'finished_s': time.perf_counter() - started, 'status': status}

    future = None
    workload_error = None
    start_recorded = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            req = urllib.request.Request(args.url + '/v1/chat/completions',
                                         json.dumps(request_body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=120) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if line == 'data: [DONE]':
                        stream_done = True
                        continue
                    if not line.startswith('data: '):
                        continue
                    event = json.loads(line[6:])
                    if event.get('usage'):
                        usage = event['usage']
                    choices = event.get('choices', [])
                    if not choices:
                        continue
                    delta = choices[0].get('delta', {})
                    content = delta.get('content') or ''
                    thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if content or thought:
                        record['events'].append({'seconds': time.perf_counter() - started,
                                                 'completion_tokens': usage.get('completion_tokens')})
                        fragments.append(content)
                        thoughts.append(thought)
                        if future is None:
                            future = pool.submit(profile_call, '/start_profile')
                            print('First output received; requesting one bounded GPU profile', flush=True)
            if future is None:
                raise RuntimeError('No generated output; profiler was not started')
            record['profile_calls'].append(future.result(timeout=35))
            start_recorded = True
            if not stream_done or usage.get('completion_tokens') != request_body['max_tokens']:
                raise RuntimeError('Profiling workload ended before its complete 2048-token stream')
        except BaseException as error:
            workload_error = error
            record['workload_error'] = repr(error)
            raise
        finally:
            cleanup_error = None
            if future is not None:
                # vLLM normally stops itself after its configured 16 active rounds.
                # A failed/lost start response can still leave some ranks profiling.
                # Settle that call, then attempt stop independently of its result.
                try:
                    start_call = future.result(timeout=35)
                    if not start_recorded:
                        record['profile_calls'].append(start_call)
                except Exception as error:
                    record['profile_start_error'] = repr(error)
                try:
                    record['profile_calls'].append(profile_call('/stop_profile'))
                except Exception as error:
                    cleanup_error = error
                    record['profile_cleanup_error'] = repr(error)
            record.update(content=''.join(fragments), reasoning=''.join(thoughts), usage=usage,
                          stream_done=stream_done,
                          workload_complete=stream_done and usage.get('completion_tokens') == request_body['max_tokens'],
                          wall_seconds=time.perf_counter() - started)
            try:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
            except Exception as error:
                if workload_error is None:
                    raise
                print(f'Could not save profiling workload: {error!r}', file=sys.stderr)
            if cleanup_error is not None and workload_error is None:
                raise RuntimeError('Profiling cleanup failed; workload artifact saved') from cleanup_error
    print(json.dumps({'saved': str(args.output), 'usage': usage,
                      'profile_calls': record['profile_calls'],
                      'cleanup_error': record.get('profile_cleanup_error')}), flush=True)


if __name__ == '__main__':
    main()
