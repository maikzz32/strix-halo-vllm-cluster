"""Inspect or reversibly select the measured four-node GPU performance mode."""
import argparse
import json
from pathlib import Path
from bench_gpu_profile import remote

HOSTS = ['192.168.1.' + str(i) for i in range(15, 19)]


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


def restore(original):
    states, errors = {}, []
    for host, state in original.items():
        try:
            states[host] = remote(host, 'restore', state)
        except Exception as exc:
            errors.append({'host': host, 'error': str(exc)})
    return {'states': states, 'errors': errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('read', 'high', 'restore'), default='read', nargs='?')
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    if args.mode == 'read':
        print(json.dumps({h: remote(h, 'read') for h in HOSTS}, indent=2))
        return
    if args.snapshot is None:
        parser.error('--snapshot is required for high and restore')
    if args.mode == 'restore':
        record = json.loads(args.snapshot.read_text(encoding='utf-8'))
        assert set(record['original']) == set(HOSTS)
        result = restore(record['original'])
        record['restoration'] = result
        save(args.snapshot, record)
        if result['errors']:
            raise RuntimeError('Restoration needs attention: ' + str(result['errors']))
        print(json.dumps(result, indent=2))
        return
    if args.snapshot.exists():
        parser.error('Snapshot already exists; preserve the original restoration record')
    original = {h: remote(h, 'read') for h in HOSTS}
    assert all(s['level'] == 'auto' for s in original.values())
    record = {'original': original, 'applied': {}}
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    save(args.snapshot, record)
    attempted = {}
    try:
        for host in HOSTS:
            attempted[host] = original[host]
            record['applied'][host] = remote(host, 'set', original[host])
        save(args.snapshot, record)
    except BaseException:
        record['rollback'] = restore(attempted)
        save(args.snapshot, record)
        if record['rollback']['errors']:
            print('ROLLBACK NEEDS ATTENTION: ' + str(record['rollback']['errors']))
        raise
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
