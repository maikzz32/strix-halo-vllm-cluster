#!/usr/bin/env python3
"""Opt-in Qwen4Exp AMD MTP local-argmax delegation; dry-run by default.

The implementation delegates to vLLM's existing quantization-aware reduction.
It does not change the target model, logits, weights, or vocabulary.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import stat
import tempfile

DEFAULT_TARGET = Path('/usr/local/lib64/python3.12/site-packages/vllm/models/qwen4_exp/amd/mtp.py')
MARKER = '# strix-cluster: mtp-local-argmax-v1'
ADDITION = '''
    # strix-cluster: mtp-local-argmax-v1
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transform(data: bytes) -> bytes:
    source = data.decode('utf-8')
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen4ExpMTP']
    if len(classes) != 1:
        raise ValueError('Expected exactly one Qwen4ExpMTP class')
    cls = classes[0]
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    existing = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_top_tokens']
    if existing:
        expected = ast.parse('class Stub:\n' + ADDITION).body[0].body[0]
        if len(existing) != 1 or MARKER not in source or ast.dump(existing[0]) != ast.dump(expected):
            raise ValueError('Existing get_top_tokens differs; refusing to overwrite')
        return data
    if MARKER in source:
        raise ValueError('Patch marker present without expected method')
    anchor = methods.get('compute_logits')
    if anchor is None:
        raise ValueError('Missing compute_logits anchor')
    expected_body = ast.parse('return self.logits_processor(self.lm_head, hidden_states)').body
    if ast.dump(ast.Module(body=anchor.body, type_ignores=[])) != ast.dump(ast.Module(body=expected_body, type_ignores=[])):
        raise ValueError('compute_logits body changed; review required')
    if len([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_logits']) != 1:
        raise ValueError('Non-unique compute_logits anchor')
    lines = source.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in source else '\n'
    insertion = ADDITION.replace('\n', newline)
    result = ''.join(lines[:anchor.end_lineno]) + insertion + ''.join(lines[anchor.end_lineno:])
    ast.parse(result)
    return result.encode('utf-8')


def backup_path(target: Path, original: bytes) -> Path:
    return target.with_name(target.name + '.before-mtp-local-argmax.' + digest(original)[:16] + '.bak')


def replace_file(target: Path, data: bytes) -> None:
    mode = stat.S_IMODE(target.stat().st_mode)
    fd, name = tempfile.mkstemp(prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, mode)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run(target: Path, apply: bool, restore: bool = False) -> str:
    target = target.resolve(strict=True)
    current = target.read_bytes()
    if restore:
        newline = '\r\n' if b'\r\n' in current else '\n'
        block = ADDITION.replace('\n', newline).encode()
        if current.count(block) != 1:
            raise ValueError('Cannot restore: exact patch block is not present once')
        original = current.replace(block, b'', 1)
        backup = backup_path(target, original)
        if not backup.is_file() or backup.read_bytes() != original or transform(original) != current:
            raise ValueError('Cannot restore: original backup missing or source changed')
        if apply:
            replace_file(target, original)
        return f'{"RESTORED" if apply else "WOULD RESTORE"} {target} from {backup}'
    updated = transform(current)
    if updated == current:
        return f'ALREADY PATCHED {target} sha256={digest(current)}'
    backup = backup_path(target, current)
    if apply:
        if backup.exists():
            if backup.read_bytes() != current:
                raise ValueError('Backup path exists with different content')
        else:
            with backup.open('xb') as handle:
                handle.write(current)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
        replace_file(target, updated)
    return f'{"PATCHED" if apply else "WOULD PATCH"} {target} backup={backup} before={digest(current)} after={digest(updated)}'


def self_test() -> None:
    sample = b'class Qwen4ExpMTP:\n    def compute_logits(self, hidden_states, spec_step_idx=0):\n        return self.logits_processor(self.lm_head, hidden_states)\n'
    assert transform(transform(sample)) == transform(sample)
    for bad in [sample.replace(b'Qwen4ExpMTP', b'Other'), sample.replace(b'self.lm_head,', b'self.other_head,')]:
        try:
            transform(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed anchor was accepted')
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / 'mtp.py'
        p.write_bytes(sample)
        run(p, apply=False)
        assert p.read_bytes() == sample and not list(Path(folder).glob('*.bak'))
        run(p, apply=True)
        assert backup_path(p, sample).read_bytes() == sample
        assert 'ALREADY PATCHED' in run(p, apply=True)
        run(p, apply=True, restore=True)
        assert p.read_bytes() == sample
    print('PASS: anchor rejection, dry-run, idempotence, backup, exact restore')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--apply', action='store_true', help='Write the patch or restoration; otherwise inspect only')
    parser.add_argument('--restore', action='store_true', help='Restore the exact original backup (requires --apply to write)')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(run(args.target, args.apply, args.restore))


if __name__ == '__main__':
    main()
