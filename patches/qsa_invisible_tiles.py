#!/usr/bin/env python3
"""Opt-in QSA scoring early exit; dry-run unless --apply, exact --restore.

Default VLLM_QSA_SKIP_INVISIBLE_TILES=0 preserves the original compiled path.
Invisible tiles still write negative infinity and tile 0 writes visibility.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import os
from pathlib import Path
import stat
import tempfile

DEFAULT_TARGET = Path('/usr/local/lib64/python3.12/site-packages/vllm/models/qwen4_exp/amd/ops/qsa.py')
MARKER = '# strix-cluster: qsa-invisible-tiles-v1'
EDITS = [
    ('import math\n', 'import math\nimport os\n' + MARKER + '\n'),
    ('''    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
''', '''    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SKIP_INVISIBLE_TILES: tl.constexpr,
) -> None:
'''),
    ('''    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
''', '''    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    if SKIP_INVISIBLE_TILES:
        if (tl.program_id(1) * BLOCK_N >= visible) | (request < 0) | (request >= num_requests):
            tl.store(
                logits_ptr + row * stride_logits_row + columns,
                -float("inf"),
                mask=(row < num_rows) & (columns < num_columns),
            )
            return
    logical_page = columns // PAGE_SIZE
'''),
    ('''        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    return logits, visible_blocks
''', '''        COMPRESS_RATIO=compress_ratio,
        SKIP_INVISIBLE_TILES=os.environ.get("VLLM_QSA_SKIP_INVISIBLE_TILES", "0") == "1",
        num_warps=4,
    )
    return logits, visible_blocks
'''),
]


def edits_for(source):
    newline = '\r\n' if '\r\n' in source else '\n'
    return [(old.replace('\n', newline), new.replace('\n', newline)) for old, new in EDITS]


def transform(data: bytes) -> bytes:
    source = data.decode('utf-8')
    tree = ast.parse(source)
    funcs = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('_qsa_mqa_paged_kernel', 'qsa_mqa_paged'):
        if name not in funcs:
            raise ValueError(f'Missing source function {name}')
    edits = edits_for(source)
    if MARKER in source:
        if source.count(MARKER) == 1 and all(source.count(new) == 1 for _, new in edits):
            return data
        raise ValueError('Existing patch differs; refusing overwrite')
    if source.count('import os\n') or source.count('import os\r\n'):
        raise ValueError('Unexpected existing os import; re-audit')
    for i, (old, _) in enumerate(edits):
        if source.count(old) != 1:
            raise ValueError(f'Anchor {i} changed or not unique')
        if i in (1, 2) and old not in funcs['_qsa_mqa_paged_kernel']:
            raise ValueError('Kernel anchor outside expected function')
        if i == 3 and old.rstrip('\r\n') not in funcs['qsa_mqa_paged']:
            raise ValueError('Launch anchor outside expected function')
    for old, new in edits:
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source.encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def backup_path(target, data):
    return target.with_name(target.name + '.before-qsa-invisible.' + sha(data)[:16] + '.bak')


def replace_file(target, data):
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


def run(target, apply=False, restore=False):
    target = target.resolve(strict=True)
    current = target.read_bytes()
    if restore:
        original = current.decode('utf-8')
        for old, new in reversed(edits_for(original)):
            if original.count(new) != 1:
                raise ValueError('Exact patched block missing or changed')
            original = original.replace(new, old, 1)
        original = original.encode('utf-8')
        backup = backup_path(target, original)
        if not backup.is_file() or backup.read_bytes() != original or transform(original) != current:
            raise ValueError('Backup missing or source changed; refusing restore')
        if apply:
            replace_file(target, original)
        return f'{"RESTORED" if apply else "WOULD RESTORE"} {target}'
    updated = transform(current)
    if updated == current:
        return f'ALREADY PATCHED {target} sha256={sha(current)}'
    backup = backup_path(target, current)
    if apply:
        if backup.exists():
            if backup.read_bytes() != current:
                raise ValueError('Backup content differs')
        else:
            with backup.open('xb') as handle:
                handle.write(current)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
        replace_file(target, updated)
    return f'{"PATCHED" if apply else "WOULD PATCH"} {target} backup={backup} before={sha(current)} after={sha(updated)} gate=VLLM_QSA_SKIP_INVISIBLE_TILES(default0)'


def self_test(target):
    original = target.read_bytes()
    assert transform(transform(original)) == transform(original)
    with tempfile.TemporaryDirectory() as folder:
        local = Path(folder) / 'qsa.py'
        local.write_bytes(original)
        run(local)
        assert local.read_bytes() == original and not list(Path(folder).glob('*.bak'))
        run(local, True)
        assert backup_path(local, original).read_bytes() == original
        assert 'ALREADY PATCHED' in run(local, True)
        run(local, True, True)
        assert local.read_bytes() == original
    try:
        transform(original.replace(b'block_n = 32', b'block_n = 64').replace(b'    return logits, visible_blocks', b'    return (logits, visible_blocks)'))
    except ValueError:
        pass
    else:
        raise AssertionError('Modified launch anchor accepted')
    print('PASS: anchors, dry-run, idempotence, backup, exact restore')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.target)
    else:
        print(run(args.target, args.apply, args.restore))


if __name__ == '__main__':
    main()
