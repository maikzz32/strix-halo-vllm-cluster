"""Hash-pinned, opt-in TP4 INT4 prefill tile candidate; preview by default."""
import argparse
import hashlib
from pathlib import Path

BASE_SHA = '8b0d48a769b03c62e786a880941531ebdfb1bc582d53b6e03c5e49e73b3b2493'
OLD = '''    return config


def fused_experts_op('''
NEW = '''    # Opt-in experiment: retain the default decode and large-prefill paths.
    import os
    if (
        os.environ.get("STRIX_MOE_PREFILL_TILES", "0") == "1"
        and current_platform.is_rocm()
        and not override_config
        and dtype == "int4_w4a16"
        and tuple(w1_shape) == (512, 320, 1280)
        and tuple(w2_shape) == (512, 2560, 80)
        and top_k == 10
        and block_shape is not None
        and list(block_shape) == [0, 32]
        and 64 <= M <= 1024
    ):
        config = dict(config, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32)
    return config


def fused_experts_op('''

def transform(source):
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA:
        raise ValueError('Unexpected fused_moe.py source SHA')
    if source.count(OLD) != 1:
        raise ValueError('Expected exactly one config anchor')
    result = source.replace(OLD, NEW)
    compile(result, '<prefill-tiles>', 'exec')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    before = a.source.read_bytes()
    after = transform(before.decode()).encode()
    print('before', BASE_SHA, 'after', hashlib.sha256(after).hexdigest())
    if a.apply:
        backup = a.source.with_name(a.source.name + '.pre-strix-prefill-' + BASE_SHA)
        if backup.exists() and backup.read_bytes() != before:
            raise ValueError('Backup mismatch')
        backup.write_bytes(before)
        temporary = a.source.with_name(a.source.name + '.strix-prefill-tmp')
        temporary.write_bytes(after)
        temporary.replace(a.source)
