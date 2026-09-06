"""ROCm profiler inheritance reproducer: CUDA-init parent then clean spawn.

Run each invocation under an external <=45-second timeout. Child executes the
tiny graph profiler harness only; no serving process or installed file changes.
"""
import argparse
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import sys

KEYS = ('ROCPROFILER_REGISTER_LIBRARY', 'HSA_TOOLS_ROCPROFILER_V1_TOOLS')


def env_snapshot():
    # C++ setenv may not update Python's already-created os.environ mapping.
    libc = ctypes.CDLL(None)
    libc.getenv.argtypes = [ctypes.c_char_p]
    libc.getenv.restype = ctypes.c_char_p
    return {key: (value.decode() if (value := libc.getenv(key.encode())) else None)
            for key in KEYS}


def child(clear, mode):
    before = env_snapshot()
    if clear:
        for key in KEYS:
            os.environ.pop(key, None)
            os.unsetenv(key)
    print(json.dumps({'child_pid': os.getpid(), 'clear_env': clear,
                      'env_before_torch_import': before, 'env_after_clear': env_snapshot()}), flush=True)
    script = Path(__file__).with_name('smoke_torch_profiler_schedule.py')
    sys.argv = [str(script), '--mode', mode]
    runpy.run_path(str(script), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clear-child-env', action='store_true')
    parser.add_argument('--mode', default='raw_unscheduled',
                        choices=['raw_unscheduled', 'raw_scheduled', 'vllm_unscheduled', 'vllm_scheduled'])
    args = parser.parse_args()
    before = env_snapshot()
    import torch
    torch.set_num_threads(1)
    assert torch.version.hip and torch.cuda.is_available()
    torch.cuda.init()
    print(json.dumps({'parent_pid': os.getpid(), 'parent_cuda_initialized': True,
                      'parent_env_before_init': before, 'parent_env_after_init': env_snapshot()}), flush=True)
    process = multiprocessing.get_context('spawn').Process(target=child,
                                                          args=(args.clear_child_env, args.mode))
    process.start()
    process.join(timeout=35)
    if process.is_alive():
        process.terminate()
        process.join(timeout=3)
        raise SystemExit('Child exceeded 35-second bound')
    print(json.dumps({'child_exitcode': process.exitcode, 'spawn_complete': True}), flush=True)
    raise SystemExit(process.exitcode or 0)


if __name__ == '__main__':
    main()
