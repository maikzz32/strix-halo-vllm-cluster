"""Pin one hash-identified gfx1151 norm/gate kernel to baseline geometry.

Install before generated Inductor modules are loaded. Cached tuning choices
are overridden in memory; no shared autotune cache entry is overwritten.
"""
import ast
import hashlib
import json
import os
from pathlib import Path

BODY_SHA256 = '082455cf69b7b450d76dba6566cd876d43eb18edf67542e20b2c1cb926ec68d1'
_INSTALLED = False


def body_hash(source):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1:
        return None
    fn = functions[0]
    fn.name = 'kernel'
    fn.decorator_list = []
    # Python 3.12 adds an empty type_params field absent from 3.10 ASTs.
    if hasattr(fn, 'type_params'):
        if fn.type_params:
            return None
        del fn.type_params
    return hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest()


def _record(obj, stage, config):
    folder = os.environ.get('STRIX_NORM_GEOMETRY_LOG_DIR')
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        row = dict(stage=stage, pid=os.getpid(), body_sha256=BODY_SHA256,
                   kernel_name=obj.fn.__name__, filename=obj.filename,
                   kwargs=config.kwargs, num_warps=config.num_warps,
                   num_stages=config.num_stages)
        with (path / f'{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from triton import Config
    original_init = CachingAutotuner.__init__
    original_precompile = CachingAutotuner._precompile_config

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._strix_norm_geometry_pinned = False
        if getattr(self.device_props, 'cc', None) != 'gfx1151':
            return
        if body_hash(self.fn.src) != BODY_SHA256:
            return
        self.configs = [Config({'XBLOCK': 4}, num_warps=2, num_stages=1)]
        self.save_cache_hook = None
        self.inductor_meta['coordinate_descent_tuning'] = False
        self._strix_norm_geometry_pinned = True
        _record(self, 'selected', self.configs[0])

    def precompile(self, config):
        if getattr(self, '_strix_norm_geometry_pinned', False):
            assert config.kwargs == {'XBLOCK': 4}
            assert config.num_warps == 2 and config.num_stages == 1
            _record(self, 'precompile', config)
        return original_precompile(self, config)

    CachingAutotuner.__init__ = initialize
    CachingAutotuner._precompile_config = precompile
    _INSTALLED = True
