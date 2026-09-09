"""Generate the opt-in, shape-restricted dense INT4 launch-geometry trial."""
import hashlib
BASE_SHA256 = 'a136168a08484c2b3e9d18ba2fcfbd9c067a5a70dc1ebb9e319b89ddb1e5a48e'
def generate(source: bytes) -> str:
    assert hashlib.sha256(source).hexdigest() == BASE_SHA256, 'unexpected installed source'
    text = source.decode()
    anchor = 'from contextlib import nullcontext\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + 'import os as _strix_dense_os\n_STRIX_DENSE_GRID = _strix_dense_os.environ.get("STRIX_DENSE_INT4_GRID", "0") == "1"\n')
    anchor = '        with ctx:\n            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, w_zp, bias)'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '        # Only the two measured shapes, with the tested data types/layout contract.\n        if (_STRIX_DENSE_GRID and cu_count == 20 and group_size == 32\n                and bias is None and w_zp is not None\n                and x_2d.dtype == torch.bfloat16 and w_s.dtype == torch.bfloat16\n                and w_zp.dtype == torch.bfloat16 and w_q.dtype == torch.int8\n                and (M, N, K) in ((1, 512, 2560), (4, 2560, 160))\n                and _on_gfx1151()):\n            cu_count = 40\n        with ctx:\n            return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, w_zp, bias)')
    compile(text, '<dense-int4-grid-candidate>', 'exec')
    return text
