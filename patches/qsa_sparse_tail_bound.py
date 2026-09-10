"""Generate the measured QSA tail bound, pinned to the verified source."""
import hashlib

ORIGINAL_SHA = '0aed889a98011d75d62e4ed6fe911821088e99b7aa1dd2921de2ef41730e76fd'
CANDIDATE_SHA = 'e02ae3503c07229904d3dcc3f3d3732a75664dee6f38f31334da6df1c53c02f4'


def generate(source: bytes) -> bytes:
    assert hashlib.sha256(source).hexdigest() == ORIGINAL_SHA
    text = source.decode()
    marker = '    for tile in range(split_tile_start, split_tile_end):'
    assert text.count(marker) == 1
    bound = '''    selection_columns = tl.arange(0, triton.next_power_of_2(TOPK))
    selection_tokens = tl.load(
        indices_ptr + row * stride_indices_row + selection_columns,
        mask=selection_columns < TOPK, other=-1,
    )
    selection_end = tl.max(tl.where(selection_tokens >= 0, selection_columns + 1, 0), axis=0)
    split_tile_end = tl.minimum(split_tile_end, tl.cdiv(selection_end, BLOCK_N))
'''
    candidate = text.replace(marker, bound + marker).encode()
    assert hashlib.sha256(candidate).hexdigest() == CANDIDATE_SHA
    return candidate
