"""Opt-in expert-major W1 scheduling, retaining W2 and original arithmetic."""
import hashlib
from pathlib import Path

BASE_SHA = '25ac1a380733b9f3ee3735cc58779e2d1215d5f87ed9e601cfc839d49c1403b7'


def patched(base):
    assert hashlib.sha256(base).hexdigest() == BASE_SHA
    source = base.decode()
    anchor = '    _glm53_moe_int4_gemv_partial[grid]('
    assert source.count(anchor) == 1
    replacement = '''    partial_kernel = _glm53_moe_int4_gemv_partial
    if (os.environ.get("STRIX_MOE_W1_EXPERT_ORDER", "0") == "1"
            and M == 4 and N == 320 and K == 2560 and top_k == 10
            and GS == 32 and BM == 16 and BN == 64 and BK == 128
            and SPLITK == 4 and num_pid_m == 40 and B.size(0) == 512
            and not mul_routed_weight
            and A.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
            and B_scale.dtype == torch.bfloat16
            and B.is_contiguous() and B_scale.is_contiguous() and B_zp.is_contiguous()):
        partial_kernel = w1_expert_kfast
    partial_kernel[grid]('''
    source = source.replace(anchor, replacement)
    kernel = Path(__file__).resolve().parents[1].joinpath('tests/moe_w1_order_kernel.py').read_text()
    source += '\n' + kernel
    compile(source, 'fused_moe.w1_order.py', 'exec')
    return source.encode()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = patched(args.base_source.read_bytes())
    with args.output.open('xb') as stream:
        stream.write(candidate)
    print(hashlib.sha256(candidate).hexdigest())
