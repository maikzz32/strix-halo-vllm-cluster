"""Hash-guarded opt-in W2 fusion, restricted to the tested TP4 MTP3 shape."""
import hashlib
from pathlib import Path
BASE_SHA='4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070'
ANCHOR='    BM = config["BLOCK_SIZE_M"]\n    EM = min(sorted_token_ids.size(0), num_valid * BM)'
ADDITION='''    if (os.environ.get("STRIX_MOE_W2_SERIAL", "0") == "1"
            and M == 40 and top_k == 1 and N == 2560 and K == 160
            and GS == 32 and BM == 16 and B.size(0) == 512
            and mul_routed_weight and topk_weights is not None
            and topk_weights.numel() == 40 and topk_weights.is_contiguous()
            and topk_weights.dtype == torch.float32
            and A.dtype == torch.bfloat16 and C.dtype == torch.bfloat16
            and B.is_contiguous() and B_scale.is_contiguous() and B_zp.is_contiguous()
            and tuple(B.shape) == (512, 2560, 80)
            and tuple(B_scale.shape) == (512, 2560, 5)
            and tuple(B_zp.shape) == (512, 1280, 5)
            and B_scale.dtype == torch.bfloat16 and B_zp.dtype == torch.uint8):
        w2_serial[(40, 80)](
            A, B, B_scale, B_zp, C, topk_weights, sorted_token_ids,
            expert_ids, num_tokens_post_padded, 40, 160,
            N=2560, K=160, GS=32, TOPK=1, BM=16, BN=32, BK=32,
            SPLITK=5, num_warps=2, num_stages=1, enable_fp_fusion=False)
        return True
'''

def patched(base):
 if hashlib.sha256(base).hexdigest()!=BASE_SHA:raise ValueError('Unexpected fused MoE source')
 text=base.decode('utf-8')
 if text.count(ANCHOR)!=1:raise ValueError('Nonunique W2 insertion anchor')
 kernel=Path(__file__).resolve().parents[1].joinpath('tests/moe_w2_serial_kernel.py').read_text(encoding='utf-8')
 if hashlib.sha256(kernel.encode()).hexdigest()!='b170f05e08df619d435c5c5592573103c53b060b8bbb53aace6a50623610a1f1':raise ValueError('Unexpected W2 kernel')
 text=text.replace(ANCHOR,ANCHOR.split('\n')[0]+'\n'+ADDITION+'\n'+ANCHOR.split('\n')[1],1)
 text+='\n'+kernel
 compile(text,'fused_moe.w2_candidate.py','exec')
 return text.encode('utf-8')
