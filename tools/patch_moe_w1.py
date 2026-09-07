"""Hash-guarded opt-in W1 tile patch; no weight or arithmetic changes."""
import hashlib
BASE_SHA='4f460bcc4c6c074dc8c08c3881bccdbdbd941490821bac458c9782c1404a0070'
ANCHOR='        SPLITK, BN, BK, num_warps, num_stages = 4, 64, 128, 4, 2'
ADDITION="""
        # Validated only for the TP4 MTP3 W1 verification shape.
        if (os.environ.get("STRIX_MOE_W1_BN32", "0") == "1"
                and M == 4 and top_k == 10 and N == 320 and K == 2560
                and GS == 32 and config["BLOCK_SIZE_M"] == 16):
            BN, num_warps = 32, 2
"""
def patched(base):
 if hashlib.sha256(base).hexdigest()!=BASE_SHA:raise ValueError('Unexpected fused MoE source')
 text=base.decode('utf-8')
 if text.count(ANCHOR)!=1:raise ValueError('Nonunique W1 geometry anchor')
 text=text.replace(ANCHOR,ANCHOR+ADDITION,1)
 compile(text,'fused_moe.w1_candidate.py','exec')
 return text.encode('utf-8')
