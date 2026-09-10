"""Extend measured MoE paths to neighboring verification depths, opt-in only."""
import hashlib
BASE_SHA256 = "8b0d48a769b03c62e786a880941531ebdfb1bc582d53b6e03c5e49e73b3b2493"

def generate(source: bytes) -> str:
    assert hashlib.sha256(source).hexdigest() == BASE_SHA256, "unexpected installed source"
    text = source.decode()
    replacements = {
        "and M == 40 and top_k == 1": 'and (M == 40 or (os.environ.get("STRIX_MOE_NEIGHBOR_DEPTHS", "0") == "1" and M in (30, 50))) and top_k == 1',
        "and topk_weights.numel() == 40": "and topk_weights.numel() == M",
        "w2_serial[(40, 80)](": "w2_serial[(M, 80)](",
        "expert_ids, num_tokens_post_padded, 40, 160,": "expert_ids, num_tokens_post_padded, M, 160,",
        "and M == 4 and N == 320": 'and (M == 4 or (os.environ.get("STRIX_MOE_NEIGHBOR_DEPTHS", "0") == "1" and M in (3, 5))) and N == 320',
        "and SPLITK == 4 and num_pid_m == 40": "and SPLITK == 4 and num_pid_m == M * 10",
    }
    for old, new in replacements.items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    compile(text, "<moe-neighbor-depths>", "exec")
    return text
