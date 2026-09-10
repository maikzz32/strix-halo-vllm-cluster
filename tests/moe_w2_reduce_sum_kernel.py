"""Isolated W2 epilogue experiment; no runtime hook or weight/matmul change.

Caller must pass enable_fp_fusion=False. Splits, routing multiplication, BF16
intermediate rounding, and expert additions retain the reference order.
"""
import triton
import triton.language as tl


@triton.jit
def w2_reduce_sum(
    partial_ptr, weight_ptr, output_ptr,
    ROUTES: tl.constexpr, N: tl.constexpr,
    TOPK: tl.constexpr, SPLITK: tl.constexpr, BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < N
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for expert_slot in tl.static_range(TOPK):
        route = token * TOPK + expert_slot
        base = route.to(tl.int64) * N + columns
        partial_sum = tl.zeros((BLOCK,), dtype=tl.float32)
        for split in tl.static_range(SPLITK):
            partial_sum = partial_sum + tl.load(
                partial_ptr + split * (ROUTES * N) + base,
                mask=mask, other=0.0,
            )
        weighted = partial_sum * tl.load(weight_ptr + route)
        # This conversion replaces a real BF16 store+load. Removing it would
        # change the reference arithmetic even though it may improve precision.
        rounded = weighted.to(tl.bfloat16).to(tl.float32)
        total = total + rounded
    tl.store(output_ptr + token.to(tl.int64) * N + columns,
             total.to(tl.bfloat16), mask=mask)
