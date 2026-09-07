# W2 reduction and expert-sum fusion

An isolated Triton fusion reduces the tested W2 epilogue from approximately 14.6–15.1 us to 11.4–11.8 us. No runtime hook or serving configuration was changed. The complete routed W2 matrix multiply is outside this measurement.

The existing path reduces five FP32 partials in order, multiplies the FP32 routing weight, casts each expert result to BF16, then sums ten experts in FP32 and casts to BF16. The candidate preserves these steps and their order, including the BF16 intermediate, with floating-point fusion explicitly disabled. It eliminates the intermediate tensor store/load and one launch. The reference calls the actually installed `_glm53_moe_int4_gemv_reduce` and `ops.moe_sum`; the installed fused-MoE Python source is hash-guarded.

| Synthetic partials | Installed two kernels (us) | Fused block256 (us) |
|---|---:|---:|
| Normal | 14.606 | 11.598 |
| Exact cancellation pairs | 15.074 | 11.845 |
| Exponent-scaled values | 14.630 | 11.440 |

Twenty-four independent buffers per case contain about 47 MiB of partials. Each method uses five samples of twenty graph replays, with 24 operations per replay. Blocks64/128/256 all produced zero BF16 bit mismatches for 72 synthetic inputs, including replay after input changes. These tests are finite synthetic cases, not actual model activation traces or proof for every floating-point value. They do not measure cache misses.

At 48 target MoE blocks, the approximately 3 us reduction suggests only 0.15 ms per target round if maintained in context. This is an estimate, not a full-model throughput result. The common-path candidate has no expert-map/padded-expert support and must not be enabled indiscriminately. A production integration would also need to return the final `[M,N]` result through the existing caller without duplicating the expert sum, and validate real model results.

All samples, source hashes, terminal process evidence and unchanged idle-service counters are in the [record](../bench/records/2026-09-07-moe-w2-epilogue.json). TP4/UMA remains active. The bounded driver defaults to dry run:

```sh
python tools/run_moe_w2_epilogue.py --output NEW_DIRECTORY
# Add --execute for the isolated Node18 test in the existing patched container.
```
