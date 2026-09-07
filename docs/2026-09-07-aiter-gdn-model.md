# Packed GDN model trial

The optional packed GDN candidate shows a small positive result, but is not promoted because control-run drift is larger than its throughput gain against the stronger control. The original GDN source is restored and serving on all four nodes.

| Run | Output tokens/s | Mean TPOT (ms) |
|---|---:|---:|
| Before, original GDN | 53.9141 | 16.4228 |
| Packed GDN candidate | 54.4418 | 16.2195 |
| After, original GDN | 53.2594 | 16.6999 |

Each run completes ShareGPT48/C1 with zero failures, 14767 input tokens and 11839 output tokens. All 48 texts, input/output lengths and speculation summaries match between candidate and both controls. Each completed-request counter advances by exactly 48. No extra warmup or post-hoc prompt exclusion was added.

Against the stronger control the candidate improves overall throughput by 0.979% and decode speed by 1.253%. Against the after control these are 2.220% and 2.962%. Control throughput drift is -1.214%, so this single bracket does not establish a robust sub-percent overall gain. Six matched candidate/after streams have identical content, reasoning, usage and finish reason, with median decode gain 1.504%; these are not a substitute for the ShareGPT success criterion.

This trial retains W2 serial and the AVX2 UMA library. It does not include the separate AVX512 communication optimization. All four candidate and restored runtime source/helper hashes, run IDs and environment flags have been verified. An initial cache search looked in default locations and found no files. Inspecting the worker environment identified `/triton-cache`; that directory contains the packed GDN compilation artifact created during the candidate run. This proves compilation, not a direct GPU execution trace.

The candidate run was stopped cleanly. Original source `cb819264c1e791177d866f484ca0579a7d57141926705a1f75dd4a79de4d0125` is restored on all four nodes. Canonical run `e27143e51a624f96a06921e49bf90345` is healthy and idle after the after-control benchmark and six streams, with 54 completed requests. The opt-in candidate remains staged for future experiments.

The 60 output token/s goal remains unmet. See [full record](../bench/records/2026-09-07-aiter-gdn-model.json).
