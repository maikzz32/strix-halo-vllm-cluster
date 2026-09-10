# HC-down weight prefetch: isolated result

A shape-restricted experiment preloads the next weight tile before the current
tile's dot products. It retains wave32, YTILE=1, UNRL=4, grid20, the activation
staging logic, and the accumulation/reduction order from the pinned vLLM source.
No runtime files or model weights are replaced.

Node 4 passes BF16 bitwise comparison against the installed kernel and an
isolated original, including changed-input HIP graph replay. Timing uses 24
rotating synthetic matrices per shape, six alternating method-order rounds,
and 20 graph replays per round. Serving counters remain unchanged.

| Shape (batch, output, K) | Installed, us | Isolated original, us | Prefetch, us |
| --- | ---: | ---: | ---: |
| 4, 336, 10240 | 43.0813 | 42.9986 | 41.7933 |
| 4, 320, 10240 | 49.3052 | 49.4514 | 45.5173 |

The candidate uses 138 VGPRs versus 132 for the isolated original. Both use
44 SGPRs, 64 KiB LDS and report zero private memory and zero register spills.
Metadata and timings refer to the same compiled library SHA256
`06c4be4e14ab4197f0865ebe918adaa746096afaad96b823f62de4f54b5754e8`.

The frequent merged projection improves by about 3.0%; the final projection
by about 7.7%. Applying these isolated deltas to 96 merged calls plus one final
call suggests only 0.127 ms per target graph. That is an estimate, not measured
model performance. The experiment remains isolated and does not meet the
60 tokens/s serving objective.

Before this test, the restored original TP4/MTP3 control completed at 56.2244
output tokens/s, 550.5388 ms mean TTFT and 15.3972 ms mean TPOT. All 48 responses,
lengths and speculation counters match the earlier original reference; all
Hermes checks pass. The original runtime remains active.

Code: `tools/build_hc_down_prefetch.py`, `tests/bench_hc_down_prefetch.py`, and
`tools/run_hc_down_prefetch.py`. Records: `2026-09-10-hc-down-prefetch*.json`
and `2026-09-10-hc-pinned-restored-*.json` under `bench/records/`.
