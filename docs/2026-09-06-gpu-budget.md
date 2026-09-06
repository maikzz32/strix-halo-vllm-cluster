# Qwen TP4 M4: measured GPU budget

The four valid model traces show a **40.10–40.16 ms target graph** inside a **48.850–48.856 ms complete decoding iteration**. The largest costs are RCCL, routed MoE, and the replicated BF16 hyper-connections. The evidence does not support a remaining single small kernel change yielding another 20% overall speedup.

Run: root-managed `e65006cd151e45bca5dbc6b82dfcf63b`, `config/qsa-torch-queuefix.json`, 2026-09-06. All four original compressed traces are under `results/torch-queuefix-node{15,16,17,18}-20260906/`. The complete workload returned 2048 tokens and DONE; start/stop profiling succeeded. This report analyzes CPU and GPU timestamps locally and does not run inference. The QSA invisible-tile skip is enabled; the optional local draft argmax reduction optimization is disabled. Ordinary greedy argmax sampling still appears in the trace.

## Completeness and attribution

All four traces contain exactly 16 complete target graphs, 16 complete small graphs, and 16 complete decoding iterations. Each target graph contains **2694 GPU dispatches**; each complete iteration contains **3151**: 2694 target, 112 small-graph and 345 eager/metadata/sampling dispatches. All dispatch durations are positive. An additional 1896–2458 kernels per rank fall outside the recorded CPU steps, consistent with asynchronous capture boundaries; these are excluded from the iteration budget.

Target identity is established by the explicit CPU annotation `execute_context_0(0)_generation_1(4)`: the CPU `hipGraphLaunch` lies within that annotation on the same process/thread. Its correlation ID then identifies the asynchronous GPU kernels. GPU timestamps are never cut at CPU annotation boundaries. The same method joins every other GPU dispatch to its CPU HIP API and ProfilerStep.

Grid/block dimensions are absent. W1/W2 attribution instead requires, on **every one of the 64 target graphs**, exactly 48 adjacent sequences:

```text
_glm53_moe_int4_gemv_partial
_glm53_moe_int4_gemv_reduce
act_and_mul_kernel ... silu
_glm53_moe_int4_gemv_partial
_glm53_moe_int4_gemv_reduce
moe_sum
```

This sequence matches the independently verified runtime source order: routed W1, split reduction, SiLU, routed W2, split reduction, expert sum. HC attribution separately requires 97 BF16 down → HC-SiLU → BF16 up → HC-gate sequences in every target graph. All checks pass. The target also has 98 RCCL calls and 12 QSA scoring calls.

## Per-target budget

Times below are means of 16 graphs on each rank. The trace contains no overlapping target kernel intervals, so the summed GPU durations also equal their interval unions in this capture.

| GPU time per target | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---:|---:|---:|---:|
| Entire target span | 40.096 ms | 40.156 ms | 40.151 ms | 40.141 ms |
| RCCL, including spin/wait | 8.794 | 8.734 | 8.733 | 9.130 |
| Routed W1 partial + reduction | 4.618 | 4.625 | 4.628 | 4.600 |
| Routed W2 partial + reduction | 3.342 | 3.720 | 3.621 | 3.324 |
| HC BF16 down + up | 7.150 | 7.106 | 7.093 | 7.112 |
| HC elementwise operations | 0.669 | 0.688 | 0.677 | 0.689 |
| Other dense/router/shared INT4 linears | 3.617 | 3.583 | 3.586 | 3.596 |
| Named QSA attention + indexing | 0.803 | 0.776 | 0.805 | 0.789 |
| GDN recurrence + convolution | 0.675 | 0.667 | 0.675 | 0.674 |
| Routed top-k + expert alignment | 0.540 | 0.521 | 0.533 | 0.529 |
| Intervals without a traced kernel | 6.560 | 6.424 | 6.498 | 6.438 |

The remaining roughly 3.3 ms consists of other BF16 linears, routed activation/expert sums, copies, reductions and other elementwise kernels; all detailed groups are retained in [the compact measured budget](../bench/records/2026-09-06-gpu-budget.json).

W1 is the largest individual compute kernel family: partial alone averages 94.1–94.7 µs per layer. W2 partial is 65.4–73.7 µs per layer, plus about 3.8 µs reduction. HC down/up average about 40.4–40.8 / 32.6–32.9 µs, respectively. This confirms the real-model HC bandwidth cost without extrapolating a warm-cache microbenchmark.

The target mean span differs by only 0.060 ms across ranks (<0.15%); full-iteration mean spans differ by less than 0.006 ms. W2 compute differs between ranks, with larger RCCL waiting partly compensating on the faster ranks. These durations do not establish absolute cross-host start skew because host clocks were not synchronized for that purpose.

## What the uncovered 6.4–6.6 ms contains

There are exactly **2693 positive gaps per target graph**, or 43,088 gaps across the 16 graphs on each rank. The median gap is **2.16–2.20 µs**, p90 **2.72–2.76 µs**, maximum **59–65 µs**. The largest gaps appear near the initial embedding/fill-to-RCCL transition.

| Gap range | Fraction of gaps, across ranks | Fraction of gap time, across ranks |
|---|---:|---:|
| 0–1 µs | 0–0.012% | 0–0.004% |
| 1–5 µs | 96.53–97.10% | 90.85–92.35% |
| 5–10 µs | 2.73–3.31% | 6.46–7.76% |
| 10–50 µs | 0.144–0.167% | 1.06–1.20% |
| 50–100 µs | 0.002–0.012% | 0.06–0.27% |
| >100 µs | 0% | 0% |

Of the total gaps, 2498 per target are between kernels on the same stream, consuming **5.45–5.56 ms**. Another 195 stream transitions consume **0.97–1.00 ms**. These are gaps across the merged, nonoverlapping GPU timeline; time occupied by another stream's kernel is not counted as an empty interval.

Thus the shape points to many small graph dispatch/dependency intervals and potential operator fusion, rather than a small set of long stalls. It does **not** establish 6.5 ms of recoverable idle: profiling changes queue handling, and the trace does not directly measure scheduler/dependency mechanics. The SDK explicitly warns that the profiling fallback uses system-memory intercept queues and does not preserve queue priority/CU masks.

## Remaining iteration work and practical priorities

The complete iteration's mean GPU kernel sum is 40.46–40.67 ms within a 48.85 ms span. The small graph contributes 2.19–2.20 ms of kernels and has a 2.47–2.48 ms span; it contains one QSA block, three HC pairs, six RCCL calls, an LM head and argmax. Its confirmed identity is **draft step 0: M4 model forward, then a last-valid-token gather and M1 LM head/sampling**. Draft steps 1 and 2 run M1 model forwards eagerly. The remaining 345 eager/metadata/sampling kernels, which include those two steps and other work, contribute 4.74–4.76 ms. They must not be omitted when estimating overall TPS.

This mapping comes from the active V2 runner and a source audit: seven relevant active worker/speculator files have SHA256 hashes identical to the local snapshot. `AutoRegressiveSpeculator.propose` preserves the target's four padded rows for draft prefill; `_prefill` calls `_run_model(num_tokens)` before gathering `last_token_indices` and sampling one request. Subsequent positions 1 and 2 dispatch `num_tokens=num_reqs=1`. Here “draft prefill” is the first draft pass of every speculative decode iteration, not a fresh prompt prefill. The source audit covers `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py:241–317,349–358,462–480`; the active autoregressive speculator SHA256 is `c7fb6ae82a6a9cc79ad88b007e73f773102c41c14e619e12596925f2c252aca3`.

1. **RCCL and graph scheduling:** roughly 8.9 ms of collective execution/wait plus 6.5 ms of small dependency gaps makes this the largest combined region to investigate. Existing channel/protocol tests and a real unprofiled A/B decide whether any improvement is usable. Do not treat all RCCL time as network transfer or all gaps as removable launch overhead.
2. **Fusion across repeated elementwise/reduction stages:** the 2694-dispatch target explains why small local launches matter. Fusion candidates should be selected by verified adjacent source sequences and validated for the original BF16 rounding. Eliminating a few launches per layer yields fractions of a millisecond, not an automatic 20% gain.
3. **Keep W1/W2 and HC expectations bounded:** W1's measured ~94 µs matches the cold-weight microbenchmark rather than exposing a hidden fallback. HC remains ~7.1 ms; earlier cold-weight alternatives offer limited measured improvement. W2 can matter if a faster kernel works across realistic expert distributions, but conditional hot-cache wins cannot be promoted from this trace alone.

A 20% increase in iteration throughput would require removing approximately **8.14 ms** from the measured 48.85 ms iteration. Even eliminating the entire HC matrix cost would fall short, and neither HC nor MoE can actually be removed. Several substantial compatible changes would be necessary; the evidence does not justify promising that gain under unchanged model arithmetic and workload.

Analysis code: [summarize_qwen_target_trace.py](../tools/summarize_qwen_target_trace.py). It aborts on incomplete graphs or mismatched MoE/HC sequences. The [compact budget record](../bench/records/2026-09-06-gpu-budget.json) includes trace SHA256 hashes, completeness checks, per-rank budgets and gap histograms. Original traces and generic `gpu-summary.json` files are retained in the analysis workspace, outside this repository. Pass those four trace directories as positional arguments and `--output <report.json>` to reproduce the analysis.

Related documentation: [repository README](../README.md), [profiler setup and limitations](2026-09-06-profiler.md), [unprofiled single-stream results](2026-09-06-single-stream.md).
