# MTP acceptance transitions and current depth-cost follow-up

Three sequential 1,024-token probes used unchanged TP4/MTP3 weights, temperature0,
seed42 and default thinking. The code, German explanation and analysis prompts
come from `bench/bench_stream.py`. All token IDs match reported usage counts;
request success counters increased by exactly one and queues were idle before
and after each request. All tests are complete.

| Prompt | Nonempty SSE token groups | Draft iterations | Final accepted count consistent with counters |
| --- | ---: | ---: | ---: |
| Code | 432 | 431 | 3 |
| German | 450 | 449 | 0 |
| Analysis | 433 | 432 | 1 |

In each case the initial group is a singleton, remaining groups contain1–4 tokens,
and per-position counters reconcile exactly after allowing final output-limit
truncation. This is consistent with one group per verification step, but is not
an independent engine trace. Initial and final groups are excluded from transition
statistics. No model patch or additional GPU synchronization was used.

## Two previous weak groups

A weak group emits at most two tokens. These are descriptive observations from
K3 trajectories, not measured outcomes of an adaptive policy:

| Prompt | Observations after two weak groups | Next third-token acceptance | Idealized required step-time saving for K2 |
| --- | ---: | ---: | ---: |
| Code | 145 | 19.31% | 8.70% |
| German | 169 | 16.57% | 7.87% |
| Analysis | 150 | 19.33% | 8.58% |

The last column is `p(third accepted)/E(next output length)`: the fractional time
reduction needed to compensate for dropping the third accepted token under an
idealized same-state comparison. It assumes E2=E3-p3. Actually switching depth
changes subsequent boundaries and trajectories; these numbers do not predict
end-to-end speedup. Longer weak streaks did not consistently improve prediction
across the three prompts. The small sample and correlated observations do not
justify selecting a production policy.

`bench/analyze_speculation_groups.py` validates accounting and records all raw
counts and group sizes in `bench/records/2026-09-10-speculation-transitions.json`.
Raw token IDs and generated text are not published. Use `--prompt-file` with
`bench/speculation_stream_probe.py` to repeat a bounded request.

## Next experiment: remeasure K2 cost on the current runtime

The earlier warm K2 results in PERFORMANCE.md used a roughly49-token/s runtime,
before the current UMA and kernel improvements. They are not a current step-cost
measurement. To decide whether adaptive engineering is worthwhile, a separate
configuration changes only `num_speculative_tokens` from3 to2. Weights, original
INT4 dispatch, UMA transport, GPU high mode and graph settings remain the same.

Run `6a3f657a109a4e60a6fa3c5e0e463620` completed ShareGPT48/C1, but failed
the exact-output comparison. Only13 of48 texts are identical, with11,695 output
tokens versus11,839 for K3. The35 differing texts have a median common prefix
of158 characters (minimum8); these are character counts, not token positions.
No cause of divergence is established. No checkpoint or source file changed.

| Configuration | Output tokens/s | Mean TTFT ms | Mean TPOT ms | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Prior MTP3 control | 55.676 | 564.287 | 15.482 | 11,839 |
| MTP2 experiment | 49.249 | 570.789 | 18.113 | 11,695 |

These are observed results with different generated outputs, not a valid
same-output speedup estimate. K2 mean/median ITL were40.22/40.41ms; they do not
establish a useful step saving on identical trajectories. Acceptance62.77%
and mean length2.26 do not compensate for the observed throughput reduction.

The controller intentionally stopped at the text-parity assertion before the
Hermes check. Consequently no K2 Hermes success is claimed, and the planned
three matched streaming probes were skipped. All K2 workers were stopped.
MTP3 restoration run `aebc6c06e5a249c3a072d4f302317c9f` is ready. Its fresh
ShareGPT48/C1 control reached55.933tokens/s, mean TTFT561.396ms and TPOT15.439ms.
All48 texts, lengths and speculative statistics match the previous MTP3 control;
all Hermes checks pass. MTP3 remains active. No K2 setting is
promoted. Adaptive depth remains unproven; further work must first resolve or
characterize the output differences rather than treating the K2 timings as
valid same-output cost measurements.

## Step-interval comparison preparation

`bench/compare_speculation_streams.py` checks identical requests, output token IDs
and usage before comparing K3/K2 intervals. It invokes the group-accounting
validator at each measured depth, drops the initial and final boundary groups,
and reports both mean and median client-observed intervals. No outliers are
silently removed; the maximum and count above twice the median are retained.

The existing K3 streams have medians40.319/40.277/40.360ms for code/German/
analysis. Means are40.292/40.540/43.396ms. Counts above twice the median are1/1/10,
with maxima82.05/148.22/198.09ms. Their cause is not established, and client SSE
intervals must not be described as isolated GPU execution time. The matched MTP2 streaming comparison
was skipped because the main benchmark failed exact-output parity.


## Follow-up: optimized-path coverage confound

The [active MoE dispatch audit](2026-09-10-moe-neighbor-depths.md) found the
promoted W1 and W2 optimizations restricted to four verification rows. MTP2
therefore also loses those optimizations. Its measured result is valid for that
configuration, but cannot isolate draft-depth cost with equal optimized-path
coverage. Extending coverage requires its own parity and serving tests; it does
not explain the output differences by itself.
