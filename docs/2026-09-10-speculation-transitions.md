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

Run `6a3f657a109a4e60a6fa3c5e0e463620` is loading on all four nodes. ShareGPT48/C1
and Hermes checks are queued after readiness. K2 acceptance statistics are
expected to differ; output text/length parity is checked separately. No K2 gain
or adaptive-policy gain is claimed. MTP3 must be restored for a fresh control
before choosing a production depth.
