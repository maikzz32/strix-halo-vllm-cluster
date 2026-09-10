# Bounded per-peer RDMA timing instrumentation

The generated diagnostic transport records local post, completion and reduction
times without changing payloads or the existing AVX512/BF16 reduction order.
Both instrumented and capture-disabled isolated four-rank tests passed the
32-bank eager and changed-input graph parity checks against active PyNccl.
The original serving library is preserved.

## What the records mean

Each 128-byte row records a collective sequence, entry time, completion of WR
posting, readiness of all sends/receives, completion of reduction, per-peer CQ
observation times and poll count. Each nonempty CQ batch gets one timestamp;
peers observed in the same batch remain tied. These are software observations,
not hardware packet-arrival timestamps. A receive can be observed during the
previous collective because the transport permits one sequence of lookahead.
Pre-arm observations with no timestamp remain unknown, not time zero.

The mapped file holds at most 2048 rows and never wraps. Records become visible
through a release-published count after completion. With requested count zero,
capture is disabled. A separate reader can arm an unused map once; it refuses
to reset published data. No file I/O, allocation or logging occurs in the
collective loop. Diagnostic mapping and files are created only if
`STRIX_RDMA_TRACE_DIR` is set; the unmodified production source is not replaced.

## Isolated validation

The initial 32-replay run filled its buffer before all timing samples finished,
so its mixed enabled/disabled samples are excluded from overhead comparison.
The matched follow-up used 8 replays, 3 samples and 32 operations per graph.
It captured all 962 collectives per rank, below the 2048-record capacity.
Disabled controls produced zero records. All temporary test processes exited,
and the serving request counters remained unchanged during the test windows.

Median complete HIP-event times were approximately 48.0–48.3 microseconds with
capture and 48.2–48.9 without. This short pair shows no measurable added cost;
it does not prove zero overhead or that instrumentation speeds up the system.
Captured CPU phase medians were about 0.4 microseconds posting, 35.5–35.6
waiting for completions and 3.7–3.8 for reduction/fence work. They are isolated
probe measurements and cannot identify a slow rank during model inference.

```sh
python tools/run_hip_rdma_graph.py --backend-test --uma-backend \
  --avx512-reduce --uma-threads 1024 --mode ring4 \
  --containers qwen029-tp2 qwen029-tp2 qwen029-tp4 qwen029-tp4 \
  --replays 8 --trace-count 2048 --execute --output results/trace-enabled
python tools/collect_rdma_trace.py results/trace-enabled --output results/trace-summary.json
```

Use `--trace-count 0` and a fresh output directory for the capture-disabled
control. `build_rdma_trace.py` checks each insertion site before generating a
separate source. `read_rdma_trace.py` validates the binary ABI, sequence and
timestamp bounds; its one-shot arm guard was also checked on a copied fixture.

The validated sources were staged separately on all nodes at
`/opt/strix-halo-next/rdma-trace-20260910-r1`, with identical library SHA256
`64c3190fc0392ae8b7d4fc04dd860ffedb962e3e20c1009fc22ac50ebc7a7b06`.
Model diagnostic run `a9d64229841044858529ff7413150882` started with capture
disabled and passed the subsequent Hermes tool checks.

## Model capture and its limitation

An untraced and traced 2048-token response matched in request, content,
reasoning and usage. All ranks produced 2048 valid records. However, Node 3's
arming command completed late: its sequence window does not intersect the
other three. The three remaining ranks share 1946 sequences. The controller
correctly rejected the attempted four-rank comparison; do not treat this as
a synchronized four-rank capture or infer that Node 3 is the slowest sender.
For the next fresh map, arm all ranks before submitting the request, rather
than launching arming commands after the first streamed token.

Local median CPU transport times were approximately 45.9–46.3 microseconds,
including about 4.9–5.0 posting WRs. This differs from the isolated probe and
does not establish a cause. On receiver rank 0, within the shared three-rank
window, the latest observed receive preceded the latest send completion by a
median 10.31 microseconds. CQ batching remains a limitation; this observation
motivates overlapping reduction with outstanding sends, while still waiting
for every send before returning. Summary:
`bench/records/2026-09-10-rdma-model-summary.json`.

The subsequent diagnostic ShareGPT48/C1 run measured 54.82 output tokens/s,
545.41 ms mean TTFT and 15.96 ms mean TPOT. It is not a promoted runtime and
is not an uninstrumented control for the early-reduction candidate. The
diagnostic run was stopped before starting that candidate. This also means
the short isolated overhead comparison must not be used to claim zero
model-level instrumentation cost. Record:
`bench/records/2026-09-10-rdma-diagnostic-serving.json`.

Evidence: `bench/records/2026-09-10-rdma-trace-{enabled,disabled,stage}.json`.

## Four-rank capture with arming before request submission

The follow-up captured the same 2,048 collective sequences on all four ranks
(73193 through 75240). Each map was armed before request submission and read
back independently: requested=2048, published=0, and distinct ranks 0 through 3.
The two 2,048-token responses match in request, content, reasoning and usage.
Hermes auto selection, roundtrip, streaming and default thinking checks pass.

| Rank | Post, us | Wait, us | Reduction/fence, us | Total, us |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.940 | 35.929 | 4.130 | 44.888 |
| 1 | 4.920 | 36.419 | 3.780 | 45.188 |
| 2 | 4.930 | 35.979 | 3.750 | 44.664 |
| 3 | 4.929 | 35.159 | 3.940 | 44.074 |

These are medians of CPU-side transport observations; component medians do
not have to sum to the median total. They exclude GPU staging and scheduling.
Rank 0 was the sole last-observed receive in 1,230/2,048 observations on rank 2
and 1,242/2,048 on rank 3. This is not proof of a slow sender: receiver-side CQ
polling and batching also affect the observations, and host clocks are not
aligned. No uniquely slow node is established.

Median last-send minus last-receive observation gaps were 9.16, 1.39, 1.185,
and 0.0 microseconds for ranks 0 through 3. This supports the already examined
early-reduction motivation; that implementation's matched serving result
remains the separate small, unproven improvement documented in the overlap
report. It is not a new performance result.

`tools/capture_model_rdma_prearmed.py` accepts explicit `--run-id`, `--config`
and `--output` arguments. It requires the exact validated diagnostic library
and refuses reused maps or a four-rank sequence intersection below 512 rows.
The completed run was `c653325fc1e64e419405c8440c5ba471`. Records are
`2026-09-10-rdma-model-prearmed-{summary,verified,tools}.json`.
The diagnostic run has been stopped; normal transport restoration is underway.
