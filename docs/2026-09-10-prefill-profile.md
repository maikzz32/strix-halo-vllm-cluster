# Four-rank prefill profile

Diagnostic run `8454376332f9468589c57f7b96c3f592` used the existing TP4/MTP3
weights and kernels. A 512-token prompt and 32-token completion were executed
without profiling, then repeated after `/start_profile` completed. Outputs and
usage match. The profiler stopped successfully and one compressed trace was
collected per rank with its SHA256. Profiling has zero warmup and a four-step
limit; this is not an unprofiled serving performance result.

All ranks contain a 504-token prefill, an 8-token prefill, and two M4 decode
scopes. The attribution tool matches runtime API calls to CPU scope/thread
containment, then selects GPU kernels by correlation ID. It checks disjoint
correlation sets, exact expected kernel counts and 98 RCCL kernels per prefill
scope. Host clocks are not compared across machines.

| Rank | 504-token GPU span, ms | RCCL kernel sum including waits, ms | 8-token RCCL sum, ms |
| --- | ---: | ---: | ---: |
| 0 | 544.782 | 171.705 | 12.642 |
| 1 | 540.532 | 163.983 | 12.846 |
| 2 | 544.167 | 152.428 | 11.557 |
| 3 | 544.532 | 174.613 | 12.884 |

On rank 0 the large prefill also sums to 131.129 ms in routed MoE GEMMs,
66.205 ms in the sparse QSA attention kernel, 17.959 ms in its MQA indexer,
and 53.718 ms in the dense INT4 kernel. These sums are not an additive estimate
of achievable savings. The record includes union times and uncovered intervals
as well as names/counts for further attribution.

The installed UMA wrapper explicitly requires `tensor.numel() == 10240` and
BF16. The native transport fixes N=10240 and BYTES=20480, the arena layout is
fixed, and numerical validation covers that exact size. Larger prefill payloads
therefore use RCCL. Removing only the Python guard would violate the native
contract. Extending the backend requires capacity/layout changes, collective
agreement on message size, bounded synchronization, and comparison against
the actual RCCL reduction at each size. The small-message rank/chunk reduction
order cannot be assumed valid for larger RCCL messages.

The next transport investigation should first measure actual prefill payload
sizes and matched isolated RCCL latency. Large payloads impose bandwidth costs;
the 20-us CPU probe and 53.93-us small UMA collective cannot be extrapolated to
these tensors. The 504+8 split also merits source inspection, but is not itself
proof of wasted work.

The diagnostic was stopped cleanly. Normal TP4 restoration is in progress as
`cb44c293b28746e69faad3890f96798e`; no new serving TPS is claimed.

Evidence: `bench/records/2026-09-10-prefill-attribution.json` and
`2026-09-10-prefill-profile-parity.json`; tools `capture_prefill_profile.py`,
`collect_prefill_profile.py`, and `summarize_prefill_trace.py`. The capture and
collection tools currently encode this diagnostic's hosts and output directory.
Raw response text remains local.

Restoration completed as `cb44c293b28746e69faad3890f96798e`. All four original
worker source hashes and UMA library mappings passed verification. Hermes auto
selection, tool-result roundtrip, streaming and default-thinking checks pass.
Evidence: `2026-09-10-prefill-restored-{workers,tools}.json`. No new serving
throughput is claimed for this restoration.
