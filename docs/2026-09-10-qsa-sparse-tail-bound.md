# Bound sparse QSA attention to the occupied selection prefix

The first empty-tile candidate regressed fully occupied prefill by 12–14%.
This replacement scans selection indices once per program, finds the last
nonnegative entry, and caps the existing split's loop end at its tile boundary.
It keeps the original split start, split count, tile geometry and populated-tile
arithmetic. Negative holes within the occupied prefix still take the original
path. No host readback or graph-specific fixed context length is introduced.

The scan is conservative: nonnegative indices that point to invalid pages can
extend the bound unnecessarily, but cannot exclude a valid entry. The existing
kernel still validates pages and requests. Only trailing tiles whose indices
are all negative are removed. Finite synthetic parity tests do not constitute
a proof for arbitrary NaN/infinity inputs.

## Initial screening

| Rows | Selection | Original (us) | Bounded loop (us) |
| --- | --- | ---: | ---: |
| 4 | Causal | 38.13 | 20.64 |
| 8 | Causal | 68.43 | 35.41 |
| 504 | Causal | 4902.52 | 878.45 |
| 504 | Full | 5248.82 | 5271.37 |
| 1024 | Causal | 9966.88 | 2875.91 |
| 1024 | Full | 10526.72 | 10472.04 |

All 20 initial cases match bit for bit. Changed-input graph replay matches
candidate eager execution. These are kernel-level timings, not model TPS.

## Expanded validation

The second run adds two requests with independently permuted page tables,
noncontiguous K/V views from a packed allocation, and three patterns generated
by the installed `expand_qsa_block_indices_cuda`: short positions, mixed short
and long requests, and positions spanning the 2048-token selection boundary.
Selected compressed blocks are synthetic random permutations; the full scoring
and top-k pipeline is not executed by this test.

All 32 cases pass exact output comparison and changed-input HIP graph checks.
Both runs alternate timing order over six samples of eight replays and verify
that serving request counters remain unchanged. The installed QSA source hash
was independently rechecked before testing. No serving source was changed.

The bounded-loop candidate is suitable for the next integration validation and
matched TP4 serving trial; it has not yet been deployed or shown a serving gain.
Evidence records: `bench/records/2026-09-10-qsa-sparse-tail-r1.json` and
`bench/records/2026-09-10-qsa-sparse-tail-r2.json`.
The current test is `tests/bench_qsa_sparse_tail_bound.py`; it includes the
expanded second-run coverage, while both records retain the candidate hash.
