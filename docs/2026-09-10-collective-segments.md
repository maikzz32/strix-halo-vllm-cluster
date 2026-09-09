# Local intervals around UMA collectives

The September 9 traces contain 16 M4 target graphs on each rank, each with
98 UMA exchanges. `tools/analyze_collective_segments.py` partitions the local
timeline from each preceding exchange's end to the next exchange's start.
Traced work uses the clipped union of event intervals, since profiler
timestamps can overlap. This avoids treating overlapping events as additional
elapsed time. Work after the final exchange is excluded.

| Mean milliseconds per target graph | Node 1 | Node 2 | Node 3 | Node 4 |
|---|---:|---:|---:|---:|
| Intervals before exchanges | 28.470 | 28.635 | 30.246 | 28.372 |
| Covered by traced work | 22.355 | 22.746 | 22.834 | 22.392 |
| Not covered by traced work | 6.115 | 5.888 | 7.411 | 5.980 |
| UMA exchanges, including waiting | 7.364 | 7.318 | 5.673 | 7.526 |

Node 3 spends more local time between exchanges and less inside them. This
is consistent with later arrival but is not causal evidence that Node 3 is
the bottleneck. Profiler overhead, event coverage, CPU progress, GPU scheduling
and the collective's own work have not been separated. Do not identify the
rank with the largest UMA duration as the slowest transport automatically.

These are local durations and position averages; host timestamps have not
been synchronized and ordinal position does not prove simultaneous arrivals.
No cross-rank start-time differences or unprofiled TPS are inferred. Each
input trace is identified by SHA256 in the evidence record.

The next bounded serving experiment tests automatic versus high GPU power
state selection. It can establish a performance effect but cannot by itself
explain this trace asymmetry.

Evidence: `bench/records/2026-09-10-collective-segments.json`.
