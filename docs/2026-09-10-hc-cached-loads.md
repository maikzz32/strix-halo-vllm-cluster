# HC ordinary weight loads: rejected

A separate native library replaces the non-temporal weight loads with ordinary
vector loads while retaining arithmetic, layout and reduction order. Generated
gfx1151 assembly confirms `global_load_b128` without the reference's `slc dlc`
modifiers. The library was not installed in serving.

Twenty-four rotating synthetic matrices per shape passed bitwise eager and
changed-input graph checks. Six alternating timing rounds gave:

| Shape / matched variant | Reference, us | Ordinary loads, us |
|---|---:|---:|
| M4/N336/K10240, U4/grid40 | 40.918 | 41.315 |
| M4/N320/K10240, U4/grid30 | 40.034 | 40.785 |
| M4/N10240/K320, U2/grid20 | 31.520 | 32.334 |

No tested ordinary-load variant improved its matched reference. Keep the
existing load policy. This isolates an operator workload, not full-model cache
behavior; no serving speedup is claimed.

Binary SHA256: `6e13f4d413bbbb48fb838928931d530a81256108caea288ed016b773ccd73216`.
Test: `tests/bench_hc_cached_loads.py`. Record:
`bench/records/2026-09-10-hc-cached-loads.json`.
