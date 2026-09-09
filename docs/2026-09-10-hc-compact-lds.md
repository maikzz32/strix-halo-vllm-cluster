# HC-up compact LDS: no additional speedup

The native HC-up kernel reserves 65536 bytes of LDS for an M4/K320 input
requiring only 2560 bytes. Shape-restricted variants reduce that allocation
without changing weight layout, dot-product order or reduction arithmetic.
Compiler metadata confirms 2560 bytes and no private-memory spill allocation.
The U2 variant also uses 106 VGPRs versus 122 in its full-LDS counterpart.

Tests ran on Node 4 after the serving control completed, with 24 rotating
synthetic BF16 weight matrices, changed-input graph parity, six alternating
timing rounds and 20 replays per round. All completed variants matched bitwise.

| Variant / grid | Median microseconds |
|---|---:|
| Installed U4 | 34.179 |
| Original U2 / 20 | 31.710 |
| Compact LDS U2 / 20 | 31.776 |
| Compact LDS U2, 8 waves / 80 | 31.839 |

A separate U1 control confirmed the same conclusion: original U2 measured
31.715 versus compact U2 31.689 microseconds. Original U1/grid40 measured
31.950; compact U1/grid40 measured 31.997. Smaller LDS and workgroups did
not produce a meaningful improvement over the already-tested U2 variant.
These variants are not deployed. The theoretical occupancy motivation did
not translate into a measured benefit; this does not establish the limiting
hardware resource without further counters.

An initial 8-wave launch failed on the CPU: the reused `mindiv` helper loops
over 13 divisors and divides by zero when given 8. Kernel logs identified
integer divide traps in the test library. The corrected variant uses a fixed
8-wave count. This was not a GPU memory fault. The serving tool checks passed
after the failed isolated attempts. R3 binary SHA256:
`cd31f096a2160847b01d282fff92e60854658a7242ee3080372b93d3d53d15dd`;
R4 adding U1 controls:
`762a857dafe81c27ea19aac79f54d8f4aaed2e9aa02374a293c08e2a0586d7af`.

Tests: `tests/bench_hc_compact_lds.py` and `tests/bench_hc_compact_lds_u1.py`.
Records under `bench/records/`: `2026-09-10-hc-compact-lds.json`,
`2026-09-10-hc-compact-lds-u1.json`, and `2026-09-10-hc-compact-resources.json`.
Resource metadata is from R2, before the host-helper correction and added U1
variants; it is not presented as R4 binary metadata.
