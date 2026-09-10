# MoE optimized-path coverage at neighboring MTP depths

Inspection of the active `fused_moe.py` found that the promoted W1 expert-order
kernel is gated on M=4, and W2 serial reduction on40 routed rows. Thus both
optimizations apply to MTP3's four-row verification, but neither applies to MTP2
or MTP4's three/five-row verification. This does not establish the cause of the
MTP2 output differences. It does mean the current MTP2 test cannot isolate the
cost of one fewer draft step while retaining equivalent optimized-path coverage.

V2 uniform graph capture rounds token counts to multiples of decode query length.
Do not assume an MTP4 single-request verification is automatically padded to eight
rows just because capture sizes include powers of two. Actual dispatch must still
be verified in any future model experiment.

## W2: neighboring shapes pass isolated parity

The existing `w2_serial` kernel was used unchanged at30 and50 routed rows. Each
shape was tested with two seeds, three routing patterns,24 rotating banks per
pattern, eager output checks and graph replay after changed activations. Reference
is the installed fallback at the same shape, with serial selection disabled in
the isolated process. All checks pass. Serving counters stayed unchanged and no
owned test processes remained. No serving source or configuration changed.

Representative median complete W2 operator times, seed9560:

| Verification rows | Routing | Installed fallback us | Serial us |
| --- | --- | ---: | ---: |
| 3 | Reuse10 | 28.961 | 22.399 |
| 3 | Overlapping experts | 52.519 | 35.817 |
| 3 | Disjoint experts | 65.877 | 46.437 |
| 5 | Reuse10 | 31.773 | 23.296 |
| 5 | Overlapping experts | 69.019 | 46.671 |
| 5 | Disjoint experts | 95.492 | 71.250 |

The second seed also passes and improves each matched median. Latency decreases
are approximately23–32% across the observed cases. These are isolated operator
results, not token-throughput gains or a proof for arbitrary inputs.

Reproduce with `tools/run_moe_w2_depths.py --execute --tokens 3 --seed 9560
--output results/unique-directory` (use5 for five rows). Full samples and hashes
are in `bench/records/2026-09-10-moe-w2-neighbor-depths.json`. Test/controller retain
the existing kernel and its license attribution.

Next steps: test the W1 expert-order variant at neighboring shapes, then validate
the actual shape-restricted dispatch before staging a model trial. Preserve the
existing M4 path. MTP3 remains the active validated service; no MTP4 trial has
started and no new checkpoint is needed for this investigation.


## W1: neighboring shapes also pass isolated parity

Two seeds (9562 and 9563), three routing patterns and 24 rotating banks per
pattern pass exact eager and changed-input graph checks at three and five
verification rows. The test calls the installed wrapper with its W1 selection
flag disabled, then substitutes the unchanged expert-order partial kernel.
This tests operator output, not a proposed production dispatch guard.
Serving counters stayed unchanged and all owned test processes exited.

| Rows | Routing | Seed 9562 fallback / expert-order us | Seed 9563 fallback / expert-order us |
| --- | --- | ---: | ---: |
| 3 | Reuse10 | 38.973 / 39.217 | 39.496 / 36.412 |
| 3 | Overlap | 71.618 / 65.497 | 67.646 / 61.059 |
| 3 | Disjoint | 97.505 / 91.323 | 90.678 / 83.732 |
| 5 | Reuse10 | 39.246 / 39.575 | 36.679 / 36.409 |
| 5 | Overlap | 97.893 / 91.046 | 90.889 / 83.742 |
| 5 | Disjoint | 151.295 / 144.799 | 138.608 / 132.530 |

Overlap/disjoint cases improve by approximately 4-10%. Reuse10 is mixed,
including a small regression in the first seed. Do not describe W1 as a
universal speedup. Full samples, source hashes and service checks are in
`bench/records/2026-09-10-moe-w1-neighbor-depths.json`.
Reproduce with `tools/run_moe_w1_depths.py --execute --tokens 3 --seed 9562
--output results/unique-directory`; also test tokens 5 and seed 9563.

The next unresolved step is actual dispatch integration with opt-in neighboring
shapes, preserving the existing M4 path and all dtype/layout checks. Only after
that validation should an MTP4 model comparison be staged. These results do not
identify the cause of MTP2's different generated outputs.


## Actual dispatch validation and staged model trial

`patches/moe_neighbor_depths.py` generates an opt-in extension from installed
source SHA256 `8b0d48a769b03c62e786a880941531ebdfb1bc582d53b6e03c5e49e73b3b2493`.
Candidate SHA256 is `ab22219d75f1add289f9418fdc2158f482418e45864dc5e2771157de456874e6`.
`STRIX_MOE_NEIGHBOR_DEPTHS=1` admits W1 M3/M5 and W2 M30/M50. Existing M4/M40
selection remains active regardless of the new flag. Other guards and kernel
implementations are unchanged. W2 grid and valid-row count follow the admitted M.

The isolated test compiles the actual generated function into the installed
module globals and observes which kernel is launched. All 48 cases pass:
W1/W2, verification rows 2/3/4/5, BM16/BM32, and three routing patterns.
Flag-off, flag-on and original optimization flags disabled are checked for
correct selection and exact output. Graphs replay exactly after changed inputs.
This covers M2 and BM32 fallback as well as the existing M4 path. It does not
establish which shapes the serving runner will request.

Full dispatch evidence is in `bench/records/2026-09-10-moe-neighbor-dispatch.json`.
Generate candidate source with the generator's `generate(original_bytes)` and run
`tools/run_moe_neighbor_dispatch.py --execute --candidate-source candidate.py
--output results/unique-directory` on an idle cluster.

The candidate was staged with original-source backups and identical hashes on
all four nodes. After an idle stop, an MTP4 trial was started as run
`b51737e200f643e5b419d46174a6f2bc`. Only draft depth (3 to 4), the new flag, and
this source extension differ from the MTP3 control. Startup and serving comparison
are pending. This is an experiment, not a promoted configuration or TPS result.
The model checkpoint and arithmetic remain unchanged.
