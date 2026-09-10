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
