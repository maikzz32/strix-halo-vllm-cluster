# MoE output alias: no demonstrated serving improvement

The candidate removes the final WNA16 MoE output copy for the audited TP4
small-batch path. It changes the output destination, not expert computation,
quantization, BF16 rounding or reduction order. It does not enable AITER.

**Decision: do not promote on current evidence.** The isolated improvement
does not translate into a measurable gain in the complete model. The original
source has been restored on all four nodes. Run
`4ee5203a4a8e40dbb98c282e4a8cbd92` reached READY with all-rank UMA/PyNccl parity.
Its post-restoration ShareGPT control completed all 48 requests at 55.6666
tokens/s, 548.28 ms mean TTFT and 15.6497 ms mean TPOT. All answers, output
lengths and speculation statistics match the first control. Control drift is
-0.140% in throughput. The candidate's +0.064% against the stronger control
and negative fixed-stream result do not justify promotion.
[Final control](../bench/records/2026-09-09-alias-control.json),
[control comparison](../bench/records/2026-09-09-alias-controls.json).

## Isolated test

Actual `FusedMoEKernelModularImpl.apply`, E512, hidden2560, TP4 local
intermediate160, top-k10, asymmetric INT4 GS32. Synthetic weights; eight
alternating captured-graph timing samples per mode. These are warm-operator
measurements, not a cold-expert full-model forecast.

| Tokens M | Original median | Alias median | Correctness |
|---:|---:|---:|---|
| 1 | 70.510 us | 68.318 us | Bit-exact eager and changed-input graph |
| 4 | 192.018 us | 188.386 us | Bit-exact eager and changed-input graph |
| 8 | 493.120 us | 490.043 us | Bit-exact eager and changed-input graph |
| 16 | 940.513 us | 939.646 us | Unchanged fallback; timing variation |

[Every sample](../bench/records/2026-09-09-moe-output-alias-micro.json).
The initial harness attempt stopped before inference because its workspace
manager was not initialized; the corrected harness uses the actual manager.
No live serving code was changed for this isolated test.

## Full-model comparison

| Metric | Before | Candidate |
|---|---:|---:|
| Successful / failed | 48 / 0 | 48 / 0 |
| Output tokens/s | 55.7447 | 55.7804 |
| Mean TTFT | 541.77 ms | 545.37 ms |
| Mean TPOT | 15.6261 ms | 15.6357 ms |
| Generated tokens | 11,839 | 11,839 |
| Draft iterations | 4,541 | 4,541 |
| Accepted draft tokens | 7,297 | 7,297 |

All 48 answer strings, token lengths and speculative statistics match.
Throughput changes by only +0.064%, while mean TTFT and TPOT do not improve.
Six fixed 512-token streams also match exactly, with a median paired decode
change of -0.183%. This does not establish a usable performance improvement.
The before run precedes the candidate by approximately 36 minutes, including
an operator interruption and model reload; a following control is needed to
quantify drift; its result is reported above. No statistical significance is claimed.

[ShareGPT comparison](../bench/records/2026-09-09-alias-vs-before.json),
[fixed-stream comparison](../bench/records/2026-09-09-alias-stream-compare.json),
[candidate record](../bench/records/2026-09-09-qwen029-tp4-alias.json),
[all-rank activation evidence](../bench/records/2026-09-09-alias-activation.json).

[Exact source restoration](../bench/records/2026-09-09-alias-source-restore.json)
and [candidate UMA runtime checks](../bench/records/2026-09-09-alias-uma-runtime.json)
are retained separately.

## Source and rollback

Original `modular_kernel.py` SHA256:
`1e60aca6ed0dd4fcb46d577897ff1651f27a6130b3449d22265c0c791beec5d5`.
Candidate SHA256:
`319e15663934b826f72695554af4643695fccde94c8d57c34534c6980577477a`.
The patch requires the original hash and defaults off. It limits activation to
WNA16, NoDPEP finalize, BF16, M<=8, the exact TP4 tensor shapes and no LoRA.

To reproduce the isolated experiment in a compatible container with the
original source, place this repository on its filesystem and run:

```bash
python3 tests/bench_moe_output_alias.py --output /tmp/alias-result.json
```

The test replaces a method only in its own process; it does not patch the
installed module. For a model trial, `patches/moe_output_alias.py SOURCE`
previews the hash-checked patch, and `--apply` writes a SHA-named backup.
Enable `STRIX_MOE_OUTPUT_ALIAS=1` only in an explicit experiment config and
restart all four workers. Restore the verified backup after stopping the
candidate and restart with the original four-node config. Saved pre-trial
container images also remain available.
