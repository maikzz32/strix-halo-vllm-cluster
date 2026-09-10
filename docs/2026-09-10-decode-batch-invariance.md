# M3/M4 row-prefix characterization after the MTP2 trial

The restored MTP3 service passed all48 text/length/speculation comparisons and
Hermes checks before this isolated test. No model source or configuration was
changed by the test. Node4 ran the installed operations on identical first-three
rows of M3 and M4 input matrices, using four synthetic weight banks per shape.

| Operation | N | K | Compared BF16 elements, eager plus changed-input graph | Unequal |
| --- | ---: | ---: | ---: | ---: |
| HC down/injection | 336 | 10240 | 8064 | 0 |
| HC down | 320 | 10240 | 7680 | 0 |
| HC up | 10240 | 320 | 245760 | 0 |
| Shared scalar gate | 1 | 2560 | 24 | 0 |
| INT4 router | 512 | 2560 | 12288 | 0 |
| INT4 shared gate/up | 320 | 2560 | 7680 | 0 |
| INT4 shared down | 2560 | 160 | 61440 | 0 |

Every fixed shape was also repeatable. Graph outputs after modifying inputs
matched the corresponding eager outputs. All compared values were finite.
The bounded controller checked idle queues and unchanged serving success
counters before and after the isolated process; it completed successfully.

The pinned HIP source suggests a dispatch difference at K10240: three BF16
activation rows fit the32768-element LDS limit, while four exceed it. Its
`WVSPLIT_TILE` and `WVSPLITK_CFG` paths select different unroll factors and
small/big kernel families. That source-level difference did not produce unequal
outputs in this sampled test. It is not evidence for the cause of the35 differing
MTP2 model responses.

This covers synthetic stateless linears, not actual model hidden states, all
possible inputs, routed MoE, attention or recurrent-state rollback. It neither
proves full batch invariance nor establishes a GDN bug. MTP2 output differences
remain unexplained; no speculative runtime patch is justified by this result.

Reproduce with `tools/run_decode_batch_invariance.py --host 192.168.1.18` after
ensuring no benchmark is active. The test is
`tests/check_decode_batch_invariance.py`; full results and the benchmark
source hash are in `bench/records/2026-09-10-decode-batch-invariance.json`.


## M4/M5 follow-up after the MTP4 trial

After the restored MTP3 control completed at 56.286 tokens/s, with all 48 outputs,
lengths and speculation counters identical to its earlier control and Hermes
checks passing, the same seven stateless operators were checked at M4 versus M5.
Four synthetic banks per shape again produced zero unequal elements for identical
first-four input rows, both eagerly and in graph replay after input changes.
Fixed-shape repeatability and unchanged serving counters also passed. This does
not explain the 36 differing responses in the MTP4 trial and does not cover
routed MoE, attention or recurrent-state rollback.

Run `tools/run_decode_batch_invariance.py --short-rows 4 --long-rows 5` to
reproduce on an idle cluster. The controller now stores the test hash as
`benchmark_source_sha256`, preserving the separately reported installed-module
`source_sha256` mapping. The older controller overwrote that mapping; do not
attribute module hashes to the older M3/M4 public record. The new full record is
`bench/records/2026-09-10-decode-batch-invariance-4-5.json`.
