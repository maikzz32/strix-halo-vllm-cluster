# Packed GDN model trial (in progress)

The first candidate run reaches 54.4418 output token/s versus 53.9141 for the immediately preceding control, a 0.98% improvement. Mean TPOT is 16.2195 ms versus 16.4228 ms. Both runs complete all 48 ShareGPT prompts with zero failures, 11839 generated tokens, and identical output texts, lengths and speculation summary fields. Completed-request counters advance by exactly 48 in both runs.

The candidate is the optional GDN patch on top of W2 serial and the original AVX2 UMA library. This is not the separate W2+AVX512 combination. All four runtime source/helper hashes, run IDs and environment flags have been verified. Actual candidate-branch execution has not yet been directly traced; a default Triton cache lookup found no matching files.

Six candidate streams completed. Their matched control streams and the after ShareGPT control are still pending. The candidate processes were cleanly stopped on all four nodes; the original GDN source hash was restored on all four. Canonical configuration is restarting as run `e27143e51a624f96a06921e49bf90345`.

No promotion decision or achievement of 60 token/s follows from the current partial bracket. See [record](../bench/records/2026-09-07-aiter-gdn-model.json).
