# Actual RCCL BF16 reference preparation

The rank script `tools/rccl_bf16_reference.py` archives the actual four-rank RCCL output for the exact 32 input banks defined by `cpu_rdma_reference.input_bits`. It is a numerical reference collector, with no timing or model-quality claim. Preparation and four local CPU tests pass; GPU/network execution is reserved for the parent after its coupled HIP test.

The existing owned `run_rccl_microbench.py` launcher has a `--bf16-reference` mode. It embeds and hashes the local generator source, retains the existing four-rank RCCL environment and HCA interfaces, forces exactly 20,480 BF16 bytes and graph capture, and limits each rank to45 seconds with an outer55-second TERM/5-second KILL bound. It is not a new process runner or tuning sweep.

The graph captures one all-reduce on a stable GPU pointer. Before every replay, the rank copies a fresh bank from its unchanged CPU tensor and verifies GPU input bits against that CPU bank. Outputs therefore do not accumulate as repeated powers of four. Every result must be finite. After all32 banks, bank0 is freshly copied and replayed again; it must reproduce its first output bitwise. The graph is reset before communicator destruction, and the payload is emitted only after teardown.

Run after obtaining all four GPU/network leases:

```sh
python tools/run_rccl_microbench.py --bf16-reference --port 29891 --output results/rccl-bf16-reference-20260907
```

Add `--dry-run` for local command preparation without any SSH or GPU action. Choose a new output directory for repeated execution.

Each rank produces `rankR-nodeN.bf16.bin`, exactly655360 bytes: 32 consecutive banks of10240 little-endian BF16 words. The existing runner captures the payload through stdout and then decodes it locally, validating length, SHA256, per-bank hashes and finiteness. Associated `.reference.json` records include generator hash, input/output-bank hashes, rank/run ID, Torch/HIP versions, selected environment and input-reset/repeat proof. `reference-summary.json` verifies all four binary outputs are identical; unequal outputs or missing post-teardown completion yield a failed run while preserving evidence. This does not force either CPU accumulation mode to match RCCL: that comparison follows separately from the saved actual bits.

Local validation: four stdlib tests passed for exact bank generation/source hash, bounded four-rank dry-run command construction, invalid configuration rejection before launch, binary archival and deliberate cross-rank mismatch rejection. No remote command, GPU or network test was run during preparation.
