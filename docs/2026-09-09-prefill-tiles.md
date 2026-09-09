# Prefill tile investigation — not deployed

The installed TP4 service remains on its original kernels and weights. A narrower
WNA16 prefill tile is promising for short prefills, but has not yet been tested
in full-model serving. These are operator timings, not output tokens/second.

The isolated node 4 process exercises the actual modular MoE apply path with
512 experts, top-10 routing, TP4 local W1/W2 shapes, asymmetric INT4 group size
32 and BF16 activations. We compare the installed configuration against
BM32/BN32/BK32 and BM64/BN32/BK32. Weights and routing are synthetic. Eight
alternating timing rounds each replay a GPU graph 20 times. Eager outputs and
graph outputs after changing the input must be bit-identical to the default.

Second run (seed 9402), median microseconds per operator:

| Input rows | Default | BM32/BN32/BK32 | BM64/BN32/BK32 |
|---:|---:|---:|---:|
| 64 | 3868.75 | 3204.24 | 4025.73 |
| 128 | 4887.73 | 4069.34 | 5081.24 |
| 256 | 5674.52 | 4713.02 | 5883.02 |
| 512 | 5941.89 | 5042.82 | 6130.52 |
| 1024 | 6358.34 | 5759.54 | 6527.09 |
| 2048 | 7797.24 | 10097.13 | 8091.55 |
| 4096 | 13937.17 | 16940.87 | 15349.39 |
| 8192 | 24401.78 | 31329.91 | 27561.37 |

All tested parity checks passed. The first run, seed 9401, independently showed
roughly 10–17% shorter operator time for BM32/BN32/BK32 at 64–1024 rows. Large
prefills regress, so an unconditional override is unsuitable. Actual expert
load imbalance may also change the result. Decode uses a separate small-row
path and no decode gain has been established.

The next candidate should be restricted to the validated tensor shapes and
row range, preserve BK32, and undergo before/candidate/after serving trials
with output comparisons, TTFT, ShareGPT C1 throughput and tool-call checks.

### Concrete candidate validation

`patches/moe_prefill_tiles.py` now implements that bounded, default-off override
under `STRIX_MOE_PREFILL_TILES=1`. It preserves explicit configuration overrides
and requires ROCm, the exact TP4 weight shapes, top-10 routing and INT4 GS32.
It pins the installed source SHA256 to
`8b0d48a769b03c62e786a880941531ebdfb1bc582d53b6e03c5e49e73b3b2493`;
the resulting candidate SHA256 is
`66b775e00ecbb5d2f320dda82d3cd93f7baa607bb77f0a39c248bcb64d4c01a8`.

`tests/bench_moe_prefill_candidate.py` evaluates the actual transformed
configuration function inside an isolated process without editing installed
files. With seed 9403, all ten cases (4, 32, 63, 64, 65, 128, 512, 1024,
1025 and 2048 rows) passed eager and changed-input graph bitwise comparisons.
At 64 rows operator time fell from 3809.60 to 3144.13 microseconds; at 1024
rows from 6309.02 to 5835.78 microseconds. Cases outside the selected range
retain their original configuration. Timing variation on those unchanged
paths illustrates why full-model bracketed trials remain necessary.

The source patch is **not installed** in any serving container. Candidate
results are in `bench/records/2026-09-09-prefill-candidate.json`. The test
requires the repository's `patches` directory on its Python import path.

An earlier isolated larger-K experiment (BM32/BN64, BK64 or BK128) terminated
with `HSA_STATUS_ERROR_MEMORY_FAULT` before producing its first case result.
The precise cause and failing variant remain unproven. No installed source
was changed. A subsequent 32-token serving request completed successfully;
this variant is not a deployment candidate. Raw failure stderr was observed
in the execution log; no standalone raw failure artifact was retained.

Reproduction in the compatible original Strix container:

```bash
python3 tests/bench_moe_prefill_ntiles.py --output /tmp/prefill-r1.json
python3 tests/bench_moe_prefill_ntiles_r2.py --output /tmp/prefill-r2.json
```

Do not run these GPU microbenchmarks concurrently with serving measurements.
The scripts change the configuration selector only inside their own process.
Raw samples are in `bench/records/2026-09-09-prefill-ntiles-r{1,2}.json`.

## Context latency baseline

`bench/context_latency.py` issued three exact-prefix requests at each input
length with 32 output tokens. The installed service already has prefix caching
enabled. TTFT and observed prefix-cache hit counter deltas were:

| Prompt tokens | First use TTFT / hits | First reuse TTFT / hits | Second reuse TTFT / hits |
|---:|---:|---:|---:|
| 2048 | 2020 ms / 0 | 2039 ms / 0 | 1246 ms / 800 |
| 8192 | 10453 ms / 0 | 9197 ms / 0 | 1098 ms / 7200 |
| 32768 | 38262 ms / 0 | 9026 ms / 24800 | 1723 ms / 31200 |

The first reuse does not always obtain the later cache coverage. Hybrid state
checkpoint boundaries and scheduling need investigation before attributing a
cause. Some repeated output hashes differ at 2K and 32K; text was not retained
by this probe, so this is neither a semantic correctness verdict nor proof of
a cache bug. A targeted reproduction should retain text and control cache
state. See `bench/records/2026-09-09-context-baseline.json` for measurements.
