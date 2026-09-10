# Remaining CPU arithmetic and lossless HC weight survey

The deployed UMA cluster remains unchanged at the validated 52.04 output
tokens/s. This round measures two possible next changes; neither result is a
new serving speedup.

## CPU ring4 arithmetic cost

The exact production `cpu_rdma_reduce_host(..., mode=2)` was compiled with
GCC C11/O3, without fast-math or LTO, inside each preserved container. The
existing numerical selftest passed first. Five samples of 2048 calls each
used one hot buffer bank and 32 rotating banks. All samples were retained.
The model service was idle; its existing UMA progress threads remained alive.

| Node | Hot median (microseconds) | Rotating median (microseconds) |
|---|---:|---:|
| 15 | 9.101 | 9.211 |
| 16 | 8.902 | 8.920 |
| 17 | 8.764 | 8.785 |
| 18 | 8.725 | 8.758 |

For 97 ordinary target collectives, eliminating this arithmetic entirely would
save about 0.85-0.89ms if it lay entirely on the critical path. This is an
opportunity estimate: malloc buffers do not reproduce GPU/CPU cache-coherence
costs, and a GPU replacement has its own work and memory traffic. It cannot
alone establish the large single-answer improvement requested by the user.

[Sources, checksums and every sample](../bench/records/2026-09-07-cpu-ring4-cost.json).
Reproduce on an idle Linux node with the existing headers and libibverbs:

```sh
gcc -std=c11 -O3 -Wall -Wextra -Werror -Itools \
  tests/bench_cpu_ring4_cost.c tools/cpu_rdma_transport.c \
  -libverbs -lm -o /tmp/bench-cpu-ring4-cost
timeout 20s /tmp/bench-cpu-ring4-cost
```

The executable does not open a verbs context or call HIP.

## Lossless BF16 HC packing feasibility

A read-only checkpoint survey inspected all 298 hyper-connection input-mix
and block-injection weight tensors, totaling 1,318,748,160 bytes. It did not
import Torch/HIP or change any checkpoint or running worker.

Within each contiguous block, if the BF16 exponent range spans at most 16
values, retain the eight original sign/mantissa bits and encode the exponent
as a four-bit offset from the block minimum. Other blocks retain the original
16-bit representation. Every eligible block passed an actual byte-packed CPU
roundtrip with all BF16 bits identical, including sign bits. This is lossless
storage encoding, not lower-precision quantization.

Estimated storage includes a four-byte descriptor per block:

| Values per block | Estimated bytes | Saving |
|---|---:|---:|
| 32 | 1,072,169,376 | 18.70% |
| 64 | 1,031,905,824 | 21.75% |
| 128 | 1,013,539,904 | 23.14% |
| 256 | 1,008,399,232 | 23.53% |

[Per-tensor hashes and roundtrip evidence](../bench/records/2026-09-07-hc-lossless-survey.json).
The same installed Python/NumPy can reproduce the read-only survey:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tools/analyze_hc_lossless.py \
  --model /home/cluster-user/qwen38_rest --output /tmp/hc-lossless-survey.json
```

The subsequent [native GPU experiment](2026-09-07-hc-native.md) implemented and tested in-kernel lossless decoding. It preserved the tested BF16 results but slowed the common merged-down and up projections; no serving integration was promoted. An uncompressed unroll adjustment improved the isolated up operator by 8%, with an estimated full-model opportunity below 1% that remains unvalidated.
