# Four-rank HIP graph with CPU-RDMA, 2026-09-07

The complete **GPU input → D2H → native CPU-RDMA callback → H2D → GPU output** path measured **58.14 µs with FP32 accumulation** and **59.14 µs with BF16 rounding after each addition**. These are the maximum of four rank medians measured with HIP events. Wall-clock equivalents were 58.39 and 59.41 µs. All four ranks passed both modes, including registration of the actual HIP-pinned arena, changing-input numerical checks and teardown.

This is an isolated collective benchmark. It changes no model weights or serving configuration. Bitwise equality was established against each mode's scalar CPU reference; equality to RCCL's reduction order and model-quality preservation remain separate checks. Production still uses RCCL.

## Measured results

Each rank processes 10,240 BF16 values, exactly 20 KiB. One captured graph contains two copies and one native host callback. The main run used four individually checked graph warmups followed by three samples of 32 graph replays. Every number below is a median in microseconds per complete collective.

| Node / rank | FP32 HIP event | FP32 wall | BF16 each-add HIP event | BF16 each-add wall |
|---|---:|---:|---:|---:|
| 15 / 0 | 58.136 | 58.390 | 58.937 | 59.119 |
| 16 / 1 | 57.933 | 58.196 | 58.775 | 59.031 |
| 17 / 2 | 57.900 | 58.173 | 59.137 | 59.409 |
| 18 / 3 | 57.835 | 58.092 | 58.712 | 58.972 |
| Maximum rank median | **58.136** | **58.390** | **59.137** | **59.409** |

The earlier four-replay, one-sample smoke is retained too: maximum rank event times were 70.298/73.728 µs and wall times 72.276/75.768 µs. It is not mixed into the longer-run medians. Three timed samples are a bounded proof, not a long-run latency distribution.

The smoke and main run took place on September 6 UTC, September 7 local time. Main run `b6d3a4ebdacb4c569d654d24340330b0` recorded its initial idle snapshot at **22:58:18.029153 UTC** and final snapshot at **22:58:21.266755 UTC**. Across both runs, all eight native rank processes and launchers exited zero; all temporary builds were removed and final run-ID scans were empty. The service had zero running/waiting requests before and after each test, and its completed-request counter remained 66. The [compact evidence record](../bench/records/2026-09-07-hip-rdma-coupled.json) preserves all rank/sample times, proof counts, timestamps, hashes and both runs.

## What the measurement includes

All four Strix Halo GPUs and Intel 25 Gb/s RoCE interfaces participate. The callback performs three parallel RC `RDMA_WRITE_WITH_IMM` sends per rank, followed by the local AVX2 sum in fixed rank order. Each rank therefore sends 60 KiB of payload for a 20 KiB input. This is the allgather protocol described in the [CPU-RDMA experiment](2026-09-07-cpu-rdma.md), with two alternating receive slots and all three send completions checked before return.

A single 200 KiB `hipHostMalloc` allocation is borrowed by the C transport and registered once with `ibv_reg_mr`. Its first eight slots hold two sets of four received rank buffers; slots eight and nine hold input and output. GPU input and output are separate allocations totaling 40 KiB. Four direct changed-input collectives prove this exact pinned-memory registration before capture. Registration, TCP/QP setup, allocation and graph construction are excluded from timing.

The native callback calls neither HIP nor Python. It uses the already-created transport, preallocated buffers and bounded completion polling. GPU input changes outside the graph for every checked warmup and timing sample; each sample then repeats the same input out-of-place. Source preparation and output readback checks are excluded from timing. HIP events surround all 32 launches and synchronization; wall time also includes event submission and the end-event wait. No GPU producer or consumer kernel is included beyond the two copies.

This direct measurement replaces the earlier estimate obtained by adding the separate [host-callback microbenchmark](2026-09-07-hip-host-callback.md) to CPU collective time. Medians from independent tests should not be added as if they were one observed execution.

## Numerical and lifetime checks

Both accumulation modes send BF16 over the network. `fp32_then_bf16` sums in FP32 and rounds once; `bf16_each_add` rounds after each addition in the specified rank order. Each of the eight main-run cases checked four direct inputs and seven graph outputs against its scalar reference: four changed warmups plus one result after each sample. Every BF16 word matched, every positive-case output was finite and maximum absolute error was zero. The 100 expected successful graph callbacks per case were observed exactly; no extra callback ran during capture.

The scalar/AVX2 self-test also runs in both the main thread and the actual HIP callback thread. It checks 32 finite banks and eight sets of 65,536 raw BF16 patterns. Both threads retained round-to-nearest/ties-to-even with FTZ and DAZ disabled. MXCSR exception status flags changed during exceptional-value tests; the arithmetic control bits did not change.

Two terminal negative replays deliberately pass an invalid count of 10,239. The first fails before sending; the second retains the failure without another collective. Both propagate canonical NaN output instead of stale valid data. Thus each case records 102 total callbacks but only 100 successful collectives. Teardown destroys the synchronized graph/stream and transport before freeing borrowed memory; every arena was released. These checks do not establish behavior under a failed NIC, hung driver or real multi-request serving workload.

## Comparison with RCCL and remaining scope

The earlier four-rank 20 KiB RCCL controls measured maximum rank event medians of **85.268, 85.250 and 85.289 µs**, using automatic Ring/LL with four channels. Comparing event times gives a roughly 26–27 µs lower isolated collective time here. The benchmarks share payload, dtype, ranks and network, but their boundaries differ:

| Detail | Historical RCCL control | Coupled HIP/CPU-RDMA |
|---|---|---|
| Captured graph | 32 all-reduces | One copy/callback/copy collective |
| Timed sampling | 20 graph replays | Three samples of 32 graph replays |
| Warmup | 50 eager collectives and three graph replays | Four direct checks and four graph replays |
| Timed inputs | In-place zero-buffer SUM; separate nonzero eager/captured proof | Out-of-place finite nonzero inputs |
| Runtime | Torch distributed / RCCL | Native HIP / C++ callback / C verbs |

These were historical controls rather than an interleaved A/B with identical graph construction. Under the additional assumption that **96 layer collectives of this size** can be replaced per M4 target step, the difference projects to approximately **2.51 ms** for repeated BF16 rounding or **2.60 ms** for FP32 accumulation. This is a potential saving, not a measured model speedup. Matching the actual RCCL output and checking model logits, acceptance and throughput are required before treating it as a production replacement. The [RCCL reference collector](2026-09-07-rccl-bf16-reference.md) addresses the numerical comparison separately.

## Reproduction and provenance

The six executed sources are [HIP wrapper](../tests/hip_rdma_graph.cpp), [rank harness](../tests/bench_hip_rdma_graph.py), [C transport](../tools/cpu_rdma_transport.c), [ABI header](../tools/cpu_rdma_transport.h), [AVX2 arithmetic](../tools/cpu_rdma_reduce_avx2.h) and [reference generator](../tools/cpu_rdma_reference.py). The [coordinator](../tools/run_hip_rdma_graph.py) builds C and C++ inside each preserved `ray-head`/`ray-worker` container, using its own libc/HIP/verbs environment. No Fedora 45 host-built library is injected into the Fedora 44 container userspace. All eight builds produced library SHA-256 `256ff7e4a763ab0f46673a27765b66efc80fd93655cf6baf8ea6f56851bd3e8b`.

From the repository root, local preparation and CPU-only regression checks are:

```sh
python -m unittest discover -s tests -p test_hip_rdma_driver.py -v
python tools/run_hip_rdma_graph.py --replays 32 --samples 3 --mode both
```

The second command is a dry run with no SSH or GPU activity. During an explicitly coordinated idle window across all four GPUs and the RDMA fabric:

```sh
python tools/run_hip_rdma_graph.py --execute --replays 32 --samples 3 --mode both --output results/hip-rdma-new-run
```

Use a new output directory. The default bootstrap ports are 29881 and 29882, separately assigned to the two modes to avoid a transition race. The native mode deadline is 10 seconds, each transport-call deadline is at most five seconds, and the coordinator's child limit is 45 seconds with outer TERM/KILL and SSH bounds. Cleanup uses the unique inherited run marker plus process identity/pidfd checks, including descendants after leader exit. API idle/counter checks and ownership scans preserve evidence when a run fails; they do not reserve the cluster against an unrelated client submitting work mid-test.

All 16 CPU mock regression tests pass. The published coordinator adds validation of each case's rank and run ID after the measurements; the six remote source files are unchanged. Source hashes in the record identify the exact UTF-8/LF bytes sent to the containers. Windows copies of those sources may have different raw hashes because of CRLF line endings. Raw manifests, status, release and service snapshots remain in the workspace's two named result directories; their hashes are retained in the repository record. This experiment adds no autostart implementation or serving hook.
