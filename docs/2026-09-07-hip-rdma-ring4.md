# Calibrated four-rank HIP/RDMA backend

The isolated descriptor backend measured **56.42 µs per 20-KiB all-reduce**
(maximum of four rank median HIP-event measurements), including GPU staging,
native CPU verbs, ordered BF16 arithmetic and copying back to the GPU.
The wall-clock equivalent was 56.48 µs. The historical RCCL control was about
85.3 µs. These figures are collective timings, not model throughput.

The backend now reproduces the observed RCCL Ring/LL/four-channel arithmetic:
four fixed rings and a source-derived 3072/3072/3072/1024-element partition.
It preserves BF16 rounding after each rank addition. All 327,680 archived RCCL
output values match. The new native arithmetic additionally passed 32 fresh CPU
input banks, totaling 655,360 checked output values across both sets.

Before becoming usable, each backend compares 32 independent new finite banks
and rotating cancellation cases with its actual PyNccl communicator and the CPU
reference, both eagerly and in a captured graph. Each phase repeats bank zero.
All four ranks must pass. This is a fixed shape/topology profile; it is not a
general promise about arbitrary RCCL configurations or NaN payload propagation.

The isolated runtime test retains 32 distinct outputs in each graph and checks
every output after four changing-input warmups and each timed sample. Three
samples replay that graph 32 times each. Both the preliminary smoke and longer
test passed on all ranks, including source preservation, unsupported-shape
fallback, graph destruction, native teardown, PyNccl destruction and empty final
process scans. The model service remained idle throughout each test.

The [evidence record](../bench/records/2026-09-07-hip-rdma-ring4-backend.json)
contains both runs, source hashes, calibration proofs, samples and cleanup data.
The earlier [direct callback experiment](2026-09-07-hip-rdma-coupled.md) used two
different fixed arithmetic modes and should not be confused with this calibrated
ring profile.

## Model experiment

An opt-in adapter has been staged in the four preserved containers. The canonical
configuration remains separate from `hip-rdma-model-r1.json`. The adapter uses
the existing registered vLLM all-reduce operator, accepts only contiguous BF16
SUM with 10,240 elements in the four-rank TP group, and leaves other operations
on existing dispatch. It performs the same mandatory startup calibration against
the model's own PyNccl communicator.

The first six matched 512-token responses are text-identical to the freshly
restarted baseline. Their median paired decode-rate gain is **3.00%**; individual
gains range from 2.74% to 3.85%. Coding answers measured 67.58–68.15 versus
65.62–65.67 decode tokens/s. These coding prompts are a different workload from
the historical 48.60-token/s ShareGPT benchmark.

The complete ShareGPT48 run measured **49.789 output tokens/s**, 17.987 ms mean
TPOT and 544.385 ms mean TTFT. All 48 responses, input/output token counts and
speculative decoding statistics match the earlier QSA production run exactly:
11,839 output tokens and 54.1427% acceptance. That earlier run measured 50.302
output tokens/s and 17.734 ms TPOT, so the short-prompt gain is not sufficient
evidence of a ShareGPT improvement. The six short responses after restoring RCCL are also identical, but the
candidate is now 1.34% slower at the median. This reversal means the initial
+3.00% is not a robust backend gain. The restored-RCCL ShareGPT control measured **50.490 output tokens/s**,
17.610 ms TPOT and 548.995 ms TTFT. All 48 texts and speculation statistics
remain identical. The candidate has 1.39% lower throughput and 2.14% higher
TPOT than this control. It is **not promoted**. All four original communicator
sources were restored, and the canonical QSA/RCCL model is running again.
[Model evidence and comparisons](../bench/records/2026-09-07-hip-rdma-model.json).

Native callback failures terminate the experimental worker rather than returning
stale output. Descriptor allocation is bounded and performed before capture;
captured descriptors remain reserved for the backend lifetime. The adapter has
no automatic resource finalizer: explicit teardown requires graph destruction,
or resources are reclaimed when the worker process exits. This work does not
add a service supervisor, automatic reboot recovery or arbitrary concurrent
stream support.

## Reproducing the bounded experiment

`tools/run_hip_rdma_graph.py --backend-test --mode ring4` is a local dry run.
Adding `--execute` runs the isolated four-rank test only when the existing model
service is idle. The driver builds inside the preserved containers and records
all rank outputs, source identities and cleanup evidence. It does not install
the serving adapter.

`tools/deploy_hip_rdma_experiment.py` is specific to the four documented hosts,
container names and reviewed communicator SHA. `stage --base-source PATH`
requires a byte-exact copy of that original `cuda_communicator.py`; it rejects
other revisions. The default is dry-run; `--execute` stages and builds in
`/opt/strix-halo-next/hip-rdma-ring4-20260907-r1`. `activate --execute` installs
the optional hook, while `restore --execute` restores its hash-checked original.
Neither action restarts workers. Stop the owned model run before changing the
source, then start the desired configuration through the head-node controller.
Do not mix enable settings between ranks: the controller's identical-config
check is a prerequisite of this adapter.

The model experiment used a separate config with `STRIX_HIP_RDMA_ENABLE=1`,
`STRIX_HIP_RDMA_MODE=ring4`, `STRIX_HIP_RDMA_MASTER=192.168.100.1`,
`STRIX_HIP_RDMA_PORT=29921`, `STRIX_HIP_RDMA_TIMEOUT_MS=5000`,
`STRIX_HIP_RDMA_DESCRIPTORS=2048`, and `PYTHONPATH` pointing to the addon.
`STRIX_HIP_RDMA_LIBRARY` names its `libhip_rdma_backend.so`. All original model
arguments and other environment values remained unchanged. The canonical
`cluster.json` has no backend enable flag.

CPU verification: the 16 driver tests and three evidence-rejection tests pass.
The four reference tests pass in the research workspace; a clean checkout
explicitly skips the one test requiring four external archived binary files.
The native fixture test also passed on Node18 without GPU or network calls.
Two source-patch tests verify rejection of changed sources and missing anchors;
the portable deploy restore dry-run does not require local archived runtime files.
