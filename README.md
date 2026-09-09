# Four-node Strix Halo vLLM cluster

Low-latency Qwen3.8 Flash Next inference across **4 × AMD Ryzen AI Max+ 395**,
each with **128 GB unified memory**, connected through **25 GbE RoCEv2**.
All four gfx1151 GPUs cooperate using tensor parallelism.

The focus is a faster **single answer**, with the existing INT4 weights and
computation preserved. The acceptance target is **at least 60 output tokens/s
on ShareGPT48 at concurrency 1**, together with improved response latency.
That target is still open.

## Measured performance

| Configuration / milestone | ShareGPT output tokens/s | Mean TTFT | Mean TPOT | Evidence |
|---|---:|---:|---:|---|
| Original TP4 baseline | 48.60 | See report | See report | [Baseline](docs/2026-09-06-single-stream.md) |
| TP4 + UMA production checkpoint | 52.04 | See report | See report | [UMA](docs/2026-09-07-hip-uma.md) |
| TP4 + W2 serial production checkpoint | 53.85 | See report | 16.43 ms | [W2](docs/2026-09-07-moe-w2-serial-model.md) |
| TP4 + AVX512 + W1 expert order | 56.04 | See report | See report | [Bracketed W1 comparison](docs/2026-09-07-moe-w1-order.md) |
| vLLM 0.29 custom TP4, September 9 baseline | **55.74** | **541.77 ms** | **15.63 ms** | [Record](bench/records/2026-09-09-qwen029-tp4-baseline.json) |

These are historical milestones, **not a controlled comparison between every
row**. Individual reports contain matched controls, output parity and known
confounders. The September 9 baseline completed 48/48 requests, generated
11,839 tokens and took 212.38 seconds. No version-only speedup is claimed.

Output throughput includes prefill and request overhead. It is not pure
decode speed or aggregate throughput at high concurrency. MTP emits token
bursts, so stream event intervals must not be confused with per-token time.
See [benchmark methodology](bench/README.md).

## Current runtime

| Component | Configuration |
|---|---|
| Hosts | Fedora 45, `192.168.1.15`–`192.168.1.18` |
| GPU | Radeon 8060S, gfx1151 / RDNA 3.5 |
| Network | Intel 25 GbE RDMA NICs through a switch |
| Model | Qwen3.8 Flash Next, `/home/maik/qwen38_rest` |
| Quantization | Existing asymmetric INT4, group size 32 |
| Runtime | Custom `0.29.0+strix.rocm100`, preserved gfx1151 patches |
| Parallelism | TP4, native vLLM `mp` executor |
| Speculation | Native MTP, 3 draft tokens |
| Graphs | `FULL_DECODE_ONLY`, draft prefill graph |
| Context limit | 262,144 tokens |
| Communication | Validated UMA/RDMA backend, AVX512 BF16 reduction |
| Tool calls | Automatic tool selection with `qwen3_xml` parser |

The custom release installation applies the complete upstream runtime delta
while preserving the working Strix stack. It is not the stock ROCm wheel.
The live container overlays and saved node images contain additional patches;
building the generic base image does not reproduce this deployment.
[Deployment details](docs/2026-09-09-qwen029-tp4.md).

## Connect an OpenAI-compatible client

```text
Base URL: http://192.168.1.15:8000/v1
Model:    /home/maik/qwen38_rest
API key:  local (placeholder if the client requires one)
```

Hermes Agent can use this custom endpoint. Automatic and streamed tool calls
have been checked, including a tool-result roundtrip. The model uses the Qwen
XML parser; the agent's name does not determine the server parser.

## Operate the existing cluster

Run on node 1:

```bash
python3 /home/maik/strix-halo-next/tools/cluster.py status \
  --config /home/maik/strix-halo-next/config/cluster.json

python3 /home/maik/strix-halo-next/tools/cluster.py start \
  --config /home/maik/strix-halo-next/config/cluster.json \
  --tag production --timeout 900
```

The controller checks all ranks and records run IDs, commands and logs.
Normal stop/restart requires idle request queues. Startup and crash recovery
are manual. Experiments use separate configuration files and may temporarily
replace the canonical run; inspect the controller status before operating it.
[Native controller guide](scripts/native/README.md).

## Reproduce the serving benchmark

On node 1, with the September 9 containers running:

```bash
bash /home/maik/strix-halo-next/tools/bench_sharegpt_qwen029.sh my-unique-run
```

This uses 48 ShareGPT prompts, seed 42, temperature 0, concurrency 1 and
streamed chat completions. The script and dataset requirements are in
[bench/README.md](bench/README.md). Results are collected outside the model
container. Public records retain timings and response hashes; weights and
generated text are not committed.

## Research and experiments

- [September 9 GitHub and paper review](docs/2026-09-09-research-roadmap.md)
- [Hardware strengths and kernel hypotheses](docs/2026-09-07-strix-hardware-strengths.md)
- [Measured GPU critical-path budget](docs/2026-09-06-gpu-budget.md)
- [DGX Spark comparison and workload differences](docs/2026-09-06-spark-comparison.md)
- [UCCL investigation and the NIC SRQ limitation](docs/2026-09-07-uccl-review.md)
- [TP2 versus TP4 comparison](docs/2026-09-07-tp2-comparison.md)
- [AITER module review](docs/2026-09-07-aiter-modules.md)

Negative experiments remain documented. Isolated kernel improvements are not
advertised as model-level gains. Candidate changes require numerical checks,
full-model measurements and controls before promotion.

## Repository layout

| Directory | Purpose |
|---|---|
| `scripts/native/` | Run-scoped lifecycle and deployment for the existing cluster |
| `bench/` | Benchmark clients, exporters and public measurement records |
| `patches/` | Version-specific runtime and kernel patches |
| `tests/` | Numerical, graph and lifecycle checks |
| `tools/` | Runtime evidence collection and bounded experiments |
| `docs/` | Deployment, research and experiment reports |
| `docker/`, `ansible/`, `models/` | General build/provisioning framework and model registry |

The older general build/provisioning path covers additional models and is
separate from the measured Qwen Flash Next runtime. See the historical
[runbook](docs/RUNBOOK.md) and [performance log](docs/PERFORMANCE.md) for that work.
