# Benchmark methodology

## Primary acceptance workload

- Model: existing `/home/cluster-user/qwen38_rest`, asymmetric INT4 group32.
- TP4 across four 128 GB gfx1151 hosts; MTP3 and 262144 context limit.
- Dataset: `ShareGPT_V3_unfiltered_cleaned_split.json` at
  `/home/cluster-user/datasets/` on node 1. The dataset is not bundled.
- 48 prompts, seed 42, temperature 0, maximum concurrency 1.
- OpenAI chat-completions endpoint, streamed output, default chat template.
- vLLM's preliminary single-request check runs before the timed workload.
- Keep client, dataset, prompt ordering, template and generation parameters
  unchanged within a comparison.

Run `bench_sharegpt_qwen029.sh UNIQUE_TAG` on node 1. Its container name is
`qwen029-tp2` for historical reasons; the server must actually be TP4.
Inspect runtime status and config, not the container name. Earlier
`bench_sharegpt.sh` targets the preserved old `ray-head` container.

The September 9 baseline is a single run, not a distribution of repeated
trials. It is the control for current vLLM 0.29 experiments, not evidence
that an upstream version change improved or regressed performance.

## Metrics

| Metric | Definition / interpretation |
|---|---|
| Output tokens/s | Generated tokens divided by wall time, including prefill and request overhead |
| TTFT | Client time to first streamed token; mean, median and p99 |
| TPOT | Per-request time per output token excluding the first token |
| ITL | Delay between stream events; MTP bursts mean this is not one-token latency |
| Acceptance length | Mean accepted draft tokens plus the normal target token per iteration |
| Successful/failed | Must be reported alongside speed |

The CLI's derived peak concurrency can disagree with the configured request
limit. Preserve it in raw evidence and use configured concurrency plus
server counters to validate the workload; do not claim a higher load level.

## Comparing a candidate

Use control/candidate/control where practical, retain all measurements,
and compare text hashes, token lengths and MTP statistics. Also run six fixed
512-token streams (`bench_stream.py`, three prompts, two repeats) for a decode
comparison. Do not compare these shorter non-thinking prompts directly with
the default-template ShareGPT headline.

Use changed-input GPU graph tests for kernel correctness. After a restart,
verify config and loaded libraries on all ranks. Check auto tool calls and
concurrency-2 requests. Cold-prefix and warm-prefix latency should be reported
separately in context-depth tests; maximum context is not measured prompt depth.

Export a public record without generated text:

```bash
python3 bench/export_serving_record.py PATH/result.json \
  --config PATH/config.json --run-id RUN_ID \
  --output bench/records/DATE-CONFIG.json
```

Records retain raw-result and config SHA256 plus per-response hashes. Keep
raw answers locally for exact comparison. Microbenchmark gains, different
quantizations and multi-client aggregate throughput are separate measurements.

## Other benchmark tools

The older registry-driven harness remains available for other deployments:

```bash
python3 bench/run_matrix.py --models qwen36-35b-a3b --profiles tp4,ep \
  --concurrencies 1,16 --prompt-lengths 512
python3 bench/report.py 'bench/results/*.json'
```

It depends on `scripts/serve.sh`, Python, PyYAML and requests. It can use a
fallback load generator with approximate prompt lengths; records produced by
different generators must not be pooled. Its older broad process teardown is
not the run-scoped native controller and should not manage the current Qwen
Flash Next experiment.

`prefix_probe.py` measures prefix reuse. `kpool_triton_validate.py` and the
INT4 prototype tools cover older GLM experiments. `compare_eth_vs_rdma.sh`
and `iommu_ab.sh` target the general deployment; the latter involves host
configuration/reboots. Their historical measurements remain in
[`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md) and Git history.
