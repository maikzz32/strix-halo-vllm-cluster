# HIP graph dispatch diagnostic

A fresh-process ABBA comparison on Node 4 tested the default queue limit
against `GPU_MAX_HW_QUEUES=1`. The serving process was unchanged and idle;
request counters remained unchanged before and after each probe. Every probe
checked two changed integer inputs for each graph length before timing.

The synthetic graph repeats an in-place integer addition over 256 elements.
It preserves a strict dependency chain, with no compiler fusion, model weights,
RDMA, or profiler. Seven samples each replay the captured graph 16 times.
Both HIP event timings and synchronized host wall times are recorded.

| Variant, in execution order | 128 launches, us | 1024 launches, us | 2646 launches, us |
| --- | ---: | ---: | ---: |
| Default A | 231.990 | 1844.516 | 4792.006 |
| Queue limit 1, B | 231.998 | 1843.798 | 4791.069 |
| Queue limit 1, B | 232.076 | 1844.244 | 4791.039 |
| Default A | 232.057 | 1844.115 | 4791.550 |

Values are median HIP event times. The queue limit supplies no meaningful
improvement in this serial workload. It is not a candidate for a model restart
on this evidence. About 1.81 microseconds per dependent operation includes
kernel execution and its memory dependency, not just dispatch. The result
cannot be subtracted from model latency or used to label profiled gaps as idle.
The model has mixed kernels, copies and distributed synchronization; 2646 here
is only a synthetic chain length, not a reconstruction of its graph.

ROCm 10 documentation lists `GPU_MAX_HW_QUEUES=4` and
`HIP_FORCE_DEV_KERNARG=1` as defaults. An absent environment variable does not
mean that optimization is disabled. See the
[official environment variable reference](https://rocm.docs.amd.com/en/latest/reference/environment-variables/).
This test changes only the queue limit, not kernel-argument placement.

Evidence: `bench/records/2026-09-10-graph-dispatch-chain.json` includes raw
sample arrays, Torch/HIP versions, the loaded HIP library hash and test hash.
The executable test is `tests/bench_graph_dispatch_chain.py`; run each variant
in a fresh process with its environment set before importing Torch.
