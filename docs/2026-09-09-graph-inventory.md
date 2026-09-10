# Direct HIP graph inventory

The existing four-rank decode traces contain 32 graph launches per rank but
no `gpu_memcpy` or `gpu_memset` events anywhere in the trace. This is not
sufficient evidence that captured graphs contain no copy nodes. The reproducible
event inventory is in `bench/records/2026-09-09-graph-copy-trace.json` and
`tools/count_decode_copies.py`.

`tools/hip_graph_inventory.cpp` intercepts successful HIP graph instantiation,
then queries node types, dependencies, memcpy dimensions/direction and child
graphs. It does not intercept replay or modify graph topology. Source graphs
need not be retained after instantiation. Both the legacy and flags APIs are
covered; a build using a different API needs separate validation. Nested API
calls can produce duplicate inventories, so file count is not a replay count.

The module was compiled against the installed HIP headers with:

```sh
g++ -std=c++17 -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include hip_graph_inventory.cpp \
  -ldl -o inventory.so
```

The staged binary SHA256 on all four nodes is
`c85cbef9949990357f34776bbaa7333c1e5e67edaefd5fc1320400a56354fdcd`.
The first build linked directly against `/opt/rocm/lib/libamdhip64` and failed
in the isolated process with duplicate LLVM option registration. Revision 2
resolves symbols from the already loaded HIP library, without loading another
HIP dependency. This revision passed on all four nodes.

Before model use, run `tests/check_hip_graph_inventory.py` in an idle isolated
process with `LD_PRELOAD` pointing to the module and
`STRIX_GRAPH_INVENTORY_DIR` pointing to an existing writable directory. The test
requires observed graph nodes and dependencies and checks replay output. Tensor
copies may use kernels; this smoke does not establish memcpy metadata coverage.

Each isolated smoke observed one two-node graph, validated dependencies and
matched the replay output. See `bench/records/2026-09-09-graph-inventory-smoke.json`.
The dedicated diagnostic model run `735d65c7cb964ecc9a5edead04cf0579` is starting;
actual model graph inventory has now completed. Diagnostic startup overhead must
not be presented as serving performance, and copy-only replay controls cannot
be subtracted directly from full-model latency.

## Model findings

All four verified worker processes yielded 18 graph inventories after startup
and one successful 128-token answer. Each rank has the same 2646-node graph:
2558 kernel nodes and 88 memcpy nodes, forming a dependency chain of length
2646. There are no host/event nodes in this particular graph. Other captured
graphs include startup/debug host and event nodes; inventory count alone does
not establish their replay frequency. Identifying a specific graph's live replay
frequency remains separate from counting its nodes.

All 88 copies use D2D kind 3: 50 x 20480 bytes, 36 x 12288 bytes,
one x 245760 bytes and one x 81920 bytes, totaling 1794048 bytes. This corrects
the interpretation of the 2646 profiler `kernel` events as exclusively compute
kernels. The profiler event category did not expose the 88 HIP memcpy nodes.

A bounded Node 1 microbenchmark with the same copy-size multiset measured
178.72 microseconds for the entire serial 88-copy graph (five samples of 32
replays). Two fresh input patterns reproduced the output exactly. This uses
contiguous synthetic buffers and excludes interleaved compute, communication
and model cache behavior. It does not prove the copies cost exactly 0.179 ms
in the model, but does not support assuming 10–20 microseconds of unavoidable
dispatch cost per copy. See `tests/bench_graph_copy_chain.py` and records
`2026-09-09-copy-chain.json`, `2026-09-09-copy-sizes.json` and
`2026-09-09-model-graph-summary.json` under `bench/records/`.

All raw graph files were validated against the collected SHA256s. The initial
Windows text writer changed LF to CRLF; original LF bytes were restored only
after verifying that they match those SHA256s. The collector now writes bytes.

Next priority: use source/trace correlation to select adjacent compute-kernel
fusion, rather than assume several milliseconds of copy-only dispatch savings.
