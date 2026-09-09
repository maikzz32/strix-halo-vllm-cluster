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
  -L/opt/rocm/lib -lamdhip64 -ldl -o inventory.so
```

The staged binary SHA256 on all four nodes is
`51f7f1ae495b75639af578b5ee71f811255b354ec82ce4a28d8ff426ca632c70`.
Staging is not activation. The canonical runtime remains free of this preload.

Before model use, run `tests/check_hip_graph_inventory.py` in an idle isolated
process with `LD_PRELOAD` pointing to the module and
`STRIX_GRAPH_INVENTORY_DIR` pointing to an existing writable directory. The test
requires observed graph nodes and dependencies and checks replay output. Tensor
copies may use kernels; this smoke does not establish memcpy metadata coverage.

Status: compilation and identical staging passed. Isolated runtime validation
and actual model graph inventory are pending. Diagnostic startup overhead must
not be presented as serving performance, and copy-only replay controls cannot
be subtracted directly from full-model latency.
