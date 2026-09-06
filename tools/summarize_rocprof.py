#!/usr/bin/env python3
"""Summarize rocprofv3 CSV/SDK JSON or Torch Chrome traces on CPU only.

CSV is preferred when both formats exist, avoiding duplicate dispatch counts.
SDK timestamps are nanoseconds. Chrome timestamps/durations are microseconds,
irrespective of displayTimeUnit; the Chrome reader converts them explicitly.
"""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re


def normalize(key):
    return re.sub(r"[^a-z0-9]", "", key.lower())


def get(row, *names, default=None):
    data = {normalize(k): v for k, v in row.items()}
    return next((data[normalize(n)] for n in names if normalize(n) in data), default)


def category(name):
    s = name.lower()
    if any(n in s for n in ("nccl", "rccl")):
        return "collective_kernel_includes_waits"
    if "qsa_mqa" in s:
        return "qsa_scoring"
    if "qsa_sparse" in s or "qsa_merge" in s:
        return "qsa_sparse_attention"
    if "top_k_per_row_decode" in s:
        return "topk_qsa_likely_verify_callsite"
    if "qsa" in s:
        return "qsa_other"
    if "moe" in s:
        return "moe"
    if "act_and_mul_kernel" in s or "silu_and_mul" in s:
        return "silu_shared_or_routed_unattributed"
    if any(n in s for n in ("delta_rule", "gated_delta", "causal_conv")):
        return "gdn_recurrent_or_conv"
    if "wvsplitk" in s and "int4" in s:
        return "dense_int4"
    if "wvsplitk" in s or "gemm" in s:
        return "linear_unattributed_not_proven_hc"
    if "hyper_connection" in s:
        return "hc_named"
    return "other_unattributed"


def trace_files(path, extension):
    if path.is_file():
        return [path] if path.name.endswith(extension) else []
    return sorted(path.rglob("*" + extension))


def read_json(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def interval_union(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(stop, end) if end is not None else stop
    return total


def event_from_row(row, kind, source, symbols=None):
    start = get(row, "Start_Timestamp", "Start_Timestamp_ns")
    end = get(row, "End_Timestamp", "End_Timestamp_ns")
    if start is None or end is None:
        return None
    try:
        start, end = int(start), int(end)
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    info = get(row, "dispatch_info", default={})
    if not isinstance(info, dict):
        info = {}
    name = get(row, "Kernel_Name", "formatted_kernel_name", "Operation", "Function", "Name", default="")
    if not name and symbols:
        name = symbols.get(str(get(info, "kernel_id", default=get(row, "kernel_id"))), "")
    if not name:
        name = ("hipGraphLaunch" if kind == "hip_graph" else
                "unknown_kernel_id=" + str(get(info, "kernel_id", default=get(row, "kernel_id", default="?"))))
    def val(*keys):
        return str(get(row, *keys, default=get(info, *keys, default="?")))
    return {"name": str(name), "kind": kind, "start": start, "end": end,
            "pid": val("Process_Id", "pid"), "agent": val("Agent_Id"),
            "queue": val("Queue_Id"), "grid": val("Grid_Size"),
            "workgroup": val("Workgroup_Size"), "source": source,
            "graph_exec_id": val("graph_exec_id"),
            "kernel_dispatch_count": val("kernel_dispatch_count"),
            "correlation_id": val("correlation_id")}


def load_csv(path):
    events = []
    for f in trace_files(path, ".csv"):
        if "kernel" in f.name.lower() and "trace" in f.name.lower():
            kind = "kernel"
        elif "hip" in f.name.lower() and "trace" in f.name.lower():
            kind = "hip_api_or_graph"
        elif "memory_copy" in f.name.lower():
            kind = "memory_copy"
        else:
            continue
        with f.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                e = event_from_row(row, kind, str(f))
                if e:
                    events.append(e)
    return events


def load_json(path):
    events = []
    for f in trace_files(path, ".json"):
        if f.name.startswith("attach-"):
            continue
        try:
            data = read_json(f)
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and "traceEvents" in data:
            continue
        symbols = {}
        def symbol_walk(obj):
            if isinstance(obj, dict):
                kid = get(obj, "kernel_id")
                name = get(obj, "formatted_kernel_name", "demangled_kernel_name", "kernel_name")
                if kid is not None and isinstance(name, str):
                    symbols[str(kid)] = name
                for value in obj.values():
                    symbol_walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    symbol_walk(value)
        symbol_walk(data)
        def walk(obj, context=""):
            if isinstance(obj, dict):
                lower = context.lower()
                kind = ("hip_graph" if "hip_graph" in lower else
                        "kernel" if "kernel_dispatch" in lower else
                        "hip_api_or_graph" if "hip" in lower else
                        "memory_copy" if "memory_copy" in lower else None)
                if kind:
                    e = event_from_row(obj, kind, str(f), symbols)
                    if e:
                        events.append(e)
                        return
                for key, value in obj.items():
                    walk(value, context + "/" + key)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value, context)
        walk(data)
    return events


def load_chrome(path):
    """Read complete GPU intervals; CPU launch APIs never count as kernels."""
    events = []
    for f in trace_files(path, ".json") + trace_files(path, ".json.gz"):
        try:
            data = read_json(f)
        except (ValueError, OSError):
            continue
        if isinstance(data, dict):
            rows = data.get("traceEvents", [])
        elif isinstance(data, list):
            rows = data
        else:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("ph") != "X":
                continue
            cats = {s.strip().lower() for s in str(row.get("cat", "")).split(",")}
            name = str(row.get("name", ""))
            if cats & {"kernel", "gpu_kernel"}:
                kind = "kernel"
            elif cats & {"gpu_memcpy", "gpu_memset"}:
                kind = "memory_copy" if "gpu_memcpy" in cats else "memory_set"
            elif cats & {"cuda_runtime", "cuda_driver", "hip_runtime", "hip_driver"}:
                kind = "hip_api_or_graph"
            elif cats & {"cpu_op", "user_annotation", "python_function"}:
                kind = "cpu_scope"
            else:
                continue  # Metadata, flow links and overall Trace spans are not work.
            try:
                start = round(float(row["ts"]) * 1000)
                duration = round(float(row["dur"]) * 1000)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if duration < 0:
                continue
            args = row.get("args", {})
            if not isinstance(args, dict):
                args = {}
            def val(*keys, default="?"):
                value = get(args, *keys, default=default)
                return json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
            events.append({"name": name, "kind": kind, "start": start,
                           "end": start + duration, "pid": str(row.get("pid", "?")),
                           "agent": val("device", "device_id"),
                           "queue": val("stream", default=row.get("tid", "?")),
                           "grid": val("grid", "grid_size"),
                           "workgroup": val("block", "block_size", "workgroup_size"),
                           "source": str(f), "graph_exec_id": val("graph_exec_id", "graph id"),
                           "kernel_dispatch_count": "?", "correlation_id": val("correlation", "correlation_id"),
                           "external_id": val("External id"), "input_shapes": val("Input Dims", "Input Shapes"),
                           "source_format": "torch_chrome"})
    return events


def moe_role(event):
    """Only identify W1/W2 when full TP4 split-K grid geometry is available."""
    name = event["name"].lower()
    if "_glm53_moe_int4_gemv_partial" in name:
        try:
            grid = json.loads(event["grid"])
        except (ValueError, TypeError):
            grid = None
        if isinstance(grid, list) and len(grid) == 3:
            if grid[1:] == [5, 4]:
                return "w1_partial_tp4_grid_match"
            if grid[1:] == [40, 5]:
                return "w2_partial_tp4_grid_match"
        return "routed_partial_w1_or_w2_grid_missing_or_unmatched"
    if "_glm53_moe_int4_gemv_reduce" in name:
        return "routed_split_reduce_w1_or_w2"
    if "topkgating" in name:
        return "router_topk"
    if "moe_align_block_size" in name:
        return "expert_alignment"
    if "moe_sum" in name:
        return "expert_sum"
    if "act_and_mul_kernel" in name or "silu_and_mul" in name:
        return "silu_shared_or_routed"
    if "moe" in name:
        return "other_moe_including_possible_fallback"
    return None


def correlate_graphs(events):
    """Use same-file correlation IDs, never CPU containment of async GPU work."""
    launches = collections.defaultdict(list)
    kernels = collections.defaultdict(list)
    for e in events:
        if e.get("source_format") != "torch_chrome" or e["correlation_id"] == "?":
            continue
        key = (e["source"], e["correlation_id"])
        if e["kind"] == "hip_api_or_graph" and "graphlaunch" in e["name"].lower():
            launches[key].append(e)
        elif e["kind"] == "kernel":
            kernels[key].append(e)
    rows = []
    shapes = collections.defaultdict(list)
    shape_kernel_ns = collections.defaultdict(collections.Counter)
    shape_kernel_calls = collections.defaultdict(collections.Counter)
    for key, ls in launches.items():
        if len(ls) != 1:
            continue  # Do not guess if a trace reused a correlation identifier.
        es = kernels.get(key, [])
        if not es:
            continue
        signature = sorted((name, grid, block, n) for (name, grid, block), n in
                           collections.Counter((e["name"], e["grid"], e["workgroup"]) for e in es).items())
        digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
        times = [(e["start"], e["end"]) for e in es]
        cat_times = collections.Counter()
        for e in es:
            cat_times[category(e["name"])] += e["end"] - e["start"]
            sk = (e["name"], e["grid"], e["workgroup"])
            shape_kernel_ns[(key[0], digest)][sk] += e["end"] - e["start"]
            shape_kernel_calls[(key[0], digest)][sk] += 1
        row = {"source": key[0], "correlation_id": key[1], "cpu_pid": ls[0]["pid"],
               "signature": digest, "kernel_dispatches": len(es),
               "cpu_launch_us": (ls[0]["end"] - ls[0]["start"]) / 1000,
               "summed_kernel_ms": sum(b-a for a, b in times) / 1e6,
               "gpu_span_ms": (max(b for a, b in times) - min(a for a, b in times)) / 1e6,
               "kernel_interval_union_ms": interval_union(times) / 1e6,
               "categories": dict(collections.Counter(category(e["name"]) for e in es)),
               "category_kernel_ms": {k: v/1e6 for k, v in cat_times.items()}}
        rows.append(row)
        shapes[(key[0], digest)].append(row)
    return {"correlated_graph_launches": len(rows),
            "graph_launch_samples": rows[:100],
            "graph_signatures": [{"source": source, "signature": digest, "calls": len(rs),
                                   "kernel_dispatches_per_launch": rs[0]["kernel_dispatches"],
                                   "categories_per_launch": rs[0]["categories"],
                                   "mean_category_kernel_ms": {k: sum(r["category_kernel_ms"].get(k, 0) for r in rs)/len(rs)
                                                               for k in rs[0]["category_kernel_ms"]},
                                   "mean_gpu_span_ms": sum(r["gpu_span_ms"] for r in rs)/len(rs),
                                   "mean_summed_kernel_ms": sum(r["summed_kernel_ms"] for r in rs)/len(rs),
                                   "top_kernel_shapes_mean_per_launch": [{"name": k[0], "grid": k[1], "workgroup": k[2],
                                                                          "calls": shape_kernel_calls[(source,digest)][k]/len(rs),
                                                                          "sum_ms": ns/1e6/len(rs)}
                                                                         for k, ns in shape_kernel_ns[(source,digest)].most_common(40)]}
                                  for (source, digest), rs in shapes.items()],
            "uncorrelated_or_non_graph_kernel_dispatches": sum(e.get("source_format") == "torch_chrome"
                                                               and e["kind"] == "kernel" for e in events)
                                                           - sum(r["kernel_dispatches"] for r in rows)}


def summarize(events, iterations=None):
    kernels = [e for e in events if e["kind"] == "kernel"]
    if not kernels:
        raise ValueError("No raw kernel dispatch intervals found; refuse CPU-only attribution")
    groups = collections.defaultdict(list)
    symbols = collections.defaultdict(list)
    agents = collections.defaultdict(list)
    for e in kernels:
        groups[category(e["name"])].append(e)
        symbols[(e["name"], e["grid"], e["workgroup"])].append(e)
        agents[(e["source"], e["pid"], e["agent"])].append(e)
    total = sum(e["end"] - e["start"] for e in kernels)
    def budget(key, es):
        durations = sorted(e["end"] - e["start"] for e in es)
        row = {"name": key, "calls": len(es), "sum_ms": sum(durations) / 1e6,
               "median_us": durations[len(durations) // 2] / 1e3,
               "share_of_summed_kernel_time_pct": 100 * sum(durations) / total}
        if iterations:
            row["sum_ms_per_user_supplied_iteration"] = row["sum_ms"] / iterations
        return row
    windows = []
    for (source, pid, agent), es in agents.items():
        span = max(e["end"] for e in es) - min(e["start"] for e in es)
        union = interval_union((e["start"], e["end"]) for e in es)
        windows.append({"source": source, "pid": pid, "agent": agent, "span_ms": span / 1e6,
                        "kernel_interval_union_ms": union / 1e6,
                        "not_covered_by_traced_kernel_intervals_ms": (span - union) / 1e6})
    moe = collections.defaultdict(list)
    host = collections.defaultdict(list)
    for e in events:
        if e["kind"] == "kernel" and (role := moe_role(e)):
            moe[role].append(e)
        elif e["kind"] in {"cpu_scope", "hip_api_or_graph"}:
            host[(e["kind"], e["name"])].append(e)
    return {"kernel_dispatches": len(kernels), "summed_kernel_ms": total / 1e6,
            "chrome_graph_correlation": correlate_graphs(events),
            "moe_observed_roles": sorted([budget(k, es) for k, es in moe.items()], key=lambda r: -r["sum_ms"]),
            "moe_map_reference": "research/moe-profiler-map.md; TP4 M4 target: W1 partial (40,5,4), reduce40; W2 partial (40,40,5), reduce200. Identical names alone do not distinguish W1/W2 or shared/routed SiLU.",
            "host_inclusive_time_top40": sorted([{"kind": k[0], "name": k[1], "calls": len(es),
                                                   "inclusive_sum_ms": sum(e["end"]-e["start"] for e in es)/1e6}
                                                  for k, es in host.items()], key=lambda r: -r["inclusive_sum_ms"])[:40],
            "categories": sorted([budget(k, es) for k, es in groups.items()], key=lambda r: -r["sum_ms"]),
            "top_kernel_shapes": sorted([dict(budget(k[0], es), grid=k[1], workgroup=k[2])
                                         for k, es in symbols.items()], key=lambda r: -r["sum_ms"])[:40],
            "agent_windows": windows,
            "hip_graph_launch_records": [{"name": e["name"], "start_ns": e["start"],
                                          "duration_us": (e["end"]-e["start"])/1e3,
                                          "graph_exec_id": e["graph_exec_id"],
                                          "kernel_dispatch_count": e["kernel_dispatch_count"],
                                          "correlation_id": e["correlation_id"]}
                                         for e in events if e["kind"] != "kernel" and "graphlaunch" in e["name"].lower()][:100],
            "graph_shape_counts": [{"graph_exec_id": k[0], "kernel_dispatch_count": k[1], "calls": n}
                                   for k, n in collections.Counter((e["graph_exec_id"], e["kernel_dispatch_count"])
                                   for e in events if e["kind"] == "hip_graph").items()],
            "other_event_counts": dict(collections.Counter(e["kind"] for e in events if e["kind"] != "kernel")),
            "notes": ["Kernel times can overlap; sums are not a disjoint wall-time budget.",
                      "RCCL kernel time includes communication/spin waits, not pure arithmetic.",
                      "Uncovered intervals are not proven GPU idle: copies, untraced work and profiler effects may exist.",
                      "Target and draft remain combined without externally verified iteration/graph boundaries.",
                      "Chrome graph correlation uses same-file launch IDs; signatures group matching kernel shapes but do not independently prove target/draft identity.",
                      "Host times are inclusive and nested; never add CPU launch/scope times to GPU kernel times.",
                      "Kernel-name categories are heuristic; retain unattributed linears rather than label them HC.",
                      "Do not extrapolate a traced slowdown as production performance."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, help="Trace directory or one trace file")
    p.add_argument("--format", choices=("auto", "rocprof", "torch"), default="auto")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--iterations", type=int)
    a = p.parse_args()
    if a.iterations is not None and a.iterations <= 0:
        p.error("iterations must be positive")
    events = load_csv(a.directory) if a.format != "torch" else []
    source_format = "csv"
    if a.format != "torch" and not any(e["kind"] == "kernel" for e in events):
        events = load_json(a.directory)
        source_format = "sdk_json"
    elif a.format != "torch":
        # rocprofv3 --help: detailed HIP graph records are emitted only to JSON
        # and rocpd. Add just those, never duplicate JSON kernel dispatches.
        events.extend(e for e in load_json(a.directory) if e["kind"] == "hip_graph")
    if a.format == "torch" or (a.format == "auto" and not any(e["kind"] == "kernel" for e in events)):
        events = load_chrome(a.directory)
        source_format = "torch_chrome"
    report = summarize(events, a.iterations)
    report["source_format"] = source_format
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("kernel_dispatches", "summed_kernel_ms", "categories", "agent_windows")}, indent=2))


if __name__ == "__main__":
    main()
