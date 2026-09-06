import importlib.util
import json
import gzip
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).resolve().parents[1] / "tools/summarize_rocprof.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_overlap_is_not_idle_or_additive_wall_time(self):
        events = [summary.event_from_row({"Kernel_Name": "ncclDevKernel", "Start_Timestamp": 1000000,
                                         "End_Timestamp": 3000000, "Agent_Id": 1}, "kernel", "a"),
                  summary.event_from_row({"Kernel_Name": "_qsa_mqa_paged_kernel", "Start_Timestamp": 2000000,
                                         "End_Timestamp": 4000000, "Agent_Id": 1}, "kernel", "a")]
        report = summary.summarize(events)
        self.assertEqual(report["summed_kernel_ms"], 4)
        self.assertEqual(report["agent_windows"][0]["kernel_interval_union_ms"], 3)
        self.assertEqual(report["agent_windows"][0]["not_covered_by_traced_kernel_intervals_ms"], 0)

    def test_cpu_only_rejected(self):
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            summary.summarize([])

    def test_csv_raw_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "worker_kernel_trace.csv").write_text('"Kernel_Name","Start_Timestamp","End_Timestamp","Grid_Size"\n"wvSplitK<int4>",1000,5000,64\n')
            (p / "worker_kernel_stats.csv").write_text('"Kernel_Name","Calls","TotalDurationNs"\n"wvSplitK<int4>",1,4000\n')
            es = summary.load_csv(p)
            self.assertEqual(len(es), 1)
            self.assertEqual(summary.summarize(es)["summed_kernel_ms"], 0.004)

    def test_sdk_json_symbol_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            data = {"rocprofiler-sdk-tool": [{"kernel_symbols": [{"kernel_id": 7,
                "formatted_kernel_name": "_glm53_moe_int4_gemv_partial"}],
                "buffer_records": {"kernel_dispatch": [{"start_timestamp": 1000,
                "end_timestamp": 6000, "dispatch_info": {"kernel_id": 7, "agent_id": 1}}]}}]}
            (p / "worker_results.json").write_text(json.dumps(data))
            es = summary.load_json(p)
            self.assertEqual(len(es), 1)
            self.assertEqual(summary.summarize(es)["categories"][0]["name"], "moe")

    def test_bf16_not_mislabelled_hc(self):
        self.assertEqual(summary.category("wvSplitK<bf16>"), "linear_unattributed_not_proven_hc")

    def test_sdk_graph_records_keep_dispatch_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            data = {"rocprofiler-sdk-tool": [{"buffer_records": {"hip_graph": [{
                "start_timestamp": 1000, "end_timestamp": 6000,
                "graph_exec_id": 12, "kernel_dispatch_count": 700}]}}]}
            (p / "worker_results.json").write_text(json.dumps(data))
            es = summary.load_json(p)
            self.assertEqual(es[0]["kind"], "hip_graph")
            self.assertEqual(es[0]["graph_exec_id"], "12")
            self.assertEqual(es[0]["kernel_dispatch_count"], "700")

    def chrome(self, rows, display="ms", compressed=False):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ("trace.json.gz" if compressed else "trace.json")
            data = json.dumps({"displayTimeUnit": display, "traceEvents": rows})
            if compressed:
                with gzip.open(p, "wt") as stream:
                    stream.write(data)
            else:
                p.write_text(data)
            return summary.load_chrome(p)

    def test_chrome_microseconds_and_large_timestamp(self):
        es = self.chrome([{"ph": "X", "cat": "kernel", "name": "kernel1", "pid": 0,
                           "ts": 5759666805111.928, "dur": 79.037,
                           "args": {"device": 0, "stream": 1, "correlation": 3}}])
        self.assertEqual(es[0]["end"] - es[0]["start"], 79037)
        self.assertAlmostEqual(summary.summarize(es)["summed_kernel_ms"], .079037)

    def test_chrome_cpu_launch_and_flow_are_not_gpu(self):
        es = self.chrome([{"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel",
                           "ts": 0, "dur": 20},
                          {"ph": "s", "cat": "ac2g", "name": "kernel", "ts": 0, "dur": 99}])
        self.assertEqual(len(es), 1)
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            summary.summarize(es)

    def test_chrome_graph_join_is_async_correlation_not_containment(self):
        rows = [{"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch", "pid": 80,
                 "ts": 0, "dur": 5, "args": {"correlation": 7}}]
        rows.extend({"ph": "X", "cat": "kernel", "name": "_glm53_moe_int4_gemv_partial",
                     "pid": 0, "ts": t, "dur": 20, "args": {"correlation": 7,
                     "device": 0, "grid": g, "block": [128,1,1]}}
                    for t,g in [(100,[40,5,4]), (130,[40,40,5])])
        es = self.chrome(rows, compressed=True)
        report = summary.summarize(es)
        graph = report["chrome_graph_correlation"]["graph_signatures"][0]
        self.assertEqual(graph["kernel_dispatches_per_launch"], 2)
        self.assertAlmostEqual(graph["mean_gpu_span_ms"], .05)
        self.assertEqual(len(graph["top_kernel_shapes_mean_per_launch"]), 2)
        self.assertEqual({r["name"] for r in report["moe_observed_roles"]},
                         {"w1_partial_tp4_grid_match", "w2_partial_tp4_grid_match"})

    def test_moe_identical_names_without_grid_stay_unattributed(self):
        es = self.chrome([{"ph": "X", "cat": "kernel", "name": "_glm53_moe_int4_gemv_partial",
                           "ts": 0, "dur": 20}])
        self.assertEqual(summary.moe_role(es[0]), "routed_partial_w1_or_w2_grid_missing_or_unmatched")

    def test_separate_worker_files_do_not_merge_device_zero_time(self):
        es = self.chrome([{"ph": "X", "cat": "kernel", "name": "k", "ts": 0,
                           "dur": 20, "pid": 0, "args": {"device": 0}}])
        es.append(dict(es[0], source="another-worker.json"))
        report = summary.summarize(es)
        self.assertEqual(len(report["agent_windows"]), 2)
        self.assertAlmostEqual(report["summed_kernel_ms"], .04)

    def test_ambiguous_graph_correlation_is_not_guessed(self):
        es = self.chrome([{"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch",
                           "ts": 0, "dur": 3, "args": {"correlation": 1}},
                          {"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch",
                           "ts": 10, "dur": 3, "args": {"correlation": 1}},
                          {"ph": "X", "cat": "kernel", "name": "k", "ts": 50,
                           "dur": 20, "args": {"correlation": 1}}])
        corr = summary.summarize(es)["chrome_graph_correlation"]
        self.assertEqual(corr["correlated_graph_launches"], 0)
        self.assertEqual(corr["uncorrelated_or_non_graph_kernel_dispatches"], 1)


if __name__ == "__main__":
    unittest.main()
