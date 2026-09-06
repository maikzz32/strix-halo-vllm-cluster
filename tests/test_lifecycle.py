"""Local lifecycle regression tests; no SSH, containers, signals, or GPUs."""
import contextlib
import importlib.util
import io
import json
import pathlib
import signal
import sys
import tempfile
import types
import unittest
from unittest import mock

# node_runtime is deployed to Linux; only its platform-independent logic is
# exercised locally on Windows. No lock function is called by these tests.
if sys.platform == "win32":
    sys.modules.setdefault("fcntl", types.SimpleNamespace())

BASE = pathlib.Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE / "scripts" / "native" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module("tested_runtime", "node_runtime.py")
cluster = load_module("tested_cluster", "cluster.py")


def proc(args, ppid=1, start="100", run_id=None):
    return {"args": args, "ppid": ppid, "start": start, "run_id": run_id, "state": "S"}


class RuntimeTests(unittest.TestCase):
    def test_profiler_config_is_preserved_on_every_rank(self):
        config = json.loads((BASE / "models" / "cluster.runtime.json").read_text())
        self.assertNotIn('--profiler-config', runtime.command(config, 0))
        config['profiler_config'] = {'profiler': 'torch', 'torch_profiler_dir': '/tmp/path with spaces'}
        for rank in range(len(config['nodes'])):
            args = runtime.command(config, rank, 'test-run')
            value = args[args.index('--profiler-config') + 1]
            self.assertEqual(json.loads(value), config['profiler_config'])

    def setUp(self):
        self.kill_patch = mock.patch.object(signal, "SIGKILL", 9, create=True)
        self.kill_patch.start()

    def tearDown(self):
        self.kill_patch.stop()

    def test_stat_starttime_and_zombie_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            for pid, state in ((31, "S"), (32, "Z")):
                directory = root / str(pid)
                directory.mkdir()
                # Fields 3 through 22: state, ppid, ..., starttime.
                fields = [state, "17"] + ["0"] * 17 + ["987654"]
                (directory / "stat").write_text(f"{pid} (comm with ) inside) " + " ".join(fields))
                (directory / "cmdline").write_bytes(b"VLLM::Worker\0")
                (directory / "environ").write_bytes(b"STRIX_CLUSTER_RUN_ID=test-run\0")
            result = runtime.processes(root)
            self.assertEqual(set(result), {31})
            self.assertEqual(result[31]["ppid"], 17)
            self.assertEqual(result[31]["start"], "987654")
            self.assertEqual(result[31]["run_id"], "test-run")

    def test_orphan_root_includes_descendants_and_titled_api(self):
        data = {10: proc(["VLLM::EngineCore"]), 20: proc(["python3", "helper"], ppid=10),
                30: proc(["VLLM::APIServer"]), 40: proc(["unrelated"])}
        with mock.patch.object(runtime, "processes", return_value=data):
            self.assertEqual(set(runtime.inference_processes()), {10, 20, 30})

    def test_run_scoped_cleanup_excludes_previous_service(self):
        data = {10: proc(["VLLM::Worker"], run_id="old"),
                20: proc(["VLLM::Worker"], run_id="new"),
                21: proc(["python3", "helper"], ppid=20)}
        with mock.patch.object(runtime, "processes", return_value=data):
            self.assertEqual(set(runtime.inference_processes("new")), {20, 21})

    def test_pid_reuse_never_signals_replacement(self):
        with mock.patch.object(runtime.os, "pidfd_open", return_value=9, create=True), \
             mock.patch.object(runtime.os, "close") as close, \
             mock.patch.object(runtime.signal, "pidfd_send_signal", create=True) as send, \
             mock.patch.object(runtime, "processes", return_value={20: proc(["other"], start="999")}):
            runtime.signal_process(20, "100", signal.SIGTERM)
            send.assert_not_called()
            close.assert_called_once_with(9)

    def test_stop_returns_failure_for_uninterruptible_remaining_process(self):
        ticks = iter(range(100))
        with mock.patch.object(runtime, "inference_processes", return_value={20: proc(["VLLM::Worker"])}), \
             mock.patch.object(runtime, "signal_process") as send, \
             mock.patch.object(runtime.time, "monotonic", side_effect=lambda: next(ticks)), \
             mock.patch.object(runtime.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.internal_stop("new"), 1)
            self.assertIn(mock.call(20, "100", signal.SIGTERM), send.call_args_list)
            self.assertIn(mock.call(20, "100", signal.SIGKILL), send.call_args_list)

    def test_stop_waits_for_disappearance_and_catches_late_child(self):
        ticks = iter(range(100))
        snapshots = iter([
            {20: proc(["VLLM::Worker"])},
            {20: proc(["VLLM::Worker"])},
            {21: proc(["python3", "helper"], run_id="new")},
            {}, {}, {}
        ])
        with mock.patch.object(runtime, "inference_processes", side_effect=lambda run: next(snapshots, {})), \
             mock.patch.object(runtime, "signal_process") as send, \
             mock.patch.object(runtime.time, "monotonic", side_effect=lambda: next(ticks)), \
             mock.patch.object(runtime.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.internal_stop("new"), 0)
            self.assertIn(mock.call(21, "100", signal.SIGTERM), send.call_args_list)


class CoordinatorTests(unittest.TestCase):
    config = {"nodes": [{"host": "one"}, {"host": "two"}], "api_port": 8000, "model": "/model"}

    def common(self):
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(sys, "argv", ["cluster.py", "start"]))
        stack.enter_context(mock.patch.object(pathlib.Path, "read_text", return_value=json.dumps(self.config)))
        stack.enter_context(mock.patch.object(cluster.uuid, "uuid4", return_value=types.SimpleNamespace(hex="unique-new-run")))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        return stack

    def test_preflight_failure_starts_nothing_and_cleans_nothing(self):
        with self.common(), mock.patch.object(cluster, "all_nodes", side_effect=RuntimeError("busy")) as calls:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                cluster.main()
            self.assertEqual([call.args[1] for call in calls.call_args_list], ["preflight"])

    def test_partial_start_failure_cleans_new_run_only(self):
        def answer(config, action, args, run_id=None, quiet=False):
            if action == "start":
                raise RuntimeError("rank1 failed")
            return {}
        with self.common(), mock.patch.object(cluster, "all_nodes", side_effect=answer) as calls:
            with self.assertRaisesRegex(RuntimeError, "rank1 failed"):
                cluster.main()
            self.assertEqual([call.args[1] for call in calls.call_args_list], ["preflight", "start", "cleanup"])
            self.assertEqual(calls.call_args_list[-1].args[3], "unique-new-run")

    def test_dead_rank_before_api_check_triggers_owned_rollback(self):
        def answer(config, action, args, run_id=None, quiet=False):
            if action == "status":
                return {0: {"run_id": "unique-new-run", "launcher_alive": True, "processes": {1: {}}},
                        1: {"run_id": "unique-new-run", "launcher_alive": False, "processes": {}}}
            return {}
        with self.common(), mock.patch.object(cluster, "all_nodes", side_effect=answer) as calls, \
             mock.patch.object(cluster.urllib.request, "urlopen") as api:
            with self.assertRaisesRegex(RuntimeError, "Rank 1 launcher died"):
                cluster.main()
            api.assert_not_called()
            self.assertEqual(calls.call_args_list[-1].args[1:4:2], ("cleanup", "unique-new-run"))


if __name__ == "__main__":
    unittest.main()
