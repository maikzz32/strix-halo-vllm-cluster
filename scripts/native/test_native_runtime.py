"""Local deployment/restart regressions; no SSH, containers, signals or GPUs."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

if sys.platform == "win32":
    sys.modules.setdefault("fcntl", types.SimpleNamespace())


def load(name):
    spec = importlib.util.spec_from_file_location("native_test_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cluster = load("cluster")
runtime = load("node_runtime")
deploy = load("deploy_release")


class CoordinatorTests(unittest.TestCase):
    def test_config_digest_ignores_key_order_but_rejects_drift(self):
        digest = runtime.verify_config({"model": "one", "nodes": [1, 2]}, None)
        self.assertEqual(runtime.verify_config({"nodes": [1, 2], "model": "one"}, digest), digest)
        with self.assertRaisesRegex(RuntimeError, "differs from coordinator"):
            runtime.verify_config({"model": "two", "nodes": [1, 2]}, digest)

    def test_remote_start_gets_digest_but_recovery_does_not_require_it(self):
        args = types.SimpleNamespace(config="/same/path.json", tag="test", config_sha256="abc")
        response = types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
        with mock.patch.object(cluster.subprocess, "run", return_value=response) as run, contextlib.redirect_stdout(io.StringIO()):
            cluster.remote({"host": "host"}, 0, "start", args, "run")
            self.assertIn("--expected-config-sha256 abc", run.call_args.args[0][-1])
            cluster.remote({"host": "host"}, 0, "cleanup", args, "run")
            self.assertNotIn("--expected-config-sha256", run.call_args.args[0][-1])

    def test_container_preflight_receives_same_digest(self):
        args = types.SimpleNamespace(config="/same/path.json", rank=1, run_id=None, expected_config_sha256="abc")
        response = types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
        with mock.patch.object(runtime.subprocess, "run", return_value=response) as run:
            runtime.internal_call({"container": "ray-worker"}, "internal-preflight", args)
            self.assertEqual(run.call_args.args[0][-2:], ["--expected-config-sha256", "abc"])

    def test_restart_validation_failure_preserves_running_service(self):
        config = {"model": "/model", "api_port": 8000, "nodes": [{"host": "one"}]}
        with mock.patch.object(sys, "argv", ["cluster.py", "restart"]), \
             mock.patch.object(Path, "read_text", return_value=json.dumps(config)), \
             mock.patch.object(cluster, "all_nodes", side_effect=RuntimeError("config drift")) as remote, \
             mock.patch.object(cluster, "ensure_idle") as idle:
            with self.assertRaisesRegex(RuntimeError, "config drift"):
                cluster.main()
            self.assertEqual([call.args[1] for call in remote.call_args_list], ["validate"])
            idle.assert_not_called()

    def test_request_gauges_include_labels_multiple_series_and_timestamps(self):
        metrics = '\n'.join([
            '# TYPE vllm:num_requests_running gauge',
            'vllm:num_requests_running{model_name="a b"} 0 1234567',
            'vllm:num_requests_running{model_name="other"} 2',
            'vllm:num_requests_waiting 0',
        ])
        self.assertEqual(cluster.request_counts(metrics), {"running": 2.0, "waiting": 0.0})

    def test_unknown_metrics_cannot_authorize_stop(self):
        for metrics in ('', '# only metadata', 'vllm:num_requests_running 0',
                        'vllm:num_requests_running NaN\nvllm:num_requests_waiting 0',
                        'vllm:num_requests_running -1\nvllm:num_requests_waiting 0'):
            with self.subTest(metrics=metrics), self.assertRaises(RuntimeError):
                cluster.request_counts(metrics)


class DeploymentTests(unittest.TestCase):
    def stage(self, root):
        destination = Path(root).resolve()
        stage = destination / 'deployments' / 'release.test'
        stage.mkdir(parents=True)
        for source in deploy.FILES:
            if source.endswith('.json'):
                data = json.dumps({'model': '/model', 'nodes': [{'host': 'one'}]}).encode()
            elif source.endswith('.sh'):
                data = b'#!/usr/bin/env bash\nset -euo pipefail\n'
            else:
                data = b'print("new release")\n'
            (stage / source).write_bytes(data)
        return stage, destination

    def test_promote_restore_and_repeated_release_preserve_operator_config(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            tool = destination / 'tools/cluster.py'
            tool.parent.mkdir()
            tool.write_bytes(b'# original operator controller\n')
            config = destination / 'config/cluster.json'
            config.parent.mkdir()
            config.write_bytes(b'{"operator":"custom"}\n')
            deploy.promote(stage, destination)
            self.assertEqual(tool.read_bytes(), (stage / 'cluster.py').read_bytes())
            self.assertEqual(config.read_bytes(), b'{"operator":"custom"}\n')
            deploy.restore(stage, destination)
            self.assertEqual(tool.read_bytes(), b'# original operator controller\n')
            self.assertEqual(config.read_bytes(), b'{"operator":"custom"}\n')
            self.assertFalse((destination / 'tools/node_runtime.py').exists())
            with self.assertRaisesRegex(ValueError, 'already attempted'):
                deploy.promote(stage, destination)
            # Restoration itself is idempotent.
            deploy.restore(stage, destination)

    def test_initial_default_config_created_then_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            deploy.promote(stage, destination)
            self.assertEqual((destination / 'config/cluster.json').read_bytes(), (stage / 'cluster.runtime.json').read_bytes())
            deploy.restore(stage, destination)
            self.assertFalse((destination / 'config/cluster.json').exists())

    def test_validation_failure_does_not_overwrite_deployment(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            tool = destination / 'tools/cluster.py'
            tool.parent.mkdir()
            tool.write_bytes(b'# untouched\n')
            (stage / 'quality_probe.py').write_bytes(b'def broken(:\n')
            with self.assertRaises(SyntaxError):
                deploy.promote(stage, destination)
            self.assertEqual(tool.read_bytes(), b'# untouched\n')
            self.assertFalse((stage / 'receipt.json').exists())

    def test_restore_refuses_post_deploy_edits_before_restoring_any_file(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            deploy.promote(stage, destination)
            changed = destination / 'tools/node_runtime.py'
            changed.write_bytes(b'# edited after deploy\n')
            with self.assertRaisesRegex(ValueError, 'changed since deployment'):
                deploy.restore(stage, destination)
            self.assertEqual(changed.read_bytes(), b'# edited after deploy\n')
            self.assertTrue((destination / 'tools/cluster.py').exists())

    def test_interrupted_promotion_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            actual_replace = deploy.replace

            def fail_after_first(target, data, mode=0o644):
                if target.name == 'node_runtime.py':
                    raise OSError('simulated disk failure')
                actual_replace(target, data, mode)

            with mock.patch.object(deploy, 'replace', side_effect=fail_after_first):
                with self.assertRaisesRegex(OSError, 'simulated disk failure'):
                    deploy.promote(stage, destination)
            self.assertTrue((destination / 'tools/cluster.py').exists())
            deploy.restore(stage, destination)
            self.assertFalse((destination / 'tools/cluster.py').exists())

    def test_crlf_shell_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            stage, destination = self.stage(folder)
            (stage / 'bench_sharegpt.sh').write_bytes(b'#!/bin/bash\r\n')
            with self.assertRaisesRegex(ValueError, 'CRLF'):
                deploy.validate(stage)


if __name__ == '__main__':
    unittest.main()
