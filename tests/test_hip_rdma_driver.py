"""CPU-only lifecycle checks for the isolated four-rank HIP/RDMA launcher."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('hip_rdma_driver_under_test', TOOLS / 'run_hip_rdma_graph.py')
driver = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(TOOLS))
try:
    spec.loader.exec_module(driver)
finally:
    sys.path.pop(0)


class DriverTests(unittest.TestCase):
    def source_fixture(self, root):
        target = Path(root) / 'placeholder.c'
        target.write_text('/* CPU-only fixture; never compiled or sent. */\n')
        return {'placeholder.c': 'placeholder.c'}

    def test_dry_run_builds_four_container_configs_without_network(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(driver, 'ROOT', Path(root)), patch.object(
                    driver, 'SOURCES', self.source_fixture(root)), patch.object(
                    driver.sys, 'argv', ['driver']), patch.object(driver, 'serving_snapshot') as serving, patch.object(
                    driver.subprocess, 'run') as remote, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(driver.main(), 0)
            manifest = json.loads(stdout.getvalue())
            self.assertEqual([c['container'] for c in manifest['configs']],
                             ['ray-head', 'ray-worker', 'ray-worker', 'ray-worker'])
            self.assertEqual([c['rank'] for c in manifest['configs']], [0, 1, 2, 3])
            self.assertEqual(len({c['run_id'] for c in manifest['configs']}), 1)
            for config in manifest['configs']:
                self.assertIn('--iterations', config['arguments'])
                self.assertIn('--deadline', config['arguments'])
                self.assertIn('--warmup', config['arguments'])
            serving.assert_not_called()
            remote.assert_not_called()

    def test_busy_server_blocks_every_remote_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(driver, 'ROOT', Path(root)), patch.object(
                    driver, 'SOURCES', self.source_fixture(root)), patch.object(
                    driver.sys, 'argv', ['driver', '--execute']), patch.object(
                    driver, 'serving_snapshot', return_value={'requests':{'running':1, 'waiting':0}}), patch.object(
                    driver.subprocess, 'run') as remote:
                with self.assertRaisesRegex(RuntimeError, 'Serving is busy'):
                    driver.main()
            remote.assert_not_called()

    def test_launcher_is_valid_python_and_uses_container_timeout(self):
        compile(driver.REMOTE.replace('CONFIG', repr({'run_id':'a' * 32}), 1), '<launcher>', 'exec')
        command = driver.ssh(18, 'ray-worker', timeout=True)
        self.assertIn('podman exec -i ray-worker timeout', command[-1])
        self.assertIn('--kill-after=5s', command[-1])
        self.assertIn('python3 -S -', command[-1])

    def test_metric_counter_is_required_and_must_be_finite(self):
        class Response:
            def __init__(self, data): self.data = data
            def read(self): return self.data.encode()
            def __enter__(self): return self
            def __exit__(self, *args): return False
        gauges = 'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
        for extra in ('', 'vllm:request_success_total NaN\n', 'vllm:request_success_total -1\n'):
            with self.subTest(extra=extra), patch.object(driver.urllib.request, 'urlopen', return_value=Response(gauges + extra)):
                with self.assertRaises(RuntimeError):
                    driver.serving_snapshot('http://unused')
        with patch.object(driver.urllib.request, 'urlopen', return_value=Response(
                gauges + 'vllm:request_success_total{reason="stop"} 4\nvllm:request_success_total{reason="length"} 6\n')):
            self.assertEqual(driver.serving_snapshot('http://unused')['successful_requests'], 10)


class ArtifactTests(unittest.TestCase):
    """Accept complete evidence; reject independent corruption in either mode."""

    def setUp(self):
        self.config = {'rank': 2, 'run_id': 'a' * 32}
        self.modes = {'fp32_then_bf16', 'bf16_each_add'}
        self.expected = 4 + 32 * 3
        case = {
            'rank': 2, 'run_id': self.config['run_id'], 'event': 'result',
            'iterations': 32, 'samples': 3, 'warmup': 4,
            'runtime_version': 70200000, 'arena_bytes': 204800,
            'exact': 1, 'finite': 1, 'pinned_mr_ok': 1,
            'negative_test_passed': 1, 'expected_negative_code': -1,
            'expected_negative_error': 'invalid fixed input/output alias, count, mode or exhausted sequence',
            'arena_released': 1, 'returncode': 0, 'cleanup_rc': 0,
            'main_selftest_rc': 0, 'callback_selftest_rc': 0,
            'max_abs_error': 0.0, 'error': '', 'direct_pinned_checks': 4,
            'node_count': 3, 'host_nodes': 1, 'memcpy_nodes': 2,
            'graph_numeric_checks': 7, 'successful_collectives': self.expected,
            'expected_collectives': self.expected, 'callback_calls': self.expected + 2,
            'wall_median_us': 74.0, 'hip_event_median_us': 72.0,
            'wall_samples_us': [73.0, 74.0, 75.0],
            'hip_event_samples_us': [71.0, 72.0, 73.0],
        }
        self.artifact = {
            'event': 'complete', 'status': 'passed', **self.config,
            'cases': [dict(copy.deepcopy(case), mode=mode) for mode in sorted(self.modes)],
        }

    def verify(self, artifact):
        return driver.verify_artifact(artifact, self.config, self.modes, self.expected)

    def test_complete_two_mode_evidence_passes(self):
        self.assertTrue(self.verify(self.artifact))
        self.artifact['cases'].reverse()
        self.assertTrue(self.verify(self.artifact))

    def test_artifact_identity_and_status_must_match(self):
        for key, value in [('rank', 0), ('run_id', 'b' * 32), ('status', 'failed')]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.artifact)
                bad[key] = value
                self.assertFalse(self.verify(bad))

    def test_each_case_identity_must_match(self):
        for index in range(2):
            for key, value in [('rank', 0), ('run_id', 'b' * 32)]:
                with self.subTest(case=index, key=key):
                    bad = copy.deepcopy(self.artifact)
                    bad['cases'][index][key] = value
                    self.assertFalse(self.verify(bad))

    def test_missing_duplicate_or_unexpected_mode_fails(self):
        variants = [[], self.artifact['cases'][:1],
                    [self.artifact['cases'][0], self.artifact['cases'][0]],
                    [self.artifact['cases'][0], dict(self.artifact['cases'][1], mode='unknown')]]
        for cases in variants:
            with self.subTest(modes=[case['mode'] for case in cases]):
                self.assertFalse(self.verify(dict(self.artifact, cases=cases)))
        missing = copy.deepcopy(self.artifact)
        del missing['cases']
        self.assertFalse(self.verify(missing))

    def test_counts_and_graph_structure_are_required_in_each_mode(self):
        fields = ('successful_collectives', 'expected_collectives', 'callback_calls',
                  'direct_pinned_checks', 'node_count', 'host_nodes', 'memcpy_nodes')
        for index in range(2):
            for key in fields:
                for value in (None, self.artifact['cases'][index][key] - 1):
                    with self.subTest(case=index, key=key, value=value):
                        bad = copy.deepcopy(self.artifact)
                        if value is None:
                            del bad['cases'][index][key]
                        else:
                            bad['cases'][index][key] = value
                        self.assertFalse(self.verify(bad))

    def test_timings_must_be_present_finite_and_positive(self):
        for index in range(2):
            for key in ('wall_median_us', 'hip_event_median_us'):
                for value in (None, 0, -1, float('nan'), float('inf'), '74.0'):
                    with self.subTest(case=index, key=key, value=value):
                        bad = copy.deepcopy(self.artifact)
                        bad['cases'][index][key] = value
                        self.assertFalse(self.verify(bad))

    def test_teardown_and_native_failures_reject_the_artifact(self):
        for index in range(2):
            for key, value in [('cleanup_rc', -1), ('arena_released', 0),
                               ('returncode', 1), ('error', 'native failure')]:
                with self.subTest(case=index, key=key):
                    bad = copy.deepcopy(self.artifact)
                    bad['cases'][index][key] = value
                    self.assertFalse(self.verify(bad))

    def test_numerical_or_negative_case_failures_reject_the_artifact(self):
        corruptions = [('exact', 0), ('finite', 0), ('pinned_mr_ok', 0),
                       ('negative_test_passed', 0), ('main_selftest_rc', -1),
                       ('callback_selftest_rc', -1), ('max_abs_error', 0.03125),
                       ('max_abs_error', float('nan'))]
        for index in range(2):
            for key, value in corruptions:
                with self.subTest(case=index, key=key):
                    bad = copy.deepcopy(self.artifact)
                    bad['cases'][index][key] = value
                    self.assertFalse(self.verify(bad))


class CleanupTests(unittest.TestCase):
    """Exercise Linux cleanup decisions with fake processes, signals and time."""

    def helpers(self):
        tree = ast.parse(driver.REMOTE.replace('CONFIG', repr({'run_id': 'a' * 32}), 1))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {
            'proc': Mock(), 'os': Mock(), 'time': Mock(),
            'signal': types.SimpleNamespace(SIGTERM=15, SIGKILL=9),
        }
        namespace['proc'].poll.return_value = 0  # Leader already exited.
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<launcher helpers>', 'exec'), namespace)
        return namespace

    def test_exited_leader_does_not_hide_live_descendant(self):
        namespace = self.helpers()
        live = [True]
        clock = [0.0]

        def monotonic():
            clock[0] += 0.4
            return clock[0]

        def send(expected, signum):
            self.assertEqual(expected, (1234, '5678'))
            if signum == 9:
                live[0] = False

        namespace['time'].monotonic.side_effect = monotonic
        namespace['owned_processes'] = Mock(side_effect=lambda: [(1234, '5678')] if live[0] else [])
        namespace['signal_owned'] = Mock(side_effect=send)
        self.assertEqual(namespace['stop_owned'](), [])
        self.assertEqual([call.args[1] for call in namespace['signal_owned'].call_args_list], [15, 9])

    def test_pidfd_signal_requires_matching_start_time_and_marker(self):
        for actual, marked, allowed in [((1234, 'new'), True, False),
                                        ((1234, '5678'), False, False),
                                        ((1234, '5678'), True, True)]:
            with self.subTest(actual=actual, marked=marked):
                namespace = self.helpers()
                namespace['identity'] = Mock(return_value=actual)
                namespace['has_marker'] = Mock(return_value=marked)
                namespace['signal'] = Mock()
                namespace['os'].pidfd_open.return_value = 91
                namespace['signal_owned']((1234, '5678'), 15)
                self.assertEqual(namespace['signal'].pidfd_send_signal.called, allowed)
                namespace['os'].close.assert_called_once_with(91)

    def test_every_pipe_drain_has_a_timeout(self):
        tree = ast.parse(driver.REMOTE.replace('CONFIG', '{}', 1))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'communicate']
        self.assertTrue(calls)
        for call in calls:
            self.assertIn('timeout', [keyword.arg for keyword in call.keywords])

    def test_release_scan_finds_marker_only_compiler_and_named_harness(self):
        module = ast.parse((TOOLS / 'run_hip_rdma_graph.py').read_text(encoding='utf-8'))
        scan = next(node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == 'scan')
        expression = next(node.value for node in scan.body
                          if isinstance(node, ast.Assign) and node.targets[0].id == 'script')
        run_id = 'a' * 32
        script = eval(compile(ast.Expression(expression), '<release script>', 'eval'), {'run_id': run_id})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = [(101, 'STRIX_HIP_RDMA_RUN_ID=' + run_id, 'g++', 'S'),
                        (102, '', 'python\0--run-id\0' + run_id, 'S'),
                        (103, 'OTHER=x', 'unrelated', 'S'),
                        (104, 'STRIX_HIP_RDMA_RUN_ID=' + run_id, 'dead', 'Z')]
            for pid, environment, command, state in fixtures:
                path = root / str(pid)
                path.mkdir()
                (path / 'environ').write_bytes(environment.encode() + b'\0')
                (path / 'cmdline').write_bytes(command.encode() + b'\0')
                (path / 'stat').write_text(str(pid) + ' (child) ' + ' '.join([state] + ['0'] * 18 + ['5678']))
            original_path = Path

            def mapped_path(path):
                return root if path == '/proc' else original_path(path)

            with patch('pathlib.Path', mapped_path), contextlib.redirect_stdout(io.StringIO()) as stdout:
                with self.assertRaises(SystemExit) as ended:
                    exec(compile(script, '<release scan>', 'exec'), {})
            self.assertEqual(ended.exception.code, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual([item['pid'] for item in report['remaining']], [101, 102])
            self.assertEqual(report['scan_errors'], [])


if __name__ == '__main__':
    unittest.main()
