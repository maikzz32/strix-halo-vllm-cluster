"""Local guard checks: preparation must never contact hosts or start a rank."""
import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cpu_rdma_driver', ROOT / 'tools/run_cpu_rdma_allreduce.py')
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class DriverTests(unittest.TestCase):
    def test_default_only_prepares_four_owned_ranks(self):
        stream = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run_cpu_rdma_allreduce.py']), \
             mock.patch.object(DRIVER.subprocess, 'run', side_effect=AssertionError('remote launch')), \
             contextlib.redirect_stdout(stream):
            self.assertEqual(DRIVER.main(), 0)
        manifest = json.loads(stream.getvalue())
        self.assertEqual(len(manifest['configs']), 4)
        self.assertEqual(manifest['payload_bytes'], 20480)
        self.assertEqual(manifest['tx_payload_bytes_per_rank'], 61440)
        for rank, cfg in enumerate(manifest['configs']):
            self.assertEqual(cfg['rank'], rank)
            self.assertEqual(cfg['run_id'], manifest['run_id'])
            args = cfg['arguments']
            self.assertEqual(args[args.index('--run-id') + 1], manifest['run_id'])

    def test_remote_launcher_is_valid_python(self):
        compile(DRIVER.REMOTE.replace('CONFIG', repr({'source':'/* local syntax only */'}), 1), '<launcher>', 'exec')

    def test_limits_reject_before_execution(self):
        for option, value in [('--port', '80'), ('--iterations', '0'), ('--deadline', '999'), ('--warmup', '0')]:
            with self.subTest(option=option), mock.patch.object(sys, 'argv', ['driver', '--execute', option, value]), \
                 mock.patch.object(DRIVER.subprocess, 'run', side_effect=AssertionError('remote launch')), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    DRIVER.main()
                self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
