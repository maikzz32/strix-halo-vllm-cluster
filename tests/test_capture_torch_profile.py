"""CPU-only profiling-client failure tests; no files, HTTP or GPU operations."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BENCH = Path(__file__).resolve().parents[1] / 'bench'
spec = importlib.util.spec_from_file_location('capture_profile_under_test', BENCH / 'capture_torch_profile.py')
capture = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(BENCH))
try:
    spec.loader.exec_module(capture)
finally:
    sys.path.pop(0)


class Response:
    status = 200

    def __init__(self, data=b'', lines=(), error=None):
        self.data, self.lines, self.error = data, lines, error

    def read(self):
        return self.data

    def __iter__(self):
        yield from self.lines
        if self.error is not None:
            raise self.error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class CaptureCleanupTests(unittest.TestCase):
    def run_capture(self, *, start_error=None, stop_error=None, stream_error=None):
        calls, artifacts = [], []

        def urlopen(request, timeout):
            url = request if isinstance(request, str) else request.full_url
            endpoint = url.removeprefix('http://127.0.0.1:8000')
            calls.append(endpoint)
            if endpoint == '/v1/models':
                return Response(json.dumps({'data': [{'id': 'test-model'}]}).encode())
            if endpoint == '/v1/chat/completions':
                event = {'choices': [{'delta': {'content': 'test answer'}}],
                         'usage': {'completion_tokens': 16 if stream_error else 2048}}
                lines = [('data: ' + json.dumps(event) + '\n').encode()]
                if stream_error is None:
                    lines.append(b'data: [DONE]\n')
                return Response(lines=lines, error=stream_error)
            if endpoint == '/start_profile':
                if start_error is not None:
                    raise start_error
                return Response()
            if endpoint == '/stop_profile':
                if stop_error is not None:
                    raise stop_error
                return Response()
            raise AssertionError(f'Unexpected HTTP request: {url}')

        def save(path, text, **kwargs):
            artifacts.append(json.loads(text))
            return len(text)

        error = None
        with patch.object(sys, 'argv', ['capture', '--output', 'memory-only.json']), \
             patch.object(capture.urllib.request, 'urlopen', side_effect=urlopen), \
             patch.object(Path, 'mkdir'), patch.object(Path, 'write_text', save), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                capture.main()
            except BaseException as caught:
                error = caught
        self.assertEqual(len(artifacts), 1, 'Failure must still preserve the workload artifact')
        self.assertEqual(calls.count('/start_profile'), 1)
        self.assertEqual(calls.count('/stop_profile'), 1)
        return error, artifacts[0]

    def test_failed_start_still_attempts_stop_and_preserves_original_error(self):
        original = TimeoutError('start response lost after partial activation')
        error, artifact = self.run_capture(start_error=original)
        self.assertIs(error, original)
        self.assertEqual(artifact['profile_start_error'], repr(original))
        self.assertEqual(artifact['profile_calls'][-1]['endpoint'], '/stop_profile')

    def test_cleanup_failure_fails_after_saving_complete_workload(self):
        cleanup = OSError('stop response lost')
        error, artifact = self.run_capture(stop_error=cleanup)
        self.assertIsInstance(error, RuntimeError)
        self.assertIs(error.__cause__, cleanup)
        self.assertTrue(artifact['workload_complete'])
        self.assertEqual(artifact['profile_cleanup_error'], repr(cleanup))

    def test_cleanup_failure_does_not_replace_broken_stream_error(self):
        original, cleanup = OSError('stream disconnected'), OSError('stop failed')
        error, artifact = self.run_capture(stream_error=original, stop_error=cleanup)
        self.assertIs(error, original)
        self.assertFalse(artifact['stream_done'])
        self.assertFalse(artifact['workload_complete'])
        self.assertEqual(artifact['workload_error'], repr(original))
        self.assertEqual(artifact['profile_cleanup_error'], repr(cleanup))

    def test_complete_workload_and_cleanup_succeed_without_duplicate_calls(self):
        error, artifact = self.run_capture()
        self.assertIsNone(error)
        self.assertTrue(artifact['workload_complete'])
        self.assertEqual([call['endpoint'] for call in artifact['profile_calls']],
                         ['/start_profile', '/stop_profile'])
        self.assertNotIn('profile_cleanup_error', artifact)


if __name__ == '__main__':
    unittest.main()
