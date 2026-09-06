"""CPU-only preparation/archival checks; no GPU imports or SSH allowed."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = load('rccl_bf16_reference')
DRIVER = load('run_rccl_microbench')
CPU = load('cpu_rdma_reference')


class ReferenceTests(unittest.TestCase):
    def test_bank_bits_and_source_hash(self):
        banks, digest = REF.input_banks(2)
        self.assertEqual(digest, hashlib.sha256((ROOT / 'tools/cpu_rdma_reference.py').read_bytes()).hexdigest())
        self.assertEqual(len(banks), 32)
        self.assertEqual({len(b) for b in banks}, {20480})
        self.assertNotEqual(banks[0], banks[1])
        for bank in (0, 1, 17, 31):
            for index in (0, 1, 2, 123, 10239):
                self.assertEqual(struct.unpack_from('<H', banks[bank], index*2)[0], CPU.input_bits(2, bank, index))

    def test_existing_launcher_prepares_owned_bounded_graph(self):
        stream = io.StringIO()
        with mock.patch.object(sys, 'argv', ['run', '--bf16-reference', '--dry-run']), \
             mock.patch.object(DRIVER.subprocess, 'run', side_effect=AssertionError('SSH forbidden')), \
             contextlib.redirect_stdout(stream):
            self.assertEqual(DRIVER.main(), 0)
        result = json.loads(stream.getvalue())
        self.assertTrue(result['bf16_reference'])
        self.assertEqual(result['sizes_bytes'], [20480])
        self.assertEqual(len(result['commands']), 4)
        for rank, node, command in result['commands']:
            self.assertIn('--deadline 45', command[-1])
            self.assertIn('--graph', command[-1])
            self.assertIn(result['run_id'], command[-1])
            self.assertIn('55s', command[-1])
            self.assertIn('--kill-after=5s', command[-1])

    def test_invalid_combinations_never_launch(self):
        for args in [['--nodes', '15,17'], ['--sizes', '4096'], ['--profile-smoke']]:
            with self.subTest(args=args), mock.patch.object(sys, 'argv', ['run', '--bf16-reference'] + args), \
                 mock.patch.object(DRIVER.subprocess, 'run', side_effect=AssertionError('SSH forbidden')), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    DRIVER.main()
                self.assertEqual(exc.exception.code, 2)

    def test_binary_extraction_and_cross_rank_rejection(self):
        run_id = 'a' * 32; generator = 'b' * 64
        payload = struct.pack('<H', 0x3f80) * (32 * 10240)
        status = [{'rank':r, 'node':15+r, 'exit_code':0} for r in range(4)]
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp)
            def write(rank, data):
                metadata = {'kind':'reference_metadata', 'rank':rank, 'run_id':run_id, 'world_size':4,
                    'generator_sha256':generator, 'input_bank_sha256':['c'*64]*32}
                record = {'kind':'reference_payload','rank':rank,'run_id':run_id,'generator_sha256':generator,
                    'values':10240,'banks':32,'byte_order':'little','finite':True,'all_gpu_inputs_checked':True,
                    'input_resets':33,'bank0_repeat_equal':True,'output_sha256':hashlib.sha256(data).hexdigest(),
                    'output_bank_sha256':[hashlib.sha256(data[i*20480:(i+1)*20480]).hexdigest() for i in range(32)],
                    'payload_b64':base64.b64encode(data).decode()}
                complete = {'kind':'complete','rank':rank,'run_id':run_id,'graph_reset':True,'communicator_destroyed':True}
                (output / f'rank{rank}-node{15+rank}.log').write_text('\n'.join(json.dumps(r) for r in (metadata,record,complete)))
            for rank in range(4): write(rank, payload)
            self.assertTrue(DRIVER.extract_bf16_reference(output, status, run_id, generator))
            self.assertEqual((output / 'rank0-node15.bf16.bin').read_bytes(), payload)
            write(3, struct.pack('<H', 0x4000) + payload[2:])
            self.assertFalse(DRIVER.extract_bf16_reference(output, status, run_id, generator))
            summary = json.loads((output / 'reference-summary.json').read_text())
            self.assertFalse(summary['all_rank_outputs_equal'])


if __name__ == '__main__':
    unittest.main()
