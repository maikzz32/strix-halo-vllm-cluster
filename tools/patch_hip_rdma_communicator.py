"""Prepare the exact two opt-in hook sites; rejects unknown source revisions."""
import hashlib


def patched(source, expected_sha256):
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('CudaCommunicator source differs from reviewed revision')
    text = source.decode()
    init_anchor = '    def _log_all_reduce_backend_selection(self) -> None:\n'
    call_anchor = '    def all_reduce(self, input_):\n'
    if text.count(init_anchor) != 1 or text.count(call_anchor) != 1:
        raise ValueError('CudaCommunicator hook anchors are not unique')
    text = text.replace(init_anchor,
        '        from .strix_hip_rdma_hook import make_backend\n'
        '        self._strix_hip_rdma = make_backend(self)\n\n' + init_anchor, 1)
    text = text.replace(call_anchor, call_anchor +
        '        if self._strix_hip_rdma is not None:\n'
        '            result = self._strix_hip_rdma.try_all_reduce(input_)\n'
        '            if result is not None:\n'
        '                return result\n', 1)
    compile(text, 'cuda_communicator.py', 'exec')
    return text.encode()
