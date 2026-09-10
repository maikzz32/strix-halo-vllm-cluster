"""Uninstalled, opt-in TP4 BF16/10240 SUM backend prototype.

No monkeypatch or automatic initialization occurs on import. A future hook calls
``create_if_enabled`` unconditionally on all members of the existing CPU TP
group, after its usual communicator initialization and before capture. The
existing registered vllm.all_reduce/fake implementation remains unchanged:

    result = backend.try_all_reduce(input_) if backend is not None else None
    if result is not None:
        return result
    # Existing CudaCommunicator backend dispatch follows.

Only use inside an owned experimental worker with identical collective order
and serialized graph replays on all four ranks. Callback/partial-enqueue errors
terminate that worker; they never fall back after a collective has started.
The numerical mode is REQUIRED explicitly. Neither mode is assumed equivalent
to installed RCCL, nor approved for serving by this source-only prototype.

Environment: STRIX_HIP_RDMA_ENABLE=1, STRIX_HIP_RDMA_LIBRARY=/absolute/lib.so,
STRIX_HIP_RDMA_RUN_ID=<32 lowercase hex>, STRIX_HIP_RDMA_DEVICE=<local HCA>,
STRIX_HIP_RDMA_MASTER=<rank0 IPv4>, STRIX_HIP_RDMA_MODE=fp32|bf16.
Optional STRIX_HIP_RDMA_PORT=29891, STRIX_HIP_RDMA_TIMEOUT_MS=5000,
STRIX_HIP_RDMA_DESCRIPTORS=1024 (40 MiB pinned staging plus 200 KiB MR).

OFF performs no CDLL load, HIP allocation or verbs initialization. A CPU-group
configuration vote still prevents inconsistent enable flags across TP ranks.
Native resources and CDLL remain strongly rooted until explicit teardown AFTER
graphs are destroyed, or until process exit. No __del__/atexit cleanup is used.
An explicit reference_validator callable is mandatory when enabled. It must
compare this candidate against the CURRENT PyNccl communicator, with 16--32
fresh finite/cancellation-rich banks, 10240 values per rank, eager AND graph.
Historical ring logs or a numerical-mode name do not satisfy this gate.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import ipaddress
import os
from pathlib import Path
import re

_LIVE_BACKENDS = []


class Params(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('abi_version', 'rank', 'world_size')]
    _fields_ += [(name, C.c_char_p) for name in ('run_id', 'device', 'master_ipv4')]
    _fields_ += [(name, C.c_uint32) for name in ('master_port', 'gid_index', 'timeout_ms')]


def _fail_stop(message):
    data = ('HIP_RDMA_BACKEND_FATAL: ' + str(message) + '\n').encode(errors='replace')[:2048]
    try:
        os.write(2, data)
    finally:
        os._exit(86)


def _settings(environment, group_name):
    """Read-only preparation; called only after the all-rank enable vote."""
    library = Path(environment['STRIX_HIP_RDMA_LIBRARY'])
    if not library.is_absolute() or not library.is_file():
        raise ValueError('STRIX_HIP_RDMA_LIBRARY must name an existing absolute file')
    run_id = environment['STRIX_HIP_RDMA_RUN_ID']
    if not re.fullmatch('[0-9a-f]{32}', run_id):
        raise ValueError('STRIX_HIP_RDMA_RUN_ID must be 32 lowercase hexadecimal characters')
    mode = environment['STRIX_HIP_RDMA_MODE']
    if mode not in ('fp32', 'bf16', 'ring4'):
        raise ValueError('An explicit fp32, bf16 or calibrated ring4 numerical mode is required')
    master = str(ipaddress.IPv4Address(environment['STRIX_HIP_RDMA_MASTER']))
    device = environment['STRIX_HIP_RDMA_DEVICE']
    if not device or len(device.encode()) >= 64 or '\0' in device:
        raise ValueError('Invalid local verbs device name')
    port = int(environment.get('STRIX_HIP_RDMA_PORT', '29891'))
    timeout = int(environment.get('STRIX_HIP_RDMA_TIMEOUT_MS', '5000'))
    slots = int(environment.get('STRIX_HIP_RDMA_DESCRIPTORS', '1024'))
    gid = int(environment.get('STRIX_HIP_RDMA_GID_INDEX', '1'))
    if not (1024 <= port <= 65535 and 100 <= timeout <= 120000 and
            128 <= slots <= 8192 and 0 <= gid <= 255):
        raise ValueError('Backend port/deadline/descriptor/GID bounds are invalid')
    transport_id = hashlib.sha256((run_id + ':' + group_name).encode()).hexdigest()[:32]
    return dict(library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                run_id=transport_id, mode=mode, master=master, device=device,
                port=port, timeout=timeout, slots=slots, gid=gid)


class HipRdmaBackend:
    def __init__(self, torch, cpu_group, rank, device, settings):
        self.torch, self.cpu_group, self.rank = torch, cpu_group, rank
        self.device, self.settings = device, settings
        self.validated = False
        self._calibrating = False
        self.handle = C.c_void_p()
        self.library = C.CDLL(settings['library'])
        self.library.hip_rdma_backend_abi_version.restype = C.c_uint
        self.library.hip_rdma_backend_params_size.restype = C.c_size_t
        if (self.library.hip_rdma_backend_abi_version() != 1 or
                self.library.hip_rdma_backend_params_size() != C.sizeof(Params)):
            raise RuntimeError('HIP backend C/Python ABI mismatch')
        create = self.library.hip_rdma_backend_create
        create.argtypes = [C.POINTER(Params), C.c_int, C.c_int, C.c_uint,
                           C.POINTER(C.c_void_p), C.c_char_p, C.c_size_t]
        create.restype = C.c_int
        enqueue = self.library.hip_rdma_backend_enqueue
        enqueue.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t,
                            C.c_void_p, C.c_char_p, C.c_size_t]
        enqueue.restype = C.c_int
        close = self.library.hip_rdma_backend_close
        close.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_size_t]
        close.restype = C.c_int
        # Root ownership BEFORE create, including partial-create failures.
        _LIVE_BACKENDS.append(self)

    def initialize(self):
        s = self.settings
        params = Params(1, self.rank, 4, s['run_id'].encode(), s['device'].encode(),
                        s['master'].encode(), s['port'], s['gid'], s['timeout'])
        message = C.create_string_buffer(512)
        with self.torch.cuda.device(self.device):
            if self.torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Backend initialization must precede graph capture')
            rc = self.library.hip_rdma_backend_create(C.byref(params), self.device.index,
                {'fp32': 0, 'bf16': 1, 'ring4': 2}[s['mode']], s['slots'], C.byref(self.handle), message, len(message))
        if rc:
            raise RuntimeError('Native backend create failed: ' + message.value.decode(errors='replace'))

    def try_all_reduce(self, tensor, *, op=None, stream=None):
        """None means unsupported BEFORE enqueue; any enqueue error is fatal."""
        torch = self.torch
        if (not tensor.is_cuda or tensor.device != self.device or tensor.dtype != torch.bfloat16 or
                tensor.numel() != 10240 or not tensor.is_contiguous() or
                (op is not None and op != torch.distributed.ReduceOp.SUM)):
            return None
        if not self.handle.value:
            _fail_stop('supported collective used after backend shutdown')
        if not self.validated and not self._calibrating:
            _fail_stop('supported collective used before active-communicator numerical validation')
        with torch.cuda.device(self.device):
            if stream is None:
                stream = torch.cuda.current_stream(self.device)
            if stream.device != self.device:
                _fail_stop('collective stream belongs to a different device')
            with torch.cuda.stream(stream):
                output = torch.empty_like(tensor)
                # Preserve allocator lifetime for asynchronous D2H/H2D on this stream.
                tensor.record_stream(stream)
                output.record_stream(stream)
                message = C.create_string_buffer(512)
                rc = self.library.hip_rdma_backend_enqueue(self.handle, tensor.data_ptr(),
                    output.data_ptr(), tensor.numel(), stream.cuda_stream, message, len(message))
            if rc:
                _fail_stop('enqueue rejected: ' + message.value.decode(errors='replace'))
        return output

    def close_after_graphs_destroyed(self):
        """Caller must release ALL associated graph objects before this call."""
        if not self.handle.value:
            return
        message = C.create_string_buffer(512)
        rc = self.library.hip_rdma_backend_close(self.handle, 1, message, len(message))
        if rc:
            _fail_stop('teardown failed; retain borrowed arena: ' + message.value.decode(errors='replace'))
        self.handle = C.c_void_p()
        _LIVE_BACKENDS.remove(self)


def create_if_enabled(cpu_group, device, unique_name, ranks, *, environment=None,
                      reference_validator=None):
    """Collective startup-only decision; never call lazily from all_reduce.

    Intended exact group is this deployment's TP ranks [0,1,2,3]. DP/EP/world
    groups and other TP memberships remain on existing backends. The hook must
    be called consistently for each TP group even when locally OFF.

    reference_validator(backend) runs collectively and returns a proof dict with
    banks (16--32), elements_per_rank=10240, exact_eager=True, exact_graph=True,
    and validated_against_active_pynccl=True. Its closure must use the existing
    communicator, fresh input banks and immutable input resets. It must compare
    uint16 output bits and all four ranks, and raise/return failure on mismatch.
    No implementation is silently supplied here: without it enable fails before
    native loading. Model integration must provide and independently test it.
    """
    if not unique_name.startswith('tp:') or list(ranks) != [0, 1, 2, 3]:
        return None
    import torch
    import torch.distributed as dist
    if dist.get_world_size(cpu_group) != 4 or dist.get_backend(cpu_group) != 'gloo':
        raise RuntimeError('Experimental backend requires the four-rank CPU Gloo TP group')
    env = os.environ if environment is None else environment

    def vote(value):
        result = [None] * 4
        dist.all_gather_object(result, value, group=cpu_group)
        return result

    enabled = env.get('STRIX_HIP_RDMA_ENABLE', '0') == '1'
    flags = vote(enabled)
    if not any(flags):
        return None  # No native library load, allocation or verbs operation.
    if not all(flags):
        raise RuntimeError('TP ranks disagree on STRIX_HIP_RDMA_ENABLE')
    rank = dist.get_rank(cpu_group)
    backend = None
    try:
        if not callable(reference_validator):
            raise ValueError('Active PyNccl eager+graph numerical validator is required')
        settings = _settings(env, unique_name)
        normalized_device = torch.device(device if not isinstance(device, int) else f'cuda:{device}')
        if normalized_device.type != 'cuda' or normalized_device.index is None:
            raise ValueError('An explicit local CUDA/HIP device ordinal is required')
        # HCA and file path can differ per host; content/network/numerics must agree.
        shared = {k: v for k, v in settings.items() if k not in ('device', 'library')}
        metadata = {'ok': True, 'shared': shared}
    except Exception as exc:
        metadata = {'ok': False, 'error': repr(exc)}
    metadata_all = vote(metadata)
    if not all(item['ok'] for item in metadata_all):
        raise RuntimeError('TP backend preflight failed: ' + repr(metadata_all))
    if any(item['shared'] != metadata_all[0]['shared'] for item in metadata_all):
        raise RuntimeError('TP ranks disagree on backend binary, network or numerical settings')
    # Resolve the native ABI collectively before any rank enters verbs create.
    try:
        backend = HipRdmaBackend(torch, cpu_group, rank, normalized_device, settings)
        loaded = {'ok': True}
    except Exception as exc:
        loaded = {'ok': False, 'error': repr(exc)}
    loaded_all = vote(loaded)
    if not all(item['ok'] for item in loaded_all):
        _fail_stop('collective native ABI preflight failed: ' + repr(loaded_all))
    try:
        backend.initialize()
        initialized = {'ok': True}
    except Exception as exc:
        initialized = {'ok': False, 'error': repr(exc)}
    initialized_all = vote(initialized)
    if not all(item['ok'] for item in initialized_all):
        _fail_stop('collective native initialization failed: ' + repr(initialized_all))
    try:
        backend._calibrating = True
        proof = reference_validator(backend)
        passed = (isinstance(proof, dict) and isinstance(proof.get('banks'), int)
                  and 16 <= proof['banks'] <= 32 and proof.get('elements_per_rank') == 10240
                  and proof.get('exact_eager') is True and proof.get('exact_graph') is True
                  and proof.get('validated_against_active_pynccl') is True)
        validated = {'ok': passed, 'proof': proof}
    except Exception as exc:
        validated = {'ok': False, 'error': repr(exc)}
    finally:
        backend._calibrating = False
    validated_all = vote(validated)
    if not all(item['ok'] for item in validated_all):
        _fail_stop('active PyNccl numerical parity failed: ' + repr(validated_all))
    backend.validated = True
    return backend
