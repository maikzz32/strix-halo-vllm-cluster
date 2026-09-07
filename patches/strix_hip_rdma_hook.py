"""Opt-in experimental adapter. Default production dispatch remains unchanged."""
import os


def make_backend(communicator):
    if os.environ.get('STRIX_HIP_RDMA_ENABLE', '0') != '1':
        return None
    if not communicator.unique_name.startswith('tp:') or communicator.world_size != 4:
        return None
    from hip_rdma_backend import create_if_enabled
    from validate_rccl_ring4 import validate
    reference = communicator.pynccl_comm
    if reference is None or reference.disabled:
        raise RuntimeError('Experimental ring4 backend requires the active PyNccl communicator')
    env = dict(os.environ)
    env['STRIX_HIP_RDMA_RUN_ID'] = env['STRIX_CLUSTER_RUN_ID']
    env['STRIX_HIP_RDMA_DEVICE'] = {
        'enp197s0f3np3':'rocep197s0f3', 'enp197s0f1np1':'rocep197s0f1'
    }[env['NCCL_SOCKET_IFNAME']]
    backend = create_if_enabled(communicator.cpu_group, communicator.device,
        communicator.unique_name, communicator.ranks, environment=env,
        reference_validator=lambda b: validate(b, reference, communicator.cpu_group))
    print('STRIX_HIP_RDMA_ENABLED rank=%d profile=%s' % (
        communicator.rank, backend.parity_proof['profile_sha256']), flush=True)
    return backend
