#!/usr/bin/env bash
# CPU-runnable smoke test for the vllm-gfx1151 image.
#
# Runs inside the container; needs no GPU and no RDMA hardware, so it works
# on GitHub Actions ubuntu-latest runners and on any node:
#   podman run --rm <image> smoke_test.sh
#
# Verifies the image contract: vLLM importable + CLI works, ray installed,
# pyyaml available, rdma-core/perftest tooling present. RDMA devices and the
# gfx1151 GPU are validated later by bench/ on real hardware, not here.

set -euo pipefail

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "== python / vllm =="
python3 -c "import vllm; print('vllm', vllm.__version__)" \
    || fail "import vllm"
vllm --help >/dev/null || fail "vllm --help"
echo "vllm --help OK"

echo "== torch (import only, no GPU) =="
python3 -c "import torch; print('torch', torch.__version__)" \
    || fail "import torch"

echo "== ray =="
ray --version || fail "ray --version"

echo "== pyyaml =="
python3 -c "import yaml; print('pyyaml', yaml.__version__)" || fail "import yaml"

echo "== rdma-core / perftest tooling =="
# Presence check only: on a CPU-only runner ibv_devices lists zero devices,
# which is fine. On the cluster nodes it must list the RoCE interfaces.
command -v ibv_devices >/dev/null || fail "ibv_devices not found (libibverbs-utils missing)"
ibv_devices || fail "ibv_devices failed to run"
command -v ib_write_bw >/dev/null || fail "ib_write_bw not found (perftest missing)"
# Report the rdma-core version for the record (must be >= v62 on the nodes).
if command -v rpm >/dev/null; then
    rpm -q rdma-core || true
fi

echo "SMOKE OK"
