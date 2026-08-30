# librccl.so for gfx1151 (RCCL over RoCEv2)

Upstream RCCL is fragile on gfx1151
(<https://github.com/ROCm/rocm-systems/issues/5229>). This directory holds a
drop-in `librccl.so` for image builds with `RCCL_IMPL=custom`.

## Which RCCL to use — decision order

1. **Test the stock ROCm RCCL first.** Its gfx1151 status may have improved;
   check <https://github.com/ROCm/rocm-systems/issues/5229> before assuming
   the custom build is still needed. Validate on the real 4-node cluster
   with a multi-node vLLM `tp4` run (Ray + RCCL over RoCEv2), not just a
   single-node smoke test.
2. If stock fails, use the known-good build from
   `kyuz0/rocm-systems`, branch `gfx1151-rccl`
   (<https://github.com/kyuz0/rocm-systems/tree/gfx1151-rccl>).

## How to obtain the library

`librccl.so` is **not committed to git** (binary, large). It is built by the
`build-rccl.yml` GitHub Actions workflow (see `.github/workflows/`), which
compiles the `gfx1151-rccl` branch for `gfx1151` and publishes the artifact.
Place the resulting `librccl.so` in this directory before building the image
with `RCCL_IMPL=custom`; the Dockerfile copies it over the stock library.

## Runtime prerequisites

RCCL over RDMA fails cryptically without these (see ansible/ and docker/):

- host `memlock` ulimit unlimited, otherwise RCCL aborts with
  `ibv_reg_mr_iova2 ... Cannot allocate memory`
- containers started with `--device /dev/infiniband`, `--ulimit memlock=-1`,
  `--network host`

## Relevant RCCL/NCCL env knobs for RoCEv2 on gfx1151

| Variable | Value | Why |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | cluster NIC (e.g. `eno1`) | bootstrap/control interface |
| `NCCL_IB_GID_INDEX` | `1` | RoCEv2 GID (v2 index) |
| `NCCL_NET_GDR_LEVEL` | `0` | no GPUDirect RDMA on the gfx1151 iGPU |
| `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES` | `1` | keep Ray from remapping devices under RCCL |
| `NCCL_IB_HCA` | (optional) | pin the RoCE HCA if auto-detection picks the wrong one |
| `NCCL_DEBUG` | `INFO` (debugging) | verbose transport selection logs |

Caution: jumbo frames (MTU 9000) broke one real 4-node setup — stay on MTU
1500 unless 9000 is validated end-to-end on the actual switches/NICs.

`amd_iommu=off` is 5–12 % faster but may break RDMA; it is parametrized
(`iommu_mode`) and A/B-tested via `bench/iommu_ab.sh` — do not hardcode a
choice here.
