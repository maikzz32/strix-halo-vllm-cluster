#!/usr/bin/env bash
# Guided A/B test: amd_iommu=off vs. iommu=pt.
#
# Background: amd_iommu=off is reportedly 5-12% faster on Strix Halo, but the
# open question is whether it breaks RDMA (RoCEv2) on this NIC. This script is
# a CHECKLIST because each mode needs a reboot on all nodes:
#
#   1. shows the current kernel cmdline and detected IOMMU mode,
#   2. runs an RCCL-over-RDMA functional smoke (TP=4 serve across all nodes
#      exercises RCCL init; ibv_reg_mr / "Cannot allocate memory" failures
#      indicate broken RDMA) plus one benchmark cell,
#   3. prints the exact grubby commands to switch modes and reboot.
#
# Run once per mode, then compare the two result files with bench/report.py.
#
# Usage: bench/iommu_ab.sh [model] [profile]
set -euo pipefail

MODEL=${1:-qwen36-35b-a3b}
PROFILE=${2:-tp4}   # multi-node profile so RCCL/RDMA is actually used

step() { echo; echo "### $*"; }

step "1/3 Current kernel cmdline"
CMDLINE=$(cat /proc/cmdline)
echo "$CMDLINE"
if grep -qw 'amd_iommu=off' <<<"$CMDLINE"; then
    MODE="amd_iommu_off"
elif grep -qw 'iommu=pt' <<<"$CMDLINE"; then
    MODE="iommu_pt"
else
    MODE="default"
fi
echo "detected mode: $MODE"

step "2/3 Functional smoke: RDMA device visible + one benchmark cell"
if command -v ibv_devinfo >/dev/null 2>&1; then
    ibv_devinfo -l || true
else
    echo "ibv_devinfo not found (rdma-core not installed on this host?)"
    echo "Run this script on node1 or inside the cluster container."
fi

echo "Reminder: host needs 'ulimit -l unlimited' / memlock unlimited, otherwise"
echo "RCCL fails with 'ibv_reg_mr_iova2 ... Cannot allocate memory'."
ulimit -l || true

OUT="bench/results/$(date -u +%Y%m%dT%H%M%SZ)_iommu_${MODE}_${MODEL}_${PROFILE}.json"
echo "results -> $OUT"
python3 bench/run_matrix.py \
    --models "$MODEL" --profiles "$PROFILE" \
    --concurrencies 1,16 --prompt-lengths 512 \
    --output "$OUT"

cat <<EOF

If the serve step failed with ibv_reg_mr / "Cannot allocate memory" or RCCL
init errors, RDMA is broken under the current IOMMU mode ($MODE) - record
this in the run notes; it answers the open question for this NIC.

step 3/3: Switch modes (requires reboot, on ALL nodes)
======================================================

Switch TO amd_iommu=off:
  sudo grubby --update-kernel=ALL --remove-args="iommu"
  sudo grubby --update-kernel=ALL --args="amd_iommu=off"
  sudo systemctl reboot

Switch BACK to iommu=pt:
  sudo grubby --update-kernel=ALL --remove-args="amd_iommu"
  sudo grubby --update-kernel=ALL --args="iommu=pt"
  sudo systemctl reboot

All nodes at once via ansible (after editing the args line as above):
  ansible all -i ansible/inventory.yaml -b -m shell \\
      -a 'grubby --update-kernel=ALL --remove-args="iommu" && grubby --update-kernel=ALL --args="amd_iommu=off"'
  ansible all -i ansible/inventory.yaml -b -m reboot

Then re-run this script in the new mode and compare:
  python3 bench/report.py 'bench/results/*iommu*.json'
EOF
