#!/usr/bin/env bash
# Run on node1 from this repository; deploy scripts/config without restarting.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST=/home/maik/strix-halo-next
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)
HOSTS=(192.168.1.15 192.168.1.16 192.168.1.17 192.168.1.18)
FILES=("$REPO/scripts/native/cluster.py" "$REPO/scripts/native/node_runtime.py"
       "$REPO/scripts/native/deploy_release.py" "$REPO/bench/bench_stream.py"
       "$REPO/bench/quality_probe.py" "$REPO/bench/bench_sharegpt.sh"
       "$REPO/bench/compare_streams.py" "$REPO/models/cluster.runtime.json"
       "$REPO/patches/mtp_local_argmax.py" "$REPO/patches/qsa_invisible_tiles.py")
for file in "${FILES[@]}"; do
  test -f "$file" || { echo "Missing deployment input: $file" >&2; exit 1; }
done
STAGES=()
# Upload and validate every node before replacing any deployed file. Stages and
# exact prior files are retained for recovery, including interrupted deployment.
for host in "${HOSTS[@]}"; do
  remote="maik@$host"
  stage=$(ssh "${SSH_OPTS[@]}" "$remote" "mkdir -p '$DEST/deployments' '$DEST/results' && mktemp -d '$DEST/deployments/release.XXXXXXXX'")
  [[ "$stage" =~ ^/home/maik/strix-halo-next/deployments/release\.[a-zA-Z0-9]{8}$ ]] || { echo "Invalid release directory" >&2; exit 1; }
  STAGES+=("$stage")
  scp -q "${SSH_OPTS[@]}" "${FILES[@]}" "$remote:$stage/"
  ssh "${SSH_OPTS[@]}" "$remote" "python3 '$stage/deploy_release.py' validate --stage '$stage' --destination '$DEST'"
  echo "Staged $remote:$stage"
done
for i in "${!HOSTS[@]}"; do
  stage=${STAGES[$i]}
  ssh "${SSH_OPTS[@]}" "maik@${HOSTS[$i]}" "python3 '$stage/deploy_release.py' promote --stage '$stage' --destination '$DEST'"
done
echo "Deployed to $DEST on four nodes. Serving processes were not restarted."
