#!/bin/bash
# mp_all.sh <stop|start TAG SCRIPT> -- stoppt/startet den mp-Verbund auf allen 4 Nodes, raeumt Prozessleichen weg
stop_all() {
  pkill -9 -f "[s]bin/vllm serve" 2>/dev/null; podman exec -i ray-head bash < /tmp/kill_vllm.sh
  for h in 192.168.100.2 192.168.100.3 192.168.100.4; do
    scp -q -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/kill_vllm.sh $h:/tmp/kill_vllm.sh
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h 'pkill -9 -f "[s]bin/vllm serve" 2>/dev/null; podman exec -i ray-worker bash < /tmp/kill_vllm.sh' | sed "s/^/$h /"
  done
  sleep 8
}
start_all() {  # start_all TAG SCRIPT
  r=1; for h in 192.168.100.2 192.168.100.3 192.168.100.4; do scp -q -o BatchMode=yes -o StrictHostKeyChecking=no "$2" $h:/tmp/; ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h "chmod +x $2; setsid nohup $2 $r ray-worker $h $1 >/dev/null 2>&1 < /dev/null &"; r=$((r+1)); done
  setsid nohup "$2" 0 ray-head 192.168.100.1 "$1" >/dev/null 2>&1 < /dev/null & sleep 3
}
case "$1" in stop) stop_all;; start) start_all "$2" "$3";; esac
