#!/bin/bash
# dump_stacks.sh <tag> -- py-spy-Dumps (mit nativen Frames) der VLLM-Prozesse in ray-head (node1) und ray-worker (node2)
T="${1:-dump}"
D="for p in /proc/[0-9]*; do c=\$(tr \"\0\" \" \" < \$p/cmdline 2>/dev/null); case \"\$c\" in *VLLM::Worker*|*VLLM::EngineCore*) echo \"##### PID \${p#/proc/} \$c\"; timeout 60 py-spy dump --native --pid \${p#/proc/} 2>&1 | head -60;; esac; done"
{ echo "===== node1 ray-head ($(date +%H:%M:%S))"; podman exec ray-head bash -c "$D"
  echo "===== node2 ray-worker"; ssh -o BatchMode=yes -o StrictHostKeyChecking=no 192.168.100.2 "podman exec ray-worker bash -c '$D'"; } > "/tmp/stacks_$T.txt" 2>&1
echo "Dumps: /tmp/stacks_$T.txt ($(wc -l < /tmp/stacks_$T.txt) Zeilen)"
