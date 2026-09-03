#!/bin/bash
# start_prod.sh [tag] -- Produktion (mp-Verbund, Graphen, MTP k=3) sauber starten: erst alle Reste beenden,
# auf freien GPU-Speicher (>100 GiB) je Node warten, dann starten und READY-Zeit messen.
T="${1:-prod}"
/tmp/mp_all.sh stop >/dev/null 2>&1
FREE="import torch;f,t=torch.cuda.mem_get_info();print(int(f/2**30))"
for i in $(seq 1 24); do ok=1; line=""
  for h in local 192.168.100.2 192.168.100.3 192.168.100.4; do C=ray-worker; [ $h = local ] && C=ray-head
    if [ $h = local ]; then g=$(podman exec $C python3 -c "$FREE" 2>/dev/null | tail -1); else g=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h "podman exec $C python3 -c '$FREE' 2>/dev/null | tail -1"); fi
    line="$line $h:${g:-?}GiB"; [ "${g:-0}" -ge 100 ] 2>/dev/null || ok=0; done
  echo "frei:$line"; [ $ok -eq 1 ] && break; sleep 10; done
[ $ok -eq 1 ] || { echo "GPU-Speicher nicht frei -> Abbruch"; exit 1; }
export CG_MODE=FULL_DECODE_ONLY MTP_K=3 EXTRA_E="-e VLLM_GFX1X_SPEC_CUDAGRAPH=0 -e NCCL_GRAPH_MIXING_SUPPORT=1 -e NCCL_LAUNCH_MODE=GROUP"
r=1; for h in 192.168.100.2 192.168.100.3 192.168.100.4; do ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h "CG_MODE=$CG_MODE MTP_K=$MTP_K EXTRA_E='$EXTRA_E' setsid nohup /tmp/start_mp_exp.sh $r ray-worker $h $T >/dev/null 2>&1 < /dev/null &"; r=$((r+1)); done
t0=$(date +%s); setsid nohup /tmp/start_mp_exp.sh 0 ray-head 192.168.100.1 "$T" >/dev/null 2>&1 < /dev/null & sleep 3
if bash /tmp/wait_ready.sh "/tmp/tp4_${T}_r0.log"; then echo "READY nach $(( $(date +%s)-t0 )) s"; else echo "START FEHLGESCHLAGEN"; exit 1; fi
grep -E "torch.compile took|Loading weights took|Graph capturing finished" "/tmp/tp4_${T}_r0.log" | grep "Worker_TP0" | tail -3 | cut -c40-160
curl -s -m 60 localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"/home/maik/qwen38_flashnext\",\"messages\":[{\"role\":\"user\",\"content\":\"Sag kurz hallo.\"}],\"max_tokens\":30,\"temperature\":0}" | python3 -c "import sys,json;d=json.load(sys.stdin);print('Probe ok:',d['choices'][0]['message']['content'][-60:].replace(chr(10),' '))"
