#!/bin/bash
# graph_exp.sh <tag> <CG_MODE> <MTP_K> <CAP_SIZES|-> <EXTRA_E...>  -- ein Graph-Experiment auf dem mp-Verbund mit Waechter
T="$1"; export CG_MODE="$2" MTP_K="$3"; [ "$4" != "-" ] && export CAP_SIZES="$4"; shift 4; export EXTRA_E="$*"
echo "===== EXPERIMENT $T: CG_MODE=$CG_MODE MTP_K=$MTP_K CAP_SIZES=${CAP_SIZES:-} EXTRA_E=$EXTRA_E  ($(date +%H:%M:%S))"
/tmp/mp_all.sh stop >/dev/null
r=1; for h in 192.168.100.2 192.168.100.3 192.168.100.4; do scp -q -o BatchMode=yes -o StrictHostKeyChecking=no /tmp/start_mp_exp.sh $h:/tmp/; ssh -o BatchMode=yes -o StrictHostKeyChecking=no $h "chmod +x /tmp/start_mp_exp.sh; CG_MODE='$CG_MODE' MTP_K='$MTP_K' CAP_SIZES='${CAP_SIZES:-}' EXTRA_E='$EXTRA_E' setsid nohup /tmp/start_mp_exp.sh $r ray-worker $h $T >/dev/null 2>&1 < /dev/null &"; r=$((r+1)); done
setsid nohup /tmp/start_mp_exp.sh 0 ray-head 192.168.100.1 "$T" >/dev/null 2>&1 < /dev/null & sleep 3
bash /tmp/wait_ready.sh "/tmp/tp4_${T}_r0.log"; RC=$?
grep -iE "Graph capturing finished" "/tmp/tp4_${T}_r0.log" | tail -1 | cut -c1-140
if [ $RC -ne 0 ]; then echo "RESULT $T: KEIN READY (rc=$RC)"; bash /tmp/dump_stacks.sh "$T"; grep -vE "rocSHMEM|agent.cpp|^W0|amdsmi|E-001h" "/tmp/tp4_${T}_r0.log" | grep -E "Error|Traceback|shared memory" | tail -3 | cut -c1-200; /tmp/mp_all.sh stop >/dev/null; exit 1; fi
OUT=$(timeout 150 curl -s -m 140 localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"/home/maik/qwen38_flashnext\",\"messages\":[{\"role\":\"user\",\"content\":\"Nenne drei Hauptstädte in Europa und je einen Fluss. Kurz.\"}],\"max_tokens\":80,\"temperature\":0}")
if echo "$OUT" | python3 -c "import sys,json;d=json.load(sys.stdin);print('Probe ok:',d['usage']['completion_tokens'],'Tokens:',d['choices'][0]['message']['content'][-120:].replace(chr(10),' '))" 2>/dev/null; then
  echo "RESULT $T: LAEUFT -> Bench"; bash /tmp/bench_c1.sh --temperature 0 | sed -n "/Serving Benchmark Result/,/=====$/p" | grep -E "throughput|TPOT|Acceptance|duration|Failed"; exit 0
else echo "RESULT $T: HANG bei erster Anfrage"; bash /tmp/dump_stacks.sh "$T"; grep -vE "rocSHMEM|agent.cpp|^W0|amdsmi|E-001h" "/tmp/tp4_${T}_r0.log" | grep -E "shared memory|Error" | tail -2 | cut -c1-160; /tmp/mp_all.sh stop >/dev/null; exit 2; fi
