#!/bin/bash
# soak.sh [sekunden] [parallel] -- Dauerlast gegen den laufenden Server, zaehlt Fehler, prueft /health
S="${1:-600}"; P="${2:-4}"; END=$(( $(date +%s) + S )); mkdir -p /tmp/soak; rm -f /tmp/soak/*
worker() { ok=0; bad=0; while [ $(date +%s) -lt $END ]; do
  code=$(curl -s -m 120 -o /tmp/soak/r$1 -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"/home/maik/qwen38_flashnext\",\"messages\":[{\"role\":\"user\",\"content\":\"Schreibe $((RANDOM%5+2)) Sätze über das Thema Nummer $RANDOM.\"}],\"max_tokens\":64,\"temperature\":$([ $((RANDOM%2)) -eq 0 ] && echo 0 || echo 0.7)}")
  if [ "$code" = "200" ]; then ok=$((ok+1)); else bad=$((bad+1)); fi; done; echo "$ok $bad" > /tmp/soak/w$1; }
for i in $(seq 1 $P); do worker $i & done
hf=0; while [ $(date +%s) -lt $END ]; do sleep 30; curl -s -m 10 localhost:8000/health -o /dev/null || hf=$((hf+1)); done; wait
OK=0; BAD=0; for f in /tmp/soak/w*; do read a b < $f; OK=$((OK+a)); BAD=$((BAD+b)); done
echo "SOAK ${S}s x$P: ok=$OK fehler=$BAD health-timeouts=$hf"
