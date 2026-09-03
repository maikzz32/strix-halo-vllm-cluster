# wird per "podman exec -i <container> bash" ueber stdin ausgefuehrt: beendet alle vLLM-Prozesse im Container
for p in /proc/[0-9]*; do
  c=$(tr "\0" " " < "$p/cmdline" 2>/dev/null)
  case "$c" in *VLLM::*|*vllm*serve*|*Worker_TP*) kill -9 "${p#/proc/}" 2>/dev/null && echo "killed ${p#/proc/} ${c:0:40}";; esac
done
