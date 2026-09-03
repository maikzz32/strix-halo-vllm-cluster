# scripts/mp — Betrieb des Ray-freien mp-Verbunds (TP4 über 4 Nodes)

Alle Skripte laufen auf node1 (Head) und erwarten sich selbst unter `/tmp/` (so wurden sie am 03.09.2026 entwickelt);
nach einem Reboot ist `/tmp` leer — `scp scripts/mp/*.sh maik@node1:/tmp/` genügt. Modell/Argumente sind auf
`qwen38-flash-next-int4` festgelegt (siehe `start_mp_exp.sh`); Env-Schalter: `CG_MODE`, `MTP_K`, `CAP_SIZES`, `EXTRA_E`.

| Skript | Zweck |
|---|---|
| `start_prod.sh [tag]` | Produktion: Reste beenden, auf >100 GiB freien GPU-Speicher je Node warten, mp-Verbund mit Graphen + MTP k=3 + RCCL-Envs starten, READY-Zeit messen, Probe |
| `mp_all.sh stop` | alle vLLM-Prozesse auf allen 4 Nodes beenden (auch verwaiste `VLLM::Worker`, die GPU-Speicher halten) |
| `kill_vllm.sh` | wird per `podman exec -i <container> bash < kill_vllm.sh` in jedem Container ausgeführt |
| `start_mp_exp.sh <rank> <container> <ip> <tag>` | ein Rang des Verbunds (Follower automatisch `--headless`) |
| `wait_ready.sh <log>` | wartet auf `/v1/models` (10 min) oder Prozessende |
| `graph_exp.sh <tag> <CG_MODE> <MTP_K> <CAP_SIZES|-> <-e ...>` | ein Experiment mit Wächter, Probe, Bench; bei Hang py-spy-Dumps (`dump_stacks.sh`) |
| `bench_c1.sh [--temperature 0]` | ShareGPT, seed 42, 48 Prompts, c=1 |
| `soak.sh [s] [parallel]` | Dauerlast mit Fehlerzählung und Health-Checks |

Gemessen (03.09.2026, dev-rocm10): READY 234 s mit warmen Caches (415 s kalt), 41,0 tok/s greedy / 33,8 Modell-Default.

Falle: verwaiste `VLLM::Worker` werden an PID 1 des Containers (`ray start --block`) umgehängt; stirbt so ein Kind, fährt Rays Subprozess-Monitor den Container herunter (`Exited (1)`, Log „received SIGTERM“). `mp_all.sh stop` startet die Container danach automatisch wieder; für den mp-Verbund ist Ray selbst nicht nötig, nur der laufende Container.

Bilder/262k (03.09.2026): `MAX_LEN=262144 MM_LIMIT='{"image": 2, "video": 0}' EXTRA_ARGS='--max-num-batched-tokens 8192 --max-num-seqs 8 --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":1003520}}' start_prod.sh` → READY 314 s, KV 3,44 M Tokens (13× 262k). Falle: JEDE Option muss an alle Ränge gehen — läuft nur Rang 0 mit Bildern, hängt die Encoder-Profilierung in den TP-Kollektiven des Vision-Towers, während die Follower schon im nächsten Warmup sind (py-spy: BusyWaitSignal in torch_sdpa_wrapper).
