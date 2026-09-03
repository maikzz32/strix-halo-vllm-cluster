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
