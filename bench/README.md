# bench/ — Benchmark-Harness

Dieses Harness entscheidet, welches Parallel-Profil (`tp4`, `pp4`, `tp2pp2`,
`ep`, `solo`) pro Modell Cluster-Default wird. Es misst Single-Stream-tok/s
(Concurrency 1) und Aggregat-Durchsatz (Concurrency 4–32) über eine
Modell × Profil × Prompt-Länge × Concurrency-Matrix.

## Voraussetzungen

- Läuft **auf node1** — entweder im Cluster-Container oder auf dem Host gegen
  den Container (Host-Network, Endpoint via `BENCH_BASE_URL`).
- Nur `python3` + `pyyaml` + `requests` (alles im Image vorhanden).
- Bevorzugt das CLI `vllm bench serve`; falls nicht verfügbar, greift ein
  eingebauter OpenAI-kompatibler Lastgenerator (Streaming `/v1/completions`)
  als Fallback — dessen Prompt-Länge ist nur approximiert (kein Tokenizer).
- Setzt `scripts/serve.sh <modell> <profil>` voraus (startet den Server,
  reicht die aktuelle Umgebung inkl. `NCCL_*` an vLLM durch).
- Teardown erfolgt per `pkill -f 'vllm serve'` lokal und per SSH auf den
  Workern (`BENCH_WORKER_NODES`, Default `node2 node3 node4`).
  **TODO:** auf `scripts/cluster_down.sh` umstellen, sobald vorhanden —
  `pkill` erwischt ggf. keine Ray-Worker mit anderer Kommandozeile.

## Benchmark-Matrix laufen lassen

```bash
# ganze Matrix: alle nicht-blockierten Modelle, alle erlaubten Profile,
# Concurrency 1/4/8/16/32, Prompt-Längen 512/4096/32768
python3 bench/run_matrix.py

# gezielt: ein Modell, zwei Profile, kurze Zellen
python3 bench/run_matrix.py \
    --models qwen36-35b-a3b --profiles tp4,ep \
    --concurrencies 1,16 --prompt-lengths 512

# nach Abbruch fortsetzen (bereits gemessene Zellen werden übersprungen)
python3 bench/run_matrix.py --resume bench/results/20260101T120000Z.json
```

Wichtige Optionen: `--output-len` (Default 128 Tokens), `--base-url`
(Default `http://127.0.0.1:8000`, Env `BENCH_BASE_URL`),
`--health-timeout` (Default 20 min — der erste Boot JIT-et Triton ~170 s),
`--output` (eigene Ergebnisdatei).

Der erste Serve-Vorgang eines Modells lädt die Gewichte und kompiliert
Triton-Kernels — Geduld, solange der Triton-Cache nicht persistiert ist.

## Ergebnisse

Ein JSON-Record pro Zelle (JSONL) in `bench/results/<utc-timestamp>.json`:

| Feld | Bedeutung |
|---|---|
| `model`, `profile` | Registry-Key, Parallel-Profil |
| `prompt_len`, `output_len`, `concurrency`, `num_prompts` | Zell-Parameter |
| `ttft_ms`, `itl_ms` | Time-to-First-Token / Inter-Token-Latency (Mittelwert) |
| `output_toks` | Aggregat-Ausgabe-tok/s; bei `concurrency=1` = Single-Stream-Rate |
| `tool` | `vllm-bench` oder `fallback` |
| `sanity` | G1: Korrektheitsprobe — `ok`, je Prompt `finish_reason`, `unique_ratio`, `sha256` |
| `acceptance_len` | G3: MTP-Acceptance (Δaccepted/Δdraft aus `/metrics`); nur wenn der Server die Counter exponiert |
| `image` | G4: Image-Tag aus `VLLM_IMAGE` (nur wenn gesetzt) |
| `env` | G4: relevante `VLLM_GFX1X_*`-/`NCCL_*`-Variablen aus der Messumgebung |
| `sampling` | G4: `temperature`/`ignore_eos` der Messung (`temperature: null` = vllm-bench-CLI-Default) |
| `prompt_len_exact` | G4: `false` beim Fallback-Generator (Prompt-Länge nur approximiert) |
| `error` | gesetzt, wenn die Zelle/das Serve fehlschlug |

## Qualitäts-Gates (Geschwindigkeit nur mit Korrektheit)

- **G1 Output-Sanity:** einmal pro (Modell, Profil) nach der ersten Zelle
  gehen 4 fixe kurze Prompts (de/en) mit `temperature=0` und **ohne**
  `ignore_eos` gegen den laufenden Server. Geprüft werden
  `finish_reason == "stop"`, nicht-leerer Output und Unique-Token-Ratio
  > 0.5 (Repetitions-Detektor); pro Output wird der SHA256 abgelegt. Das
  Ergebnis steckt als `sanity` in jeder Zelle des Profils.
- **G3 MTP/Spec-Acceptance:** exponiert der Server
  `vllm:spec_decode_num_accepted_tokens_total` und
  `vllm:spec_decode_num_draft_tokens_total` unter `/metrics`, werden die
  Counter vor/nach jeder Zelle gelesen; `acceptance_len` = Δaccepted/Δdraft.
- **G4 Messvertrag:** jeder Record trägt `image`, `env`, `sampling` und
  `prompt_len_exact`. `report.py` warnt, wenn Zellen desselben Modells
  Tools (`vllm-bench` vs. `fallback`) oder Verträge mischen, und zeigt
  Sanity/Acceptance im Ergebnis-Abschnitt.

Serve-Logs pro Modell/Profil liegen daneben als
`bench/results/serve_<timestamp>_<modell>_<profil>.log`.

## Report erzeugen

```bash
python3 bench/report.py 'bench/results/*.json'
python3 bench/report.py bench/results/20260101T120000Z.json --output report.md
```

Pro Modell und Prompt-Länge eine Tabelle Profil × Concurrency (tok/s),
der Gewinner für Single-Stream (C=1) und für Aggregat ist **fett**. Der
Abschnitt **Spark-Vergleich** unter jedem Modell setzt die Cluster-Bestwerte
ins Verhältnis zu den externen DGX-Spark-Referenzwerten (Quelle:
[maci0/qwen3.8-flash-next-spark](https://github.com/maci0/qwen3.8-flash-next-spark))
und markiert jede Metrik als `BEATEN` oder `NOT-YET`. Der Abschnitt
**Toolbox-C-Vergleich** vergleicht pro Concurrency-Stufe (C=1/8/32) die
beste Cluster-Zelle als tok/s pro Request (Aggregat/C) gegen die offiziellen
kyuz0-Toolbox-C-Werte (Qwen3.8-27B, 1 Node, MTP: 43,55 / 16,84 / 7,46).
Die Referenzwerte am
Ende sind externe Literaturwerte (andere Hardware/Interconnect/Quantisierung)
— Strategie, Begründung und Akzeptanzkriterium stehen in
[docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

## Prefix-Cache-Probe

```bash
python3 bench/prefix_probe.py --model /home/maik/qwen38_ablit --prefix-tokens 2048
```

APC (Automatic Prefix Caching) ist in vLLM V1 defaultmäßig an; dieses Skript
misst den Nachweis und den Nutzen: TTFT einer langen, geteilten Prefix kalt
vs. warm. Gemessen 2026-08-31 (qwen38-27b-ablit, tp4, MTP): kalt 5517 ms →
warm Ø 1958 ms bei ~2K-Token-Prefix (2,8×).


## kpool-Sparse-Indexer: Triton-Lane (GLM-5.3-Flash, gfx1151)

```bash
# im Container auf node1 (ray-head), kein Server nötig:
python3 bench/kpool_triton_validate.py --lane /tmp/kpool_triton_lane.py
python3 bench/kpool_triton_validate.py --installed   # gepatchtes Modul
```

Validiert die Triton-Hotpaths aus `patches/runtime_glm53_kpool_triton.py`
(Standalone-Spiegel: `bench/kpool_triton_lane.py`) gegen die Torch-Lane
(v1.4) als Goldene Referenz und misst beide: Decode-Paged-Logits ~20–37×
schneller, Prefill-Logits ~4,5–5,4×, Gather bit-exakt, Top-k/Expand-Indizes
exakt identisch, Logits max. rel. Diff ~3e-7. Vertrag: Cache
`[nb, 32, 132]` uint8 (fp8 + fp32-Scale, 16×16-Preshuffle), block_table in
288-Pool-Einheiten (num_states = 1152/4). Laufzeit-Gate:
`VLLM_GFX1X_KPOOL_TRITON=1` (Default aus).


## A/B: Ethernet/TCP vs. RDMA (RoCE)

```bash
bench/compare_eth_vs_rdma.sh qwen36-35b-a3b tp4
```

Läuft dieselbe Matrix (Concurrency 1,16) zweimal: einmal mit RCCL über RoCE
(`NCCL_IB_GID_INDEX=1`), einmal erzwungen über TCP-Sockets
(`NCCL_IB_DISABLE=1`), und druckt das Delta in Prozent.

**Voraussetzung:** `podman exec` erbt die Client-Umgebung nicht —
`scripts/serve.sh` muss `NCCL_IB_DISABLE` in seiner `-e`-Liste an den
Container durchreichen. Das Skript prüft das vorab und bricht sonst mit
einer konkreten Anleitung ab.

## A/B: IOMMU-Modus (braucht Reboots)

```bash
bench/iommu_ab.sh qwen36-35b-a3b tp4
```

Checklisten-Skript: zeigt den aktuellen Kernel-Cmdline-Modus
(`amd_iommu=off` vs. `iommu=pt`), prüft die RDMA-Sichtbarkeit
(`ibv_devinfo`) und fährt eine Benchmark-Zelle mit einem Multi-Node-Profil —
das beantwortet die offene Frage, ob `amd_iommu=off` RDMA auf dieser NIC
bricht (Fehlerbild: `ibv_reg_mr` / „Cannot allocate memory“ beim RCCL-Init).
Danach druckt es die exakten `grubby`-Kommandos zum Umschalten inkl.
Reboot auf allen Nodes. Einmal pro Modus ausführen, dann vergleichen:

```bash
python3 bench/report.py 'bench/results/*iommu*.json'
```
