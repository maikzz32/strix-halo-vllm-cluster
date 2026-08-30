# Runbook — Fehlerkatalog und Betrieb (4× Strix Halo, gfx1151, vLLM/Ray/RCCL über RoCEv2)

Kurzreferenz für den Cluster-Betrieb. Reihenfolge: bekannte Fehlerbilder mit
Symptom → Ursache → Behebung, danach Routine-Checks. Build-Themen (Dockerfile,
`RCCL_IMPL`, Patch-Schicht) sind hier nur insoweit erwähnt, wie sie zur
Diagnose nötig sind.

## Routine-Checks

- Cluster hochfahren: `scripts/cluster_up.sh` (idempotent, mit Pre-Flight:
  memlock, `/dev/infiniband`, `ibv_devices`).
- Gesundheit: `scripts/status.sh` (Ray-Status, GPU-Speicher je Node,
  RDMA-Link-State).
- Modell starten: `scripts/serve.sh <modell> <profil>` — Profile und Status
  kommen aus `models/registry.yaml`.
- Cluster runterfahren: `scripts/cluster_down.sh`.
- Compile-Cache wärmen und verteilen: `scripts/warmup.sh <modell> <profil>`
  auf node1, dann `scripts/dist_cache.sh <modell> <profil>` (siehe Eintrag 9).

## Fehlerkatalog

### 1. `ibv_reg_mr_iova2 ... Cannot allocate memory` (RCCL bricht beim Start ab)

- **Symptom:** vLLM/Ray-Worker sterben beim Multi-Node-Start; im Log die
  RCCL/IB-Verben-Fehlermeldung `ibv_reg_mr_iova2` mit `Cannot allocate memory`.
- **Ursache:** `memlock`-Limit des Prozesses ist nicht `unlimited`. RCCL
  registriert Puffer beim RDMA-Stack (pinned memory); mit dem Default-Limit
  (typ. 8 MiB) schlägt die Registrierung fehl.
- **Behebung:**
  - Host: `ulimit -l` prüfen (muss `unlimited` sein). Persistent über
    `/etc/security/limits.d/99-memlock.conf` mit `* - memlock unlimited`
    (danach neu einloggen; für systemd-Services `LimitMEMLOCK=infinity`).
  - Container: `--ulimit memlock=-1` (in `cluster_up.sh` bereits gesetzt).
  - `cluster_up.sh` prüft das Limit im Pre-Flight und bricht mit Hinweis ab.

### 2. Triton: `invalid device ordinal`

- **Symptom:** vLLM stürzt beim Kompilieren der Triton-Kernel mit
  `invalid device ordinal` ab, obwohl die GPU sichtbar ist.
- **Ursache:** Tritons Geräte-Enumeration passt nicht zur gfx1151-Umgebung
  (u. a. Wechselspiel mit `ROCR_VISIBLE_DEVICES` unter Ray — deshalb setzt
  `serve.sh` `RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`).
- **Behebung:** Patch 10 aus `patches/` muss im Image aktiv sein (Patch-Schicht
  prüft fail-closed beim Build). Falls der Fehler nach einem Image-Update
  wieder auftaucht: zuerst prüfen, ob der Patch gegen die neue vLLM-/Triton-
  Version noch greift.

### 3. Hängender Start / Deadlock beim HIP-Graph-Capture

- **Symptom:** vLLM hängt beim Warmup/Capture-Phase komplett (kein Fortschritt,
  keine Fehlermeldung), typischerweise direkt nach „Capturing CUDA graphs“ o. ä.
- **Ursache:** HIP-Graph-Capture deadlocked auf gfx1151 (vllm#32180).
- **Behebung:** `--enforce-eager` ist Pflicht. Es steht in
  `models/registry.yaml` unter `defaults.extra_args` und wird von `serve.sh`
  automatisch angehängt — beim manuellen `vllm serve` nicht vergessen.
  Möglicher Ausstieg auf dem Prüfstand: `cudagraph_mode NONE` (siehe
  Eintrag 11).

### 4. Durchsatz-Einbruch / Timeouts nach MTU-Umstellung auf 9000

- **Symptom:** Nach Umstellung des RoCE-Fabrics auf Jumbo Frames (MTU 9000):
  schlechterer Durchsatz, RCCL-Timeouts oder hängende Kollektivoperationen.
- **Ursache:** MTU 9000 hat in einem realen 4-Node-Setup die Stabilität
  gebrochen (inkonsistente MTU auf Pfad-Segmenten, PFC/ECN nicht sauber
  abgestimmt).
- **Behebung:** Sofort zurück auf MTU 1500 (Default) — auf **allen** Nodes und
  Switch-Ports. MTU 9000 erst wieder aktivieren, nachdem der Pfad Ende-zu-Ende
  validiert ist (`ping -M do -s 8972 <ziel>` je Node-Paar, dann perftest/
  `ib_write_bw` mit 9000). Danach RCCL-Messung wiederholen.

### 5. RCCL: `hipErrorInvalidKernelFile`

- **Symptom:** RCCL-Init oder erste Kollektivoperation schlägt fehl mit
  `hipErrorInvalidKernelFile`.
- **Ursache:** Upstream-RCCL liefert keine/inkompatible Kernel für gfx1151
  (ROCm/rocm-systems#5229).
- **Behebung:** Image mit dem bekannt guten Custom-RCCL bauen:
  `RCCL_IMPL=custom` (Branch `gfx1151-rccl` aus kyuz0/rocm-systems). Im laufenden
  Container prüfen, welche RCCL geladen wird (Build-Protokoll /
  `ldd` auf die RCCL-Bibliothek). Nach einem ROCm-Upgrade im Basis-Image
  erneut verifizieren, dass nicht wieder Stock-RCCL im Pfad liegt.

### 6. Tool-Call-/Reasoning-Parser-Fehler bei TP=4

- **Symptom:** Tool-Calls werden falsch/abgeschnitten geparst oder das
  Reasoning-Format bricht — nur unter Tensor-Parallelismus (TP=4), single-node
  ok. Bekannte Upstream-Bugs: vllm#51914, vllm#48931, vllm#53831.
- **Ursache:** Parser-Bugs in vLLM bei verteilten Workern.
- **Behebung:**
  - Parser-Konfiguration kommt aus `models/registry.yaml` (`parsers:`) —
    dort anpassen, nicht im Aufruf.
  - Workaround: Profil wechseln (`solo` oder `pp4` statt `tp4`) oder auf ein
    `:dev`-Image mit vLLM main testen (Fix landet oft zuerst dort).
  - Vor Bug-Meldung prüfen, ob eines der drei genannten Issues den Fall
    bereits abdeckt.

### 7. Erster Start dauert ~170 s länger (Triton-JIT)

- **Symptom:** Jeder Start des Containers / Modells dauert Minuten in der
  Kompilierphase, obwohl frühere Starts schnell waren.
- **Ursache:** Triton-Cache nicht persistiert — nach Container-Neustart wird
  der komplette JIT erneut kompiliert (~170 s).
- **Behebung:** Cache als Volume mounten. `cluster_up.sh` setzt
  `TRITON_CACHE_DIR=/triton-cache` mit Volume `triton-cache`. Wer Container
  manuell startet: Volume nicht vergessen; Cache-Verzeichnis nicht in
  ephemeren Container-Layer legen. Auf den *Worker*-Nodes zusätzlich den
  warmen Snapshot verteilen (Eintrag 9), sonst zahlt jeder Node den JIT
  einmal selbst.

### 8. Modell lässt sich nicht serven: Status `blocked` in der Registry

- **Symptom:** `serve.sh` bricht ab mit „model ... is blocked upstream“ und
  einer Liste von Tracking-URLs.
- **Ursache:** Upstream-Support fehlt (Beispiel: GLM-5.3-Flash ist
  gfx950-gated, siehe `models/registry.yaml`). Das ist Absicht — kein Bug im
  Cluster.
- **Behebung:** Die Tracking-URLs verfolgen; erst serven, wenn die PRs auf
  vLLM main sind und das Modell in der Registry auf `dev`/`supported` steht.
  Status `dev` verlangt ein `:dev`-Image (`VLLM_IMAGE` entsprechend setzen).

### 9. Worker-Nodes kompilieren beim ersten Serve trotzdem (kalter JIT pro Node)

- **Symptom:** Erster Start eines Modells dauert auf node2–4 genauso lange
  wie ein Kaltstart auf node1 (gemessener Präzedenz: 294 s kalt vs. 82 s mit
  warmem Cache), obwohl der `triton-cache`-Volume überall gemountet ist.
- **Ursache:** Das Volume ist pro Node lokal — der JIT-Cache wird auf jedem
  Node einzeln aufgebaut. Die Triton-Cache-Keys enthalten die GPU-Architektur,
  daher trifft ein Snapshot von einem gfx1151-Node auf **allen** identischen
  gfx1151-Nodes (gleiche triton-/torch-Version und gleiches Env vorausgesetzt).
- **Behebung:**
  - Einmal auf node1 wärmen: `scripts/warmup.sh <modell> <profil>` — startet
    `serve.sh` einmal mit der exakten Registry-Config (bewusst **ohne**
    abweichende Args: der `torch_compile_cache`-Hash deckt die gesamte Config
    ab, ein abweichendes Warmup würde den Hash forken), wartet auf `/health`,
    feuert eine kurze Generierung und legt den Snapshot unter
    `/var/lib/vllm/warm-cache/<modell>/<profil>/` ab. Idempotent: bei
    unverändertem Fingerprint (Image + Registry-Env/Args) wird übersprungen,
    `--force` erzwingt.
  - Verteilen: `scripts/dist_cache.sh <modell> <profil>` — rsync (Fallback
    scp) in den Volume-Mountpoint von node2–4 (per `podman volume inspect`
    aufgelöst, Volume wird fehlendfalls angelegt), danach
    sha256-Verifikation je Node.
  - `TRITON_STORE_BINARY_ONLY=1` im Image-Env schrumpft den Cache um ~77 %
    (schnelleres Verteilen).

### 10. Ein Node kompiliert neu, die anderen nicht (Compile-Cache-Hash-Fork)

- **Symptom:** Nach dem Verteilen des Snapshots startet genau ein Node wieder
  langsam und kompiliert erneut — ohne Fehlermeldung, still.
- **Ursache:** Der `torch_compile_cache`-Hash von vLLM deckt **alle**
  Config-Sektionen ab. Jede Drift — abweichendes `VLLM_*`-Env, geänderte
  CLI-Args, eine abweichende `scripts/defaults.env` auf einem Node, ein
  anderes Image-Tag — forkt den Hash und erzwingt einen Recompile.
- **Behebung:** Die Registry (`models/registry.yaml`) ist die einzige Quelle.
  Drift finden: `scripts/lib/registry.py show <modell>` und die effektiven
  Env/Args auf den Nodes vergleichen (Image-Tag, `defaults.env`, manuelle
  `VLLM_*`-Exports). Angleichen, dann `warmup.sh --force` + `dist_cache.sh`
  neu laufen lassen. Keine per-Node-Overrides pflegen.

### 11. Experiment: `cudagraph_mode NONE` statt `--enforce-eager`

- **Hintergrund:** `--enforce-eager` ist Pflicht, weil das HIP-Graph-**Capture**
  auf gfx1151 deadlocked (vllm#32180, Eintrag 3).
  `--compilation-config '{"cudagraph_mode":"NONE"}'` lässt nur das Capture
  weg und behält die Inductor-Fusion (Präzedenz vllm#44988) — potenziell
  gleiche Stabilität bei höherem Durchsatz. Gilt als unbestätigt, bis der
  Soak auf diesem Cluster sauber durchläuft.
- **Ausführen:** `bench/cudagraph_ab.sh [modell] [profil]` auf node1.
  Ablauf: (1) Eager-Baseline via `run_matrix.py --concurrencies 1,16`,
  (2) Registry temporär auf NONE umgeschaltet (Backup + Trap-Restore —
  Rollback ist automatisch), (3) 10-minütiger Soak (`SOAK_SECONDS`) mit
  Dauerlast und Hang-Detektor, (4) NONE-Matrix mit denselben Zellen.
  Ergebnisse in `bench/results/`, Vergleich per
  `python3 bench/report.py 'bench/results/*cudagraph_*.json'`.
- **Hang-Bild:** `/health` antwortet nicht mehr (drei aufeinanderfolgende
  Timeouts), der vLLM-Prozess steht, `dmesg` zeigt neue `amdgpu`-Reset- oder
  Ring-Timeout-Zeilen. Das Skript bricht dann laut ab, schreibt
  `bench/results/<ts>_cudagraph_NONE_HANG_dmesg.log` und markiert NONE als
  FAILED; danach kann ein Reboot des betroffenen Nodes nötig sein (GPU im
  Reset-Zustand).
- **Rollback:** kein manueller Eingriff nötig — die Registry wird per Trap in
  jedem Fall auf `--enforce-eager` zurückgesetzt. Wer manuell testet:
  `--compilation-config`-Eintrag entfernen, `--enforce-eager` wieder setzen.
  Bei Erfolg promoten: in `models/registry.yaml` unter `defaults.extra_args`
  `--enforce-eager` durch `--compilation-config` +
  `'{"cudagraph_mode":"NONE"}'` ersetzen und Eintrag 3 anpassen.

## Weitere Hinweise

- **`amd_iommu=off`:** 5–12 % schneller, kann aber RDMA brechen — der Trade-off
  ist ungelöst. Über `iommu_mode` in den Ansible-Group-Vars parametriert,
  A/B-Test via `bench/iommu_ab.sh`. Nach Änderung der Kernel-Cmdline
  (`amdgpu.gttsize=126976 ttm.pages_limit=32505856`, ggf. `amd_iommu=off`)
  Reboot nötig.
- **RCCL-Umgebung:** `NCCL_IB_GID_INDEX=1` (RoCEv2), `NCCL_NET_GDR_LEVEL=0`
  (kein GPU-Direct-Pfad auf der iGPU), `NCCL_SOCKET_IFNAME` aus dem RoCE-Iface —
  alle gesetzt durch `serve.sh`; bei manuellen Starts selbst exportieren.
