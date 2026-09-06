# HIP-Graph: lokaler CPU-Callback und 20-KiB-Kopien

Der isolierte Test auf Node18 bestätigt, dass ein nativer CPU-Callback im
installierten HIP-Runtime als Graph-Knoten erfasst und korrekt wiederholt wird.
Mit je 20 KiB D2H/H2D und einer exakten BF16-Negation benötigt der gesamte
Graph im Median **15,98 µs**. Das ist noch kein RDMA-All-reduce; siehe den
separaten [CPU-RDMA-Vergleich](../docs/2026-09-07-cpu-rdma.md).

| Graph | Knoten | Wall-Median | HIP-Event-Median |
|---|---|---:|---:|
| Nativer Host-Callback | 1 Host | 7,89 µs | 7,72 µs |
| 20 KiB D2H → 20 KiB H2D | 2 Kopien | 6,23 µs | 6,12 µs |
| D2H → CPU-Negation → H2D | 2 Kopien + 1 Host | 15,98 µs | 15,83 µs |

Ausführung: **6. September 2026, 22:19:16.385–22:19:16.662 UTC**
(7. September, 00:19 Uhr Berlin), exklusiv auf Node18 im erhaltenen
`ray-worker`, Modell-API währenddessen idle. Der Prozess endete mit Exit 0;
danach waren keine zugehörigen Testprozesse mehr vorhanden.

Der [Messdatensatz](../bench/records/2026-09-07-hip-host-callback.json) enthält
alle fünf Timing-Samples, Quellcode-SHA256, UTC-Zeiten und numerischen Prüfungen.
Der Binary-SHA256 wurde beim Lauf nicht erfasst und ist ausdrücklich `null`;
bekannt sind Größe (32.408 Bytes) und Änderungszeit
`2026-09-06T22:12:27.335706Z`. Die Bibliothek wurde erst beim freigegebenen
GPU-Test geladen. Ihre vorherige CPU-Kompilation dauerte 1,07 Sekunden.

## Messverfahren und Grenzen

Der [C++-Harness](../tests/hip_graph_host_callback.cpp) verwendet nur native
HIP-Host-APIs und wird vom [Python-Wrapper](../tests/bench_hip_graph_host_callback.py)
über `ctypes` aufgerufen. Der Callback ruft weder HIP noch Python auf und
benötigt weder GIL, Speicherallokation, Netzwerkzugriff noch blockierende Warteoperationen.
Stream, Graph, Events, gepinnter Hostspeicher und Device-Puffer werden vor dem
Timing angelegt. Der Callback-Kontext bleibt bis zur Stream-Synchronisation
und anschließenden Freigabe gültig.

Vier anfängliche Einzel-Replays prüfen jeweils geänderte, nichtnull BF16-Daten.
Danach folgen fünf Samples mit je 50 Replays. Vor jedem Sample wird die Quelle
erneut außerhalb des Graphen geändert und die Ausgabe mit Nullen überschrieben.
Kopierkontrolle und Negation sind für alle 10.240 Werte bitgenau, endlich und
ohne Abweichung zur FP32-Identität beziehungsweise Negation. Die beiden
Callback-Fälle hatten exakt 254 erwartete Aufrufe und keine fehlerhaften
Eingabewerte. Der reine Host-Callback prüft einen Zähler und `Zähler+42`, ohne
20 KiB zu bearbeiten.

Der erste Host-only-Sample lag bei 17,41 µs und bleibt im Datensatz; die anderen
vier lagen bei 7,79–7,96 µs. Die Tabelle verwendet den Median aller fünf.
Explizit allokierter gepinnter Host- plus Device-Speicher: höchstens 60 KiB,
zusätzlich ein 20-KiB-CPU-Ausgabepuffer. Das schließt den HIP-Kontext nicht ein.

Gemessen wird der Durchsatz seriell geordneter Graph-Replays einschließlich
Callback-Abschluss. RDMA, eine echte Summenreduktion, vier beteiligte Ranks,
deren Synchronisationsversatz und Einbettung in vLLM/Torch wurden nicht getestet.
Die 16 µs dürfen daher nicht als vollständige All-reduce-Latenz oder bereits
erreichter Modellgewinn verwendet werden.

## Wiederholen im vorhandenen Container

Nur während einer exklusiv freigehaltenen GPU auf Node18 ausführen. Dieser
Test ist keine Änderung der Serving-Konfiguration und lädt kein Modell.
Aus einem Repository-Checkout die zwei Quelldateien übertragen:

```bash
scp tests/hip_graph_host_callback.cpp tests/bench_hip_graph_host_callback.py \
  maik@192.168.1.18:/home/maik/strix-halo-next/tests/
```

Auf Node18 mit dem vorhandenen Compiler bauen; dies ist CPU-Arbeit:

```bash
podman exec ray-worker timeout --signal=TERM --kill-after=5s 40s \
  g++ -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include \
  /home/maik/strix-halo-next/tests/hip_graph_host_callback.cpp \
  -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 \
  -o /tmp/libstrix_hip_graph_host_callback.so
podman exec ray-worker sha256sum /tmp/libstrix_hip_graph_host_callback.so
```

Erst mit freier GPU den begrenzten Test starten:

```bash
date -u +%FT%TZ
podman exec ray-worker timeout --signal=TERM --kill-after=5s 45s \
  python3 /home/maik/strix-halo-next/tests/bench_hip_graph_host_callback.py \
  --library /tmp/libstrix_hip_graph_host_callback.so \
  --output /tmp/hip-host-callback-repeat.json
date -u +%FT%TZ
```

`--replays` ist auf 5–100 und `--samples` auf 1–5 begrenzt. Ein HIP-Capture-/API-
oder Numerikfehler liefert einen Fehlerstatus und ein JSON mit dem konkreten
API-Fehler. Ein harter Timeout kann das Schreiben eines vollständigen JSON
verhindern; nur Exit 0 **und** `status=passed` gelten als erfolgreicher Lauf.
