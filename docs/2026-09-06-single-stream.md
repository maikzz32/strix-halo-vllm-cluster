# Qwen3.8 Flash Next: Messungen vom 6. September 2026

Ziel ist eine einzelne laufende Antwort deutlich über dem bisherigen
ShareGPT-Ergebnis von etwa 48 Token/s. Die bisherigen Änderungen erreichen
dieses Ziel noch nicht. Alle hier verglichenen Modellläufe verwenden dieselben
INT4-Gewichte unter `/home/maik/qwen38_rest`, vier TP-Ranks und MTP3.

## Vergleichbare Modellmessungen

ShareGPT48, Seed 42, Temperatur 0, eine Anfrage gleichzeitig, reservierter
Kontext 262144. Die Output-Rate umfasst auch Prompt-Verarbeitung und
Request-Overhead. TPOT beschreibt die Zeit je Ausgabetoken nach dem ersten Token.

| Variante | Output-Token/s | Mittlere TPOT | Output-Tokens | MTP-Akzeptanz |
|---|---:|---:|---:|---:|
| Unveränderte Ausgangslaufzeit | 48,60 | 18,08 ms | 11839 | 54,14 % |
| MTP Local Argmax | 48,44 | 18,12 ms | 11839 | 54,14 % |
| QSA unsichtbare Tiles überspringen | 48,92 | 17,69 ms | 11839 | 54,14 % |
| QSA, aufgewärmter Wiederholungslauf | 49,56 | 17,75 ms | 11839 | 54,14 % |
| TP4 + Expert Parallelism + QSA + K640-Kernel | 46,22 | 19,09 ms | 11871 | 53,73 % |

Alle 48 Requests waren erfolgreich. Die vollständigen ShareGPT-Ausgaben der
QSA- und Argmax-Läufe einschließlich der QSA-Wiederholung sind identisch. Für die ursprüngliche Ausgangsmessung
liegen Benchmark-Log und Metriken vor; detaillierte Texte wurden erst im
nachfolgenden Argmax-Lauf gespeichert.

Sechs separate Streaming-Tests mit kurzen Prompts und jeweils 512 Ausgabetokens
liefern vor/nach QSA exakt gleiche Texte. Der Median der gepaarten Verbesserung
der Decode-Rate ist **2,01 %**: Coding etwa 66,9 auf 68,2, Deutsch 55,0 auf 56,2,
Analyse 64,3 auf 65,6 Token/s. Diese höheren Raten stammen von anderen Prompts
als ShareGPT und sind kein Sprung von 48 auf 68 Token/s durch die Änderung.

Der erste Coding-Request nach dem QSA-Neustart brauchte 2,27 Sekunden bis zum
ersten Token; spätere vergleichbare Requests etwa 0,3–0,4 Sekunden. Deshalb
werden Decode-Rate und Anlaufzeit getrennt dokumentiert. Der erste vollständige
QSA-ShareGPT-Lauf enthält diese Anlaufkosten. Die Wiederholung erreicht
49,56 Token/s bei 609 ms mittlerer TTFT und bestätigt den kleinen Gewinn.
Ein vollständig alternierender A/B/A-Nachweis liegt noch nicht vor.

Expert Parallelism war im vollständigen Modelltest langsamer. Die sechs kurzen
Streaming-Ausgaben änderten sich; ihre Decode-Rate sank gegenüber QSA ohne EP
im Median um 10,37 %. Alle 48 ShareGPT-Anfragen gelangen, aber bei veränderten
Texten und Tokenzahlen. Der dafür getestete K640-Kernel wurde auf allen vier
Nodes exakt auf seine gesicherte Originaldatei zurückgesetzt. Die laufende
Konfiguration verwendet wieder gewöhnliches TP4.

Eine kleine zusätzliche JSON-/Logikprobe bestand 11 von 12 Aufgaben. Das Modell
antwortete auf die Rekurrenz `x=2; x=3*x+1` nach drei Schritten mit 79 statt 67.
Diese Probe wurde nicht vor der Änderung ausgeführt; der Fehler kann damit
nicht der Änderung zugeordnet werden. Die erwartete Lösung bleibt unverändert.

## Ergebnisse isolierter Versuche

| Versuch | Befund | Konsequenz |
|---|---|---|
| QSA unsichtbare Score-Tiles | 32 GPU-Fälle bitgleich; kurzer Kontext etwa 95 auf 11,5 µs pro Scorer | Opt-in-Patch, vollständiger Modellvergleich oben |
| QSA Radix TopK | Bei kurzen Kontexten etwa doppelt so langsam; Gleichstände können andere Sets liefern | Nicht übernehmen |
| QSA vollständige Sparse Attention | Etwa 38–43 µs je Layer, zwölf Target-Layer | Kein großer Restengpass belegt |
| HC BF16 über BLAS | Warmer Cache täuschte Gewinn vor; wechselnde Gewichte langsamer | Nicht übernehmen |
| HC W8 | Verlustbehaftet; isolierte Einsparung grob 1,54 ms pro Verify, zusätzlicher Speicher | Nicht übernehmen |
| GDN Verify-Geometrie | Beste Variante spart insgesamt nur etwa 17 µs über 36 Layer | Nicht übernehmen |
| Routed MoE W1-Geometrie | Fünf Varianten in vier Routerverteilungen langsamer; 60 Numerikfälle bestanden | Bestehende Geometrie beibehalten |
| RCCL Tree/LL, vier Ranks | 8/16 KiB etwa 19/37 % langsamer; Korrektheit und Teardown bestanden | Automatische Auswahl beibehalten |
| MTP Local Argmax | Texte gleich, kein belegter Gesamtgewinn | Default bleibt aus |

K5/K6 können den Vorteil zusätzlicher akzeptierter Tokens durch mehr Draftarbeit
und den Verlust des Dense-INT4-Skinny-Pfads bei mehr als fünf Target-Tokens
aufbrauchen. Die gemessene positionsweise Akzeptanz ist 73,66/51,57/37,20 %.
Ein großer Gewinn allein durch längere Entwürfe ist daraus nicht belegt.

## Laufzeit und Wiederherstellung

Die native Steuerung wurde mit erfolgreichen Starts und Stopps auf allen vier
Nodes geprüft. 22 lokale Lifecycle-/Deployment-Tests bestehen. Logs, genaue
Startkonfigurationen und wiederherstellbare Deployment-Dateien liegen unter
`/home/maik/strix-halo-next` auf den Nodes.

Auf Node 1 ist die vorhandene Container-Laufzeit zusätzlich als lokales Image
`localhost/strix-halo-runtime:20260906-qsa-gated` gesichert:

```
718fdd7ceb8ad3fa889361f1cec132d966ed3eb21059fbf2898e2a51352cefb1
```

Dieses Image enthält die installierten Binärpakete und Python-Patches. Es ist
kein getesteter Ersatzcontainer und enthält keine Sicherung der extern
eingebundenen Modelle, Volumes oder Host-RDMA-Konfiguration. Das ursprüngliche
Container-Inspect und die Image-ID liegen auf Node 1 im `snapshots`-Verzeichnis.
Die vier ursprünglichen Container wurden nicht ersetzt.

Die QSA-Änderung wird mit `patches/qsa_invisible_tiles.py --apply` installiert
und erst durch `VLLM_QSA_SKIP_INVISIBLE_TILES=1` aktiv. Ohne Flag bleibt der
alte Rechenpfad erhalten. Der Patcher bewahrt eine SHA-gekennzeichnete Originaldatei
und unterstützt `--restore --apply`; Modellgewichte werden nicht verändert.

Weitere Untersuchung: vollständiges GPU-Kernelbudget. Externes ROCProfiler-Attach
und ein separater Start mit ROCProfiler funktionierten in dieser Laufzeit nicht;
die Fehlversuche wurden beendet. Ein kleiner integrierter Torch-Profiler-Test
zeichnet fünf GPU-Graph-Replays korrekt auf. Die anschließende Aufzeichnung von
16 Decode-Runden aus dem echten Modell enthält jedoch nur CPU-Ereignisse. Sie
belegt keine GPU-Zeitanteile. Der Parser lehnt diese unvollständige Aufzeichnung ab.
Der passende [PyTorch-Fehlerbericht #182373](https://github.com/pytorch/pytorch/issues/182373)
beschreibt fehlende GPU-Ereignisse in gestarteten Kindprozessen nach GPU-Initialisierung
im Elternprozess. Die lokale Reproduktion und ein begrenzter Workaround werden
geprüft. Profiling-Durchsatz wird nicht mit ungestörten Benchmark-Raten vermischt.
