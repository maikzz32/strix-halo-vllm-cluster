#!/usr/bin/env python3
"""Render a Markdown benchmark report from bench/results/*.json files.

Reads one or more result files (JSONL as written by run_matrix.py; shell
globs or --output for a file) and prints, per model and prompt length, a
profile x concurrency table of output tok/s. The best single-stream cell
(concurrency 1) and the best aggregate cell are highlighted in bold.

The report text is German (repo docs convention); code and comments are
English. Records may carry the quality-gate fields `sanity` (G1 output
probe), `acceptance_len` (G3 MTP acceptance) and the G4 measurement
contract (image/env/sampling/prompt_len_exact); the report shows them and
warns when cells of one model mix benchmark tools or contracts. DGX Spark
numbers below are EXTERNAL reference points from published literature, not
measured on this cluster.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict

# External reference points from DGX Spark literature - NOT measured on this
# cluster, shown only for orientation and as the official benchmark target
# (see docs/PERFORMANCE.md section c). All rows: Qwen3.8-Flash-Next class on
# 2x DGX Spark (GB10, CX7 RoCE rail), source maci0/qwen3.8-flash-next-spark.
# single_tps / aggregate_tps are numeric midpoints for ratio computation;
# the *_display strings preserve the exact wording of the legacy table.
SPARK_REFERENCES = [
    {
        "label": "Qwen3.8-Klasse (MoE), vLLM TP=2, BF16",
        "single_tps": 31.1,      # measured 31.1 tok/s, MTP 3, 512K YaRN
        "aggregate_tps": 74.3,   # measured 74.3 tok/s @ 8 concurrent
        "single_display": "~31 tok/s",
        "aggregate_display": "~74 tok/s",
        "source_url": "https://github.com/maci0/qwen3.8-flash-next-spark",
    },
    {
        "label": "Qwen3.8-Klasse (MoE), SGLang NVFP4 TP=2",
        "single_tps": 42.0,      # midpoint of the measured 40-44 range
        "aggregate_tps": 150.0,  # midpoint of the measured 148-155 range
        "single_display": "40-44 tok/s",
        "aggregate_display": "~150 tok/s",
        "source_url": "https://github.com/maci0/qwen3.8-flash-next-spark",
    },
    {
        "label": "Qwen3.8-Klasse (MoE), llama.cpp GGUF + MTP (1x Spark)",
        "single_tps": 32.1,      # 27.4 -> 32.1 tok/s (+17%) via MTP
        "aggregate_tps": None,   # single-node run, no aggregate published
        "single_display": "~32 tok/s",
        "aggregate_display": "-",
        "source_url": "https://github.com/maci0/qwen3.8-flash-next-spark",
    },
]

# Official kyuz0 "Toolbox C" reference - NOT measured by us. Single Strix
# Halo node, Qwen3.8-27B, MTP enabled; values are per-request tok/s at the
# given concurrency (NOT aggregate). The control run is a repeat measurement
# of the same setup and shows the scatter of the official numbers (~10% at
# C=8), so BEATEN verdicts close to the bar are within noise.
TOOLBOX_REFERENCE = {
    "label": "Qwen3.8-27B, offizielle Toolbox C (1 Node, MTP)",
    "per_request_tps": {1: 43.55, 8: 16.84, 32: 7.46},
    "control_run": {1: 43.44, 8: 15.10, 32: 7.99},
    "source_url": "https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes",
}


def load_records(paths):
    records = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern)) or ([pattern] if glob.os.path.exists(pattern) else [])
        if not matches:
            print(f"warning: no files match '{pattern}'", file=sys.stderr)
            continue
        for path in matches:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" not in rec and rec.get("output_toks") is not None:
                        records.append(rec)
    return records


def contract_fingerprint(rec):
    """G4 measurement-contract fingerprint of a record. None for legacy
    records that predate the contract fields (image/env/sampling/
    prompt_len_exact)."""
    keys = ("image", "env", "sampling", "prompt_len_exact")
    if not any(k in rec for k in keys):
        return None
    return (rec.get("image"),
            json.dumps(rec.get("env") or {}, sort_keys=True),
            json.dumps(rec.get("sampling") or {}, sort_keys=True),
            rec.get("prompt_len_exact"))


def contract_label(fp):
    """Short human-readable description of a contract fingerprint."""
    if fp is None:
        return "Legacy-Records (keine Vertragsfelder)"
    image, env_json, sampling_json, exact = fp
    return (f"image={image or '-'}, env={len(json.loads(env_json))} Variablen, "
            f"sampling={sampling_json}, prompt_len_exact={exact}")


def latest_sanity_by_profile(records):
    """Latest G1 sanity probe per profile (the probe runs once per server
    start but is attached to every record of that profile)."""
    out = {}
    for r in sorted(records, key=lambda r: r.get("ts", "")):
        if r.get("sanity") is not None:
            out[r["profile"]] = r["sanity"]
    return out


def render_warnings(records, sanity_by_profile):
    """Warn when cells of one model mix benchmark tools (vllm-bench vs
    fallback) or G4 measurement contracts, or when a profile failed the
    G1 sanity probe. Speed without correctness does not count."""
    lines = []
    tools = sorted({r.get("tool") for r in records if r.get("tool")})
    if len(tools) > 1:
        lines.append("- Zellen mischen Benchmark-Tools (" +
                     ", ".join(f"`{t}`" for t in tools) + ") — tok/s-Werte "
                     "sind wegen unterschiedlicher Lastgeneratoren nicht "
                     "direkt vergleichbar.")
    contracts = defaultdict(set)
    for r in records:
        contracts[contract_fingerprint(r)].add(r.get("profile") or "?")
    if len(contracts) > 1:
        lines.append(f"- Zellen mischen {len(contracts)} Messverträge "
                     "(G4: image/env/sampling/prompt_len_exact) — nur Zellen "
                     "mit gleichem Vertrag sind fair vergleichbar:")
        for fp in sorted(contracts, key=str):
            lines.append(f"  - {contract_label(fp)} — Profile: "
                         + ", ".join(f"`{p}`" for p in sorted(contracts[fp])))
    failed = sorted(p for p, s in sanity_by_profile.items() if not s.get("ok"))
    if failed:
        lines.append("- Output-Sanity (G1) fehlgeschlagen für " +
                     ", ".join(f"`{p}`" for p in failed) +
                     " — Geschwindigkeit ohne Korrektheit zählt nicht; diese "
                     "Profile nicht als Gewinner übernehmen.")
    if not lines:
        return []
    return ["**Warnungen (Messvertrag/Qualität):**", ""] + lines + [""]


def render_quality_summary(latest, sanity_by_profile):
    """G1/G3 lines for the Ergebnis section; empty for legacy records that
    carry no quality-gate fields."""
    lines = []
    if sanity_by_profile:
        parts = []
        for profile, s in sorted(sanity_by_profile.items()):
            if "error" in s:
                parts.append(f"`{profile}` **FEHLER** (Probe nicht durchgelaufen: "
                             f"{s['error']})")
                continue
            checks = s.get("checks") or []
            passed = sum(1 for c in checks if c.get("ok"))
            verdict = "OK" if s.get("ok") else "**FEHLGESCHLAGEN**"
            parts.append(f"`{profile}` {verdict} ({passed}/{len(checks)} Prompts)")
        lines.append("- Output-Sanity (G1, greedy, finish_reason=stop, "
                     "Repetitions-Check): " + "; ".join(parts))
    acc_by_profile = defaultdict(list)
    for r in latest.values():
        if r.get("acceptance_len") is not None:
            acc_by_profile[r["profile"]].append(r["acceptance_len"])
    if acc_by_profile:
        parts = [f"`{p}` Ø {sum(v) / len(v):.2f} ({len(v)} Zellen)"
                 for p, v in sorted(acc_by_profile.items())]
        lines.append("- MTP-Acceptance-Length (G3, Δaccepted/Δdraft): "
                     + "; ".join(parts))
    return lines


def render_model(model, records):
    """records: all cells of one model. Returns Markdown text."""
    lines = [f"## Modell: `{model}`", ""]
    sanity_by_profile = latest_sanity_by_profile(records)
    lines += render_warnings(records, sanity_by_profile)
    prompt_lens = sorted({r["prompt_len"] for r in records})

    # latest record wins per (prompt_len, profile, concurrency)
    latest = {}
    for r in sorted(records, key=lambda r: r.get("ts", "")):
        latest[(r["prompt_len"], r["profile"], r["concurrency"])] = r

    # winners across all prompt lengths
    single_cells = [r for r in latest.values() if r["concurrency"] == 1]
    best_single = max(single_cells, key=lambda r: r["output_toks"], default=None)
    best_agg = max(latest.values(), key=lambda r: r["output_toks"], default=None)

    for pl in prompt_lens:
        profiles = sorted({p for (p_, p, _) in latest if p_ == pl})
        concurrencies = sorted({c for (p_, _, c) in latest if p_ == pl})
        lines.append(f"### Prompt-Länge {pl} tokens")
        lines.append("")
        header = "| Profil | " + " | ".join(f"C={c}" for c in concurrencies) + " |"
        sep = "|---|" + "---:|" * len(concurrencies)
        lines += [header, sep]
        for profile in profiles:
            row = [f"`{profile}`"]
            for c in concurrencies:
                r = latest.get((pl, profile, c))
                if r is None:
                    row.append("-")
                    continue
                cell = f"{r['output_toks']:.1f}"
                if r.get("acceptance_len") is not None:
                    cell += f" (α {r['acceptance_len']:.2f})"
                if r is best_single or r is best_agg:
                    cell = f"**{cell}**"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if any(r.get("acceptance_len") is not None for r in latest.values()):
        lines.append("_α = MTP-Acceptance-Length: Δ akzeptierte / Δ entworfene "
                     "Draft-Tokens aus /metrics (G3); höher = MTP amortisiert "
                     "sich stärker_")
        lines.append("")

    lines.append("**Ergebnis:**")
    if best_single:
        acc = (f", MTP-Acceptance {best_single['acceptance_len']:.2f}"
               if best_single.get("acceptance_len") is not None else "")
        lines.append(
            f"- Bestes Single-Stream (C=1): `{best_single['profile']}` mit "
            f"**{best_single['output_toks']:.1f} tok/s** "
            f"(Prompt {best_single['prompt_len']}, TTFT {best_single.get('ttft_ms')} ms, "
            f"ITL {best_single.get('itl_ms')} ms{acc})")
    if best_agg:
        acc = (f", MTP-Acceptance {best_agg['acceptance_len']:.2f}"
               if best_agg.get("acceptance_len") is not None else "")
        lines.append(
            f"- Bester Aggregat-Durchsatz: `{best_agg['profile']}` mit "
            f"**{best_agg['output_toks']:.1f} tok/s** bei C={best_agg['concurrency']} "
            f"(Prompt {best_agg['prompt_len']}{acc})")
    lines += render_quality_summary(latest, sanity_by_profile)
    lines.append("")
    if best_single or best_agg:
        lines += render_spark_comparison(best_single, best_agg)
    lines += render_toolbox_comparison(latest)
    return lines


def render_spark_comparison(best_single, best_agg):
    """Ratio of the best cluster cells to each Spark reference, with a
    clear BEATEN/NOT-YET verdict per metric."""
    lines = ["**Spark-Vergleich** (externe Referenz, NICHT auf diesem "
             "Cluster gemessen - Details: docs/PERFORMANCE.md):"]
    for ref in SPARK_REFERENCES:
        parts = []
        if best_single and ref["single_tps"]:
            ratio = best_single["output_toks"] / ref["single_tps"]
            verdict = "BEATEN" if ratio >= 1.0 else "NOT-YET"
            parts.append(f"Single-Stream {best_single['output_toks']:.1f} / "
                         f"{ref['single_tps']:.1f} tok/s = {ratio:.2f}x -> "
                         f"{verdict}")
        if best_agg and ref["aggregate_tps"]:
            ratio = best_agg["output_toks"] / ref["aggregate_tps"]
            verdict = "BEATEN" if ratio >= 1.0 else "NOT-YET"
            parts.append(f"Aggregat {best_agg['output_toks']:.1f} / "
                         f"{ref['aggregate_tps']:.1f} tok/s = {ratio:.2f}x -> "
                         f"{verdict}")
        if parts:
            lines.append(f"- `{ref['label']}`: " + "; ".join(parts))
    lines.append("")
    return lines


def render_toolbox_comparison(latest):
    """Per-request tok/s of the best cluster cell at each concurrency level
    vs. the official kyuz0 Toolbox C numbers (single node, MTP). Our
    output_toks is aggregate throughput, so per-request = output_toks /
    concurrency."""
    lines = ["**Toolbox-C-Vergleich** (externe Referenz, 1 Node, NICHT auf "
             "diesem Cluster gemessen - Details: docs/PERFORMANCE.md):"]
    compared = False
    for conc, ref_tps in sorted(TOOLBOX_REFERENCE["per_request_tps"].items()):
        cells = [r for r in latest.values() if r["concurrency"] == conc]
        if not cells:
            continue
        compared = True
        best = max(cells, key=lambda r: r["output_toks"])
        per_req = best["output_toks"] / conc
        ratio = per_req / ref_tps
        verdict = "BEATEN" if ratio >= 1.0 else "NOT-YET"
        lines.append(f"- C={conc}: `{best['profile']}` pro Request "
                     f"{per_req:.2f} / {ref_tps:.2f} tok/s = {ratio:.2f}x "
                     f"-> {verdict} (Aggregat {best['output_toks']:.1f} tok/s)")
    if not compared:
        lines.append("- keine Messzellen auf den Referenz-Concurrency-"
                     "Stufen (C=1/8/32) vorhanden")
    lines.append(f"- Referenz: {TOOLBOX_REFERENCE['label']}; Kontrolllauf "
                 "der Toolbox streut bis ~10 % (s. docs/PERFORMANCE.md) - "
                 "knapp über 1,00x ist noch im Rauschen")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", nargs="+",
                    help="result JSON files (globs allowed), e.g. 'bench/results/*.json'")
    ap.add_argument("--output", default=None, help="write report to file instead of stdout")
    args = ap.parse_args()

    records = load_records(args.results)
    if not records:
        print("no usable records found", file=sys.stderr)
        sys.exit(1)

    lines = ["# Benchmark-Report Strix-Halo-Cluster", ""]
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    for model in sorted(by_model):
        lines += render_model(model, by_model[model])

    lines += [
        "## Externe Referenzwerte (DGX Spark)",
        "",
        "Nur zur Orientierung - **NICHT** auf diesem Cluster gemessen",
        "(andere Hardware, anderer Interconnect, teils andere Quantisierung):",
        "",
        "| Workload | Single-Stream | Aggregat |",
        "|---|---|---|",
    ]
    for ref in SPARK_REFERENCES:
        lines.append(f"| {ref['label']} | {ref['single_display']} | "
                     f"{ref['aggregate_display']} |")
    lines.append("")
    sources = sorted({ref["source_url"] for ref in SPARK_REFERENCES})
    lines.append("Quelle(n): " + ", ".join(f"<{u}>" for u in sources) +
                 " — offizielle Benchmark-Ziele, siehe docs/PERFORMANCE.md.")
    lines.append("")
    lines += [
        "## Externe Referenzwerte (offizielle Toolbox C)",
        "",
        f"{TOOLBOX_REFERENCE['label']} — tok/s **pro Request**, ebenfalls "
        "NICHT auf diesem Cluster gemessen:",
        "",
        "| Concurrency | Toolbox C | Kontrolllauf | Abweichung |",
        "|---|---:|---:|---:|",
    ]
    per_req = TOOLBOX_REFERENCE["per_request_tps"]
    ctrl = TOOLBOX_REFERENCE["control_run"]
    for conc in sorted(per_req):
        dev = abs(per_req[conc] - ctrl[conc]) / per_req[conc] * 100
        lines.append(f"| C={conc} | {per_req[conc]:.2f} tok/s | "
                     f"{ctrl[conc]:.2f} tok/s | {dev:.1f} % |")
    lines.append("")
    lines.append(f"Quelle: <{TOOLBOX_REFERENCE['source_url']}>")
    lines.append("")

    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
