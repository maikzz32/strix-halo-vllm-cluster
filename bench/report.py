#!/usr/bin/env python3
"""Render a Markdown benchmark report from bench/results/*.json files.

Reads one or more result files (JSONL as written by run_matrix.py; shell
globs or --output for a file) and prints, per model and prompt length, a
profile x concurrency table of output tok/s. The best single-stream cell
(concurrency 1) and the best aggregate cell are highlighted in bold.

The report text is German (repo docs convention); code and comments are
English. DGX Spark numbers below are EXTERNAL reference points from
published literature, not measured on this cluster.
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


def render_model(model, records):
    """records: all cells of one model. Returns Markdown text."""
    lines = [f"## Modell: `{model}`", ""]
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
                if r is best_single or r is best_agg:
                    cell = f"**{cell}**"
                row.append(cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("**Ergebnis:**")
    if best_single:
        lines.append(
            f"- Bestes Single-Stream (C=1): `{best_single['profile']}` mit "
            f"**{best_single['output_toks']:.1f} tok/s** "
            f"(Prompt {best_single['prompt_len']}, TTFT {best_single.get('ttft_ms')} ms, "
            f"ITL {best_single.get('itl_ms')} ms)")
    if best_agg:
        lines.append(
            f"- Bester Aggregat-Durchsatz: `{best_agg['profile']}` mit "
            f"**{best_agg['output_toks']:.1f} tok/s** bei C={best_agg['concurrency']} "
            f"(Prompt {best_agg['prompt_len']})")
    lines.append("")
    if best_single or best_agg:
        lines += render_spark_comparison(best_single, best_agg)
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

    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
