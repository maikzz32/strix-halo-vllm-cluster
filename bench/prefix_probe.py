#!/usr/bin/env python3
"""Prefix-cache probe: measures TTFT cold vs warm for a shared long prefix.

Sends one cold request with a long deterministic prefix, then --repeats warm
requests that share the same prefix but end in different short questions.
With APC (automatic prefix caching, vLLM V1 default) the warm TTFT should
collapse to a fraction of the cold TTFT (1Cat PR #427: 2.642 s -> 0.164 s).

Usage: prefix_probe.py --base-url http://127.0.0.1:8000 --model <hf_repo_or_name>
Report text is German (repo docs convention); code and comments are English.
"""

import argparse
import hashlib
import json
import time

import requests


def build_prefix(approx_tokens):
    """Deterministic long prefix. ~1 token per word for this vocabulary."""
    words = ("system prompt analysiere dokument kontext antwort praezise "
             "fakten struktur tabelle vergleich ergebnis messung").split()
    reps = max(1, approx_tokens // len(words))
    return " ".join(words * reps)


def ttft_once(base_url, model, prompt, max_tokens=32):
    """Time to first content chunk via streaming. Returns (ttft_ms, usage)."""
    t0 = time.monotonic()
    r = requests.post(
        f"{base_url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": 0, "stream": True},
        stream=True, timeout=300,
    )
    r.raise_for_status()
    ttft_ms = None
    usage = {}
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        if ttft_ms is None and chunk.get("choices") and chunk["choices"][0].get("text"):
            ttft_ms = (time.monotonic() - t0) * 1000
        if chunk.get("usage"):
            usage = chunk["usage"]
    return ttft_ms, usage


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()

    prefix = build_prefix(args.prefix_tokens)
    digest = hashlib.sha256(prefix.encode()).hexdigest()[:12]
    print(f"Prefix: ~{args.prefix_tokens} tokens (sha256:{digest})")

    rows = []
    for i in range(args.repeats + 1):
        question = f"\n\nFrage {i}: Nenne eine Kernaussage des Textes."
        ttft_ms, usage = ttft_once(args.base_url, args.model, prefix + question)
        rows.append((i, ttft_ms, usage.get("prompt_tokens")))
        kind = "kalt" if i == 0 else "warm"
        print(f"  Request {i} ({kind}): TTFT {ttft_ms:.0f} ms, "
              f"prompt_tokens={usage.get('prompt_tokens')}")

    cold = rows[0][1]
    warm = [r[1] for r in rows[1:] if r[1] is not None]
    print()
    if warm and cold:
        warm_avg = sum(warm) / len(warm)
        print(f"Ergebnis: kalt {cold:.0f} ms -> warm Ø {warm_avg:.0f} ms "
              f"({cold / warm_avg:.1f}x schneller)")
        print("Hinweis: ohne APC waere warm ~ kalt; ein deutlicher Abfall "
              "belegt aktives Prefix-Caching.")
    else:
        print("Messung unvollstaendig (keine verwertbaren Warm-Laeufe).")


if __name__ == "__main__":
    main()
