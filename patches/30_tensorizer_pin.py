#!/usr/bin/env python3
"""Patch 30: relax the tensorizer dependency constraint (breaks on Python 3.14).

vLLM pins tensorizer with a constraint that does not resolve on Python 3.14
(the Fedora 44 interpreter). The patch rewrites the requirement line to a
relaxed constraint. Once a known-good tensorizer version is confirmed on the
first real py3.14 build, replace the relaxed constraint with an exact pin
and record it in MANIFEST.md.

STATUS: expected-to-need-adjustment (exact pin TBD after first build).

Usage:
    python3 30_tensorizer_pin.py --src /opt/vllm          # apply
    python3 30_tensorizer_pin.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / check passed, 1 = check failed / error,
            42 = target pattern not found (upstream moved; re-audit needed).
"""

import argparse
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 30_tensorizer_pin"
EXIT_REAUDIT = 42

# Candidate dependency files, tried in order; rglob fallback below.
CANDIDATES = [
    "requirements/common.txt",
    "requirements/rocm.txt",
    "requirements/build.txt",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
]

REQ_RE = re.compile(r"^(\s*)tensorizer(\[[^\]]*\])?[^\s#]*", re.MULTILINE)


def find_target(src: Path) -> tuple[Path, re.Match] | None:
    files = [src / c for c in CANDIDATES if (src / c).is_file()]
    if not any(REQ_RE.search(f.read_text()) for f in files):
        # Fallback: search all requirement files in the checkout.
        files = sorted(src.rglob("requirements/*.txt"))
    for f in files:
        m = REQ_RE.search(f.read_text())
        if m:
            return f, m
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    found = find_target(src)
    if found is None:
        print(f"ERROR: no 'tensorizer' requirement found under {src} "
              f"(searched {', '.join(CANDIDATES)} and requirements/*.txt). "
              f"Upstream dropped or renamed the dependency; re-audit.",
              file=sys.stderr)
        return EXIT_REAUDIT
    target, m = found
    content = target.read_text()

    if args.check:
        if MARKER in content:
            print(f"OK: patch 30 present in {target}")
            return 0
        print(f"FAIL: patch 30 marker not found in {target}", file=sys.stderr)
        return 1

    if MARKER in content:
        print(f"SKIP: patch 30 already applied to {target}")
        return 0

    indent, extras = m.group(1), m.group(2) or ""
    old_line = m.group(0)
    new_line = (
        f"{indent}tensorizer{extras}  # {MARKER}: upstream constraint "
        f"'{old_line.strip()}' does not resolve on Python 3.14; relaxed. "
        f"TODO: pin exact version once a known-good one is confirmed on the "
        f"first py3.14 build."
    )
    patched = content[: m.start()] + new_line + content[m.end():]
    target.write_text(patched)
    print(f"OK: patch 30 applied to {target} (was: '{old_line.strip()}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
