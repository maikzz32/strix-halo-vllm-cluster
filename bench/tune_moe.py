#!/usr/bin/env python3
"""Tune fused-MoE Triton configs for gfx1151 inside the cluster image.

Runs vLLM's `benchmarks/kernels/benchmark_moe.py --tune` for a given model
inside the container and writes the resulting tuned JSON(s) into a
host-mounted folder that defaults to `patches/configs/` — the folder baked
into the image as `VLLM_TUNED_CONFIG_FOLDER` (see patches/configs/README.md).
The tuner writes the vLLM config file name itself
(`E={E},N={N},device_name=AMD_Radeon_8060S[,dtype=...].json`), so the result
is drop-in correct for the hardware it ran on.

The final image carries only the vLLM wheel, not the source tree, so the
benchmark script must come from somewhere: either pass --vllm-src (local
checkout matching the image's VLLM_REF, mounted read-only) or use an image
that already contains it and point --script-path at it.

Tuning is single-GPU (the script shards sizes analytically for --tp-size);
run it on any one gfx1151 node. Repeat per TP level you actually serve with
(N in the file name = moe_intermediate_size // TP).

Dependencies: python3 stdlib only.

Examples:
    python3 bench/tune_moe.py --model Qwen/Qwen3.6-35B-A3B \
        --image ghcr.io/<org>/vllm-gfx1151:dev --vllm-src ~/src/vllm

    # extra args for benchmark_moe.py after `--`:
    python3 bench/tune_moe.py --model Qwen/Qwen3.6-35B-A3B \
        --vllm-src ~/src/vllm -- --trust-remote-code --batch-size 1 2 4 8 16
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "patches" / "configs"
CONTAINER_CONFIG_DIR = "/opt/vllm-tuned-configs"
CONTAINER_VLLM_SRC = "/opt/vllm-src"
DEFAULT_SCRIPT_PATH = "/opt/vllm/benchmarks/kernels/benchmark_moe.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Extra arguments after `--` are passed to benchmark_moe.py.",
    )
    ap.add_argument("--model", required=True,
                    help="HF repo id, e.g. Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--image", default="vllm-gfx1151:dev",
                    help="container image to run in (default: %(default)s)")
    ap.add_argument("--runtime", default="podman", choices=["podman", "docker"],
                    help="container runtime (default: %(default)s)")
    ap.add_argument("--tp-size", type=int, default=1,
                    help="tensor-parallel size to tune for (default: %(default)s)")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                    help="host folder for tuned JSONs, mounted at "
                         f"{CONTAINER_CONFIG_DIR} (default: %(default)s)")
    ap.add_argument("--vllm-src",
                    help="local vLLM checkout (mounted read-only at "
                         f"{CONTAINER_VLLM_SRC}); should match the image's "
                         "VLLM_REF")
    ap.add_argument("--script-path", default=None,
                    help="benchmark_moe.py path inside the container when "
                         "--vllm-src is not given "
                         f"(default: {DEFAULT_SCRIPT_PATH})")
    ap.add_argument("--dtype", default="bf16",
                    help="activation dtype passed to the tuner "
                         "(default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the container command and exit")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args for benchmark_moe.py after `--`")
    args = ap.parse_args()

    if shutil.which(args.runtime) is None:
        print(f"ERROR: {args.runtime} not found in PATH.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vllm_src:
        vllm_src = Path(args.vllm_src).resolve()
        script = f"{CONTAINER_VLLM_SRC}/benchmarks/kernels/benchmark_moe.py"
        if not (vllm_src / "benchmarks/kernels/benchmark_moe.py").is_file():
            print(f"ERROR: {vllm_src} does not look like a vLLM checkout "
                  f"(benchmarks/kernels/benchmark_moe.py missing).",
                  file=sys.stderr)
            return 1
    else:
        vllm_src = None
        script = args.script_path or DEFAULT_SCRIPT_PATH
        print(f"NOTE: no --vllm-src given; expecting the benchmark script at "
              f"{script} inside the image.")

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = [
        args.runtime, "run", "--rm",
        # ROCm device access (Fedora nodes, podman or docker).
        "--device", "/dev/kfd", "--device", "/dev/dri",
        "--group-add", "video",
        "--ipc=host", "--network=host",
        "--security-opt", "seccomp=unconfined",
        "-v", f"{output_dir}:{CONTAINER_CONFIG_DIR}:Z",
        "-w", "/root",
    ]
    if vllm_src is not None:
        cmd += ["-v", f"{vllm_src}:{CONTAINER_VLLM_SRC}:ro,Z"]
    # HF auth/cache passthrough: the tuner only needs the model config.
    for var in ("HF_TOKEN", "HF_HOME"):
        if os.environ.get(var):
            cmd += ["-e", f"{var}={os.environ[var]}"]
    cmd += [
        args.image,
        "python3", script,
        "--tune",
        "--model", args.model,
        "--tp-size", str(args.tp_size),
        "--dtype", args.dtype,
        "--save-dir", CONTAINER_CONFIG_DIR,
        *extra,
    ]

    print("Running:", " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"ERROR: tuner exited with {rc}. If the benchmark script was "
              f"not found in the image, pass --vllm-src.", file=sys.stderr)
        return rc

    written = sorted(output_dir.glob("E=*,N=*,device_name=*.json"))
    print(f"Tuned configs now in {output_dir}:")
    for p in written:
        print(f"  {p.name}")
    print("Rebuild the image (or bind-mount this folder over "
          f"{CONTAINER_CONFIG_DIR}) to pick them up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
