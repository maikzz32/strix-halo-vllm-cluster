"""Validate, install and restore a staged native-controller release (no services)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

FILES = {
    "cluster.py": "tools/cluster.py", "node_runtime.py": "tools/node_runtime.py",
    "bench_stream.py": "tools/bench_stream.py", "quality_probe.py": "tools/quality_probe.py",
    "bench_sharegpt.sh": "tools/bench_sharegpt.sh",
    "compare_streams.py": "tools/compare_streams.py",
    "cluster.runtime.json": "config/repository-default.json",
    "mtp_local_argmax.py": "patches/mtp_local_argmax.py",
    "qsa_invisible_tiles.py": "patches/qsa_invisible_tiles.py",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(stage):
    for source in FILES:
        data = (stage / source).read_bytes()
        if source.endswith(".py"):
            compile(data, source, "exec")
        elif source.endswith(".json"):
            config = json.loads(data)
            if not config.get("model") or not config.get("nodes"):
                raise ValueError("Runtime config must specify model and nodes")
        elif source.endswith(".sh") and b"\r\n" in data:
            raise ValueError(f"{source} contains CRLF; deploy Linux shell scripts with LF")


def replace(target, data, mode=0o644):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, mode)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def checked_target(destination, relative):
    if relative not in set(FILES.values()) | {"config/cluster.json"}:
        raise ValueError("Unknown deployment target")
    target = destination / relative
    if target.is_symlink() or not target.resolve().is_relative_to(destination):
        raise ValueError(f"Deployment target escapes destination or is a symlink: {relative}")
    return target


def promote(stage, destination):
    validate(stage)
    receipt_path = stage / "receipt.json"
    if receipt_path.exists():
        raise ValueError("Release already attempted; restore it or stage a fresh release")
    files = dict(FILES)
    if not (destination / "config/cluster.json").exists():
        files["cluster.runtime.json#initial"] = "config/cluster.json"
    entries = []
    for source, relative in files.items():
        target = checked_target(destination, relative)
        original = target.read_bytes() if target.exists() else None
        new = (stage / source.split("#", 1)[0]).read_bytes()
        mode = (target.stat().st_mode & 0o777) if original is not None else 0o644
        if original is not None:
            backup = stage / "previous" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original)
        entries.append({"target": relative, "source": source.split("#", 1)[0],
                        "before": sha(original) if original is not None else None,
                        "after": sha(new), "mode": mode})
    # All original files and the recovery manifest exist before first overwrite.
    replace(receipt_path, json.dumps(entries, indent=2).encode())
    for item in entries:
        target = checked_target(destination, item["target"])
        current = sha(target.read_bytes()) if target.exists() else None
        if current != item["before"]:
            raise ValueError(f"Target changed during deployment: {item['target']}")
        replace(target, (stage / item["source"]).read_bytes(), item["mode"])


def restore(stage, destination):
    entries = json.loads((stage / "receipt.json").read_text())
    prepared = []
    for item in entries:
        target = checked_target(destination, item["target"])
        current = sha(target.read_bytes()) if target.exists() else None
        # Accept untouched files from an interrupted promotion, but never edits
        # made after deployment or a subsequent release.
        if current not in (item["before"], item["after"]):
            raise ValueError(f"Source changed since deployment: {item['target']}")
        data = None if item["before"] is None else (stage / "previous" / item["target"]).read_bytes()
        if data is not None and sha(data) != item["before"]:
            raise ValueError("Recovery backup checksum mismatch")
        prepared.append((target, current, data, item["mode"]))
    for target, observed, data, mode in prepared:
        current = sha(target.read_bytes()) if target.exists() else None
        if current != observed:
            raise ValueError(f"Target changed during restoration: {target}")
        if data is None:
            target.unlink(missing_ok=True)
        else:
            replace(target, data, mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "promote", "restore"))
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve(strict=True)
    stage = args.stage.resolve(strict=True)
    if stage.parent != destination / "deployments":
        raise ValueError("Release directory must be directly inside destination/deployments")
    # Deployment and recovery on one host must not race another release.
    import fcntl
    with (destination / "deployments" / "deploy.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "validate":
            validate(stage)
        elif args.action == "promote":
            promote(stage, destination)
        else:
            restore(stage, destination)
    print(f"{args.action}: {stage}")


if __name__ == "__main__":
    main()
