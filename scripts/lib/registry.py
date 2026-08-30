#!/usr/bin/env python3
"""Shared helper: resolves models/registry.yaml entries for the shell scripts.

Single source of truth for serve.sh / cluster_up.sh / status.sh so the YAML
schema logic lives in exactly one place. Python 3 + PyYAML only.

Usage:
  registry.py profiles                        # all defined parallel profiles
  registry.py show <model>                    # resolved entry (merged over defaults) as JSON
  registry.py status <model>
  registry.py hf_repo <model>
  registry.py env <model>                     # merged env, one KEY=VALUE per line
  registry.py extra_args <model>              # merged extra CLI args, one per line
  registry.py parser <model> <key>            # tokenizer_mode|tool_call_parser|reasoning_parser
  registry.py allowed_profiles <model>        # space-separated; omit in YAML = all
  registry.py tracking <model>                # upstream URLs, one per line
  registry.py nodes <inventory.yaml>          # "name host user" per line, head (node1) first

Exit code 1 with a message on stderr for unknown models / bad usage.
"""

import json
import sys
from pathlib import Path

import yaml

PROFILES = ("tp4", "pp4", "tp2pp2", "ep", "solo")

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "models" / "registry.yaml"


def load_registry(path=REGISTRY_PATH):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(reg, name):
    """Return the model entry merged over `defaults`.

    - env:        model.env overlays defaults.env (model wins per key)
    - extra_args: defaults.extra_args followed by model extra_args
    - allowed_profiles: model value or all PROFILES when omitted
    """
    models = reg.get("models") or {}
    if name not in models:
        known = ", ".join(sorted(models)) or "(none)"
        die(f"unknown model '{name}'. Known models: {known}")
    defaults = reg.get("defaults") or {}
    model = dict(models[name] or {})

    env = dict(defaults.get("env") or {})
    env.update(model.get("env") or {})
    model["env"] = env

    model["extra_args"] = list(defaults.get("extra_args") or []) + list(
        model.get("extra_args") or []
    )

    allowed = model.get("allowed_profiles")
    model["allowed_profiles"] = list(allowed) if allowed else list(PROFILES)
    return model


def die(msg):
    print(f"registry.py: error: {msg}", file=sys.stderr)
    sys.exit(1)


def inventory_nodes(path):
    """Yield (name, host, user) for every host in an ansible YAML inventory.

    Walks any `hosts:` mapping recursively (all / all.children.<group>.hosts).
    ansible_host defaults to the host name, ansible_user to the group-level
    (vars.ansible_user) value or "" — the ssh wrapper then falls back to the
    caller's ssh config. Document order is preserved; the inventory lists
    node1 first, and node1 is the Ray head.
    """
    with open(path, encoding="utf-8") as fh:
        inv = yaml.safe_load(fh)
    out = []

    def walk(node, group_user):
        if not isinstance(node, dict):
            return
        user = (node.get("vars") or {}).get("ansible_user", group_user)
        hosts = node.get("hosts")
        if isinstance(hosts, dict):
            for name, hvars in hosts.items():
                hvars = hvars or {}
                out.append(
                    (name, hvars.get("ansible_host", name),
                     hvars.get("ansible_user", user or ""))
                )
        children = node.get("children")
        if isinstance(children, dict):
            for child in children.values():
                walk(child, user)

    walk(inv.get("all", inv), "")
    if not out:
        die(f"no hosts found in {path}")
    return out


def main(argv):
    if len(argv) < 2:
        die(__doc__)
    cmd = argv[1]

    if cmd == "nodes":
        if len(argv) != 3:
            die("usage: registry.py nodes <inventory.yaml>")
        for name, host, user in inventory_nodes(argv[2]):
            print(f"{name} {host} {user}")
        return

    if cmd == "profiles":
        print(" ".join(PROFILES))
        return

    if len(argv) < 3:
        die(f"usage: registry.py {cmd} <model> [<key>]")
    entry = resolve(load_registry(), argv[2])

    if cmd == "show":
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif cmd == "status":
        print(entry.get("status", "supported"))
    elif cmd == "hf_repo":
        print(entry["hf_repo"])
    elif cmd == "env":
        for key, val in entry["env"].items():
            print(f"{key}={val}")
    elif cmd == "extra_args":
        for arg in entry["extra_args"]:
            print(arg)
    elif cmd == "parser":
        if len(argv) != 4:
            die("usage: registry.py parser <model> <key>")
        print((entry.get("parsers") or {}).get(argv[3], ""))
    elif cmd == "allowed_profiles":
        print(" ".join(entry["allowed_profiles"]))
    elif cmd == "tracking":
        for url in entry.get("tracking") or []:
            print(url)
    else:
        die(f"unknown subcommand '{cmd}'")


if __name__ == "__main__":
    main(sys.argv)
