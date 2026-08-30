#!/usr/bin/env python3
"""model-watch: poll upstream tracking state for dev|blocked models.

Reads models/registry.yaml and, for every model with status dev|blocked,
resolves each `tracking` URL via the GitHub API into a per-model "signal"
(PR open/closed/merged, issue open/closed, latest release tag). The signal
is compared against the state stored in a machine-readable marker inside the
body of the model's model-watch issue - the issue body is the state store,
so the script itself is stateless and idempotent.

Models may additionally pin a `vllm_ref` (full SHA of a vLLM PR head, see
registry.yaml). For those, the source PR's current head SHA is part of the
signal; a moved head (rebase/advance) is a change event that dispatches
build-dev.yml with the pinned vllm_ref and a `model` input, so the image
gets a model-specific tag suffix (:dev-<model>).

Behaviour:
  * first run (no issue yet)   -> create a baseline issue, do NOT build
  * signal unchanged           -> no-op
  * signal changed             -> update + reopen the issue
  * change looks like progress  -> additionally dispatch build-dev.yml
    (a gating PR merged/closed, a tracking issue closed, or a new covering
    vLLM release tag appeared)

Nothing in registry.yaml is modified automatically; flipping `status` stays
a human decision after validation on the :dev image.

Required env: GITHUB_TOKEN, GITHUB_REPOSITORY (owner/name),
              GITHUB_REF_NAME (default branch; both set by Actions).
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

import yaml

API = "https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GITHUB_REPOSITORY"]
REF = os.environ.get("GITHUB_REF_NAME", "main")
REGISTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.yaml")
DEV_WORKFLOW = "build-dev.yml"

# Base repo used to resolve a pinned vllm_ref (commit SHA) to its source PR
# via the search + compare APIs (fork PR heads are reachable in the base repo).
VLLM_REPO = "vllm-project/vllm"

# Marker embedded in the issue body; carries the last-seen signal as JSON.
STATE_RE = re.compile(r"<!--\s*model-watch-state:\s*(\{.*?\})\s*-->", re.S)

GH_ITEM_URL = re.compile(r"https://github\.com/([^/]+)/([^/]+)/(pull|issues)/(\d+)")
GH_RELEASES_URL = re.compile(r"https://github\.com/([^/]+)/([^/]+)/releases/?$")

# States that mean "the gating item is resolved upstream".
RESOLVED = {"merged", "closed"}


def gh(path, method="GET", payload=None):
    """Minimal GitHub API client (avoids a PyGitHub dependency)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    with urllib.request.urlopen(req) as resp:
        if resp.status == 204:
            return None
        return json.load(resp)


def tracking_state(url):
    """Resolve one tracking URL to a (key, state) signal entry."""
    url = url.strip()
    m = GH_ITEM_URL.match(url)
    if m:
        owner, repo, kind, num = m.groups()
        if kind == "pull":
            pr = gh(f"/repos/{owner}/{repo}/pulls/{num}")
            state = "merged" if pr.get("merged_at") else pr["state"]
            return f"pr:{owner}/{repo}#{num}", state
        issue = gh(f"/repos/{owner}/{repo}/issues/{num}")
        return f"issue:{owner}/{repo}#{num}", issue["state"]  # open|closed
    m = GH_RELEASES_URL.match(url)
    if m:
        owner, repo = m.groups()
        try:
            rel = gh(f"/repos/{owner}/{repo}/releases/latest")
            return f"release:{owner}/{repo}", rel["tag_name"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return f"release:{owner}/{repo}", "none"
            raise
    # Non-GitHub or unrecognized URL: opaque constant, so editing the URL in
    # the registry still registers as a signal change.
    return f"url:{url}", "untracked"


def _pr_head_if_contains(num, sha):
    """Head SHA of vLLM PR `num` if `sha` is its head or an ancestor of it.

    None when the pin is not reachable from the PR head (e.g. the PR was
    rebased and force-pushed). Verified via the compare API: the pin must be
    an ancestor of ('ahead') or equal to ('identical') the current head.
    """
    pr = gh(f"/repos/{VLLM_REPO}/pulls/{num}")
    head = pr["head"]["sha"]
    if head == sha:
        return head
    try:
        cmp = gh(f"/repos/{VLLM_REPO}/compare/{sha}...{head}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return head if cmp.get("status") in ("ahead", "identical") else None


def vllm_ref_state(sha):
    """Resolve a pinned vllm_ref to the current head SHA of its source PR.

    The pin->PR mapping goes through the issue search API - the commit->PRs
    association endpoint does not cover fork PR heads (verified: returns []
    for both current pins). Candidates are PRs containing the pin in their
    head history; lowest PR number wins. Returns the PR's current head SHA
    (== pin, or advanced past it), 'gone' when the pin is no longer
    reachable from any candidate (rebase + force-push -> re-audit the pin),
    or 'untracked' when no PR claims the commit at all.
    """
    res = gh(f"/search/issues?q=repo:{VLLM_REPO}+is:pr+{sha}")
    candidates = sorted(
        item["number"] for item in res.get("items", [])
        if "pull_request" in item
    )
    for num in candidates:
        head = _pr_head_if_contains(num, sha)
        if head:
            return head
    return "gone" if candidates else "untracked"


def model_signal(model):
    signal = {}
    for url in model.get("tracking") or []:
        key, state = tracking_state(url)
        signal[key] = state
    ref = model.get("vllm_ref")
    if ref:
        # The key embeds the pin, so a human updating the pin in the registry
        # registers as a signal change just like a moved PR head does.
        signal[f"vllm_ref:{ref}"] = vllm_ref_state(ref)
    return signal


def is_progress(old, new):
    """True if the signal change looks like upstream support moved forward."""
    for key, state in new.items():
        if old.get(key) == state:
            continue
        if state in RESOLVED:
            return True
        # The source PR of a pinned vllm_ref moved (rebase/advance, also
        # 'gone'/'untracked') or the pin itself changed in the registry.
        if key.startswith("vllm_ref:"):
            return True
        # A new release tag on a `releases` tracking entry may cover the model.
        if key.startswith("release:") and state != "none":
            return True
    return False


def find_issue(title):
    """Find the model's model-watch issue by exact title (open or closed).

    No label filter: the 'model-watch' label is not guaranteed to exist and
    creating issues with unknown labels fails with 422. Single page is fine
    for a repo of this size.
    """
    issues = gh(f"/repos/{REPO}/issues?state=all&per_page=100")
    for it in issues:
        if it.get("title") == title and "pull_request" not in it:
            return it
    return None


def stored_signal(issue):
    m = STATE_RE.search(issue.get("body") or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def render_body(name, model, signal):
    lines = [
        f"Upstream tracking state for `{name}` (registry status: `{model.get('status')}`).",
        "",
        "| tracking | state |",
        "|---|---|",
    ]
    lines += [f"| `{key}` | {state} |" for key, state in sorted(signal.items())]
    lines += [
        "",
        "If a gating item is resolved: validate the model on the `:dev` image,",
        f"then flip `status` for `{name}` in `models/registry.yaml` manually.",
        "This issue is maintained by `.github/workflows/model-watch.yml`.",
        "",
        f"<!-- model-watch-state: {json.dumps(signal, sort_keys=True)} -->",
    ]
    return "\n".join(lines)


def dispatch_dev_build(vllm_ref=None, model=None):
    payload = {"ref": REF}
    inputs = {}
    if vllm_ref:
        inputs["vllm_ref"] = vllm_ref
    if model:
        # Passed through as image tag suffix -> :dev-<model>.
        inputs["model"] = model
    if inputs:
        payload["inputs"] = inputs
    gh(f"/repos/{REPO}/actions/workflows/{DEV_WORKFLOW}/dispatches",
       method="POST", payload=payload)


def main():
    with open(REGISTRY, encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    watched = {
        name: model
        for name, model in (registry.get("models") or {}).items()
        if model.get("status") in ("dev", "blocked")
    }
    if not watched:
        print("no dev|blocked models in registry - nothing to do")
        return

    for name, model in watched.items():
        title = f"model-watch: {name} support changed"
        signal = model_signal(model)
        print(f"[{name}] signal: {json.dumps(signal, sort_keys=True)}")

        issue = find_issue(title)
        old = stored_signal(issue) if issue else None
        if old == signal:
            print(f"[{name}] unchanged - skipping")
            continue

        body = render_body(name, model, signal)
        if issue is None:
            # Baseline: record the current state without triggering a build
            # (first run must not stampede build-dev for every watched model).
            created = gh(f"/repos/{REPO}/issues", method="POST",
                         payload={"title": title, "body": body})
            print(f"[{name}] baseline issue #{created['number']} created")
            continue

        # Reopen on update: a closed issue must resurface when state changes.
        gh(f"/repos/{REPO}/issues/{issue['number']}", method="PATCH",
           payload={"body": body, "state": "open"})
        print(f"[{name}] issue #{issue['number']} updated")

        if is_progress(old or {}, signal):
            # A moved vllm_ref head rebuilds from the model's pin and tags the
            # image :dev-<model>; other progress builds the generic :dev.
            ref_moved = any(
                key.startswith("vllm_ref:") and (old or {}).get(key) != state
                for key, state in signal.items()
            )
            if ref_moved and model.get("vllm_ref"):
                dispatch_dev_build(vllm_ref=model["vllm_ref"], model=name)
                print(f"[{name}] vllm_ref PR head moved - {DEV_WORKFLOW} "
                      f"dispatched (vllm_ref={model['vllm_ref']}, model={name})")
            else:
                dispatch_dev_build()
                print(f"[{name}] progress detected - {DEV_WORKFLOW} dispatched")


if __name__ == "__main__":
    sys.exit(main())
