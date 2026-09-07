# Native TP4 runtime, September 2026

This controller operates the four existing patched containers (`ray-head` and
`ray-worker`) using vLLM's native `mp` executor. Its configuration reproduces the
measured `/home/maik/qwen38_rest` production setup: TP4, MTP3, 262144 context,
target decode graphs, draft prefill graphs, four RCCL channels and RoCEv2 GID1.

**The existing `ray-head`/`ray-worker` containers are a prerequisite.** Their
approximately 6 GB writable overlays contain installed packages and patches;
the `dev-rocm10` base tag does not reproduce this runtime. A snapshot of 2,357
Python sources, `.pth` files and the package list exists. Node 1 also retains
the local binary snapshot `localhost/strix-halo-runtime:20260906-qsa-gated`
(image ID `718fdd7ceb8ad3fa889361f1cec132d966ed3eb21059fbf2898e2a51352cefb1`).
A replacement container has not been tested; models, named volumes and host
RDMA configuration remain external. Do not recreate containers from the base tag expecting an
equivalent installation. This deployment preserves existing containers and does
not rebuild images, install runtime packages, change weights or RDMA settings.

From a repository checkout on node1:

```bash
bash scripts/native/deploy.sh
python3 /home/maik/strix-halo-next/tools/cluster.py status
python3 /home/maik/strix-halo-next/tools/cluster.py dry-run
python3 /home/maik/strix-halo-next/tools/cluster.py restart --tag production
```

`restart` first validates runtime/model availability and matching configuration
on every host and inside its container while the current service remains running.
It requires both running/waiting request gauges to confirm idle, stops inference
on every rank, checks that all nodes can start, then launches the ranks and waits
for the matching model API. Startup
failure cleans up only processes carrying that attempt's run ID. Logs and exact
launch configuration are retained under `/home/maik/strix-halo-next/logs` on each
node. `--force-stop` is an explicit recovery option for a broken or busy server.

Model endpoint: `http://192.168.1.15:8000/v1`.

The QSA baseline config enables `VLLM_QSA_SKIP_INVISIBLE_TILES=1` and uses
no profiler configuration or profiler environment overrides. Install the
matching `patches/qsa_invisible_tiles.py` patch before using this setting on a
restored runtime. The previous operator config is retained on all four hosts as
`config/cluster.pre-qsa-20260906.json`. Measured ShareGPT48 performance is
49.56–50.30 versus 48.60 output tokens/s (the final run also has lower TTFT);
decode improvement remains about 2%. This is a modest gain, not the original
goal of substantially faster single-answer generation.

The current canonical host configuration additionally selects the validated
[UMA/RDMA backend](../../docs/2026-09-07-hip-uma.md), with the complete
[candidate configuration](../../bench/records/hip-uma-model-r1.json) archived.
The preserved addon and opt-in communicator hook are required on every node;
the September 6 runtime image snapshot predates this addon. The repository's
baseline deployment does not install it or overwrite an existing operator
configuration. Use the dedicated staging/activation procedure and preserve the
original communicator before enabling it. The prior QSA/RCCL configuration is
saved on all four hosts as `config/cluster.pre-uma-20260907.json`.
Allow 900 seconds for model startup; the earlier 420-second deadline was too
short for one observed load and caused deliberate worker cleanup.

Startup is manual. The detached rank launchers survive SSH disconnection, but
there is no configured boot startup, crash supervisor or automatic failover.
After a host reboot, start the preserved `ray-head` container on node 1, wait
for its Ray service on port 6379, then start the three preserved `ray-worker`
containers. Their PID 1 still runs Ray, although inference uses native `mp`.
Once all containers are running, use the native controller. This reboot sequence
has not been tested by rebooting the hosts.

The obsolete `dflash-27b.service` on node 1 was stopped and disabled after its
826 failed retries against a missing `strix-halo-llm-finetuning` container.
Its unit file was preserved. Restoring that older service requires restoring
its separate container first; it is not part of this Qwen Flash Next cluster.

Configuration is `/home/maik/strix-halo-next/config/cluster.json` **on every node**.
For an experiment, copy one identical JSON file to all four nodes and pass its
absolute path with `--config`. Revert by restarting with the original file. The
local config contains host addresses/interface names specific to this cluster.
Configuration comparisons ignore JSON formatting and key order. Drift fails
before a start; a restart checks it before stopping the existing service. `stop`
and run-scoped failure cleanup remain available when configuration copies differ.

Deployment uploads and syntax-validates all four nodes before replacing any
deployed file. SSH/SCP explicitly use `maik` and fail without a password prompt.
Existing `config/cluster.json` remains untouched; the repository version is
stored as `config/repository-default.json`. Each upload is retained under
`/home/maik/strix-halo-next/deployments/release.XXXXXXXX`, with exact previous files
and a recovery manifest. The deploy command prints each node's release path.

To restore a deployment on a node, use its printed release path:

```bash
RELEASE=/home/maik/strix-halo-next/deployments/release.XXXXXXXX
python3 "$RELEASE/deploy_release.py" restore --stage "$RELEASE" \
  --destination /home/maik/strix-halo-next
```

Restoration refuses to overwrite files edited since deployment and can recover
a partially promoted release. It restores controller/config files; it does not
stop or restart serving. Restore the corresponding release on every affected
node. If the operator config was changed separately, restart with the saved
original JSON after copying it to the same absolute path on all nodes.

```bash
python3 /home/maik/strix-halo-next/tools/bench_stream.py \
  --output /home/maik/strix-halo-next/results/stream.json --tokens 512 --repeats 2
bash /home/maik/strix-halo-next/tools/bench_sharegpt.sh sharegpt-run
python3 /home/maik/strix-halo-next/tools/quality_probe.py \
  --output /home/maik/strix-halo-next/results/quality.json
```

Run benchmarks sequentially. `bench_stream.py` stores requests, complete text,
token usage and stream timestamps. Its generation rate excludes first-token
delay. The existing ShareGPT48 comparison includes prompt processing and request
overhead; it must not be presented as the same metric. The quality probe is a
small deterministic smoke check, not a broad quality evaluation.

The opt-in `patches/mtp_local_argmax.py` adds full-vocabulary draft reduction and
has exact backup/restore. It was correctness-checked, but measured **48.44 t/s vs
48.60 baseline** on ShareGPT48: no established speed benefit. Therefore the
repository default leaves `use_local_argmax_reduction` unset. Do not change the
default based on the higher rates of unrelated short coding prompts.

Local lifecycle tests require no GPUs, SSH or containers:

```bash
python3 -m unittest discover -s tests -p test_lifecycle.py -v
python3 -m unittest discover -s scripts/native -p test_native_runtime.py -v
```

## W2 serial canonical configuration

The September 7 canonical service now enables `STRIX_MOE_W2_SERIAL=1` on all four
nodes, with the original W1 and AVX2 UMA library. The bounded serial W2 kernel
passes the complete model and post-promotion checks documented in
[the W2 report](../../docs/2026-09-07-moe-w2-serial-model.md). The configuration
is recorded in `bench/records/w2-serial-model-r1.json`; existing deployments
preserve the canonical host configuration rather than replacing it with the
repository default. Container overlays remain required: the earlier image
snapshot predates the UMA and W2 additions.

Every host retains `config/cluster.pre-w2-serial-20260907.json` under
`/home/maik/strix-halo-next`, SHA-256
`b7f78a76b557e81e86271027f4339a74c9759cb5b700faf3c2a35ee8df038546`.
The W2 source backup is inside each container at
`/opt/strix-halo-next/moe-w2-serial-20260907-r1/original.py`.
Restoring W2 leaves the prior UMA optimization enabled.

To roll back W2, run these commands sequentially from a repository checkout on
the management host with SSH access. Check each command succeeds before continuing;
the mutation tools reject unexpected source/configuration hashes and require a
stopped service for source changes. Use a new output directory for each attempt.

```bash
ssh maik@192.168.1.15 'python3 /home/maik/strix-halo-next/tools/cluster.py stop --config /home/maik/strix-halo-next/config/cluster.json'
python tools/deploy_moe_w2_serial.py restore --execute
python tools/set_w2_canonical_config.py restore --config bench/records/w2-serial-model-r1.json --output results/w2-rollback-config --execute
ssh maik@192.168.1.15 'python3 /home/maik/strix-halo-next/tools/cluster.py start --config /home/maik/strix-halo-next/config/cluster.json --tag w2-rollback --timeout 900'
```

For reactivation after that rollback, stop serving, run `deploy_moe_w2_serial.py
activate --execute`, run the configuration tool with `promote` and a new output
directory, then start with the canonical configuration and the 900-second timeout.
These tools operate on the already staged, hash-checked additions on this cluster;
they are not a clean-machine installer.
