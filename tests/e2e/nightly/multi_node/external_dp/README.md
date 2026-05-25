# External DP Nightly Tests

This directory contains the YAML-driven nightly framework for external data
parallel tests.

The first version is intentionally scoped to CI:

- `generic_dp` starts multiple local DP ranks per node and routes traffic
  through `examples/external_online_dp/dp_load_balance_proxy_server.py`.
- `disaggregated_prefill` starts prefiller and decoder ranks and routes traffic
  through the existing PD proxy examples.
- Benchmark execution reuses `tools.aisbench.run_aisbench_cases`; the result
  JSON keeps the existing benchmark fields and adds `external_dp_topology`.

`server_cmd_template` contains only arguments after `vllm serve <model>`.
The top-level `model` is used both for `vllm serve` and for OpenAI/AISBench
requests, matching the existing multi-node nightly flow.

Template env values use framework variables such as `${PORT}` and
`${VISIBLE_DEVICES}`. Command arguments can reference rendered env values with
shell-style `$VARNAME`, for example `--port $SERVER_PORT`.

The Python entrypoints under `scripts/` are intentionally kept compact:

- `external_dp_config.py`: YAML loading, schema dataclasses, cluster placeholder
  resolution, and endpoint expansion.
- `utils.py`: command rendering, process helpers, proxy command
  generation, and benchmark result JSON writing.
- `test_external_dp.py`: pytest orchestration for backend startup, proxy startup,
  AISBench execution, log collection, and cleanup.

## Local Two-Node PD Smoke

For local two-node debugging, copy `config/disaggregated_prefill_smoke.yaml` to
`/tmp/external_dp_pd_local.yaml` on both nodes and add explicit
`cluster_hosts`. This is the only External DP-specific config needed to avoid
the LWS DNS fallback.

```bash
cd /vllm-workspace/vllm-ascend
cp tests/e2e/nightly/multi_node/external_dp/config/disaggregated_prefill_smoke.yaml \
  /tmp/external_dp_pd_local.yaml
```

Add the hosts as a top-level YAML field:

```yaml
cluster_hosts:
  - "172.22.0.155"
  - "172.22.0.188"
```

Use the same temporary config path on both nodes. Set `WORKSPACE` to the parent
directory of the checkout.

Start the decoder node first:

```bash
cd /vllm-workspace/vllm-ascend
export WORKSPACE=/vllm-workspace
export IS_PR_TEST=false
export CONFIG_YAML_PATH=/tmp/external_dp_pd_local.yaml
export CONFIG_BASE_PATH=tests/e2e/nightly/multi_node/external_dp/config/
export LWS_WORKER_INDEX=1
bash tests/e2e/nightly/multi_node/scripts/run.sh
```

Then start the prefiller/proxy/benchmark node:

```bash
cd /vllm-workspace/vllm-ascend
export WORKSPACE=/vllm-workspace
export IS_PR_TEST=false
export CONFIG_YAML_PATH=/tmp/external_dp_pd_local.yaml
export CONFIG_BASE_PATH=tests/e2e/nightly/multi_node/external_dp/config/
export LWS_WORKER_INDEX=0
bash tests/e2e/nightly/multi_node/scripts/run.sh
```

When the model, benchmark data, and Ascend environment are already prepared,
these are the only extra settings required. `run.sh` sources the Ascend
environment, clears old NPU Python/VLLM processes, and then runs the pytest
entrypoint. Logs default to `/tmp/external_dp_logs`; set `LOG_PREFIX` only if
you want the framework to collect a compressed log artifact.

## Logs

Backend and proxy stdout/stderr are written to `EXTERNAL_DP_LOG_DIR`, which
defaults to `/tmp/external_dp_logs`. The directory layout is:

```text
/tmp/external_dp_logs/
  node-0/
    rank-0.log
    rank-1.log
    proxy.log
  node-1/
    rank-0.log
    rank-1.log
```

The first line of each rank log records the exact command and env used to start
that rank. `proxy.log` exists only on the configured proxy node, usually node 0.
Pytest orchestration logs and AISBench output are still printed to the terminal
running `run.sh`.

Long-running stages also print heartbeat logs every 30 seconds. The master node
prints progress while AISBench is running, including the proxy health status.
Non-master nodes print progress while waiting for the master backend to stop.

Use `EXTERNAL_DP_LOG_DIR` if you want an isolated log directory for one local
run:

```bash
export EXTERNAL_DP_LOG_DIR=/tmp/external_dp_logs_pd_local
```

To watch logs while services are starting, run these commands in another
terminal on the corresponding node:

```bash
# node 0: prefiller ranks and proxy
tail -F /tmp/external_dp_logs/node-0/rank-0.log \
        /tmp/external_dp_logs/node-0/rank-1.log \
        /tmp/external_dp_logs/node-0/proxy.log

# node 1: decoder ranks
tail -F /tmp/external_dp_logs/node-1/rank-0.log \
        /tmp/external_dp_logs/node-1/rank-1.log
```

If `EXTERNAL_DP_LOG_DIR` is set, replace `/tmp/external_dp_logs` with that
directory. `tail -F` can be started before the files exist; it will follow them
after the ranks create the files.

If the terminal keeps printing `Polling external DP endpoints`, check the rank
logs on the node mentioned in the polling line. For example, if node 1 rank 0 is
waiting, inspect:

```bash
tail -n 200 /tmp/external_dp_logs/node-1/rank-0.log
```

If a local rank process exits during startup, the pytest run fails immediately
and prints the corresponding log path. Remote rank failures are still diagnosed
from the remote node's rank logs.

For CI or local artifact collection, set `LOG_PREFIX` on each node:

```bash
export LOG_PREFIX=/tmp/external_dp_artifacts_pd_local
mkdir -p "$LOG_PREFIX"
```

At process cleanup, each node packs its raw rank/proxy logs into:

```text
$LOG_PREFIX/node_<LWS_WORKER_INDEX>_external_dp_logs.tar.gz
```

For example:

```bash
ls -lh /tmp/external_dp_artifacts_pd_local
tar -tzf /tmp/external_dp_artifacts_pd_local/node_0_external_dp_logs.tar.gz
tar -xzf /tmp/external_dp_artifacts_pd_local/node_0_external_dp_logs.tar.gz -C /tmp
```
