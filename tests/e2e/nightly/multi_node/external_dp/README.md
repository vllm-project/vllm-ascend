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
- `external_dp_utils.py`: command rendering, process helpers, proxy command
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
  - 172.17.0.3
  - 172.17.0.2
```

Use the same temporary config path on both nodes. Start the decoder node first:

```bash
cd /vllm-workspace/vllm-ascend
export CONFIG_YAML_PATH=/tmp/external_dp_pd_local.yaml
export LWS_WORKER_INDEX=1
pytest -sv --show-capture=no tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py
```

Then start the prefiller/proxy/benchmark node:

```bash
cd /vllm-workspace/vllm-ascend
export CONFIG_YAML_PATH=/tmp/external_dp_pd_local.yaml
export LWS_WORKER_INDEX=0
pytest -sv --show-capture=no tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py
```

When the model, benchmark data, and Ascend environment are already prepared,
these are the only extra settings required. Logs default to
`/tmp/external_dp_logs`; set `LOG_PREFIX` only if you want the framework to
collect a compressed log artifact.
