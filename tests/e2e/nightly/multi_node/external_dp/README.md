# External DP Nightly Tests

This directory contains the YAML-driven nightly framework for external data
parallel tests.

The first version is intentionally scoped to CI:

- `generic_dp` starts multiple local DP ranks per node and routes traffic
  through `examples/external_online_dp/dp_load_balance_proxy_server.py`.
- `disaggregated_prefill` starts prefiller and decoder ranks and routes traffic
  through the existing PD proxy examples.
- Benchmark execution reuses `tools.aisbench.run_aisbench_cases`; the result
  JSON keeps the existing benchmark fields.

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

## Local Debugging

Local two-node run and log instructions are documented in the
[multi-node contribution guide](../../../../../docs/source/developer_guide/contribution/multi_node_test.md).
Keep local debugging steps there so internal DP and external DP use the same
shared `run.sh` guidance.
