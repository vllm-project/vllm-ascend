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
The top-level `model` is the serve model path. `request_model` is optional and
is used for OpenAI/AISBench requests when `--served-model-name` is set in the
template.

The Python entrypoints under `scripts/` are intentionally kept compact:

- `external_dp_config.py`: YAML loading, schema dataclasses, cluster placeholder
  resolution, and endpoint expansion.
- `external_dp_utils.py`: command rendering, process helpers, proxy command
  generation, and benchmark result JSON writing.
- `test_external_dp.py`: pytest orchestration for backend startup, proxy startup,
  AISBench execution, log collection, and cleanup.
