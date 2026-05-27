# External DP Config Template

This document shows how to write YAML configs consumed by
`tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py`.

`server_cmd_template` contains only the arguments after
`vllm serve <model>`. The framework prepends `vllm serve` and the top-level
`model` automatically.

Do not write `proxy_node_index`, `proxy_host`, `proxy_port`, `proxy_script`, or
`dp_group` in YAML. The framework derives proxy metadata from `routing.type`,
and roles are selected by `routing.groups`.

## Generic DP Template

Use this template for generic external data parallel serving. This mode uses
`--data-parallel-rank`, so it is intended for MoE models. For dense models, use
independent vLLM instances instead of external DP rank arguments.

```yaml
test_name: "test Qwen3-30B-A3B generic external dp"
model: "Qwen/Qwen3-30B-A3B"
num_nodes: 2
npu_per_node: 16

# Optional for local debugging. In CI, cluster IPs are resolved from LWS DNS.
# cluster_hosts:
#   - "172.22.0.xxx"
#   - "172.22.0.xxx"

routing:
  type: "generic_dp"
  groups:
    worker: [0, 1]

config:
  - node_index: 0
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 4
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 1
    dp_address: "${NODE_0_IP}"

  - node_index: 1
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 4
    dp_size_local: 2
    dp_rank_start: 2
    tp_size: 1
    dp_address: "${NODE_0_IP}"

templates:
  - node_index: 0
    envs: &generic_env
      VLLM_USE_MODELSCOPE: "true"
      OMP_PROC_BIND: "false"
      OMP_NUM_THREADS: "10"
      PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
      ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"
      HCCL_BUFFSIZE: "1024"
      SERVER_PORT: "${PORT}"
    server_cmd_template: &generic_server_cmd
      - --host
      - "0.0.0.0"
      - --port
      - $SERVER_PORT
      - --data-parallel-size
      - ${DP_SIZE}
      - --data-parallel-rank
      - ${DP_RANK}
      - --data-parallel-address
      - ${DP_ADDRESS}
      - --data-parallel-rpc-port
      - ${DP_RPC_PORT}
      - --tensor-parallel-size
      - ${TP_SIZE}
      - --max-model-len
      - "4096"
      - --trust-remote-code
      - --enable-expert-parallel

  - node_index: 1
    envs:
      <<: *generic_env
    server_cmd_template: *generic_server_cmd

benchmarks:
  perf:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in3500-bs2800
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 4
    max_out_len: 16
    batch_size: 1
    request_rate: 1
    baseline: 1
    threshold: 0.1

  acc:
    case_type: accuracy
    dataset_path: vllm-ascend/gsm8k
    request_conf: vllm_api_general_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
    num_prompts: 4
    max_out_len: 16
    batch_size: 1
    baseline: 0
    threshold: 100
```

## Disaggregated Prefill Template

Use this template for PD disaggregation. `routing.groups` decides which config
entries run as prefillers or decoders. The framework derives the PD proxy script
from `routing.type`, so do not write `proxy_*` fields in YAML.

```yaml
test_name: "test DeepSeek-V2-Lite-W8A8 external dp disaggregated_prefill"
model: "vllm-ascend/DeepSeek-V2-Lite-W8A8"
num_nodes: 2
npu_per_node: 16

# Optional for local debugging. In CI, cluster IPs are resolved from LWS DNS.
# cluster_hosts:
#   - "172.22.0.xxx"
#   - "172.22.0.xxx"

routing:
  type: "disaggregated_prefill"
  groups:
    prefiller: [0]
    decoder: [1]

config:
  - node_index: 0
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 2
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 1
    dp_address: "${NODE_0_IP}"

  - node_index: 1
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 2
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 1
    dp_address: "${NODE_1_IP}"

env_common: &env_common
  HCCL_OP_
  VLLM_USE_MODELSCOPE: "true"
  OMP_PROC_BIND: "false"
  OMP_NUM_THREADS: "10"
  PYTORCH_NPU_ALLOC_CONF: "expandable_segments:True"
  ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"
  HCCL_BUFFSIZE: "256"
  SERVER_PORT: "${PORT}"
  VLLM_ASCEND_ENABLE_FLASHCOMM1: "0"

templates:
  - node_index: 0
    envs:
      <<: *env_common
    server_cmd_template:
      - --host
      - "0.0.0.0"
      - --port
      - $SERVER_PORT
      - --data-parallel-size
      - ${DP_SIZE}
      - --data-parallel-rank
      - ${DP_RANK}
      - --data-parallel-address
      - ${DP_ADDRESS}
      - --data-parallel-rpc-port
      - ${DP_RPC_PORT}
      - --tensor-parallel-size
      - ${TP_SIZE}
      - --trust-remote-code
      - --quantization
      - ascend
      - --enable-expert-parallel
      - --kv-transfer-config
      - '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": 2,
            "tp_size": 1
          },
          "decode": {
            "dp_size": 2,
            "tp_size": 1
          }
        }}'

  - node_index: 1
    envs:
      <<: *env_common
    server_cmd_template:
      - --host
      - "0.0.0.0"
      - --port
      - $SERVER_PORT
      - --data-parallel-size
      - ${DP_SIZE}
      - --data-parallel-rank
      - ${DP_RANK}
      - --data-parallel-address
      - ${DP_ADDRESS}
      - --data-parallel-rpc-port
      - ${DP_RPC_PORT}
      - --tensor-parallel-size
      - ${TP_SIZE}
      - --trust-remote-code
      - --quantization
      - ascend
      - --enable-expert-parallel
      - --kv-transfer-config
      - '{"kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_consumer",
        "kv_port": "30200",
        "engine_id": "1",
        "kv_connector_extra_config": {
          "prefill": {
            "dp_size": 2,
            "tp_size": 1
          },
          "decode": {
            "dp_size": 2,
            "tp_size": 1
          }
        }}'

benchmarks:
  perf:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in3500-bs2800
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    max_out_len: 128
    batch_size: 4
    request_rate: 1
    baseline: 1
    threshold: 0.1

  acc:
    case_type: accuracy
    dataset_path: vllm-ascend/gsm8k
    request_conf: vllm_api_general_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
    max_out_len: 48
    batch_size: 4
    baseline: 0
    threshold: 100
```

## Field Notes

- `test_name`: Human-readable test name. It is also used when writing benchmark
  result metadata.
- `model`: Model passed to `vllm serve <model>` and AISBench requests.
- `num_nodes`: Number of config entries and templates expected.
- `npu_per_node`: Device capacity validation for each node.
- `cluster_hosts`: Optional local-debug IP list. Omit it in CI unless a test
  needs fixed hosts.
- `routing.type`: Supported values are `generic_dp` and
  `disaggregated_prefill`.
- `routing.groups`: Maps config indices to roles. `generic_dp` requires
  `worker`; `disaggregated_prefill` requires `prefiller` and `decoder`.
- For `disaggregated_prefill`, use `kv_producer` for prefiller templates and
  `kv_consumer` for decoder templates.
- `config[].dp_size`: Global DP size for this DP group.
- `config[].dp_size_local`: Number of vLLM ranks started on this node.
- `config[].dp_rank_start`: First global DP rank owned by this node.
- `config[].dp_address`: DP master address. For one global DP group, use
  `${NODE_0_IP}` on all nodes. For PD disaggregation, use the prefiller master
  address for prefiller nodes and the decoder master address for decoder nodes.
- `templates`: One template per config entry. The framework expands one command
  per local DP rank.

The framework injects distributed network envs at startup:

```text
HCCL_IF_IP
HCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME
TP_SOCKET_IFNAME
LOCAL_IP
NIC_NAME
MASTER_IP
```

The framework also derives proxy metadata from `routing.type`:

```text
generic_dp -> examples/external_online_dp/dp_load_balance_proxy_server.py
disaggregated_prefill -> examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
```

The proxy runs on node 0, listens on `${NODE_0_IP}:1999`, and is used by node 0
for benchmark requests.

## Template Variables

The following variables are available in `envs` and `server_cmd_template`:

```text
${MODEL}
${PORT_START}
${PORT}
${DP_SIZE}
${DP_SIZE_LOCAL}
${DP_RANK_START}
${DP_RANK}
${LOCAL_RANK}
${TP_SIZE}
${CP_SIZE}
${SP_SIZE}
${PP_SIZE}
${DP_ADDRESS}
${DP_RPC_PORT}
${VISIBLE_DEVICES}
${NODE_INDEX}
${CONFIG_INDEX}
${NODE_0_IP}, ${NODE_1_IP}, ...
${LOCAL_IP}
${MASTER_IP}
${LWS_WORKER_INDEX}
```

Command arguments can also reference rendered environment variables with
shell-style `$VARNAME`, for example:

```yaml
envs:
  SERVER_PORT: "${PORT}"
server_cmd_template:
  - --port
  - $SERVER_PORT
```

## Checks Before Running

- Keep `len(config) == num_nodes` and `len(templates) == num_nodes`.
- Make sure each config index is assigned to exactly one routing group.
- Ensure `dp_rank_start + dp_size_local <= dp_size`.
- Ensure `dp_size_local * tp_size * cp_size * sp_size * pp_size <= npu_per_node`.
- For `generic_dp` with `--data-parallel-rank`, use an MoE model and
  `--enable-expert-parallel`.
- Set `--max-model-len` large enough for benchmark input tokens plus
  `max_out_len`.
