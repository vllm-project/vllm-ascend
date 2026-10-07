# Periodic E2E Test Cases

This directory contains test cases shared by the nightly and weekly E2E
pipelines. It is the starting point for contributors who want to add, update,
run, or schedule a periodic E2E case.

The test implementation is independent of its schedule and resource topology:

- Test files and YAML case definitions live under `tests/e2e/cases/`.
- Shared pytest runners and service orchestration live under
  `tests/e2e/common/`.
- Nightly and weekly frequency is selected in the corresponding workflow
  matrix.
- Single-node, double-node, and multi-node resources are selected by the
  matrix section that contains the case.
- Multi-node internal or external load balancing is selected by the
  `dp_load_balancing` field in the case YAML.

## Contents

- [Directory layout](#directory-layout)
- [How a case is selected](#how-a-case-is-selected)
- [Choose a test style](#choose-a-test-style)
- [Single-node YAML cases](#single-node-yaml-cases)
- [Multi-node YAML cases](#multi-node-yaml-cases)
- [Pytest-driven cases](#pytest-driven-cases)
- [Nightly and weekly scheduling](#nightly-and-weekly-scheduling)
- [Adding a case](#adding-a-case)
- [Review checklist](#review-checklist)

## Directory Layout

```text
tests/e2e/
├── cases/
│   ├── models/
│   │   └── configs/
│   │       ├── DeepSeek/
│   │       ├── GLM/
│   │       ├── Kimi/
│   │       └── Qwen/
│   └── features/
│       ├── kv_pool/
│       ├── openai_api_compatibility/
│       └── structured_output/
└── common/
    ├── single_node/
    └── multi_node/
```

Use the following placement rules:

- Put model accuracy, performance, and model-specific serving configurations
  under `models/configs/<model-family>/`.
- Put feature-focused cases under `features/<feature>/`. Examples include KV
  pooling, structured output, speculative decoding, and API compatibility.
- Keep reusable runners, configuration loaders, service lifecycle code, and
  cluster orchestration under `tests/e2e/common/`.

A YAML file does not need to be under `models/configs/`. The workflow resolves
it from the explicit `config_base_path`, so a feature YAML may stay next to the
feature tests that own it.

## How a Case Is Selected

The following dimensions are configured independently:

| Dimension | Source | Examples |
| --- | --- | --- |
| Test content | `tests/e2e/cases/` | model, KV pool, structured output |
| Frequency | workflow matrix | nightly, weekly |
| Resource topology | workflow matrix section | single node, double node, multi node |
| DP load balancing | multi-node case YAML | `internal`, `external` |
| Execution framework | `tests/e2e/common/` | pytest runner, LWS, service management |

The workflow matrices are:

- `.github/workflows/configs/nightly_config.yaml`
- `.github/workflows/configs/weekly_config.yaml`

The `name` in a matrix entry is the identifier used by `/nightly` and
`/weekly`. Keep it unique within the applicable SoC matrix and treat it as a
stable case identifier.

## Choose a Test Style

Periodic E2E supports two test styles.

### YAML-driven cases

Use a YAML-driven case when the shared framework can start the service and run
the required requests or benchmarks. The matrix entry supplies
`config_file_path` and `config_base_path`.

### Pytest-driven cases

Use a pytest-driven case when the test needs custom fixtures, assertions, or
control flow. The matrix entry supplies `tests`, pointing to a pytest file or
directory under `tests/e2e/cases/`.

Use one style for each matrix entry. Do not set both `tests` and
`config_file_path` for the same entry.

## Single-Node YAML Cases

A single-node YAML contains a `test_cases` list. The shared runner creates one
pytest parameter for each item in that list.

Minimal example:

```yaml
test_cases:
  - name: Qwen-example
    model: Qwen/Qwen3-8B
    envs:
      SERVER_PORT: DEFAULT_PORT
    server_cmd:
      - --tensor-parallel-size
      - "8"
      - --port
      - $SERVER_PORT
```

Add it to a matrix under `single_node.test_config`:

```yaml
a3:
  single_node:
    test_config:
      - name: qwen-example
        os: linux-aarch64-a3-800i-8
        config_file_path: Qwen-example.yaml
        config_base_path: tests/e2e/cases/models/configs/Qwen
```

Run it locally from the repository root:

```bash
export CONFIG_BASE_PATH=tests/e2e/cases/models/configs/Qwen
export CONFIG_YAML_PATH=Qwen-example.yaml
pytest -sv tests/e2e/common/single_node/test_single_node.py
```

### Single-node field reference

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `test_cases` | list | Yes | - | Cases loaded from this YAML file |
| `name` | string | Yes | - | Pytest case ID and log identifier |
| `model` | string | Yes | - | Model repository or local path |
| `service_mode` | string | No | `openai` | `openai` or `epd` |
| `envs` | mapping | Yes | `{}` | Environment passed to the service |
| `server_cmd` | list | Conditional | `[]` | vLLM arguments for `openai` mode |
| `server_cmd_extra` | list | No | `[]` | Arguments appended to `server_cmd` |
| `prompts` | list | No | built-in prompt | Requests used by functional checks |
| `api_keyword_args` | mapping | No | built-in values | OpenAI API request arguments |
| `test_content` | list | No | `completion` | Registered functional test phases |
| `benchmarks` | mapping | No | `{}` | AISBench accuracy or performance jobs |
| `epd_server_cmds` | list of lists | Conditional | `[]` | Encode and decode commands for EPD |
| `epd_proxy_args` | list | Conditional | `[]` | EPD proxy arguments |
| `kv_pool` | mapping | No | - | Managed Mooncake or Memcache service |

`name` must be non-empty. It appears in pytest output, for example
`test_single_node[Qwen-example]`, and in the single-node start marker.

The recognized port variables are `SERVER_PORT`, `ENCODE_PORT`, `PD_PORT`, and
`PROXY_PORT`. A missing value or `DEFAULT_PORT` is replaced with a free local
port. Commands may reference values with `$VAR` or `${VAR}`.

Use `server_cmd_extra` when cases share a base command but one case needs
additional arguments. The loader appends it to `server_cmd` before launching
the service.

Unknown fields are retained in `extra_config` for registered test handlers.
Adding a new handler or changing the service lifecycle is a framework change;
see `tests/e2e/common/single_node/GUIDE_AND_TEMPLATE.md` for those extension
points.

### Multiple single-node cases and YAML anchors

One YAML may contain several cases. YAML anchors can keep their shared service
configuration in one place:

```yaml
_envs: &envs
  SERVER_PORT: DEFAULT_PORT
  OMP_NUM_THREADS: "1"

_server_cmd: &server_cmd
  - --port
  - $SERVER_PORT
  - --tensor-parallel-size
  - "8"

_benchmarks: &benchmarks
  perf:
    case_type: performance
    dataset_path: vllm-ascend/GSM8K-in3500-bs400
    request_conf: vllm_api_stream_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_str_perf
    num_prompts: 400
    max_out_len: 1500
    batch_size: 1000
    baseline: 1
    threshold: 0.97

test_cases:
  - name: case-eager
    model: Qwen/Qwen3-8B
    envs:
      <<: *envs
    server_cmd: *server_cmd
    server_cmd_extra:
      - --enforce-eager

  - name: case-graph
    model: Qwen/Qwen3-8B
    envs:
      <<: *envs
    server_cmd: *server_cmd
    benchmarks:
      <<: *benchmarks
```

### Single-node EPD case

Use `service_mode: epd` when one case starts encode and decode services plus a
proxy:

```yaml
test_cases:
  - name: qwen-epd-example
    model: Qwen/Qwen3-8B
    service_mode: epd
    envs:
      ENCODE_PORT: DEFAULT_PORT
      PD_PORT: DEFAULT_PORT
      PROXY_PORT: DEFAULT_PORT

    epd_server_cmds:
      - [--port, $ENCODE_PORT, --model, Qwen/Qwen3-8B]
      - [--port, $PD_PORT, --model, Qwen/Qwen3-8B]

    epd_proxy_args:
      - --host
      - 127.0.0.1
      - --port
      - $PROXY_PORT
      - --encode-servers-urls
      - http://localhost:$ENCODE_PORT
      - --decode-servers-urls
      - http://localhost:$PD_PORT
      - --prefill-servers-urls
      - disable

    test_content:
      - chat_completion
```

### Single-node managed KV pool

The framework starts the selected pool before vLLM and stops it after all
service and proxy processes exit. Pool ports must be available on the host.
Keep the matching `--kv-transfer-config` in `server_cmd` or
`epd_server_cmds`; the framework passes that argument through unchanged.

Mooncake example:

```yaml
kv_pool:
  type: mooncake
  master_port: 50088
  metrics_port: 50089
  config:
    metadata_server: P2PHANDSHAKE
    protocol: ascend
    device_name: ""
    global_segment_size: 1GB
    preferred_segment: false
    prefer_alloc_in_same_node: true
```

Memcache example:

```yaml
kv_pool:
  type: memcache
  meta_service_port: 5000
  config_store_port: 6000
  config:
    meta:
      ock.mmc.log_level: error
    local:
      ock.mmc.log_level: error
      ock.mmc.local_service.world_size: 256
      ock.mmc.local_service.protocol: device_sdma
      ock.mmc.local_service.dram.size: 1GB
```

### Single-node debugging

Use a prepared NPU environment with `pytest`, PyYAML, the OpenAI client, and
AISBench installed as required by the selected case.

```bash
# Select one case from a YAML containing several test_cases.
pytest -sv tests/e2e/common/single_node/test_single_node.py -k case-name

# Stop after the first failure.
pytest -sv tests/e2e/common/single_node/test_single_node.py -x
```

Use `-s` to keep service output attached to pytest while diagnosing startup or
request failures; it is already enabled by `-sv`.

## Multi-Node YAML Cases

Multi-node cases use the shared entrypoint
`tests/e2e/common/multi_node/run.sh`. The entrypoint starts the common pytest
runner, which reads `dp_load_balancing` from the selected YAML and dispatches
to the appropriate implementation.

Every new or migrated multi-node YAML must declare one of:

```yaml
dp_load_balancing: internal
```

```yaml
dp_load_balancing: external
```

The field describes how the case starts and balances its serving processes. It
belongs to the case YAML because the required serving behavior is part of the
test scenario. Do not encode this choice in the directory name.

### Meaning of `dp_load_balancing`

`dp_load_balancing` selects who creates the DP ranks and who distributes
requests across them. It does not select the number of nodes and does not, by
itself, enable or disable PD disaggregation.

| Value | Rank startup | Request distribution | YAML layout |
| --- | --- | --- | --- |
| `internal` | Each `deployment` entry starts its configured `vllm serve` process; vLLM creates and coordinates the DP ranks declared by its DP arguments | The vLLM server performs DP dispatch internally | `deployment`, with a complete `server_cmd` per node |
| `external` | The E2E framework expands `config` and `templates`, then starts an individual `vllm serve` process for each local DP rank | A framework-managed external proxy routes requests to the rank endpoints | `config`, `templates`, and `routing.groups` |

An internal-DP case can still use PD disaggregation. In that situation the
framework starts a PD proxy to route traffic between the prefill and decode
server groups. That proxy performs PD routing; DP rank creation and balancing
inside each vLLM server remain internal.

An external-DP case starts independently addressable DP rank processes. The
framework derives their commands from `server_cmd_template`, starts the
external proxy, and sends benchmark requests through that proxy.

Choose `internal` when the scenario is intended to exercise vLLM's native DP
startup and dispatch. Choose `external` when the scenario requires the E2E
framework to launch rank endpoints separately and balance requests through its
proxy.

Add the case to the matrix section that provides the required number of nodes:

```yaml
a3:
  multi_node:
    test_config:
      - name: qwen-example-pd
        config_file_path: Qwen-example-PD.yaml
        config_base_path: tests/e2e/cases/models/configs/Qwen
        size: 4
```

The matrix controls the allocation:

- `double_node.test_config` and `multi_node.test_config` select the applicable
  reusable workflow and resource pool.
- `size` is passed to the cluster resource as the requested node count.

The YAML controls serving behavior, including internal or external load
balancing and any disaggregated prefill configuration required by the case.

The YAML `num_nodes` and the matrix `size` must describe the same allocation.
`npu_per_node` records the device capacity available to each node and is used
when validating the parallel layout.

### Internal DP configuration

Internal DP uses a `deployment` list. Each item contains the environment and a
complete `vllm serve` command for one node. The framework starts the service
directly and may start the disaggregated-prefill proxy when that mode is
enabled.

| Field | Required | Description |
| --- | --- | --- |
| `test_name` | No | Human-readable case name and result metadata |
| `model` | Yes | Model used by requests and benchmarks |
| `num_nodes` | Yes | Number of deployment entries and scheduled nodes |
| `npu_per_node` | Yes | NPU capacity available on each node |
| `dp_load_balancing` | Yes | Must be `internal` |
| `cluster_hosts` | Local only | Explicit node IPs outside LWS |
| `env_common` | No | YAML anchor for shared environment variables |
| `disaggregated_prefill` | No | Internal PD role assignment |
| `deployment` | Yes | One environment and server command per node |
| `benchmarks` | Yes | AISBench jobs; use `{}` when none are required |

Internal DP example:

```yaml
test_name: qwen-internal-pd
model: Qwen/Qwen3-235B-A22B
num_nodes: 2
npu_per_node: 16
dp_load_balancing: internal

# Set this only when running outside LWS.
# cluster_hosts: [10.0.0.10, 10.0.0.11]

env_common: &env_common
  VLLM_USE_MODELSCOPE: "true"
  SERVER_PORT: "8080"
  OMP_NUM_THREADS: "1"

disaggregated_prefill:
  enabled: true
  prefiller_host_index: [0]
  decoder_host_index: [1]

deployment:
  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve Qwen/Qwen3-235B-A22B
      --host 0.0.0.0
      --port $SERVER_PORT
      --data-parallel-size 2
      --data-parallel-size-local 2
      --tensor-parallel-size 8
      --enable-expert-parallel
      --kv-transfer-config
      '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer"}'

  - envs:
      <<: *env_common
    server_cmd: >
      vllm serve Qwen/Qwen3-235B-A22B
      --host 0.0.0.0
      --port $SERVER_PORT
      --data-parallel-size 2
      --data-parallel-size-local 2
      --tensor-parallel-size 8
      --enable-expert-parallel
      --kv-transfer-config
      '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_consumer"}'

benchmarks:
  acc:
    case_type: accuracy
    dataset_path: vllm-ascend/gsm8k-lite
    request_conf: vllm_api_general_chat
    dataset_conf: gsm8k/gsm8k_gen_0_shot_cot_chat_prompt
    max_out_len: 4096
    batch_size: 512
    baseline: 95
    threshold: 10
```

`prefiller_host_index` and `decoder_host_index` contain node indices, not IP
addresses. A headless node that only contributes distributed workers is not a
separate proxy endpoint.

### External DP configuration

External DP uses separate `config` and `templates` lists. A config entry
defines the DP ranks owned by one node, while the corresponding template
defines the environment and vLLM arguments expanded for each local rank.

`server_cmd_template` contains only arguments after `vllm serve <model>`; the
framework prepends those tokens automatically.

Do not add `proxy_node_index`, `proxy_host`, `proxy_port`, `proxy_script`, or
`dp_group` to the YAML. The framework derives proxy metadata from
`routing.type`, and `routing.groups` assigns each config index a role.

| Field | Required | Description |
| --- | --- | --- |
| `test_name` | No | Human-readable case name and result metadata |
| `model` | Yes | Model prepended to each `vllm serve` command |
| `num_nodes` | Yes | Number of config and template entries |
| `npu_per_node` | Yes | NPU capacity used for layout validation |
| `dp_load_balancing` | Yes | Must be `external` |
| `cluster_hosts` | Local only | Explicit node IPs outside LWS |
| `routing.type` | Yes | Currently `disaggregated_prefill` |
| `routing.groups` | Yes | Config indices assigned to each serving role |
| `config` | Yes | Per-node DP and parallel layout |
| `env_common` | No | YAML anchor for shared environment variables |
| `templates` | Yes | Per-config environment and command template |
| `kv_pool` | No | Managed Mooncake or Memcache service |
| `benchmarks` | Yes | AISBench jobs run on node 0; may be `{}` |

External DP example:

```yaml
test_name: qwen-external-pd
model: Qwen/Qwen3-235B-A22B
num_nodes: 2
npu_per_node: 16
dp_load_balancing: external

# Set this only when running outside LWS.
# cluster_hosts:
#   - 10.0.0.10
#   - 10.0.0.11

routing:
  type: disaggregated_prefill
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
    tp_size: 8
    dp_address: ${NODE_0_IP}

  - node_index: 1
    port_start: 7100
    dp_rpc_port: 12321
    dp_size: 2
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 8
    dp_address: ${NODE_1_IP}

env_common: &env_common
  VLLM_USE_MODELSCOPE: "true"
  SERVER_PORT: ${PORT}
  ASCEND_RT_VISIBLE_DEVICES: ${VISIBLE_DEVICES}
  OMP_NUM_THREADS: "10"

templates:
  - node_index: 0
    envs:
      <<: *env_common
    server_cmd_template:
      - --host
      - 0.0.0.0
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
      - --enable-expert-parallel
      - --kv-transfer-config
      - >-
        {"kv_connector":"MooncakeConnectorV1",
         "kv_role":"kv_producer","kv_port":"30000"}

  - node_index: 1
    envs:
      <<: *env_common
    server_cmd_template:
      - --host
      - 0.0.0.0
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
      - --enable-expert-parallel
      - --kv-transfer-config
      - >-
        {"kv_connector":"MooncakeConnectorV1",
         "kv_role":"kv_consumer","kv_port":"30200"}

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
```

The main `config` fields are:

- `node_index`: Node that owns this config entry.
- `port_start`: First API port assigned to local DP ranks.
- `dp_rpc_port`: RPC coordination port for the DP group.
- `dp_size`: Global size of this DP group.
- `dp_size_local`: Number of vLLM ranks started on this node.
- `dp_rank_start`: First global DP rank owned by this node.
- `dp_address`: DP master address. Use one group master address for every
  member of the same group.
- `tp_size`, `cp_size`, `sp_size`, and `pp_size`: Parallel sizes expanded into
  the command template. Optional parallel sizes default according to the
  external DP loader.

For disaggregated prefill, prefiller templates normally use
`kv_role: kv_producer` and decoder templates use `kv_role: kv_consumer`.
The framework derives the proxy script from `routing.type`, runs the proxy on
node 0, and currently assigns port `1999`. Benchmark requests are sent through
that proxy.

### External DP template variables

The following variables are available in `envs` and
`server_cmd_template`:

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

Command arguments may also reference rendered environment variables such as
`$SERVER_PORT`:

```yaml
envs:
  SERVER_PORT: ${PORT}
server_cmd_template:
  - --port
  - $SERVER_PORT
```

The framework injects these distributed network variables at startup:

```text
HCCL_IF_IP
HCCL_SOCKET_IFNAME
GLOO_SOCKET_IFNAME
TP_SOCKET_IFNAME
LOCAL_IP
NIC_NAME
MASTER_IP
```

### External DP managed KV pool

External DP supports an optional managed KV pool. Mooncake uses `master_port`
and `metrics_port`; Memcache uses `meta_service_port` and
`config_store_port`. All configured service ports must be available on node 0.
The `kv_pool` blocks shown in the single-node section use the same schema when
placed at the top level of an external DP YAML.

For external DP, the framework writes the generated backend configuration for
every rank. It overwrites the Mooncake master address or the Memcache service
URLs with resolved cluster values. The generated files and service logs are
stored with the node logs:

```text
<external-dp-log-root>/node-<index>/runtime/mooncake.json
<external-dp-log-root>/node-0/mooncake-master.log
<external-dp-log-root>/node-<index>/runtime/mmc-meta.conf
<external-dp-log-root>/node-<index>/runtime/mmc-local.conf
<external-dp-log-root>/node-0/memcache-meta-service.log
```

Mooncake additionally injects `MOONCAKE_CONFIG_PATH` and `MOONCAKE_MASTER`.
Memcache injects `MMC_LOCAL_CONFIG_PATH`.

When KV pooling and PD transfer are both required, use `MultiConnector` in the
server command. Set the prefiller's outer connector and child connectors to
`kv_producer`, and set the decoder equivalents to `kv_consumer`. Select
`backend: mooncake` or `backend: memcache` explicitly in the
`AscendStoreConnector` configuration.

Prefiller example:

```yaml
- --kv-transfer-config
- >-
  {
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "MooncakeConnectorV1",
          "kv_role": "kv_producer",
          "kv_port": "30000",
          "kv_connector_extra_config": {
            "prefill": {"dp_size": 2, "tp_size": 8},
            "decode": {"dp_size": 2, "tp_size": 8}
          }
        },
        {
          "kv_connector": "AscendStoreConnector",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {
            "lookup_rpc_port": "0",
            "backend": "mooncake"
          }
        }
      ]
    }
  }
```

The decoder uses the same structure with `kv_consumer` on the outer connector
and both child connectors.

### Benchmark fields

Each key under `benchmarks` names one AISBench job. Common fields include:

| Field | Description |
| --- | --- |
| `case_type` | `accuracy` or `performance` |
| `dataset_path` | Dataset repository or local path |
| `request_conf` | AISBench request configuration |
| `dataset_conf` | AISBench dataset configuration |
| `num_prompts` | Number of prompts for performance jobs |
| `max_out_len` | Maximum generated tokens |
| `batch_size` | Client concurrency or batch size |
| `request_rate` | Request arrival rate; use `0` for no rate limit where supported |
| `baseline` | Expected reference result |
| `threshold` | Allowed accuracy or performance threshold |

Optional request fields such as `temperature`, `top_k`, and `top_p` are passed
to the applicable benchmark configuration.

### Multi-node validation checklist

- Keep the matrix `size` consistent with YAML `num_nodes`.
- For external DP, keep `len(config) == num_nodes` and
  `len(templates) == num_nodes`.
- Assign every external config index to exactly one routing group.
- Keep `dp_rank_start + dp_size_local <= dp_size`.
- Keep `dp_size_local * tp_size * cp_size * sp_size * pp_size` within
  `npu_per_node`.
- Use one DP master address for all members of the same DP group.
- Give producer and consumer connectors the correct PD roles.
- Set `--max-model-len` large enough for benchmark input tokens plus
  `max_out_len`.
- Ensure API, RPC, proxy, and managed-pool ports do not conflict.

For LWS details, bare-metal execution, environment variables, and log
locations, see
`docs/source/developer_guide/contribution/multi_node_test.md`.

## Pytest-Driven Cases

Place custom pytest cases under the most appropriate model or feature
directory. A matrix entry points directly to the test file or directory:

```yaml
a3:
  single_node:
    test_config:
      - name: structured-output-qwen3
        os: linux-aarch64-a3-800t-2
        tests: tests/e2e/cases/features/structured_output
```

Run the same target locally with pytest:

```bash
pytest -sv tests/e2e/cases/features/structured_output
```

If a pytest-driven case needs multiple nodes, its cluster startup and resource
requirements must be supported by the selected reusable workflow before the
case is added to a multi-node matrix.

## Nightly and Weekly Scheduling

Nightly and weekly reuse the same case files and common runners. A case may be
listed in either or both matrices:

- Add frequently needed regression coverage to `nightly_config.yaml`.
- Add longer, more expensive, or lower-frequency coverage to
  `weekly_config.yaml`.
- If the same case appears in both matrices, point both entries to the same
  test file or YAML unless the scenarios genuinely require different
  configuration.

Do not copy a case into a `nightly/` or `weekly/` directory to select its
frequency. Scheduling comes from matrix membership.

Example commands on an authorized pull request:

```text
/nightly qwen-example
/weekly qwen-example
```

Use the exact matrix `name`, including capitalization.

## Adding a Case

1. Decide whether the case tests a model or a feature.
2. Choose YAML-driven or pytest-driven execution.
3. Add the case under `tests/e2e/cases/`.
4. For a multi-node YAML, set `dp_load_balancing` explicitly.
5. Add an entry to the nightly or weekly matrix.
6. Set `config_base_path` explicitly for every YAML-driven entry.
7. Select the resource topology and runner that match the case requirements.
8. Run the pytest entrypoint locally in a prepared NPU environment.
9. Trigger the case by its matrix `name` on a pull request.

## Review Checklist

- The case is placed by test purpose rather than scheduling frequency.
- Model and feature cases use the appropriate directory.
- The matrix `name` is unique and descriptive.
- A YAML-driven entry has both `config_file_path` and `config_base_path`.
- A pytest-driven entry has `tests` and no YAML fields.
- A multi-node YAML declares `dp_load_balancing`.
- The requested runner and node count match the test requirements.
- Nightly and weekly entries reuse the same case when their scenarios match.
- Local pytest execution and the applicable CI trigger have been verified.
