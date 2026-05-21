from pathlib import Path

import pytest

from tests.e2e.nightly.multi_node.external_dp.scripts.external_dp_config import ExternalDPConfigLoader

GENERIC_EXTERNAL_DP_YAML = """
test_name: "generic external dp unit"
model: "Qwen/Qwen3-0.6B"
num_nodes: 2
npu_per_node: 16
cluster_hosts:
  - 10.0.0.1
  - 10.0.0.2
env_common: &env_common
  VLLM_USE_MODELSCOPE: "true"
config_common: &config_common
  host: "0.0.0.0"
  port_start: 7100
  dp_rpc_port: 12321
routing:
  type: "generic_dp"
  proxy_node_index: 0
  proxy_host: "${NODE_0_IP}"
  proxy_port: 1999
  proxy_script: "examples/external_online_dp/dp_load_balance_proxy_server.py"
  groups:
    worker: [0, 1]
config:
  - node_index: 0
    <<: *config_common
    dp_group: default
    dp_size: 4
    dp_size_local: 2
    dp_rank_start: 0
    tp_size: 1
    pp_size: 1
    dp_address: "${NODE_0_IP}"
  - node_index: 1
    <<: *config_common
    dp_group: default
    dp_size: 4
    dp_size_local: 2
    dp_rank_start: 2
    tp_size: 1
    pp_size: 1
    dp_address: "${NODE_0_IP}"
templates:
  - envs:
      <<: *env_common
      ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"
      SERVER_PORT: "${PORT}"
      LOCAL_ENDPOINT: "${LOCAL_IP}:${PORT}"
    server_cmd_template: &cmd
      - --host
      - ${HOST}
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
  - envs:
      <<: *env_common
      ASCEND_RT_VISIBLE_DEVICES: "${VISIBLE_DEVICES}"
      SERVER_PORT: "${PORT}"
    server_cmd_template: *cmd
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
"""


PD_EXTERNAL_DP_YAML = GENERIC_EXTERNAL_DP_YAML.replace(
    'type: "generic_dp"\n  proxy_node_index: 0\n  proxy_host: "${NODE_0_IP}"\n'
    '  proxy_port: 1999\n  proxy_script: "examples/external_online_dp/dp_load_balance_proxy_server.py"\n'
    "  groups:\n    worker: [0, 1]",
    'type: "disaggregated_prefill"\n  proxy_node_index: 0\n  proxy_host: "${NODE_0_IP}"\n'
    '  proxy_port: 1999\n  proxy_script: "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"\n'
    "  groups:\n    prefiller: [0]\n    decoder: [1]",
)


def write_config(tmp_path: Path, content: str, name: str = "external_dp.yaml") -> Path:
    config_path = tmp_path / name
    config_path.write_text(content, encoding="utf-8")
    return config_path


@pytest.fixture
def generic_config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path, GENERIC_EXTERNAL_DP_YAML)


@pytest.fixture
def generic_config(generic_config_path: Path):
    return ExternalDPConfigLoader.from_yaml(str(generic_config_path))


@pytest.fixture
def pd_config_path(tmp_path: Path) -> Path:
    return write_config(tmp_path, PD_EXTERNAL_DP_YAML)


@pytest.fixture
def pd_config(pd_config_path: Path):
    return ExternalDPConfigLoader.from_yaml(str(pd_config_path))
