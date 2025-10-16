import os

CONFIG_PATH = "tests/e2e/multi_node/config/config.json"
ASCEND_ENV_PATH = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
LIB_PATH = "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/"

# Disaggregated prefill specific
RANKTABLE_GEN_PATH = "examples/disaggregated_prefill_v1/gen_ranktable.py"
RANKTABLE_PATH = "examples/disaggregated_prefill_v1/ranktable.json"
LOAD_BALANCER_PROXY_SCRIPT = "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
DISAGGEGATED_PREFILL_PORT = 6657
PREFILLER_START_PORT = 20002
DECODER_START_PORT = 20002


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "2"))


def get_npu_per_node() -> int:
    return int(os.getenv("NPU_PER_NODE", "16"))
