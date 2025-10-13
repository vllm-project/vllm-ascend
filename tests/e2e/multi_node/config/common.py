import os

CONFIG_PATH = "tests/e2e/multi_node/config/config.json"
RANKTABLE_GEN_PATH = "examples/disaggregated_prefill_v1/gen_ranktable.py"
RANKTABLE_PATH = "examples/disaggregated_prefill_v1/ranktable.json"
DISAGGEGATED_PREFILL_PORT = 6657
ASCEND_ENV_PATH = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
LIB_PATH = "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/"


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "2"))


def get_npu_per_node() -> int:
    return int(os.getenv("NPU_PER_NODE", "16"))
