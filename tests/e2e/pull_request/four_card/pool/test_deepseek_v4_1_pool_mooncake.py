# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""DSV4.1-Flash KV pool hit watch on A5 4 cards with mooncake.

PR pool case #4: ``four_card_a5`` + DP4 x TP1 + 16K prompt + non-layerwise
mooncake + DSpark5 + V2 model runner.

The original "output is non-empty" assertion is kept; the only addition is the
shared R1-R4 pool hit sequence from ``tests.e2e.pull_request.kv_pool_watch``.
"""

import os
from unittest.mock import patch

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import run_pool_hit_sequence

MODEL = os.environ.get("DEEPSEEK_V4_1_FLASH_MODEL_PATH", "Eco-Tech/DeepSeek-V4.1-Flash-w8a8")

DATA_PARALLEL_SIZE = 4
# 16,388 tokens = 128 full blocks plus a 4-token tail at block_size=128.
POOL_PROMPT_TOKENS = 16388
POOL_MAX_MODEL_LEN = POOL_PROMPT_TOKENS + 128

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "mooncake",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
    },
}


def _build_mooncake_pool_config() -> MooncakeKVPoolConfig:
    return MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "global_segment_size": "5GB",
            "local_buffer_size": "5GB",
            "preferred_segment": False,
            "prefer_alloc_in_same_node": True,
        },
        master_port=get_open_port(),
        metrics_port=get_open_port(),
    )


@pytest.mark.e2e_model(MODEL)
@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_1_pool_hit_mooncake() -> None:
    pool_config = _build_mooncake_pool_config()
    with (
        SingleNodeMooncakeManager(pool_config, "dsv4_1_pr_pool_mooncake") as pool,
        patch.dict(os.environ, pool.server_envs),
        DPVllmRunner(
            MODEL,
            data_parallel_size=DATA_PARALLEL_SIZE,
            tensor_parallel_size=1,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=DATA_PARALLEL_SIZE,
            max_num_batched_tokens=POOL_PROMPT_TOKENS,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.9,
            quantization="ascend",
            tokenizer_mode="deepseek_v41",
            block_size=128,
            # KV pool watch: local prefix cache must be explicitly disabled,
            # and pool metrics require log stats.
            enable_prefix_caching=False,
            disable_log_stats=False,
            # A5 4-card profile from the V4.1 deployment doc: no --ipc=host.
            engram_config={"cpu_offload": True, "dp_shared_memory": False},
            speculative_config={
                "num_speculative_tokens": 5,
                "method": "dspark",
                "enforce_eager": True,
            },
            kv_transfer_config=POOL_KV_TRANSFER_CONFIG,
        ) as runner,
    ):
        outputs = runner.generate_greedy(["Hello, my name is"], max_tokens=8)
        assert outputs and outputs[0][0], "pool case must produce non-empty output"

        run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            replicas=DATA_PARALLEL_SIZE,
            tag="dsv4.1-dp4-mooncake",
        )
