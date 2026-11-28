# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pool A/B watch for DeepSeek-V4.1-Flash DP8 x TP1 + DSpark5 on eight A5 dies."""

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psutil
import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MooncakeKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMooncakeManager
from tests.e2e.conftest import DPVllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    ensure_hixl_topology,
    run_pool_ab,
    run_pool_hit_sequence,
)

MODEL = os.environ.get(
    "DEEPSEEK_V4_1_FLASH_MODEL_PATH",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
)

DATA_PARALLEL_SIZE = 8
TENSOR_PARALLEL_SIZE = 1
NUM_SPECULATIVE_TOKENS = 5
POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
# Two lcm blocks plus a tail -> 32,768 / 36,408 = 90% partial hit.
POOL_PERF_PROMPT_TOKENS = 2 * POOL_LCM_BLOCK_SIZE + 2 * POOL_LCM_BLOCK_SIZE // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
# Keep 4,096 chunked prefill on a5-8; DSpark still yields 16,384 reusable tokens.
POOL_BATCHED_TOKENS = 4096
CASE_NAME = "dsv4_1_dp8_dspark_mooncake"

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "mooncake",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
    },
}


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "?"


def _host_memory() -> str:
    """Summarize host and cgroup memory so a pod-level kill leaves a trace."""
    memory = psutil.virtual_memory()
    shm = psutil.disk_usage("/dev/shm") if os.path.isdir("/dev/shm") else None
    shm_text = f" shm_total={shm.total >> 30}GiB shm_free={shm.free >> 30}GiB" if shm else ""
    cgroup_limit = _read("/sys/fs/cgroup/memory.max")
    cgroup_usage = _read("/sys/fs/cgroup/memory.current")
    if cgroup_limit == "?":
        cgroup_limit = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        cgroup_usage = _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    devices = [name for name in ("/dev/uburma", "/dev/ummu", "/etc/hixlep") if os.path.exists(name)]
    return (
        f"mem_available={memory.available >> 30}GiB mem_total={memory.total >> 30}GiB{shm_text}"
        f" cgroup_limit={cgroup_limit} cgroup_usage={cgroup_usage}"
        f" pool_devices={devices or 'none'}"
    )


def _build_mooncake_pool_config() -> MooncakeKVPoolConfig:
    # Each DP rank contributes its own segment and local buffer to the pool, so
    # the size is per card. 16K/36K prompts need well under 1GiB of KV per rank;
    # 2GiB + 1GiB keeps host memory low enough for the Engram offload.
    return MooncakeKVPoolConfig(
        config={
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "ascend",
            "global_segment_size": "2GB",
            "local_buffer_size": "1GB",
            "preferred_segment": False,
            "prefer_alloc_in_same_node": True,
        },
        master_port=get_open_port(),
        metrics_port=get_open_port(),
    )


@contextmanager
def _leg(use_pool: bool, case_name: str):
    """Leg A enables the mooncake pool; leg B keeps everything else identical."""
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            data_parallel_size=DATA_PARALLEL_SIZE,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            enable_expert_parallel=True,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            max_num_seqs=64,
            tokenizer_mode="deepseek_v41",
            block_size=128,
            gpu_memory_utilization=0.9,
            quantization="deepseek_v4_fp8",
            engram_config={"cpu_offload": True, "dp_shared_memory": True},
            additional_config={"multistream_overlap_shared_expert": True, "combine_quant_mode": 4},
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            # Bound pool-side hangs: workers start sequentially, and a hung
            # generate should surface as ENV_BLOCKED instead of the job timeout.
            dp_start_timeout=600,
            dp_request_timeout=600,
            speculative_config={
                "method": "dspark",
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            },
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMooncakeManager(_build_mooncake_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[kv_pool_leg] {case_name} pool={use_pool} {_host_memory()}", flush=True)
        yield stack.enter_context(DPVllmRunner(MODEL, **kwargs))


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="prefix_caching,dspark",
    parallel="DP,TP,EP",
    deploy="pd_mix",
    hardware="A5",
    quantization="FP8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTHONHASHSEED": "0",
        # The 950DT pod exposes the UB CTP (URMA) interconnect: HIXL rejects
        # ub_rtp:device (conflicts with InterconType=2) and the UBoE auto path
        # cannot generate endpoints (E19999 GenDefaultUboeEndpointConfig). UB
        # CTP pure-URMA mode is selected with the bare ub_ctp descriptor.
        "ASCEND_GLOBAL_RESOURCE_CONFIG": '{"comm_resource_config.protocol_desc":["ub_ctp"]}',
        "ASCEND_LOCAL_COMM_RES": '{"version":"1.3"}',
        # DP8 settings mirrored from PR #22258's known-good A5 config.
        "HCCL_OP_EXPANSION_MODE": "CCU_SCHED",
        "HCCL_BUFFSIZE": "1600",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "2040",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_1_flash_dp8_tp1_dspark_pool_watch() -> None:
    ensure_hixl_topology()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)

    def _accuracy(runner) -> None:
        sampling = SamplingParams(temperature=0.0, max_tokens=16)
        outputs = runner.generate(["Hello, my name is"], sampling)
        assert outputs and outputs[0][1] and outputs[0][1][0].strip()

    def _hit(runner) -> dict:
        return run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            replicas=DATA_PARALLEL_SIZE,
            lcm_block_size=POOL_LCM_BLOCK_SIZE,
            # Only the exact token-identity comparison is disabled (DSpark's
            # speculative verify can shift tokens). The long-prefix semantic
            # guard stays on: answer_secret asserts R1/R2/R3 all answer 8.
            check_output=False,
            answer_secret=POOL_SECRET,
            tag="P2",
        )

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_tokens=POOL_PROMPT_TOKENS,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        token_identity=False,
        # Build the pool leg first so a pool-side pod kill surfaces without
        # paying for the control leg.
        pool_first=True,
    )
