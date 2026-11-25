# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Pool hit watch for DeepSeek-V4 DSA-PCP MTP on four A3 dies.

The upstream DSA-PCP acceptance case in
``four_card/context_parallel/test_deepseek_v4.py`` stays untouched. This file
rebuilds both legs itself with TP2 x PCP2 + EP, so the control leg is our own
bare engine instead of the online test.

Acceptance is recorded, not gated: the upstream floors were calibrated at
``max_num_seqs=4``, while the perf segment now runs 32 requests at concurrency
16. Output prefixes, exact hit tokens, R4 and the A/B throughput gate remain.
"""

import os
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    build_pool_accuracy_prompt,
    run_pool_ab,
    run_pool_hit_sequence,
)
from tests.e2e.pull_request.one_card.model_runner_v2.utils import calculate_acceptance_per_pos
from tests.e2e.pull_request.utils import PROMPTS_SHORT

MODEL = os.environ.get(
    "DEEPSEEK_V4_MTP_MODEL_PATH",
    "gdydems/DeepSeek-V4-Flash-w4a8-mtp",
)

TENSOR_PARALLEL_SIZE = 2
PREFILL_CONTEXT_PARALLEL_SIZE = 2
NUM_SPECULATIVE_TOKENS = 3
MAX_NUM_SEQS = 32
ACCURACY_MAX_TOKENS = 1024
# Measured on both vLLM pins (2 legs x 2 pins): pos0 0.609-0.622,
# pos1 0.315-0.350, pos2 0.150-0.164. Floors keep ~15-20% headroom.
MIN_ACCEPTANCE_RATES = [0.52, 0.26, 0.12]
# The pool leg must stay within 20% of the no-pool leg at every position.
ACCEPTANCE_AB_RATIO = 0.80

POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
# Perf prompt: two lcm blocks plus a tail -> 32,768 / 36,408 = 90% hit.
POOL_PERF_PROMPT_TOKENS = 2 * POOL_LCM_BLOCK_SIZE + 2 * POOL_LCM_BLOCK_SIZE // 9
POOL_MAX_MODEL_LEN = max(POOL_PROMPT_TOKENS, POOL_PERF_PROMPT_TOKENS) + 128
POOL_BATCHED_TOKENS = 4096

# Same prefixes as the upstream DSA-PCP MTP case.
EXPECTED_OUTPUT_PREFIXES = {
    "Hello, my name is": "Hello, my name is {name} and I am {age} years old.",
    "The president of the United States is": "The president of the United States is the head of the executive branch",
    "The capital of France is": "The capital of France is Paris",
    "The future of AI is": "The future of AI is not just about technology;",
}

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
    },
}


def _build_memcache_pool_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_sdma",
                # Per-die DRAM: 5GB x 4 local services = 20GB total.
                "ock.mmc.local_service.dram.size": "5GB",
            },
        },
    )


CASE_NAME = "dsv4_pcp2_mtp_pool_watch"


@contextmanager
def _leg(use_pool: bool, case_name: str):
    """Leg A enables the memcache pool; leg B keeps everything else identical."""
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            prefill_context_parallel_size=PREFILL_CONTEXT_PARALLEL_SIZE,
            enable_expert_parallel=True,
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            tokenizer_mode="deepseek_v4",
            block_size=128,
            distributed_executor_backend="mp",
            gpu_memory_utilization=0.9,
            quantization="ascend",
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,
            enforce_eager=False,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                # 1 + 3 draft tokens -> capture sizes must be multiples of 4.
                "cudagraph_capture_sizes": [16, 32],
            },
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            },
            attention_config={"indexer_kv_dtype": "int8"},
            additional_config={
                "enable_prefill_mc2": True,
                # TODO: Re-enable after PCP decode sharding lands; DSV4 results are unstable.
                "enable_pcp_o_proj_weight_sharding": False,
            },
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[kv_pool_leg] {case_name} pool={use_pool}", flush=True)
        yield stack.enter_context(VllmRunner(MODEL, **kwargs))


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,mtp,prefix_caching",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(
    os.environ,
    {
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "2560",
        "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
        "CLOSE_MATMUL_K_SHIFT": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_dsa_pcp_mtp_pool_watch() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)

    def _accuracy(runner) -> list[float]:
        sampling = SamplingParams(temperature=0.0, max_tokens=ACCURACY_MAX_TOKENS, seed=0)
        outputs = runner.generate(PROMPTS_SHORT, sampling)
        assert len(outputs) == len(PROMPTS_SHORT)
        for prompt, (_, texts) in zip(PROMPTS_SHORT, outputs, strict=True):
            assert len(texts) == 1, f"expected one completion for {prompt!r}, got {len(texts)}"
            expected = EXPECTED_OUTPUT_PREFIXES.get(prompt)
            if expected is not None:
                assert texts[0].startswith(expected), (
                    f"unexpected output prefix for {prompt!r}: got {texts[0][: len(expected)]!r}, expected {expected!r}"
                )

        # Absolute floors: catch model/engine regressions on both legs.
        rates = calculate_acceptance_per_pos(
            runner.model.get_metrics(),
            NUM_SPECULATIVE_TOKENS,
            Counter,
            Vector,
        )
        assert len(rates) == len(MIN_ACCEPTANCE_RATES), f"expected {len(MIN_ACCEPTANCE_RATES)} positions, got {rates}"
        print(f"[kv_pool_acceptance] rates={rates} minimum={MIN_ACCEPTANCE_RATES}", flush=True)
        for position, (actual, minimum) in enumerate(zip(rates, MIN_ACCEPTANCE_RATES, strict=True)):
            assert actual >= minimum, f"acceptance at draft position {position} is {actual}, below {minimum}"
        return rates

    def _hit(runner) -> dict:
        return run_pool_hit_sequence(
            runner,
            block_size=128,
            prompt_tokens=POOL_PROMPT_TOKENS,
            prompt_ids=prompt_ids,
            lcm_block_size=POOL_LCM_BLOCK_SIZE,
            check_output=False,
            answer_secret=POOL_SECRET,
            tag="dsv4-pcp2-mtp-memcache",
        )

    run_pool_ab(
        case_name=CASE_NAME,
        build_leg=_leg,
        accuracy=_accuracy,
        hit=_hit,
        prompt_ids=prompt_ids,
        perf_prompt_tokens=POOL_PERF_PROMPT_TOKENS,
        # MTP cold/hit paths already split at the tail, so pool-hit vs
        # local-hit tokens are not bit-comparable.
        token_identity=False,
        acceptance_ratio=ACCEPTANCE_AB_RATIO,
    )
