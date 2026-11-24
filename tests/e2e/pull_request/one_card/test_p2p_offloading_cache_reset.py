# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import time
from collections.abc import Callable

import pytest
from vllm import SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.transformers_utils.utils import maybe_model_redirect
from vllm.v1.kv_offload.tiering.base import TieringOffloadingMetrics
from vllm.v1.metrics.reader import Gauge

from tests.e2e.conftest import VllmRunner

NPU_BLOCK_SIZE = 128
MAX_MODEL_LEN = 256
CPU_CACHE_BYTES = 32 << 20
POLL_INTERVAL_S = 0.05
WAIT_TIMEOUT_S = 10


def test_p2p_cache_reset_with_absent_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the prefix cache while the P2P tier is holding an unfetched store."""
    p2p_session = pytest.importorskip("vllm.v1.kv_offload.tiering.p2p.session.session")
    if not hasattr(p2p_session.P2PSession, "drain_stores"):
        pytest.skip("P2P cache reset requires vLLM PR #63988 to drain unfetched stores")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with VllmRunner(
        maybe_model_redirect("Qwen/Qwen3-8B"),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        block_size=NPU_BLOCK_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        enforce_eager=True,
        async_scheduling=False,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": CPU_CACHE_BYTES,
                "block_size": NPU_BLOCK_SIZE,
                "spec_name": "TieringOffloadingSpec",
                "secondary_tiers": [{"type": "p2p", "host": "127.0.0.1", "port": 0}],
            },
        ),
    ) as runner:
        llm = runner.model
        engine = llm.llm_engine
        effective_block_size = engine.vllm_config.cache_config.block_size
        print(f"Effective NPU KV cache block size: {effective_block_size}")
        assert effective_block_size == NPU_BLOCK_SIZE

        # The prompt has one complete offload chunk and a final token for computation.
        prompt = TokensPrompt(prompt_token_ids=list(range(100, 100 + NPU_BLOCK_SIZE + 1)))
        producer_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            detokenize=False,
            extra_args={"kv_transfer_params": {"remote_decoder": {"kv_request_id": "absent"}}},
        )

        def has_parked_cascade() -> bool:
            return any(
                isinstance(metric, Gauge)
                and metric.name == TieringOffloadingMetrics.ACTIVE_CASCADE_JOBS
                and metric.value > 0
                for metric in llm.get_metrics()
            )

        def step_until(condition: Callable[[], bool], description: str) -> None:
            deadline = time.monotonic() + WAIT_TIMEOUT_S
            while not condition():
                assert time.monotonic() < deadline, f"Timed out waiting for {description}"
                engine.step()
                time.sleep(POLL_INTERVAL_S)

        before = llm.generate(prompt, producer_params, use_tqdm=False)[0]
        step_until(has_parked_cascade, "a parked P2P store")
        print("A P2P store is parked before reset")
        step_until(lambda: llm.reset_prefix_cache(reset_connector=True), "the prefix cache reset")

        after = llm.generate(prompt, SamplingParams(temperature=0, max_tokens=1, detokenize=False), use_tqdm=False)[0]
        print(
            f"Cached tokens after reset: {after.num_cached_tokens}, "
            f"output tokens before and after: {before.outputs[0].token_ids}, {after.outputs[0].token_ids}"
        )
        assert after.num_cached_tokens == 0
        assert len(before.outputs[0].token_ids) == 1
        assert after.outputs[0].token_ids == before.outputs[0].token_ids
