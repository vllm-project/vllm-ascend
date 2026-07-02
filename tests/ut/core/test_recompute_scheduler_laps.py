#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end integration tests for LAPS wired into ``RecomputeScheduler``.

Unlike ``test_laps_scheduler.py`` (which exercises ``LapsRequestQueue`` in
isolation) and ``test_patch_laps_scheduler.py`` (mixin selection glue), these
build a real ``RecomputeScheduler`` and drive its ``schedule()`` to verify the
LAPS waiting queue is actually installed and that the short-before-long priority
(and aging promotion) shows up in the scheduling order.

Pattern mirrors ``test_scheduler_dynamic_batch.py``: a CPU-only scheduler built
with mocked ``__post_init__`` and a fake KV-cache config. The Ascend config is
stubbed via ``get_ascend_config`` so we control ``laps_config`` directly without
constructing a full ``AscendConfig``. The aging clock is driven by patching
``vllm_ascend.core.laps_scheduler.time``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import LapsConfig
from vllm_ascend.core.laps_scheduler import LapsRequestQueue

EOS_TOKEN_ID = 50256
MODEL = "Qwen3-0.6B"
THRESHOLD = 256
MAX_NUM_BATCHED_TOKENS = 10000
BLOCK_SIZE = 16


class FakeClock:
    """Monotonic clock stub; advance with ``.advance(seconds)``."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def _fake_ascend_config(**laps_overrides) -> SimpleNamespace:
    laps = {"enabled": True, "threshold": THRESHOLD}
    laps.update(laps_overrides)
    return SimpleNamespace(laps_config=LapsConfig(laps))


def create_requests(num_tokens_list, max_tokens: int = 16):
    """Build one request per entry in ``num_tokens_list`` (prompt length)."""
    init_none_hash(sha256)
    requests = []
    for i, num_tokens in enumerate(num_tokens_list):
        sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
        request = Request(
            request_id=f"{i}",
            prompt_token_ids=[i % 50] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            mm_features=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        requests.append(request)
    return requests


class TestRecomputeSchedulerLaps(TestBase):
    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    # The real property calls is_encoder_decoder(self.hf_config), whose helper
    # also probes hf_config.get_text_config(); on a MagicMock that returns a
    # fresh (truthy) mock, so patch the property instead of the mock attrs.
    @patch("vllm.config.ModelConfig.is_encoder_decoder", PropertyMock(return_value=False))
    def create_scheduler(self, **laps_overrides):
        from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

        scheduler_config = SchedulerConfig(
            max_num_seqs=16,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            is_encoder_decoder=False,
        )

        model_config = ModelConfig(
            model=MODEL,
            tokenizer=MODEL,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
        )
        model_config.pooler_config = MagicMock()
        # Text-only: keeps supports_multimodal_inputs() False so the scheduler
        # skips the MultiModalBudget path (encoder budget = 0).
        model_config.multimodal_config = None
        model_config.served_model_name = MODEL
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
        # is_hybrid_model checks substrings in model_type -> must be a real str.
        model_config.hf_text_config.model_type = "qwen3"

        cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=False,
        )

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            kv_transfer_config=None,
            speculative_config=None,
        )

        kv_cache_config = KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32),
                )
            ],
        )
        cache_config.num_gpu_blocks = 10000

        fake_cfg = _fake_ascend_config(**laps_overrides)
        with (
            patch("vllm_ascend.core.recompute_scheduler.get_ascend_config", return_value=fake_cfg),
            patch("vllm_ascend.core.laps_scheduler.get_ascend_config", return_value=fake_cfg),
        ):
            scheduler = RecomputeScheduler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                block_size=BLOCK_SIZE,
                log_stats=True,
                structured_output_manager=MagicMock(spec=StructuredOutputManager),
            )

        should_advance = MagicMock(return_value=False)
        scheduler.structured_output_manager.should_advance = should_advance
        return scheduler

    def _running_order(self, scheduler):
        return [req.request_id for req in scheduler.running]

    # ----------------------------------------------------------------- #
    # Wiring
    # ----------------------------------------------------------------- #
    def test_waiting_queue_is_laps(self):
        scheduler = self.create_scheduler()
        self.assertIsInstance(scheduler.waiting, LapsRequestQueue)

    # ----------------------------------------------------------------- #
    # Short-before-long priority through real schedule()
    # ----------------------------------------------------------------- #
    def test_short_prefill_scheduled_before_long(self):
        scheduler = self.create_scheduler()
        # Add a long prefill first, then a short one. Without LAPS, FCFS would
        # schedule the long one first; LAPS must flip the order.
        long_req, short_req = create_requests([THRESHOLD + 1000, 10])
        scheduler.add_request(long_req)  # request_id "0" (long)
        scheduler.add_request(short_req)  # request_id "1" (short)

        scheduler.schedule()

        order = self._running_order(scheduler)
        self.assertIn("1", order)
        self.assertIn("0", order)
        # Short ("1") must be scheduled ahead of long ("0").
        self.assertLess(order.index("1"), order.index("0"))

    # ----------------------------------------------------------------- #
    # Aging promotion through real schedule()
    # ----------------------------------------------------------------- #
    def test_aged_long_promoted_over_short_after_wait(self):
        clock = FakeClock()
        with patch("vllm_ascend.core.laps_scheduler.time", clock):
            # Aging bound at 100ms with a token reservation so the aged long is
            # admitted through the soft-phase bucket (the only admission channel).
            scheduler = self.create_scheduler(long_max_wait_ms=100.0, long_token_reservation=0.5)
            long_req, short_req = create_requests([THRESHOLD + 1000, 10])
            scheduler.add_request(long_req)  # "0" long, enqueued at t0
            scheduler.add_request(short_req)  # "1" short

            # Long has aged past its bound before this scheduling step.
            clock.advance(0.2)  # 200ms >= 100ms
            scheduler.schedule()

            order = self._running_order(scheduler)
            # The aged long ("0") must be admitted ahead of the short ("1").
            self.assertIn("0", order)
            self.assertLess(order.index("0"), order.index("1"))
