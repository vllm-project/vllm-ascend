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
"""Integration tests for ShortRequestFirst wired into ``RecomputeScheduler``."""

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
from vllm_ascend.ascend_config import ShortRequestFirstConfig
from vllm_ascend.core.short_request_first_scheduler import ShortRequestFirstRequestQueue

EOS_TOKEN_ID = 50256
MODEL = "Qwen3-0.6B"
THRESHOLD = 256
MAX_NUM_BATCHED_TOKENS = 10000
BLOCK_SIZE = 16


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds: float):
        self.now += seconds


def _fake_ascend_config(**overrides) -> SimpleNamespace:
    config = {"enabled": True, "threshold": THRESHOLD}
    config.update(overrides)
    return SimpleNamespace(short_request_first_config=ShortRequestFirstConfig(config))


def create_requests(num_tokens_list, max_tokens: int = 16):
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


class TestRecomputeSchedulerShortRequestFirst(TestBase):
    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm.config.ModelConfig.is_encoder_decoder", PropertyMock(return_value=False))
    def create_scheduler(self, **config_overrides):
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
        model_config.multimodal_config = None
        model_config.served_model_name = MODEL
        model_config.hf_config = SimpleNamespace(canvas_length=None)
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
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

        fake_cfg = _fake_ascend_config(**config_overrides)
        with (
            patch("vllm_ascend.core.recompute_scheduler.get_ascend_config", return_value=fake_cfg),
            patch("vllm_ascend.core.short_request_first_scheduler.get_ascend_config", return_value=fake_cfg),
        ):
            scheduler = RecomputeScheduler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                block_size=BLOCK_SIZE,
                log_stats=True,
                structured_output_manager=MagicMock(spec=StructuredOutputManager),
            )

        scheduler.structured_output_manager.should_advance = MagicMock(return_value=False)
        return scheduler

    def _running_order(self, scheduler):
        return [req.request_id for req in scheduler.running]

    def test_waiting_queue_is_short_request_first(self):
        scheduler = self.create_scheduler()
        self.assertIsInstance(scheduler.waiting, ShortRequestFirstRequestQueue)

    def test_short_prefill_scheduled_before_long(self):
        scheduler = self.create_scheduler()
        long_req, short_req = create_requests([THRESHOLD + 1000, 10])
        scheduler.add_request(long_req)
        scheduler.add_request(short_req)

        scheduler.schedule()

        order = self._running_order(scheduler)
        self.assertIn("1", order)
        self.assertIn("0", order)
        self.assertLess(order.index("1"), order.index("0"))

    def test_aged_long_promoted_over_short_after_wait(self):
        clock = FakeClock()
        with patch("vllm_ascend.core.short_request_first_scheduler.time", clock):
            scheduler = self.create_scheduler(long_max_wait_ms=100.0)
            long_req, short_req = create_requests([THRESHOLD + 1000, 10])
            scheduler.add_request(long_req)
            scheduler.add_request(short_req)

            clock.advance(0.2)
            scheduler.schedule()

            order = self._running_order(scheduler)
            self.assertIn("0", order)
            self.assertLess(order.index("0"), order.index("1"))
