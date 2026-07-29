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
"""Verify that RecomputeScheduler does not enable ShortRequestFirst."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstRequestQueue,
)

MODEL = "Qwen3-0.6B"
MAX_NUM_BATCHED_TOKENS = 10000
BLOCK_SIZE = 16


class TestRecomputeSchedulerWithoutShortRequestFirst(TestBase):
    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch(
        "vllm.config.ModelConfig.is_encoder_decoder",
        PropertyMock(return_value=False),
    )
    def test_waiting_queue_uses_upstream_policy(self):
        from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

        init_none_hash(sha256)
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
        cache_config.num_gpu_blocks = 10000
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
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
        )

        scheduler = RecomputeScheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            block_size=BLOCK_SIZE,
            log_stats=True,
            structured_output_manager=MagicMock(spec=StructuredOutputManager),
        )

        self.assertNotIsInstance(
            scheduler.waiting,
            ShortRequestFirstRequestQueue,
        )
