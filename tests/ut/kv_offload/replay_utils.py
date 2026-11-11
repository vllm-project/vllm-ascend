# SPDX-License-Identifier: Apache-2.0
"""Real vLLM 0.32 scheduler fixtures for Store replay regression tests."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs

BLOCK_SIZE = 16
WINDOW = 32


def _initialize_replay_window(scheduler):
    groups = scheduler.kv_cache_manager.kv_cache_config.kv_cache_groups
    scheduler.prefix_replay_group_ids = tuple(i for i, g in enumerate(groups) if g.kv_cache_spec.prefix_replay_tokens)
    windows = {groups[i].kv_cache_spec.prefix_replay_tokens for i in scheduler.prefix_replay_group_ids}
    assert len(windows) <= 1
    scheduler.prefix_replay_tokens = windows.pop() if windows else 0


@pytest.fixture(params=[(False, False), (True, False), (True, True)], ids=["v1", "v2", "v2-async"])
def replay_scheduler(request):
    use_v2_model_runner, async_scheduling = request.param
    with (
        patch.object(ModelConfig, "__post_init__", lambda self, *args, **kwargs: None),
        patch.object(VllmConfig, "__post_init__", lambda self, *args, **kwargs: None),
        patch.object(VllmConfig, "use_v2_model_runner", new_callable=PropertyMock, return_value=use_v2_model_runner),
        patch.object(ModelConfig, "is_encoder_decoder", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "uses_mrope", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "is_diffusion", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "is_multimodal_model", new_callable=PropertyMock, return_value=False),
    ):
        model = ModelConfig(model="test", tokenizer="test", max_model_len=256)
        model.runner_type = "generate"
        model.hf_config = MagicMock()
        model.hf_config.architectures = ["DeepseekV41ForCausalLM"]
        model.hf_config.canvas_length = None
        model.multimodal_config = None
        model.pooler_config = None
        cache = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=True, swa_bounded_replay=True)
        cache.num_gpu_blocks = 128
        config = VllmConfig(
            model_config=model,
            cache_config=cache,
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_seqs=1,
                max_num_batched_tokens=16,
                enable_chunked_prefill=True,
                is_encoder_decoder=False,
                async_scheduling=async_scheduling,
                watermark=0.0,
            ),
        )
        register_all_kvcache_specs(None)
        register_ascend_kv_cache_specs()
        full = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
        swa = AscendSlidingWindowMLASpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=WINDOW,
            bounded_replay=True,
        )
        kv = KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["full"], full),
                KVCacheGroupSpec(["swa"], swa),
            ],
        )
        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        yield scheduler_cls(config, kv, MagicMock(), BLOCK_SIZE)


def _pd_request(scheduler, length):
    init_none_hash(sha256)
    request = Request(
        "pd",
        list(range(length)),
        SamplingParams(max_tokens=4),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(request)
    return request
