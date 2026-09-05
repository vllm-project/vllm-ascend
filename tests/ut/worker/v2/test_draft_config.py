# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from vllm.config import CacheConfig

from vllm_ascend.utils import refresh_block_size
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


@dataclass
class _ParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 4
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_size: int = 2
    data_parallel_rank: int = 1
    enable_expert_parallel: bool = True
    enable_eplb: bool = True


@dataclass
class _RuntimeConfig:
    """Exercise the real Ascend cache default applied during config validation."""

    model_config: SimpleNamespace
    cache_config: CacheConfig
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    scheduler_config: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(enable_chunked_prefill=True))
    compilation_config: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(static_forward_context={}))

    def __post_init__(self):
        refresh_block_size(self)


@pytest.mark.parametrize("speculator_cls", [AscendMTPSpeculator, AscendEagleSpeculator])
def test_draft_runtime_config_preserves_hybrid_cache_layout(speculator_cls):
    # Qwen3.5's main model is hybrid; its MTP head is attention-only. The
    # platform's dense-model default must not resize their established cache.
    target_config = _RuntimeConfig(
        model_config=SimpleNamespace(is_hybrid=True, hf_config=SimpleNamespace(model_type="qwen3_5")),
        cache_config=CacheConfig(
            block_size=1536,
            mamba_block_size=1536,
            mamba_cache_mode="align",
            enable_prefix_caching=True,
        ),
    )
    draft_model_config = SimpleNamespace(is_hybrid=False, hf_config=SimpleNamespace(model_type="qwen3_5_mtp"))
    speculator = object.__new__(speculator_cls)
    speculator.vllm_config = target_config
    speculator.draft_model_config = draft_model_config

    draft_config = speculator._create_draft_vllm_config()

    assert target_config.cache_config.block_size == 1536
    assert target_config.cache_config.mamba_block_size == 1536
    assert draft_config.cache_config.block_size == 1536
    assert draft_config.cache_config.mamba_block_size == 1536
    assert draft_config.model_config is draft_model_config
    assert target_config.model_config.is_hybrid
    assert draft_config.compilation_config is target_config.compilation_config

    assert target_config.parallel_config.pipeline_parallel_size == 2
    assert target_config.parallel_config.enable_expert_parallel
    assert target_config.parallel_config.enable_eplb
    assert draft_config.parallel_config.pipeline_parallel_size == 1
    assert draft_config.parallel_config.tensor_parallel_size == 4
    assert draft_config.parallel_config.data_parallel_size == 2
    assert draft_config.parallel_config.data_parallel_rank == 1
    expected_expert_parallel = speculator_cls is AscendMTPSpeculator
    assert draft_config.parallel_config.enable_expert_parallel == expected_expert_parallel
    assert draft_config.parallel_config.enable_eplb == expected_expert_parallel

    draft_config.cache_config.num_gpu_blocks = 32
    assert target_config.cache_config.num_gpu_blocks is None
