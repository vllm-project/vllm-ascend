#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
from vllm_ascend.attention.utils import ascend_chunked_prefill_workspace_size
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile

DEFAULT_CAPACITY = 128 * 1024
ALL_HARDWARE_PROFILES = [get_hardware_profile(device_type) for device_type in AscendDeviceType]


def _model_config(architectures=("DeepseekV3ForCausalLM",), model_type="deepseek_v3", max_model_len=4096, **kwargs):
    return SimpleNamespace(
        architectures=list(architectures),
        hf_config=SimpleNamespace(model_type=model_type),
        max_model_len=max_model_len,
        **kwargs,
    )


def _vllm_config(model_config=None, max_num_seqs=16, block_size=128, pcp_size=1, dcp_size=1, interleave=128):
    return SimpleNamespace(
        model_config=model_config or _model_config(),
        cache_config=SimpleNamespace(block_size=block_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=interleave,
        ),
    )


@pytest.fixture
def ascend_config(monkeypatch):
    config = SimpleNamespace(chunked_prefill_workspace_size=DEFAULT_CAPACITY)
    monkeypatch.setattr(attention_utils, "get_ascend_config", lambda: config)
    return config


@pytest.fixture
def mock_logger(monkeypatch):
    logger = MagicMock()
    monkeypatch.setattr(attention_utils, "logger", logger)
    return logger


@pytest.mark.parametrize("hardware_profile", ALL_HARDWARE_PROFILES)
@pytest.mark.parametrize("architecture", ["DeepseekV3ForCausalLM", "KimiK3ForConditionalGeneration"])
def test_default_capacity_is_128k(ascend_config, mock_logger, monkeypatch, hardware_profile, architecture):
    monkeypatch.setattr(attention_utils, "get_current_hardware_profile", lambda: hardware_profile)
    vllm_config = _vllm_config(_model_config(architectures=(architecture,), max_model_len=128 * 1024))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == DEFAULT_CAPACITY
    assert ascend_config.chunked_prefill_workspace_size == DEFAULT_CAPACITY
    mock_logger.warning_once.assert_called_once()
    args = mock_logger.warning_once.call_args.args
    assert args[1:4] == (1024 * 1024, DEFAULT_CAPACITY, "additional_config.chunked_prefill_workspace_size")


@pytest.mark.parametrize("configured_capacity", [1, 64 * 1024, 512 * 1024, 1024 * 1024])
def test_explicit_capacity_is_used(ascend_config, mock_logger, configured_capacity):
    ascend_config.chunked_prefill_workspace_size = configured_capacity
    vllm_config = _vllm_config(
        _model_config(architectures=("KimiK3ForConditionalGeneration",), max_model_len=128 * 1024),
        max_num_seqs=1,
        block_size=1,
    )

    assert ascend_chunked_prefill_workspace_size(vllm_config) == configured_capacity
    assert ascend_config.chunked_prefill_workspace_size == configured_capacity


def test_demand_below_capacity_is_not_capped(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 64 * 1024
    # demand = max(8 * 4096, 4 * 16 * 128) = 32768 rows.
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32768
    mock_logger.warning_once.assert_not_called()


def test_demand_equal_to_capacity_is_not_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 32768
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32768
    mock_logger.warning_once.assert_not_called()


def test_demand_above_capacity_is_clamped_and_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 32767
    vllm_config = _vllm_config(_model_config(max_model_len=4096))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 32767
    mock_logger.warning_once.assert_called_once()
    message, *args = mock_logger.warning_once.call_args.args
    assert "clamps the workspace" in message
    assert args == [32768, 32767, "additional_config.chunked_prefill_workspace_size", 32767]


def test_minimum_capacity_overrides_smaller_cap_and_is_reported(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 1024
    # floor = 16 * 128 = 2048 rows > C; demand = max(8 * 64, 4 * 2048) = 8192.
    vllm_config = _vllm_config(_model_config(max_model_len=64))

    assert ascend_chunked_prefill_workspace_size(vllm_config) == 2048
    mock_logger.warning_once.assert_called_once()
    message, *args = mock_logger.warning_once.call_args.args
    assert "max_num_seqs * block_size" in message
    assert "above the capacity" in message
    assert args == [8192, 1024, "additional_config.chunked_prefill_workspace_size", 2048, 2048]


@pytest.mark.parametrize("builder_cls", [AscendMLAMetadataBuilder, AscendSFAMetadataBuilder])
@pytest.mark.parametrize(
    "configured,architectures,expected",
    [
        (DEFAULT_CAPACITY, ("DeepseekV3ForCausalLM",), DEFAULT_CAPACITY),
        (DEFAULT_CAPACITY, ("KimiLinearForCausalLM",), DEFAULT_CAPACITY),
        (256 * 1024, ("KimiLinearForCausalLM",), 256 * 1024),
    ],
)
def test_mla_and_sfa_builders_use_configured_capacity(
    ascend_config, mock_logger, builder_cls, configured, architectures, expected
):
    ascend_config.chunked_prefill_workspace_size = configured
    vllm_config = _vllm_config(_model_config(architectures=architectures, max_model_len=128 * 1024))

    assert builder_cls.determine_chunked_prefill_workspace_size(vllm_config) == expected
    assert ascend_config.chunked_prefill_workspace_size == configured


def test_pcp_dcp_minimum_raises_workspace_above_configured_capacity(ascend_config, mock_logger):
    ascend_config.chunked_prefill_workspace_size = 4096
    # floor = 32 * 128 = 4096 rows; PCP/DCP needs 2 * 32 * lcm(128, 128 * 16) rows.
    vllm_config = _vllm_config(
        _model_config(max_model_len=8192), max_num_seqs=32, block_size=128, pcp_size=2, dcp_size=16
    )

    assert AscendMlaDCPMetadataBuilder.determine_chunked_prefill_workspace_size(vllm_config) == 131072
    assert ascend_config.chunked_prefill_workspace_size == 4096


@pytest.mark.parametrize("configured_capacity", [4096, 65536, 131071])
def test_pcp_dcp_minimum_override_is_reported(ascend_config, mock_logger, configured_capacity):
    ascend_config.chunked_prefill_workspace_size = configured_capacity
    config = _vllm_config(max_num_seqs=32, pcp_size=2, dcp_size=16)

    assert AscendMlaDCPMetadataBuilder.determine_chunked_prefill_workspace_size(config) == 131072
    mock_logger.warning_once.assert_called_once()
    message, *args = mock_logger.warning_once.call_args.args
    assert "context-parallel minimum" in message
    assert args == [131072, configured_capacity, "additional_config.chunked_prefill_workspace_size", 131072]
    assert ascend_config.chunked_prefill_workspace_size == configured_capacity


@pytest.mark.parametrize(
    "builder_cls,pcp_size,dcp_size,expected,alignment,fragments",
    [
        (AscendMLAMetadataBuilder, 2, 1, 8192, 128, 64),
        (AscendMlaDCPMetadataBuilder, 1, 16, 65536, 2048, 32),
    ],
)
def test_small_capacity_fits_single_axis_context_chunks(
    ascend_config, mock_logger, monkeypatch, builder_cls, pcp_size, dcp_size, expected, alignment, fragments
):
    ascend_config.chunked_prefill_workspace_size = 4096
    config = _vllm_config(max_num_seqs=32, pcp_size=pcp_size, dcp_size=dcp_size)
    workspace_size = builder_cls.determine_chunked_prefill_workspace_size(config)
    assert workspace_size == expected
    mock_logger.warning_once.assert_called_once()
    assert "context-parallel minimum" in mock_logger.warning_once.call_args.args[0]

    # Exercise the chunk planner used by both builders with the full batch.
    builder = AscendMLAMetadataBuilder.__new__(AscendMLAMetadataBuilder)
    builder.chunked_prefill_enabled = True
    builder.chunked_prefill_workspace_size = workspace_size
    builder.chunked_prefill_workspace = torch.empty(0)
    builder.block_size = alignment
    builder.device = torch.device("cpu")
    builder.num_decodes = 0
    builder.num_prefills = fragments
    builder.query_lens = torch.ones(fragments, dtype=torch.int32)
    builder.seq_lens = torch.full((fragments,), 4097, dtype=torch.int32)
    cpu_zeros = torch.zeros

    def zeros_without_pinning(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return cpu_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_without_pinning)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    monkeypatch.setattr(torch.Tensor, "npu", lambda tensor: tensor, raising=False)
    metadata = builder.build_chunked_metadata(0, SimpleNamespace(num_reqs=fragments))
    assert builder.max_context_chunk == alignment
    assert max(metadata.seq_tot) <= workspace_size
    assert metadata.chunk_seq_lens.sum(0).tolist() == [4096] * fragments


@pytest.mark.parametrize("configured_capacity", [131072, 262144])
def test_context_parallel_floor_within_capacity_does_not_warn(ascend_config, mock_logger, configured_capacity):
    ascend_config.chunked_prefill_workspace_size = configured_capacity
    config = _vllm_config(max_num_seqs=32, pcp_size=2, dcp_size=16)

    assert AscendMlaDCPMetadataBuilder.determine_chunked_prefill_workspace_size(config) == 131072
    mock_logger.warning_once.assert_not_called()
