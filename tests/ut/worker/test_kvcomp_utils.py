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
# This file is a part of the vllm-ascend project.

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.worker.kvcomp_utils import (
    HashEncoder,
    KVCompConfig,
    bind_hashk_cache,
    recover_request_lengths,
)

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
print(f"NPU_AVAILABLE={NPU_AVAILABLE}")

# =============================================================================
# test KVCompConfig
# =============================================================================


def test_kvcomp_config_default():
    """Test KVCompConfig default values."""
    config = KVCompConfig()
    assert config.model_name == "DummyModel"
    assert config.is_mla is False
    assert config.hash_weight_type == "random"
    assert config.num_hidden_layers == 36
    assert config.seq_len_threshhold == 2048
    assert config.chunk_size == 128
    assert config.chunk_repre_method == "max"
    assert config.head_dim == 128
    assert config.hash_bits == 128
    assert len(config.top_k_ratio_per_layer) == 36
    assert len(config.top_k_index_reuse) == 36
    assert config.must_select_blocks == [0, -2, -1]


def test_kvcomp_config_to_json_from_json_roundtrip():
    """Test KVCompConfig to_json and from_json roundtrip."""
    config = KVCompConfig()
    config.model_name = "RoundtripModel"
    config.num_hidden_layers = 8
    config.chunk_size = 128

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        config.to_json(path)
        loaded = KVCompConfig.from_json(path)
        assert loaded.model_name == config.model_name
        assert loaded.num_hidden_layers == config.num_hidden_layers
        assert loaded.chunk_size == config.chunk_size
    finally:
        Path(path).unlink(missing_ok=True)


# # =============================================================================
# # test HashEncoder
# # =============================================================================


@pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
def test_hash_encoder():
    """Test HashEncoder init with valid params (NPU only)."""
    encoder = HashEncoder(
        input_dim=128,
        hash_bits=128,
        dtype=torch.float16,
        device=torch.device("npu:0"),
    )
    assert encoder.input_dim == 128
    assert encoder.hash_bits == 128
    assert encoder.hash_numbers == 16
    assert encoder.hash_weights.shape == (128, 128)

    x = torch.randn((2, 8, 128), device=torch.device("npu:0"), dtype=torch.float16)

    hash_codes = encoder.compute_hash(x)
    assert hash_codes.shape == (2, 8, 16)

    unpacked_bits = encoder._unpack_hash(hash_codes)
    assert unpacked_bits.shape == (2, 8, 128)


# # =============================================================================
# # test recover_request_lengths
# # =============================================================================


@pytest.mark.parametrize(
    "cu_num_tokens, expected",
    [
        (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
        (torch.tensor([2, 7, 10]), torch.tensor([5, 3])),
        (torch.tensor([0, 5, 12, 20]), torch.tensor([5, 7, 8])),
        (torch.tensor([100]), torch.tensor([])),
    ],
)
def test_recover_request_lengths(cu_num_tokens, expected):
    """Test recover_request_lengths from cumulative token tensor."""
    result = recover_request_lengths(cu_num_tokens)
    assert torch.equal(result, expected)
    assert result.dtype == cu_num_tokens.dtype
    assert result.device == cu_num_tokens.device


def test_recover_request_lengths_empty():
    """Test recover_request_lengths with empty input preserves device/dtype."""
    for device in ["cpu"]:
        cu = torch.tensor([], dtype=torch.int32, device=device)
        result = recover_request_lengths(cu)
        assert result.numel() == 0
        assert result.device == cu.device
        assert result.dtype == cu.dtype


# # =============================================================================
# # test bind_hashk_cache
# # =============================================================================


@patch("vllm_ascend.worker.kvcomp_utils.extract_layer_index")
def test_bind_hashk_cache_basic(mock_extract):
    """Test bind_hashk_cache populates runner and forward_context."""
    mock_extract.side_effect = lambda name, _: (0 if "layers.0" in name else (1 if "layers.1" in name else 2))

    cache0 = torch.zeros(2, 8, 128, 16, dtype=torch.uint8)
    cache1 = torch.ones(2, 8, 128, 16, dtype=torch.uint8)
    hashk_caches = {"model.layers.0.self_attn": cache0, "model.layers.1.self_attn": cache1}

    attn0 = MagicMock()
    attn1 = MagicMock()
    forward_context = {
        "model.layers.0.self_attn": attn0,
        "model.layers.1.self_attn": attn1,
    }

    runner_hashk_caches: list[torch.Tensor] = []

    bind_hashk_cache(hashk_caches, forward_context, runner_hashk_caches, num_attn_module=1)

    assert len(runner_hashk_caches) == 2
    assert runner_hashk_caches[0] is cache0
    assert runner_hashk_caches[1] is cache1
    assert attn0.hashk_cache == [cache0]
    assert attn1.hashk_cache == [cache1]
