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

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.worker.kvcomp_utils import (
    KVCompConfig,
    KVCompMetaData,
    HashEncoder,
    bind_hashk_cache,
    bind_hashk_cache_nope,
    bind_hashk_cache_rope,
    get_kvcomp_config_path_for_model,
    recover_request_lengths,
)


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

@pytest.mark.parametrize(
    "hash_weight_type,chunk_repre_method",
    [
        ("uniform", "max"),
        ("uniform", "min"),
        ("uniform", "sum"),
        ("fixed", "max"),
    ],
)
def test_kvcomp_config_generate_config_data_valid(
    hash_weight_type, chunk_repre_method
):
    """Test KVCompConfig.generate_config_data with valid inputs."""
    config = KVCompConfig()
    num_layers = 4
    top_k = [0.3] * num_layers
    top_k_reuse = [-1] * num_layers

    config.generate_config_data(
        model_name="TestModel",
        hash_weight_type=hash_weight_type,
        num_hidden_layers=num_layers,
        seq_len_threshhold=1024,
        chunk_size=256,
        chunk_repre_method=chunk_repre_method,
        head_dim=128,
        hash_bits=128,
        top_k_ratio_per_layer=top_k,
        top_k_index_reuse=top_k_reuse,
        must_select_blocks=[0, -1],
    )

    assert config.model_name == "TestModel"
    assert config.is_mla is False
    assert config.hash_weight_type == hash_weight_type
    assert config.num_hidden_layers == num_layers
    assert config.seq_len_threshhold == 1024
    assert config.chunk_size == 256
    assert config.chunk_repre_method == chunk_repre_method
    assert config.head_dim == 128
    assert config.hash_bits == 128
    assert config.top_k_ratio_per_layer == top_k
    assert config.top_k_index_reuse == top_k_reuse
    assert config.must_select_blocks == [0, -1]


# @pytest.mark.parametrize(
#     "hash_weight_type",
#     ["random", "invalid", ""],
# )
# def test_kvcomp_config_generate_config_data_invalid_hash_weight_type(
#     hash_weight_type,
# ):
#     """Test KVCompConfig.generate_config_data rejects invalid hash_weight_type."""
#     config = KVCompConfig()
#     num_layers = 4

#     with pytest.raises(ValueError, match="hash_weight_type"):
#         config.generate_config_data(
#             model_name="TestModel",
#             hash_weight_type=hash_weight_type,
#             num_hidden_layers=num_layers,
#             seq_len_threshhold=1024,
#             chunk_size=128,
#             chunk_repre_method="max",
#             head_dim=128,
#             hash_bits=128,
#             top_k_ratio_per_layer=[0.3] * num_layers,
#             top_k_index_reuse=[-1] * num_layers,
#             must_select_blocks=[0, -1],
#         )


# @pytest.mark.parametrize(
#     "chunk_size",
#     [64, 100, 255],
# )
# def test_kvcomp_config_generate_config_data_invalid_chunk_size(chunk_size):
#     """Test KVCompConfig.generate_config_data rejects chunk_size not divisible by 128."""
#     config = KVCompConfig()
#     num_layers = 4

#     with pytest.raises(ValueError, match="chunk_size"):
#         config.generate_config_data(
#             model_name="TestModel",
#             hash_weight_type="uniform",
#             num_hidden_layers=num_layers,
#             seq_len_threshhold=1024,
#             chunk_size=chunk_size,
#             chunk_repre_method="max",
#             head_dim=128,
#             hash_bits=128,
#             top_k_ratio_per_layer=[0.3] * num_layers,
#             top_k_index_reuse=[-1] * num_layers,
#             must_select_blocks=[0, -1],
#         )


# @pytest.mark.parametrize(
#     "chunk_repre_method",
#     ["avg", "mean", ""],
# )
# def test_kvcomp_config_generate_config_data_invalid_chunk_repre_method(
#     chunk_repre_method,
# ):
#     """Test KVCompConfig.generate_config_data rejects invalid chunk_repre_method."""
#     config = KVCompConfig()
#     num_layers = 4

#     with pytest.raises(ValueError, match="chunk_repre_method"):
#         config.generate_config_data(
#             model_name="TestModel",
#             hash_weight_type="uniform",
#             num_hidden_layers=num_layers,
#             seq_len_threshhold=1024,
#             chunk_size=128,
#             chunk_repre_method=chunk_repre_method,
#             head_dim=128,
#             hash_bits=128,
#             top_k_ratio_per_layer=[0.3] * num_layers,
#             top_k_index_reuse=[-1] * num_layers,
#             must_select_blocks=[0, -1],
#         )


# def test_kvcomp_config_generate_config_data_top_k_length_mismatch():
#     """Test KVCompConfig.generate_config_data rejects top_k length mismatch."""
#     config = KVCompConfig()
#     num_layers = 4

#     with pytest.raises(ValueError, match="top_k_ratio_per_layer"):
#         config.generate_config_data(
#             model_name="TestModel",
#             hash_weight_type="uniform",
#             num_hidden_layers=num_layers,
#             seq_len_threshhold=1024,
#             chunk_size=128,
#             chunk_repre_method="max",
#             head_dim=128,
#             hash_bits=128,
#             top_k_ratio_per_layer=[0.3] * 3,  # wrong length
#             top_k_index_reuse=[-1] * num_layers,
#             must_select_blocks=[0, -1],
#         )

#     with pytest.raises(ValueError, match="top_k_index_reuse"):
#         config.generate_config_data(
#             model_name="TestModel",
#             hash_weight_type="uniform",
#             num_hidden_layers=num_layers,
#             seq_len_threshhold=1024,
#             chunk_size=128,
#             chunk_repre_method="max",
#             head_dim=128,
#             hash_bits=128,
#             top_k_ratio_per_layer=[0.3] * num_layers,
#             top_k_index_reuse=[-1] * 5,  # wrong length
#             must_select_blocks=[0, -1],
#         )


# def test_kvcomp_config_generate_mla_config_data_valid():
#     """Test KVCompConfig.generate_mla_config_data with valid inputs."""
#     config = KVCompConfig()
#     num_layers = 4

#     config.generate_mla_config_data(
#         model_name="MLAModel",
#         hash_weight_type="random",
#         num_hidden_layers=num_layers,
#         seq_len_threshhold=2048,
#         chunk_size=128,
#         chunk_repre_method="max",
#         kv_lora_rank=16,
#         qk_rope_head_dim=64,
#         hash_bits_kv_lora=64,
#         hash_bits_qk_rope=64,
#         top_k_ratio_per_layer=[0.3] * num_layers,
#         top_k_index_reuse=[-1] * num_layers,
#         must_select_blocks=[0, -2, -1],
#     )

#     assert config.is_mla is True
#     assert config.model_name == "MLAModel"
#     assert config.head_dim == 64 + 16  # qk_rope_head_dim + kv_lora_rank
#     assert config.hash_bits == 64 + 64  # hash_bits_qk_rope + hash_bits_kv_lora
#     assert config.kv_lora_rank == 16
#     assert config.qk_rope_head_dim == 64


# def test_kvcomp_config_set_hash_weight_valid():
#     """Test KVCompConfig.set_hash_weight when hash_weight_type is fixed."""
#     config = KVCompConfig()
#     config.hash_weight_type = "fixed"
#     config.head_dim = 4
#     config.hash_bits = 16

#     hash_weight = [[0.1] * 16 for _ in range(4)]
#     config.set_hash_weight(hash_weight)

#     assert config.hash_weight == hash_weight


# def test_kvcomp_config_set_hash_weight_wrong_type():
#     """Test KVCompConfig.set_hash_weight raises when hash_weight_type is not fixed."""
#     config = KVCompConfig()
#     config.hash_weight_type = "random"

#     with pytest.raises(ValueError, match="hash_weight can only be set when"):
#         config.set_hash_weight([[0.1] * 16 for _ in range(4)])


# def test_kvcomp_config_set_hash_weight_wrong_shape():
#     """Test KVCompConfig.set_hash_weight raises when shape is wrong."""
#     config = KVCompConfig()
#     config.hash_weight_type = "fixed"
#     config.head_dim = 4
#     config.hash_bits = 16

#     with pytest.raises(ValueError, match="hash_weight shape"):
#         config.set_hash_weight([[0.1] * 8 for _ in range(4)])  # wrong hash_bits


# def test_kvcomp_config_set_mla_hash_weight_valid():
#     """Test KVCompConfig.set_mla_hash_weight when hash_weight_type is fixed."""
#     config = KVCompConfig()
#     config.hash_weight_type = "fixed"
#     config.kv_lora_rank = 4
#     config.hash_bits_kv_lora = 16
#     config.qk_rope_head_dim = 8
#     config.hash_bits_qk_rope = 8

#     hw_kv = [[0.1] * 16 for _ in range(4)]
#     hw_qk = [[0.2] * 8 for _ in range(8)]
#     config.set_mla_hash_weight(hw_kv, hw_qk)

#     assert config.hash_weight_kv_lora == hw_kv
#     assert config.hash_weight_qk_rope == hw_qk


# def test_kvcomp_config_to_json_from_json_roundtrip():
#     """Test KVCompConfig to_json and from_json roundtrip."""
#     config = KVCompConfig()
#     config.model_name = "RoundtripModel"
#     config.num_hidden_layers = 8
#     config.chunk_size = 256

#     with tempfile.NamedTemporaryFile(
#         mode="w", suffix=".json", delete=False
#     ) as f:
#         path = f.name

#     try:
#         config.to_json(path)
#         loaded = KVCompConfig.from_json(path)
#         assert loaded.model_name == config.model_name
#         assert loaded.num_hidden_layers == config.num_hidden_layers
#         assert loaded.chunk_size == config.chunk_size
#     finally:
#         Path(path).unlink(missing_ok=True)


# def test_kvcomp_config_from_json_existing_file():
#     """Test KVCompConfig.from_json loads from existing config."""
#     config_path = (
#         Path(__file__).resolve().parents[2]
#         / "vllm_ascend"
#         / "attention"
#         / "kvcomp_configs"
#         / "KVComp_Qwen3_32B_config.json"
#     )
#     if config_path.exists():
#         config = KVCompConfig.from_json(str(config_path))
#         assert config.model_name == "Qwen/Qwen3-32B"
#         assert config.num_hidden_layers == 64
#         assert config.chunk_size == 128


# # =============================================================================
# # test KVCompMetaData
# # =============================================================================


# def test_kvcomp_metadata_creation():
#     """Test KVCompMetaData creation with required fields."""
#     config = KVCompConfig()
#     config.num_hidden_layers = 4
#     config.vllm_hash_attention_topk = 256

#     metadata = KVCompMetaData(
#         kvcomp_config=config,
#         chunk_sizes_for_hamming_full=torch.full([4], 128, dtype=torch.int32),
#         topk_for_hamming_full=torch.full([4], 2, dtype=torch.int32),
#         topk_for_hamming_full_cpu=torch.full([4], 2, dtype=torch.int32),
#         seq_lens_for_hamming=torch.zeros([4], dtype=torch.int32),
#         hamming_output=torch.zeros([4, 8, 32], dtype=torch.int32),
#         hash_encoder=None,
#         hashk_caches=None,
#     )

#     assert metadata.kvcomp_config is config
#     assert metadata.chunk_sizes_for_hamming_full.shape == (4,)
#     assert metadata.topk_for_hamming_full.shape == (4,)
#     assert metadata.hash_encoder is None
#     assert metadata.hashk_caches is None
#     assert metadata.hash_encoder_nope is None
#     assert metadata.hash_encoder_rope is None


# # =============================================================================
# # test HashEncoder
# # =============================================================================

# NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


# @pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
# def test_hash_encoder_init_valid():
#     """Test HashEncoder init with valid params (NPU only)."""
#     encoder = HashEncoder(
#         input_dim=128,
#         hash_bits=128,
#         dtype=torch.float16,
#         device=torch.device("npu:0"),
#     )
#     assert encoder.input_dim == 128
#     assert encoder.hash_bits == 128
#     assert encoder.hash_numbers == 16
#     assert encoder.hash_weights.shape == (128, 128)


# def test_hash_encoder_init_invalid_hash_bits():
#     """Test HashEncoder init raises when hash_bits not multiple of 8."""
#     with pytest.raises(ValueError, match="hash_bits must be a multiple of 8"):
#         HashEncoder(
#             input_dim=128,
#             hash_bits=100,
#             dtype=torch.float16,
#             device=torch.device("npu:0"),  # may fail if NPU unavailable
#         )


# @pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
# def test_hash_encoder_set_hash_weight_valid():
#     """Test HashEncoder.set_hash_weight with matching tensor (NPU only)."""
#     encoder = HashEncoder(
#         input_dim=8,
#         hash_bits=16,
#         dtype=torch.float16,
#         device=torch.device("npu:0"),
#     )
#     weights = torch.randn(8, 16, dtype=torch.float16, device=torch.device("npu:0"))
#     encoder.set_hash_weight(weights)
#     assert torch.allclose(encoder.hash_weights, weights)


# @pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU not available")
# def test_hash_encoder_set_hash_weight_invalid_shape():
#     """Test HashEncoder.set_hash_weight raises on shape mismatch (NPU only)."""
#     encoder = HashEncoder(
#         input_dim=8,
#         hash_bits=16,
#         dtype=torch.float16,
#         device=torch.device("npu:0"),
#     )
#     wrong_weights = torch.randn(4, 16, dtype=torch.float16, device=torch.device("npu:0"))
#     with pytest.raises(ValueError, match="hash_weights shape"):
#         encoder.set_hash_weight(wrong_weights)


# # =============================================================================
# # test recover_request_lengths
# # =============================================================================


# @pytest.mark.parametrize(
#     "cu_num_tokens, expected",
#     [
#         (torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)),
#         (torch.tensor([2, 7, 10]), torch.tensor([5, 3])),
#         (torch.tensor([0, 5, 12, 20]), torch.tensor([5, 7, 8])),
#         (torch.tensor([100]), torch.tensor([])),
#     ],
# )
# def test_recover_request_lengths(cu_num_tokens, expected):
#     """Test recover_request_lengths from cumulative token tensor."""
#     result = recover_request_lengths(cu_num_tokens)
#     assert torch.equal(result, expected)
#     assert result.dtype == cu_num_tokens.dtype
#     assert result.device == cu_num_tokens.device


# def test_recover_request_lengths_empty():
#     """Test recover_request_lengths with empty input preserves device/dtype."""
#     for device in ["cpu"]:
#         cu = torch.tensor([], dtype=torch.int32, device=device)
#         result = recover_request_lengths(cu)
#         assert result.numel() == 0
#         assert result.device == cu.device
#         assert result.dtype == cu.dtype


# # =============================================================================
# # test bind_hashk_cache
# # =============================================================================


# @patch("vllm_ascend.worker.kvcomp_utils.extract_layer_index")
# def test_bind_hashk_cache_basic(mock_extract):
#     """Test bind_hashk_cache populates runner and forward_context."""
#     mock_extract.side_effect = lambda name, _: (
#         0 if "layers.0" in name else (1 if "layers.1" in name else 2)
#     )

#     cache0 = torch.zeros(2, 8, 128, 16, dtype=torch.uint8)
#     cache1 = torch.ones(2, 8, 128, 16, dtype=torch.uint8)
#     hashk_caches = {"model.layers.0.self_attn": cache0, "model.layers.1.self_attn": cache1}

#     attn0 = MagicMock()
#     attn1 = MagicMock()
#     forward_context = {
#         "model.layers.0.self_attn": attn0,
#         "model.layers.1.self_attn": attn1,
#     }

#     runner_hashk_caches = []

#     bind_hashk_cache(hashk_caches, forward_context, runner_hashk_caches, num_attn_module=1)

#     assert len(runner_hashk_caches) == 2
#     assert runner_hashk_caches[0] is cache0
#     assert runner_hashk_caches[1] is cache1
#     assert attn0.hashk_cache == [cache0]
#     assert attn1.hashk_cache == [cache1]


# def test_bind_hashk_cache_assert_nonempty_runner():
#     """Test bind_hashk_cache asserts runner_hashk_caches is empty."""
#     hashk_caches = {"layers.0": torch.zeros(1, 1, 1, 1, dtype=torch.uint8)}
#     forward_context = {"layers.0": MagicMock()}
#     runner_hashk_caches = [torch.zeros(1, dtype=torch.uint8)]  # not empty

#     with pytest.raises(AssertionError):
#         bind_hashk_cache(hashk_caches, forward_context, runner_hashk_caches)


# @patch("vllm_ascend.worker.kvcomp_utils.extract_layer_index")
# def test_bind_hashk_cache_nope_basic(mock_extract):
#     """Test bind_hashk_cache_nope populates runner and forward_context."""
#     mock_extract.return_value = 0

#     cache_nope = torch.zeros(2, 8, 128, 8, dtype=torch.uint8)
#     hashk_caches_nope = {"model.layers.0.self_attn": cache_nope}

#     attn = MagicMock()
#     forward_context = {"model.layers.0.self_attn": attn}
#     runner_hashk_caches_nope = []

#     bind_hashk_cache_nope(
#         hashk_caches_nope, forward_context, runner_hashk_caches_nope, num_attn_module=1
#     )

#     assert len(runner_hashk_caches_nope) == 1
#     assert attn.hashk_cache_nope == [cache_nope]


# @patch("vllm_ascend.worker.kvcomp_utils.extract_layer_index")
# def test_bind_hashk_cache_rope_basic(mock_extract):
#     """Test bind_hashk_cache_rope populates runner and forward_context."""
#     mock_extract.return_value = 0

#     cache_rope = torch.ones(2, 8, 128, 32, dtype=torch.uint8)
#     hashk_caches_rope = {"model.layers.0.self_attn": cache_rope}

#     attn = MagicMock()
#     forward_context = {"model.layers.0.self_attn": attn}
#     runner_hashk_caches_rope = []

#     bind_hashk_cache_rope(
#         hashk_caches_rope, forward_context, runner_hashk_caches_rope, num_attn_module=1
#     )

#     assert len(runner_hashk_caches_rope) == 1
#     assert attn.hashk_cache_rope == [cache_rope]


# # =============================================================================
# # test get_kvcomp_config_path_for_model
# # =============================================================================


# def test_get_kvcomp_config_path_unsupported_model():
#     """Test get_kvcomp_config_path_for_model raises for unsupported model."""
#     vllm_config = MagicMock()
#     vllm_config.model_config = MagicMock()
#     vllm_config.model_config.model = "unknown-model-xyz"

#     with pytest.raises(ValueError, match="Unsupported model for KVComp"):
#         get_kvcomp_config_path_for_model(vllm_config)


# @pytest.mark.parametrize(
#     "model_name,expected_subpath",
#     [
#         ("deepseek-r1-123", "KVComp_DeepSeek_R1_W8A8_config.json"),
#         ("Qwen3-32B", "KVComp_Qwen3_32B_config.json"),
#         ("qwen3-30b-coder", "KVComp_Qwen3_30B_A3B_config.json"),
#         ("qwen3-4b", "KVComp_Qwen3_4B_config.json"),
#     ],
# )
# def test_get_kvcomp_config_path_supported_model(model_name, expected_subpath):
#     """Test get_kvcomp_config_path_for_model returns path or None for supported models."""
#     vllm_config = MagicMock()
#     vllm_config.model_config = MagicMock()
#     vllm_config.model_config.model = model_name

#     result = get_kvcomp_config_path_for_model(vllm_config)

#     if result is not None:
#         assert expected_subpath in result
#     else:
#         # Config file may not exist in test env
#         assert result is None or expected_subpath in result