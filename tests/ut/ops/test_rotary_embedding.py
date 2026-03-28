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
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    mock_config.model_config.hf_text_config = MagicMock()
    mock_config.model_config.hf_text_config.partial_rotary_factor = 1.0
    mock_config.model_config.hf_text_config.rotary_dim = 64
    mock_config.scheduler_config.max_num_batched_tokens = 1024

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(mock_config):
        yield mock_config


class TestSetCosAndSin:
    @patch("vllm_ascend.ops.rotary_embedding._cos_mla", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin_mla", None)
    @patch("vllm_ascend.ops.rotary_embedding._cos", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin", None)
    def test_set_cos_and_sin_mla_model(self, default_vllm_config):
        from vllm_ascend.ops.rotary_embedding import set_cos_and_sin

        default_vllm_config.model_config.use_mla = True
        default_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 32

        set_cos_and_sin(
            default_vllm_config,
            max_num_reqs=16,
            decode_token_per_req=1,
            dtype=torch.float16,
            device="cpu",
        )

    @patch("vllm_ascend.ops.rotary_embedding._cos_mla", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin_mla", None)
    @patch("vllm_ascend.ops.rotary_embedding._cos", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin", None)
    @patch("vllm_ascend.ops.rotary_embedding.is_vl_model", return_value=False)
    @patch("vllm_ascend.ops.rotary_embedding.has_rope", return_value=True)
    def test_set_cos_and_sin_gqa_model(
        self,
        mock_has_rope,
        mock_is_vl,
        default_vllm_config,
    ):
        from vllm_ascend.ops.rotary_embedding import set_cos_and_sin

        default_vllm_config.model_config.use_mla = False

        set_cos_and_sin(
            default_vllm_config,
            max_num_reqs=16,
            decode_token_per_req=1,
            dtype=torch.float16,
            device="cpu",
        )


class TestGetCosAndSinMla:
    def test_get_cos_and_sin_mla_basic(self):
        from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

        with patch("vllm_ascend.ops.rotary_embedding._cos_cache") as mock_cos_cache:
            with patch("vllm_ascend.ops.rotary_embedding._sin_cache") as mock_sin_cache:
                mock_cos_cache.__getitem__ = lambda self, idx: torch.randn(1, 1, 1, 32)
                mock_sin_cache.__getitem__ = lambda self, idx: torch.randn(1, 1, 1, 32)

                positions = torch.tensor([0, 1, 2])
                cos, sin = get_cos_and_sin_mla(positions, use_cache=False)

                assert cos is not None
                assert sin is not None


class TestRopeForwardOot:
    @patch("vllm_ascend.ops.rotary_embedding.HAS_TRITON", False)
    @patch("torch_npu._npu_rotary_embedding")
    def test_rope_forward_oot_basic(self, mock_rotary_emb):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        num_tokens = 4
        head_size = 64
        rotary_dim = 64
        num_heads = 8

        positions = torch.arange(num_tokens)
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_heads * head_size)
        cos_sin_cache = torch.randn(1024, rotary_dim * 2)

        result_q, result_k = rope_forward_oot(
            positions,
            query,
            key,
            cos_sin_cache,
            head_size,
            rotary_dim,
            is_neox_style=True,
        )

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape

    def test_rope_forward_oot_with_offsets_raises(self):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        positions = torch.arange(4)
        query = torch.randn(4, 512)
        key = torch.randn(4, 512)
        cos_sin_cache = torch.randn(1024, 128)
        offsets = torch.tensor([0, 1, 2, 3])

        with pytest.raises(NotImplementedError, match="Batched rotary embedding"):
            rope_forward_oot(
                positions,
                query,
                key,
                cos_sin_cache,
                head_size=64,
                rotary_dim=64,
                is_neox_style=True,
                offsets=offsets,
            )

    @patch("vllm_ascend.ops.rotary_embedding.HAS_TRITON", False)
    @patch("torch_npu._npu_rotary_embedding")
    def test_rope_forward_oot_partial_rotary(self, mock_rotary_emb):
        from vllm_ascend.ops.rotary_embedding import rope_forward_oot

        num_tokens = 2
        head_size = 64
        rotary_dim = 32
        num_heads = 4

        positions = torch.arange(num_tokens)
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_heads * head_size)
        cos_sin_cache = torch.randn(1024, rotary_dim * 2)

        result_q, result_k = rope_forward_oot(
            positions,
            query,
            key,
            cos_sin_cache,
            head_size,
            rotary_dim,
            is_neox_style=True,
        )

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape


class TestAscendRotaryEmbedding:
    @pytest.fixture
    def rotary_emb(self, default_vllm_config):
        from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding

        return AscendRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float16,
        )

    @patch("torch.ops.vllm.npu_rotary_embedding")
    def test_forward_oot_basic(self, mock_npu_rotary, rotary_emb):
        num_tokens = 4
        num_heads = 8
        head_size = 64

        positions = torch.arange(num_tokens)
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_heads * head_size)

        mock_npu_rotary.return_value = (query.clone(), key.clone())

        result_q, result_k = rotary_emb.forward_oot(positions, query, key)

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape

    @patch("torch.ops.vllm.npu_rotary_embedding")
    def test_forward_oot_with_neox_style_override(self, mock_npu_rotary, rotary_emb):
        positions = torch.arange(4)
        query = torch.randn(4, 512)
        key = torch.randn(4, 512)

        mock_npu_rotary.return_value = (query.clone(), key.clone())

        rotary_emb.forward_oot(positions, query, key, is_neox_style_override=False)

        mock_npu_rotary.assert_called_once()


class TestAscendYaRNRotaryEmbedding:
    @pytest.fixture
    def yarn_rotary_emb(self, default_vllm_config):
        from vllm_ascend.ops.rotary_embedding import AscendYaRNRotaryEmbedding

        return AscendYaRNRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            scaling_factor=2.0,
            dtype=torch.float16,
        )

    @patch("torch.ops.vllm.npu_rotary_embedding")
    def test_forward_oot(self, mock_npu_rotary, yarn_rotary_emb):
        num_tokens = 4
        num_heads = 8
        head_size = 64

        positions = torch.arange(num_tokens)
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_heads * head_size)

        mock_npu_rotary.return_value = (query.clone(), key.clone())

        result_q, result_k = yarn_rotary_emb.forward_oot(positions, query, key)

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape


class TestAscendDeepseekScalingRotaryEmbedding:
    @pytest.fixture
    def deepseek_rotary_emb(self, default_vllm_config):
        from vllm_ascend.ops.rotary_embedding import AscendDeepseekScalingRotaryEmbedding

        with patch("vllm_ascend.platform.NPUPlatform.device_type", "cpu"):
            return AscendDeepseekScalingRotaryEmbedding(
                head_size=64,
                rotary_dim=64,
                max_position_embeddings=2048,
                base=10000,
                is_neox_style=True,
                scaling_factor=2.0,
                dtype=torch.float32,
            )

    def test_yarn_get_mscale(self, deepseek_rotary_emb):
        result = deepseek_rotary_emb._yarn_get_mscale(scale=1.0, mscale=1.0)
        assert result == 1.0

        result = deepseek_rotary_emb._yarn_get_mscale(scale=2.0, mscale=1.0)
        assert result > 1.0

    def test_rotate_half(self, deepseek_rotary_emb):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = deepseek_rotary_emb._rotate_half(x)

        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_yarn_linear_ramp_mask(self, deepseek_rotary_emb):
        result = deepseek_rotary_emb._yarn_linear_ramp_mask(0, 10, 10)

        assert result.shape == (10,)
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_yarn_find_correction_dim(self, deepseek_rotary_emb):
        result = deepseek_rotary_emb._yarn_find_correction_dim(
            num_rotations=10,
            dim=64,
            base=10000,
            max_position_embeddings=2048,
        )

        assert isinstance(result, torch.Tensor)

    def test_yarn_find_correction_range(self, deepseek_rotary_emb):
        low, high = deepseek_rotary_emb._yarn_find_correction_range(
            low_rot=1,
            high_rot=32,
            dim=64,
            base=10000,
            max_position_embeddings=2048,
        )

        assert low >= 0
        assert high <= 64

    @patch("torch.ops.vllm.npu_rotary_embedding")
    def test_forward(self, mock_npu_rotary, deepseek_rotary_emb):
        num_tokens = 4
        num_heads = 8
        head_size = 64

        positions = torch.arange(num_tokens)
        query = torch.randn(num_tokens, num_heads * head_size)
        key = torch.randn(num_tokens, num_heads * head_size)

        mock_npu_rotary.return_value = (query.clone(), key.clone())

        result_q, result_k = deepseek_rotary_emb.forward(positions, query, key)

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape


class TestAscendMRotaryEmbedding:
    @pytest.fixture
    def mrotary_emb(self, default_vllm_config):
        from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding

        mock_config = MagicMock()
        mock_config.mrope_section = [16, 24, 24]

        with patch.object(
            AscendMRotaryEmbedding,
            "__init__",
            lambda self, *args, **kwargs: None,
        ):
            emb = AscendMRotaryEmbedding.__new__(AscendMRotaryEmbedding)
            emb.mrope_section = [16, 24, 24]
            emb.head_size = 64
            emb.rotary_dim = 64
            emb.mrope_interleaved = False
            emb.cos_sin_cache = torch.randn(1024, 128)
            return emb

    def test_ascend_triton_grid_limit(self, mrotary_emb):
        assert hasattr(AscendMRotaryEmbedding, "_ASCEND_TRITON_GRID_LIMIT")
        assert AscendMRotaryEmbedding._ASCEND_TRITON_GRID_LIMIT == 65535


class TestUpdateCosSin:
    @patch("vllm_ascend.ops.rotary_embedding._cos_sin_cache", None)
    @patch("vllm_ascend.ops.rotary_embedding._cos", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin", None)
    def test_update_cos_sin_with_none_caches(self):
        from vllm_ascend.ops.rotary_embedding import update_cos_sin

        positions = torch.tensor([0, 1, 2])
        result = update_cos_sin(positions)

        assert result is None

    @patch("vllm_ascend.ops.rotary_embedding._cos_sin_cache")
    @patch("vllm_ascend.ops.rotary_embedding._cos")
    @patch("vllm_ascend.ops.rotary_embedding._sin")
    def test_update_cos_sin_updates_slices(self, mock_sin, mock_cos, mock_cache):
        from vllm_ascend.ops.rotary_embedding import update_cos_sin

        mock_cache.__getitem__ = lambda self, idx: torch.randn(4, 128)
        mock_cos.__setitem__ = MagicMock()
        mock_sin.__setitem__ = MagicMock()

        positions = torch.tensor([0, 1, 2, 3])
        update_cos_sin(positions)


class TestGetCosAndSinSlice:
    @patch("vllm_ascend.ops.rotary_embedding._cos_slice", None)
    @patch("vllm_ascend.ops.rotary_embedding._sin_slice", None)
    def test_get_cos_and_sin_slice(self):
        from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_slice

        cos, sin = get_cos_and_sin_slice()

        assert cos is None
        assert sin is None
