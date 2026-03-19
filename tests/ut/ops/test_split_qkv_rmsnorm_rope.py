#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Tests for the non-Triton fallback split_qkv_rmsnorm_rope op."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(mock_config):
        yield mock_config


class TestSplitQkvRmsnormRopeRegistration:
    """Test that the non-Triton fallback registers the op correctly."""

    def test_op_imported_when_no_triton(self):
        """When HAS_TRITON is False, the fallback module should be imported
        and the qkv_rmsnorm_rope op should be available."""
        import vllm_ascend.ops as ops_module

        fallback_module = "vllm_ascend.ops.split_qkv_rmsnorm_rope"
        previous_fallback = sys.modules.pop(fallback_module, None)
        try:
            with patch("vllm.triton_utils.HAS_TRITON", False):
                importlib.reload(ops_module)
                assert fallback_module in sys.modules
                assert hasattr(torch.ops.vllm, "qkv_rmsnorm_rope"), (
                    "qkv_rmsnorm_rope op should be registered even when Triton is not available"
                )
        finally:
            if previous_fallback is not None:
                sys.modules[fallback_module] = previous_fallback
            importlib.reload(ops_module)

    def test_head_dim_must_be_even(self):
        """head_dim must be even for RoPE to work correctly."""
        from vllm_ascend.ops.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope_impl,
        )

        odd_head_dim = 127
        batch_size = 2
        q_hidden_size = odd_head_dim * 4
        kv_hidden_size = odd_head_dim
        total = q_hidden_size + kv_hidden_size * 2

        input_tensor = torch.randn(batch_size, total)
        q_weight = torch.randn(odd_head_dim)
        k_weight = torch.randn(odd_head_dim)
        cos_sin_cache = torch.randn(100, odd_head_dim)
        positions = torch.zeros(batch_size, dtype=torch.long)

        with pytest.raises(AssertionError, match="head_dim must be even"):
            split_qkv_rmsnorm_rope_impl(
                input=input_tensor,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=q_hidden_size,
                kv_hidden_size=kv_hidden_size,
                head_dim=odd_head_dim,
                eps=1e-6,
            )

    def test_fake_impl_output_shapes(self):
        """The fake implementation should return correctly shaped tensors."""
        from vllm_ascend.ops.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope_impl_fake,
        )

        batch_size = 4
        head_dim = 128
        num_q_heads = 32
        num_kv_heads = 4
        q_hidden_size = num_q_heads * head_dim
        kv_hidden_size = num_kv_heads * head_dim
        total = q_hidden_size + kv_hidden_size * 2

        input_tensor = torch.randn(batch_size, total)
        q_weight = torch.randn(head_dim)
        k_weight = torch.randn(head_dim)
        cos_sin_cache = torch.randn(1024, head_dim)
        positions = torch.zeros(batch_size, dtype=torch.long)

        q, k, v = split_qkv_rmsnorm_rope_impl_fake(
            input=input_tensor,
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            eps=1e-6,
        )

        assert q.shape == (batch_size, q_hidden_size)
        assert k.shape == (batch_size, kv_hidden_size)
        assert v.shape == (batch_size, kv_hidden_size)

    @patch("vllm_ascend.ops.split_qkv_rmsnorm_rope.torch_npu")
    def test_impl_calls_npu_rms_norm(self, mock_torch_npu):
        """The implementation should call npu_rms_norm for Q and K."""
        from vllm_ascend.ops.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope_impl,
        )

        head_dim = 128
        batch_size = 2
        num_q_heads = 4
        num_kv_heads = 1
        q_hidden_size = num_q_heads * head_dim
        kv_hidden_size = num_kv_heads * head_dim
        total = q_hidden_size + kv_hidden_size * 2

        input_tensor = torch.randn(batch_size, total)
        q_weight = torch.randn(head_dim)
        k_weight = torch.randn(head_dim)
        cos_sin_cache = torch.randn(1024, head_dim)
        positions = torch.zeros(batch_size, dtype=torch.long)

        # Mock npu_rms_norm to return the input unchanged
        mock_torch_npu.npu_rms_norm.side_effect = lambda x, w, e: (x, None)

        # Mock npu_rotary_embedding to return inputs unchanged
        with patch("torch.ops.vllm.npu_rotary_embedding") as mock_rope:
            mock_rope.side_effect = lambda pos, q, k, cache, hd, rd, neox: (q, k)

            q, k, v = split_qkv_rmsnorm_rope_impl(
                input=input_tensor,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=q_hidden_size,
                kv_hidden_size=kv_hidden_size,
                head_dim=head_dim,
                eps=1e-6,
            )

            # npu_rms_norm should be called twice (once for Q, once for K)
            assert mock_torch_npu.npu_rms_norm.call_count == 2
            # npu_rotary_embedding should be called once
            assert mock_rope.call_count == 1

            # Output shapes should be correct
            assert q.shape == (batch_size, q_hidden_size)
            assert k.shape == (batch_size, kv_hidden_size)
            assert v.shape == (batch_size, kv_hidden_size)

    @patch("vllm_ascend.ops.split_qkv_rmsnorm_rope.torch_npu")
    def test_impl_applies_bias(self, mock_torch_npu):
        """When bias is provided, it should be added after RMSNorm."""
        from vllm_ascend.ops.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope_impl,
        )

        head_dim = 128
        batch_size = 2
        num_q_heads = 4
        num_kv_heads = 1
        q_hidden_size = num_q_heads * head_dim
        kv_hidden_size = num_kv_heads * head_dim
        total = q_hidden_size + kv_hidden_size * 2

        input_tensor = torch.randn(batch_size, total)
        q_weight = torch.randn(head_dim)
        k_weight = torch.randn(head_dim)
        q_bias = torch.randn(head_dim)
        k_bias = torch.randn(head_dim)
        cos_sin_cache = torch.randn(1024, head_dim)
        positions = torch.zeros(batch_size, dtype=torch.long)

        # Mock npu_rms_norm to return zeros so we can check bias is added
        mock_torch_npu.npu_rms_norm.side_effect = lambda x, w, e: (torch.zeros_like(x), None)

        with patch("torch.ops.vllm.npu_rotary_embedding") as mock_rope:
            # Return q, k as-is to check bias was applied
            mock_rope.side_effect = lambda pos, q, k, cache, hd, rd, neox: (q, k)

            q, k, v = split_qkv_rmsnorm_rope_impl(
                input=input_tensor,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                q_weight=q_weight,
                k_weight=k_weight,
                q_hidden_size=q_hidden_size,
                kv_hidden_size=kv_hidden_size,
                head_dim=head_dim,
                eps=1e-6,
                q_bias=q_bias,
                k_bias=k_bias,
            )

            # With zeros from rms_norm + bias, q should have bias values
            # repeated across heads
            assert q.shape == (batch_size, q_hidden_size)
            assert torch.allclose(
                q,
                q_bias.repeat(num_q_heads).expand(batch_size, -1),
            )
            assert k.shape == (batch_size, kv_hidden_size)
            assert torch.allclose(
                k,
                k_bias.repeat(num_kv_heads).expand(batch_size, -1),
            )
            assert v.shape == (batch_size, kv_hidden_size)
