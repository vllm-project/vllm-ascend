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
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_ascend.models.deepseek_v2 import (CustomDeepseekV2MLAAttention,
                                            CustomDeepseekV2RowParallelLinear)


@pytest.mark.parametrize("cls", [CustomDeepseekV2RowParallelLinear])
def test_row_parallel_linear(cls, mock_distributed):
    linear = cls(input_size=128, output_size=64, bias=False, quant_config=None)
    linear.quant_method = Mock()
    linear.quant_method.apply.return_value = torch.randn(2, 4, 64)
    input_ = torch.randn(2, 4, 128)
    with patch("vllm_ascend.models.deepseek_v2.split_tensor_along_last_dim",
               return_value=[torch.randn(2, 4, 64)]):
        linear.input_is_parallel = False
        output = linear(input_, is_prefill=True)
    assert output[0].shape == (2, 4, 64)

    linear.input_is_parallel = True
    output = linear(input_, is_prefill=False)
    assert output[0].shape == (2, 4, 64)


@patch("torch.ops.vllm.mla_forward")
@patch("torch_npu.npu_rms_norm")
def test_custom_deepseek_v2_mla_attention(mock_rms_norm, mock_mla_forward,
                                          mock_distributed, base_config):
    mock_rms_norm.return_value = (torch.randn(2, 128), torch.randn(2, 128))

    attn = CustomDeepseekV2MLAAttention(config=base_config,
                                        hidden_size=128,
                                        num_heads=8,
                                        qk_nope_head_dim=16,
                                        qk_rope_head_dim=16,
                                        v_head_dim=32,
                                        q_lora_rank=16,
                                        kv_lora_rank=16,
                                        cache_config=CacheConfig(),
                                        quant_config=None,
                                        prefix="layers.0.self_attn")
    assert attn.debug_layer_idx == 0

    x = torch.randn(2, 4, 128)
    positions = torch.arange(4).repeat(2, 1)
    with patch.object(attn.mla_attn,
                      "__call__",
                      return_value=torch.randn(2, 4, 128)):
        attn(positions, x)
        mock_mla_forward.assert_called_once()

    attn = CustomDeepseekV2MLAAttention(config=base_config,
                                        hidden_size=128,
                                        num_heads=8,
                                        qk_nope_head_dim=16,
                                        qk_rope_head_dim=16,
                                        v_head_dim=32,
                                        q_lora_rank=None,
                                        kv_lora_rank=16,
                                        prefix="layers.1.self_attn")
    assert hasattr(attn, "q_proj")


def test_deepseek_v2_lmhead(mock_distributed, vllm_config):
    # 创建一个简单的配置对象
    class SimpleConfig:

        def __init__(self):
            self.vocab_size = 10000
            self.hidden_size = 128

    config = SimpleConfig()

    # 直接创建lmhead和logits_processor
    lmhead = ParallelLMHead(config.vocab_size, config.hidden_size)
    logits_processor = LogitsProcessor(config.vocab_size)

    # 创建模拟输出
    mock_output = torch.randn(2, 4, config.hidden_size)
    mock_logits = torch.randn(2, 4, config.vocab_size)

    # 直接测试logits_processor
    with patch.object(lmhead.quant_method, "apply", return_value=mock_logits):
        with patch.object(logits_processor,
                          "_gather_logits",
                          return_value=mock_logits):
            logits = logits_processor(lmhead, mock_output)
    assert logits.shape == (2, 4, config.vocab_size)
