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

from vllm_ascend.models.deepseek_v2 import (
    CustomDeepseekV2MergedReplicatedLinear, CustomDeepseekV2MLAAttention,
    CustomDeepseekV2MLP, CustomDeepseekV2RowParallelLinear,
    CustomDeepseekV2SiluAndMul, LogitsProcessor, ParallelLMHead)


def test_custom_deepseek_v2_silu_and_mul():
    torch.set_default_device("cpu")

    silu = CustomDeepseekV2SiluAndMul()
    assert silu.weight_scale is None

    x = torch.randn(2, 4)
    output = silu.forward_oot(x)
    assert output.shape == (2, 2)

    weight_scale = Mock(return_value=torch.tensor(0.1))
    silu = CustomDeepseekV2SiluAndMul(weight_scale=weight_scale)
    quant_x = torch.randint(-128, 127, (2, 4), dtype=torch.int32)
    dynamic_scale = torch.randn(2, 1)
    with patch("torch_npu.npu_dequant_swiglu_quant",
               return_value=torch.randn(2, 4)):
        output = silu.forward_oot((quant_x, dynamic_scale))
        assert output.shape == (2, 4)


def test_custom_deepseek_v2_merged_replicated_linear(mock_distributed):
    linear = CustomDeepseekV2MergedReplicatedLinear(input_size=128,
                                                    output_sizes=[64, 64],
                                                    bias=False,
                                                    quant_config=None)
    assert linear.output_sizes == [64, 64]

    param = Mock()
    param.data = torch.zeros(128, 128)
    param.output_dim = 1
    param.is_gguf_weight = False
    param.is_gguf_weight_type = False
    loaded_weight = torch.randn(128, 64)
    linear.weight_loader(param, loaded_weight, loaded_shard_id=0)

    with pytest.raises(AssertionError):
        linear.weight_loader(param, torch.randn(128, 32), loaded_shard_id=0)


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


def test_custom_deepseek_v2_mlp(mock_distributed, base_config):
    mlp = CustomDeepseekV2MLP(hidden_size=128,
                              intermediate_size=256,
                              hidden_act="silu",
                              quant_config=None)
    assert isinstance(mlp.act_fn, CustomDeepseekV2SiluAndMul)

    x = torch.randn(2, 4, 128)
    output = mlp(x)
    assert output.shape == (2, 4, 128)

    with patch("vllm_ascend.models.deepseek_v2.QuantizationConfig"
               ) as mock_quant_config:
        mock_quant_config.name = "w8a8dynamic"
        with pytest.raises(NotImplementedError):
            CustomDeepseekV2MLP(hidden_size=128,
                                intermediate_size=256,
                                hidden_act="silu",
                                quant_config=mock_quant_config,
                                force_replicate=False)
    with pytest.raises(ValueError):
        CustomDeepseekV2MLP(hidden_size=128,
                            intermediate_size=256,
                            hidden_act="relu",
                            quant_config=None)


@patch("torch_npu.npu_rms_norm")
def test_custom_deepseek_v2_mla_attention(mock_rms_norm, mock_distributed,
                                          base_config):
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
        with pytest.raises(AssertionError):
            attn(positions, x)

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
