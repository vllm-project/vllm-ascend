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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CacheConfig
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.models.deepseek_v2 import (
    CustomDeepseekV2MergedReplicatedLinear, CustomDeepseekV2MLAAttention,
    CustomDeepseekV2MLP, CustomDeepseekV2MoE,
    CustomDeepseekV2RowParallelLinear,
    CustomDeepseekV2RowParallelLinearReplaceAllreduce,
    CustomDeepseekV2SiluAndMul, LogitsProcessor, ParallelLMHead)


@pytest.fixture
def base_config():
    config = PretrainedConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=256,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=32,
        topk_method="noaux_tc",
        scoring_func="softmax",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        vocab_size=10000,
    )
    return config


@pytest.fixture
def vllm_config(base_config):
    model_config = SimpleNamespace(
        hf_config=base_config,
        tensor_parallel_size=1,
        dtype=torch.float32,
        use_mla=False,
        quant_config=None,
        max_model_len=2048,
    )

    cache_config = CacheConfig()
    vllm_config = Mock()
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.quant_config = None
    return vllm_config


@pytest.fixture
def mock_distributed():
    tp_group = Mock(spec=GroupCoordinator)
    tp_group.rank_in_group = 0
    tp_group.world_size = 1
    tp_group.device_group = Mock()

    dp_group = Mock(spec=GroupCoordinator)
    dp_group.rank_in_group = 0
    dp_group.world_size = 1

    ep_group = Mock(spec=GroupCoordinator)
    ep_group.rank_in_group = 0
    ep_group.world_size = 1

    pp_group = Mock(spec=GroupCoordinator)
    pp_group.rank_in_group = 0
    pp_group.world_size = 1

    mock_vllm_config = Mock()
    mock_vllm_config.scheduler_config = Mock(max_num_seqs=256)
    mock_vllm_config.model_config = Mock(max_model_len=2048, quant_config=None)

    with patch("vllm_ascend.models.deepseek_v2.get_tensor_model_parallel_rank", return_value=0), \
            patch("vllm_ascend.models.deepseek_v2.get_tensor_model_parallel_world_size", return_value=1), \
            patch("vllm_ascend.models.deepseek_v2.get_tp_group", return_value=tp_group), \
            patch("vllm_ascend.models.deepseek_v2.get_ep_group", return_value=ep_group), \
            patch("vllm_ascend.models.deepseek_v2.get_dp_group", return_value=dp_group), \
            patch("vllm_ascend.models.deepseek_v2.get_pp_group", return_value=pp_group), \
            patch("vllm_ascend.models.deepseek_v2.get_pp_group",
                  return_value=Mock(is_first_rank=False, is_last_rank=False)), \
            patch("vllm_ascend.ops.fused_moe.get_current_vllm_config", return_value=mock_vllm_config), \
            patch.dict("vllm.distributed.parallel_state.__dict__", _TP=tp_group, _EP=ep_group, _DP=dp_group,
                       _PP=pp_group), \
            patch.dict("vllm_ascend.distributed.parallel_state.__dict__", _MC2=ep_group), \
            patch("torch.npu.current_device", return_value=0):
        yield


@pytest.fixture
def mock_forward_context():
    forward_context = Mock(in_profile_run=False, with_prefill=False)
    with patch("vllm_ascend.models.deepseek_v2.get_forward_context",
               return_value=forward_context):
        yield


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


@pytest.mark.parametrize("cls", [
    CustomDeepseekV2RowParallelLinearReplaceAllreduce,
    CustomDeepseekV2RowParallelLinear
])
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


def test_custom_deepseek_v2_moe(mock_distributed, base_config,
                                mock_forward_context):
    base_config.n_shared_experts = 1
    moe = CustomDeepseekV2MoE(config=base_config,
                              quant_config=None,
                              prefix="mlp")
    assert moe.top_k == 2

    x = torch.randn(2, 4, 128)
    attn_metadata = Mock(num_prefills=1)
    with patch("vllm_ascend.ops.fused_moe.AscendFusedMoE.__call__",
               return_value=(torch.randn(2, 4, 128), torch.randn(2, 4, 128))):
        output = moe(x, attn_metadata)
        assert output.shape == (2, 4, 128)


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
