from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from transformers import PretrainedConfig
from vllm.config import CacheConfig, EPLBConfig, ParallelConfig
from vllm.distributed.parallel_state import GroupCoordinator


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
        use_mla=True,
        quant_config=None,
        max_model_len=2048,
    )
    parallel_config = MagicMock(spec=ParallelConfig)
    eplb_config = MagicMock(spec=EPLBConfig)
    eplb_config.num_redundant_experts = 0
    parallel_config.eplb_config = eplb_config

    cache_config = CacheConfig()
    vllm_config = Mock()
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.quant_config = None
    vllm_config.parallel_config = parallel_config
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
    ep_group.device_group = Mock()
    ep_group.device_group.rank.return_value = 0
    ep_group.device_group.size.return_value = 1

    pp_group = Mock(spec=GroupCoordinator)
    pp_group.rank_in_group = 0
    pp_group.world_size = 1

    mock_vllm_config = Mock()
    mock_vllm_config.scheduler_config = Mock(max_num_seqs=256)
    mock_vllm_config.model_config = Mock(max_model_len=2048, quant_config=None)

    with patch("vllm_ascend.models.deepseek_v2.get_tensor_model_parallel_rank", return_value=0), \
            patch("vllm_ascend.models.deepseek_v2.get_tensor_model_parallel_world_size", return_value=1), \
            patch("vllm_ascend.models.deepseek_v2.get_tp_group", return_value=tp_group), \
            patch("vllm_ascend.models.deepseek_v2.get_pp_group", return_value=pp_group), \
            patch("vllm_ascend.models.deepseek_v2.get_pp_group",
                  return_value=Mock(is_first_rank=False, is_last_rank=False)), \
            patch("vllm_ascend.ops.fused_moe.get_current_vllm_config", return_value=mock_vllm_config), \
            patch("vllm_ascend.ops.moe.token_dispatcher.torch.distributed.get_rank", return_value=0), \
            patch("vllm_ascend._build_info.__ascend_soc_version__", return_value=""), \
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
