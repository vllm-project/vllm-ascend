from unittest.mock import MagicMock, patch

import torch
from torch import nn
from vllm.config import CacheConfig, CompilationConfig, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.mla import MLAModules

from tests.ut.base import TestBase
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention, IndexerWrapper


class TestIndexerWrapper(TestBase):

    def test_initialization(self):
        mock_indexer = MagicMock()
        mock_indexer.n_head = 64
        mock_indexer.head_dim = 128
        mock_indexer.topk_tokens = 2048
        mock_indexer.q_lora_rank = 1536
        mock_indexer.wq_b = nn.Linear(128, 128)
        mock_indexer.wk = nn.Linear(128, 128)
        mock_indexer.weights_proj = nn.Linear(128, 128)
        mock_indexer.k_norm = nn.LayerNorm(128)
        mock_indexer.softmax_scale = 0.123
        mock_indexer.topk_indices_buffer = torch.randn(10)
        mock_indexer.k_cache = torch.randn(10)

        wrapper = IndexerWrapper(mock_indexer)

        self.assertEqual(wrapper.n_head, 64)
        self.assertEqual(wrapper.head_dim, 128)
        self.assertEqual(wrapper.topk_tokens, 2048)
        self.assertEqual(wrapper.q_lora_rank, 1536)
        self.assertIs(wrapper.wq_b, mock_indexer.wq_b)
        self.assertIs(wrapper.wk, mock_indexer.wk)
        self.assertIs(wrapper.weights_proj, mock_indexer.weights_proj)
        self.assertIs(wrapper.k_norm, mock_indexer.k_norm)
        self.assertEqual(wrapper.softmax_scale, 0.123)

        self.assertIsNone(mock_indexer.topk_indices_buffer)
        self.assertIsNone(mock_indexer.k_cache)

    def test_forward(self):
        mock_indexer = MagicMock()
        wrapper = IndexerWrapper(mock_indexer)
        result = wrapper.forward()
        self.assertIsNone(result)


class TestAscendMultiHeadLatentAttention(TestBase):

    def setUp(self):
        self.hidden_size = 4096
        self.num_heads = 32
        self.scale = 0.123
        self.qk_nope_head_dim = 64
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.q_lora_rank = 1536
        self.kv_lora_rank = 128
        self.prefix = "model.layers.0.mla"

        self.mock_mla_modules = MagicMock(spec=MLAModules)
        self.mock_mla_modules.indexer = MagicMock()
        self.mock_mla_modules.is_sparse = False
        self.mock_mla_modules.rotary_emb = MagicMock()
        self.mock_mla_modules.fused_qkv_a_proj = MagicMock()
        self.mock_mla_modules.q_b_proj = MagicMock()
        self.mock_mla_modules.q_a_layernorm = MagicMock()
        self.mock_mla_modules.q_proj = MagicMock()
        self.mock_mla_modules.kv_a_proj_with_mqa = MagicMock()
        self.mock_mla_modules.kv_a_layernorm = MagicMock()
        self.mock_mla_modules.kv_b_proj = MagicMock()
        self.mock_mla_modules.o_proj = MagicMock()

        self.mock_cache_config = MagicMock(spec=CacheConfig)
        self.mock_quant_config = MagicMock()

    @patch("vllm_ascend.ops.mla.get_current_vllm_config")
    @patch("vllm_ascend.ops.mla.get_ascend_config")
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    def test_initialization(self, mock_tp_size, mock_ascend_config,
                            mock_get_vllm_config):
        # Create a proper mock for MLAAttention that has the required attributes
        mock_mla_attn = MagicMock()
        mock_mla_attn.process_weights_after_loading = MagicMock()
        mock_mla_attn.impl = MagicMock()
        mock_mla_attn.impl.process_weights_after_loading = MagicMock()

        with patch("vllm_ascend.ops.mla.MLAAttention", return_value=mock_mla_attn):
            mock_tp_size.return_value = 2
            mock_ascend_config.return_value.enable_shared_expert_dp = True
            mock_vllm_config = MagicMock(spec=VllmConfig)
            mock_vllm_config.model_config.hf_text_config = MagicMock(
                num_hidden_layers=32, first_k_dense_replace=True)
            mock_get_vllm_config.return_value = mock_vllm_config
            mock_vllm_config.compilation_config = CompilationConfig()

            attn = AscendMultiHeadLatentAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                scale=self.scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                mla_modules=self.mock_mla_modules,
                cache_config=self.mock_cache_config,
                quant_config=self.mock_quant_config,
                prefix=self.prefix,
            )

            self.assertEqual(attn.tp_size, 2)
            self.assertTrue(attn.enable_shared_expert_dp)
            self.assertIsNotNone(attn.mla_attn)

    @patch("vllm_ascend.ops.mla.torch.ops.vllm.mla_forward")
    @patch("vllm_ascend.ops.mla.get_current_vllm_config")
    @patch("vllm_ascend.ops.mla.get_ascend_config")
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.ops.mla.get_forward_context")
    def test_forward(self, mock_get_forward_context, mock_tp_size,
                     mock_ascend_config, mock_get_vllm_config,
                     mock_mla_forward):
        mock_tp_size.return_value = 1
        mock_ascend_config.return_value.enable_shared_expert_dp = False
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_vllm_config.model_config.hf_text_config = MagicMock(
            num_hidden_layers=32, first_k_dense_replace=False)
        mock_get_vllm_config.return_value = mock_vllm_config
        mock_vllm_config.compilation_config = CompilationConfig()

        # Create a proper mock for MLAAttention that has the required attributes
        mock_mla_attn = MagicMock()
        mock_mla_attn.process_weights_after_loading = MagicMock()
        mock_mla_attn.impl = MagicMock()
        mock_mla_attn.impl.process_weights_after_loading = MagicMock()

        with patch("vllm_ascend.ops.mla.MLAAttention", return_value=mock_mla_attn):
            attn = AscendMultiHeadLatentAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                scale=self.scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                mla_modules=self.mock_mla_modules,
                cache_config=self.mock_cache_config,
                quant_config=self.mock_quant_config,
                prefix=self.prefix,
            )
        positions = torch.tensor([0, 1, 2])
        hidden_states = torch.randn(3, self.hidden_size)

        mock_forward_context = MagicMock(spec=ForwardContext)
        mock_forward_context.flash_comm_v1_enabled = False
        mock_get_forward_context.return_value = mock_forward_context

        mock_mla_forward.return_value = (3, self.hidden_size)

        output = attn.forward(positions, hidden_states)

        self.assertEqual(output.shape, (3, self.hidden_size))
