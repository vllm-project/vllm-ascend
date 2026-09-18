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
        mock_indexer.wk_weights_proj = nn.Linear(128, 128)
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
        self.assertIs(wrapper.wk_weights_proj, mock_indexer.wk_weights_proj)
        self.assertIs(wrapper.k_norm, mock_indexer.k_norm)
        self.assertEqual(wrapper.softmax_scale, 0.123)
        self.assertIs(wrapper.k_cache, mock_indexer.k_cache)

        self.assertIsNone(mock_indexer.topk_indices_buffer)

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
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    def test_initialization(self, mock_tp_size, mock_get_vllm_config):
        # Create a proper mock for MLAAttention that has the required attributes
        mock_mla_attn = MagicMock()
        mock_mla_attn.process_weights_after_loading = MagicMock()
        mock_mla_attn.impl = MagicMock()
        mock_mla_attn.impl.process_weights_after_loading = MagicMock()

        with patch("vllm_ascend.ops.mla.MLAAttention", return_value=mock_mla_attn):
            mock_tp_size.return_value = 2
            mock_vllm_config = MagicMock(spec=VllmConfig)
            mock_vllm_config.model_config.hf_text_config = MagicMock(num_hidden_layers=32, first_k_dense_replace=True)
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
            self.assertIsNotNone(attn.mla_attn)

    @patch("vllm_ascend.ops.mla.torch.ops.vllm.mla_forward")
    @patch("vllm_ascend.ops.mla.get_current_vllm_config")
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.ops.mla.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward(
        self,
        mock_get_forward_context_2,
        mock_get_forward_context,
        mock_tp_size,
        mock_get_vllm_config,
        mock_mla_forward,
    ):
        mock_tp_size.return_value = 1
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_vllm_config.model_config.hf_text_config = MagicMock(num_hidden_layers=32, first_k_dense_replace=False)
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
        mock_get_forward_context.return_value = mock_forward_context
        mock_get_forward_context_2.return_value = mock_forward_context

        mock_mla_forward.return_value = (3, self.hidden_size)

        output = attn.forward(positions, hidden_states)

        self.assertEqual(output.shape, (3, self.hidden_size))

    @patch("vllm_ascend.ops.mla.torch.ops.vllm.mla_forward")
    def test_fused_o_proj_allocates_sequence_shard(self, mock_mla_forward):
        attn = AscendMultiHeadLatentAttention.__new__(AscendMultiHeadLatentAttention)
        torch.nn.Module.__init__(attn)
        attn.hidden_size = 32
        attn.output_token_shard_size = 8
        attn.prefix = self.prefix

        def write_output(hidden_states, output, prefix):
            self.assertEqual(output.shape, ((hidden_states.shape[0] + 7) // 8, 32))
            self.assertEqual(prefix, self.prefix)
            output.fill_(2)

        mock_mla_forward.side_effect = write_output
        for num_tokens in (0, 1, 8, 17):
            with self.subTest(num_tokens=num_tokens):
                output = attn.forward(torch.arange(num_tokens), torch.empty(num_tokens, 32, dtype=torch.bfloat16))
                self.assertEqual(output.shape, ((num_tokens + 7) // 8, 32))
                torch.testing.assert_close(output, torch.full_like(output, 2))
