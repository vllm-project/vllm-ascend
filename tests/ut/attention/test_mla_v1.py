from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (AscendMLABackend,
                                          AscendMLADecodeMetadata,
                                          AscendMLAImpl, AscendMLAMetadata,
                                          AscendMLAMetadataBuilder,
                                          AscendMLAPrefillMetadata)


class TestAscendMLABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendMLABackend.get_name(), "ASCEND_MLA")

    def test_get_metadata_cls(self):
        self.assertEqual(AscendMLABackend.get_metadata_cls(),
                         AscendMLAMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(AscendMLABackend.get_builder_cls(),
                         AscendMLAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendMLABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendMLABackend.get_impl_cls()
        self.assertEqual(result, AscendMLAImpl)


class TestAscendMLAPrefillMetadata(TestBase):

    def test_ascend_mla_prefill_metadata_default(self):
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendMLAPrefillMetadata(attn_mask=attn_mask,
                                            query_lens=query_lens,
                                            seq_lens=seq_lens,
                                            context_lens=context_lens,
                                            input_positions=input_positions,
                                            query_start_loc=query_start_loc,
                                            block_table=block_table,
                                            max_query_len=max_query_len,
                                            max_seq_lens=max_seq_lens)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.query_lens, query_lens)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertIs(metadata.context_lens, context_lens)
        self.assertIs(metadata.input_positions, input_positions)
        self.assertIs(metadata.query_start_loc, query_start_loc)
        self.assertIs(metadata.block_table, block_table)
        self.assertEqual(metadata.max_query_len, max_query_len)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertIsNone(metadata.chunked_context)

    def test_ascend_mla_prefill_metadata_with_chunked_context(self):
        cu_seq_lens = torch.tensor([0, 2, 4])
        starts = torch.tensor([0, 2])
        seq_tot = [2, 2]
        max_seq_lens = [2, 2]
        workspace = torch.randn(2, 4)
        chunk_seq_lens = torch.tensor([2, 2])

        chunked_context = AscendMLAPrefillMetadata.ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens,
            starts=starts,
            seq_tot=seq_tot,
            max_seq_lens=max_seq_lens,
            workspace=workspace,
            chunk_seq_lens=chunk_seq_lens)

        metadata = AscendMLAPrefillMetadata(
            attn_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
            query_lens=[1, 2],
            seq_lens=[2, 2],
            context_lens=torch.tensor([1, 2]),
            input_positions=torch.tensor([0, 1, 0, 1]),
            query_start_loc=torch.tensor([0, 1, 3]),
            block_table=torch.tensor([[0, 1], [2, 3]]),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked_context)

        self.assertIsNotNone(metadata.chunked_context)
        self.assertIs(metadata.chunked_context.cu_seq_lens, cu_seq_lens)
        self.assertIs(metadata.chunked_context.starts, starts)
        self.assertEqual(metadata.chunked_context.seq_tot, seq_tot)
        self.assertEqual(metadata.chunked_context.max_seq_lens, max_seq_lens)
        self.assertIs(metadata.chunked_context.workspace, workspace)
        self.assertIs(metadata.chunked_context.chunk_seq_lens, chunk_seq_lens)


class TestAscendMLADecodeMetadata(TestBase):

    def test_ascend_mla_decode_metadata_default(self):
        input_positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        block_table = torch.tensor([[0, 3, 2, 1], [0, 2, 1, 3]])
        seq_lens = torch.tensor([[2], [3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None

        metadata = AscendMLADecodeMetadata(input_positions, block_table,
                                           seq_lens, max_seq_lens,
                                           seq_lens_list, attn_mask)

        self.assertIs(metadata.input_positions, input_positions)
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)


class TestAscendMLAMetadata(TestBase):

    def test_ascend_mla_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        query_start_loc = torch.tensor([1, 2, 3, 4])
        seq_lens = [30, 50]
        block_tables = torch.randint(0, 100, (100, 4))

        num_decodes = 4
        num_decode_tokens = 8
        num_prefills = 8

        num_input_tokens = 2

        query_lens = None
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        decode = None
        prefill = None

        metadata = AscendMLAMetadata(num_actual_tokens, slot_mapping,
                                     query_start_loc, seq_lens, block_tables,
                                     num_decodes, num_decode_tokens,
                                     num_prefills, num_input_tokens,
                                     query_lens, head_dim, attn_mask,
                                     attn_state, decode, prefill)

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertIs(metadata.query_start_loc, query_start_loc)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertIs(metadata.block_tables, block_tables)
        self.assertEqual(metadata.num_decodes, num_decodes)
        self.assertEqual(metadata.num_decode_tokens, num_decode_tokens)
        self.assertEqual(metadata.num_prefills, num_prefills)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertEqual(metadata.query_lens, query_lens)
        self.assertEqual(metadata.head_dim, head_dim)
        self.assertEqual(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)
        self.assertEqual(metadata.decode, decode)
        self.assertEqual(metadata.prefill, prefill)


class TestAscendMLAMetadataBuilder(TestBase):

    def test_ascend_mla_metadata_builder_default(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(mock_vllm_config, mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.chunked_prefill_enabled)

    def test_reorder_batch(self):
        ascend_config = MagicMock()

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(mock_vllm_config, mock_device)
            builder.decode_threshold = 1

        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 1, 1: 3, 2: 1, 3: 2}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [],
            1: [1],
            2: [],
            3: []
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        self.assertTrue(modified)
        input_batch.swap_states.assert_called_once_with(1, 2)


class TestAscendMLAImpl(TestBase):

    @patch('vllm.distributed.parallel_state._CP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_context_model_parallel_world_size",
           return_value=1)
    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
           return_value=2)
    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def setUp(self, ascend_config, get_current_vllm_config, mock_get_tp_size,
              mock_tp, mock_get_cp_size, mock_cp):
        mock_tp.world_size = 2
        mock_cp.world_size = 1
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        vllm_config.model_config = model_config
        get_current_vllm_config.return_value = vllm_config
        vllm_config.parallel_config.enable_sequence_parallel = False

        num_heads = 256
        head_size = 1024
        scale = 0.1
        num_kv_heads = 8
        kv_cache_dtype = "auto"

        kv_a_layernorm = MagicMock()
        kv_a_layernorm.weight = torch.randn(96)
        kv_a_layernorm.variance_epsilon = 1e-6
        kwargs = {
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "qk_head_dim": 96,
            "v_head_dim": 128,
            "rotary_emb": MagicMock(),
            "q_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "kv_a_layernorm": kv_a_layernorm,
        }

        self.impl = AscendMLAImpl(num_heads=num_heads,
                                  head_size=head_size,
                                  scale=scale,
                                  num_kv_heads=num_kv_heads,
                                  alibi_slopes=None,
                                  sliding_window=None,
                                  kv_cache_dtype=kv_cache_dtype,
                                  blocksparse_params=None,
                                  logits_soft_cap=None,
                                  attn_type=None,
                                  kv_sharing_target_layer_name=None,
                                  **kwargs)

    def test_init(self):
        self.assertEqual(self.impl.num_heads, 256)
        self.assertEqual(self.impl.head_size, 1024)
        self.assertEqual(self.impl.scale, 0.1)
        self.assertEqual(self.impl.num_kv_heads, 8)
        self.assertEqual(self.impl.kv_cache_dtype, "auto")
        self.assertEqual(self.impl.q_lora_rank, 64)
        self.assertEqual(self.impl.kv_lora_rank, 32)
        self.assertEqual(self.impl.qk_nope_head_dim, 64)
        self.assertEqual(self.impl.qk_rope_head_dim, 32)
        self.assertEqual(self.impl.qk_head_dim, 96)
        self.assertEqual(self.impl.v_head_dim, 128)
        self.assertIsNotNone(self.impl.rotary_emb)
        self.assertIsNotNone(self.impl.q_proj)
        self.assertIsNotNone(self.impl.kv_b_proj)
        self.assertIsNotNone(self.impl.o_proj)
        self.assertIsNotNone(self.impl.kv_a_proj_with_mqa)
        self.assertIsNotNone(self.impl.kv_a_layernorm)
        self.assertEqual(self.impl.num_queries_per_kv, 32)
        self.assertEqual(self.impl.tp_size, 2)

    def test_v_up_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads,
                        self.impl.kv_lora_rank)

        if not hasattr(self.impl, 'W_UV') or self.impl.W_UV is None:
            self.impl.W_UV = torch.randn(self.impl.num_heads,
                                         self.impl.kv_lora_rank,
                                         self.impl.v_head_dim)
        result = self.impl._v_up_proj(x)

        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1],
                         self.impl.num_heads * self.impl.v_head_dim)

    def test_q_proj_and_k_up_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)
        q_proj_output = torch.randn(batch_size, self.impl.num_heads,
                                    self.impl.qk_head_dim)
        self.impl.q_proj.return_value = (q_proj_output, )
        if not hasattr(self.impl, 'W_UK_T') or self.impl.W_UK_T is None:
            self.impl.W_UK_T = torch.randn(self.impl.num_heads,
                                           self.impl.qk_nope_head_dim,
                                           self.impl.kv_lora_rank)
        result = self.impl._q_proj_and_k_up_proj(x)
        ql_nope, q_pe = result
        self.assertEqual(ql_nope.shape[0], batch_size)
        self.assertEqual(ql_nope.shape[1], self.impl.num_heads)
        self.assertEqual(ql_nope.shape[2], self.impl.kv_lora_rank)
        self.assertEqual(q_pe.shape[0], batch_size)
        self.assertEqual(q_pe.shape[1], self.impl.num_heads)
        self.assertEqual(q_pe.shape[2], self.impl.qk_rope_head_dim)

    def test_process_weights_after_loading(self):
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock()
        apply = MagicMock()
        quant_method.apply = apply
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim +
                                         self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        apply.return_value = layer.weight.T
        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    def test_compute_prefill_context_none(self):
        batch_size = 4
        kv_cache = torch.randn(10, 1, 1, 192)
        query = torch.randn(batch_size, self.impl.num_heads,
                            self.impl.qk_head_dim)
        metadata = MagicMock()
        metadata.prefill = None
        prefix_out = torch.randn(2, 16, 128)
        prefix_lse = torch.randn(2, 16, 8)
        q_pe = query[..., self.impl.qk_nope_head_dim:]
        q_nope = query[..., :self.impl.qk_nope_head_dim]

        out, lse = self.impl._compute_prefill_context(q_nope, q_pe, kv_cache,
                                                      32, metadata, prefix_out,
                                                      prefix_lse)

        self.assertTrue(torch.equal(prefix_out, out))
        self.assertTrue(torch.equal(prefix_lse, lse))

    @patch("torch_npu.atb.npu_paged_cache_load")
    @patch("torch_npu.atb.npu_ring_mla")
    def test_compute_prefill_context(self, mock_ring, mock_load):
        S, N, D, VD = 2, self.impl.num_heads, self.impl.qk_head_dim, self.impl.v_head_dim
        _, AND = self.impl.qk_rope_head_dim, self.impl.qk_nope_head_dim
        latent_kv_dim = self.impl.kv_lora_rank
        num_blocks, block_size = 100, 20
        query = torch.randn(S, N, D)
        q_nope = query[..., :self.impl.qk_nope_head_dim]
        q_pe = query[..., self.impl.qk_nope_head_dim:]
        kv_cache_0 = torch.randn(num_blocks, block_size, N, latent_kv_dim)
        kv_cache_1 = torch.randn(num_blocks, block_size, N, D)
        kv_cache = [kv_cache_0, kv_cache_1]
        prefix_out = torch.randn(S, N, 128)
        prefix_lse = torch.randn(S, N)

        self.impl.kv_b_proj.return_value = (torch.randn(8, N, VD + AND), )

        chunk_ctx = MagicMock()
        chunk_ctx.seq_tot = [8]
        chunk_ctx.chunk_seq_lens = [torch.tensor([8])]
        chunk_ctx.starts = [torch.tensor([0])]

        prefill_meta = MagicMock()
        prefill_meta.chunked_context = chunk_ctx
        prefill_meta.query_lens = [8]
        prefill_meta.block_table = torch.randint(0, 100, (S, 4))

        meta = MagicMock()
        meta.prefill = prefill_meta
        self.impl.prefill_mask = torch.triu(
            torch.ones(512, 512, device=q_nope.device, dtype=q_nope.dtype), 1)

        out, lse = self.impl._compute_prefill_context(q_nope, q_pe, kv_cache,
                                                      32, meta, prefix_out,
                                                      prefix_lse)

        mock_load.assert_called_once()
        mock_ring.assert_called_once()

        self.assertEqual(out.shape, prefix_out.shape)
        self.assertEqual(lse.shape, prefix_lse.shape)

    @patch("vllm_ascend.attention.mla_v1.AscendMLAImpl._v_up_proj")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode_without_graph(self,
                                          mock_npu_fused_infer_attention_score,
                                          mock_up_proj):
        num_tokens = 100
        block_size = 4
        q_nope = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, self.impl.num_heads,
                           self.impl.qk_rope_head_dim)
        k_nope = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.qk_nope_head_dim)
        k_pe = torch.randn(num_tokens, self.impl.num_heads,
                           self.impl.qk_rope_head_dim)
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.block_table = MagicMock()
        metadata.decode.seq_lens = 10
        mock_npu_fused_infer_attention_score.return_value = [
            torch.randn(num_tokens, self.impl.num_heads,
                        self.impl.kv_lora_rank), None
        ]
        mock_up_proj.return_value = torch.randn(num_tokens,
                                                self.impl.num_heads,
                                                self.impl.v_head_dim)
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe,
                                           block_size, metadata)
        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.num_heads)
        self.assertEqual(result.shape[2], self.impl.v_head_dim)
        mock_up_proj.assert_called_once()
        mock_npu_fused_infer_attention_score.assert_called_once()

    @patch("vllm_ascend.attention.mla_v1.npu_prefetch")
    def test_mla_preprocess(self, magic_npu_fetch):
        magic_npu_fetch.return_value = MagicMock()
        batch_size = 4
        seq_len = 8
        hidden_size = 1024
        hidden_states = torch.randn(batch_size * seq_len, hidden_size)

        kv_cache = MagicMock()

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_prefills = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_actual_tokens = 4
        num_prefill_tokens = 2
        attn_metadata.slot_mapping = torch.arange(4)
        attn_metadata.decode.cos = torch.randn(2, 64)
        attn_metadata.decode.sin = torch.randn(2, 64)
        attn_metadata.prefill.cos = torch.randn(2, 64)
        attn_metadata.prefill.sin = torch.randn(2, 64)

        self.impl.q_a_proj = MagicMock()
        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.num_heads,
            self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_nope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_head_dim)
        ]
        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.v_head_dim + self.impl.qk_nope_head_dim)
        ]
        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]
        self.impl.exec_kv_prefill = MagicMock()
        self.impl.exec_kv_prefill.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_rope_head_dim),
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.kv_lora_rank)
        ]
        self.impl._q_proj_and_k_up_proj = MagicMock()
        self.impl._q_proj_and_k_up_proj.return_value = [
            MagicMock(), MagicMock()
        ]
        self.impl.num_kv_heads = self.impl.num_heads

        decode_res, prefill_res = self.impl._mla_preprocess(
            hidden_states, kv_cache, attn_metadata, need_gather_q_kv=False)

        self.assertIsNotNone(decode_res)
        self.assertIsNotNone(prefill_res)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            None, None,
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank)
        ]

        k_pe, k_nope = self.impl.exec_kv_prefill(kv_no_split, cos, sin,
                                                 kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_decode(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank), None, None
        ]

        k_pe, k_nope = self.impl.exec_kv_decode(kv_no_split, cos, sin,
                                                kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch.npu.stream")
    @patch("vllm_ascend.attention.mla_v1.get_multistream_comm_context")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode(self, mock_npu_fused_infer_attention_score,
                            mock_get_multistream_comm_context,
                            mock_npu_stream):
        B = 2
        N = self.impl.num_kv_heads
        BS = 100
        HD = self.impl.v_head_dim
        self.impl.kv_lora_rank = 256
        self.impl.spec_token_num = 1
        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(B, N, HD)
        q_nope = torch.randn(B, N, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(B, N, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, N, self.impl.kv_lora_rank)
        k_pe = torch.randn(BS, N, self.impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_lengths_q = MagicMock()
        attn_metadata.decode.seq_lens_list = MagicMock()
        self.impl.enable_kv_nz = True

        mock_npu_fused_infer_attention_score.return_value = [
            torch.randn(B, N, self.impl.kv_lora_rank), None
        ]
        mock_get_multistream_comm_context.return_value = None

        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS,
                                           attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)

        self.impl.enable_kv_nz = False
        attn_metadata.attn_state = None
        mock_return_value = MagicMock()
        mock_get_multistream_comm_context.return_value = mock_return_value
        mock_return_value.before_comm_event = MagicMock()
        mock_return_value.comm_stream = MagicMock()
        mock_npu_stream.return_value = MagicMock()

        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS,
                                           attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)


# test _forward_prefill_cp and ring_mla
class TestAscendMLAImplWithCPSP(TestAscendMLAImpl):

    @patch("vllm_ascend.attention.mla_v1.npu_prefetch")
    @patch("vllm_ascend.attention.mla_v1.get_tp_group")
    def test_mla_preprocess_prefill_with_sp(self, mock_get_tp_group,
                                            mock_npu_prefetch):
        mock_group = MagicMock()
        mock_group.all_gather.side_effect = lambda x, dim: x
        mock_get_tp_group.return_value = mock_group
        mock_npu_prefetch.return_value = None

        # config impl
        self.impl.enable_sp = True
        self.impl.cp_size = 1  # do not trigger CP
        self.impl.num_kv_heads = self.impl.num_heads

        # mock metadata（minimal）
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_prefills = 1
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_actual_tokens = 2
        attn_metadata.num_input_tokens = 2
        attn_metadata.slot_mapping = torch.arange(2)
        prefill = MagicMock()
        prefill.cos = torch.randn(2, self.impl.qk_rope_head_dim)
        prefill.sin = torch.randn(2, self.impl.qk_rope_head_dim)
        attn_metadata.prefill = prefill

        # mock module/ops, shape expected to be equal
        self.impl.q_a_proj = MagicMock()
        self.impl.q_a_proj.weight = torch.randn(
            1)  # avoid real npu_prefetch error
        self.impl.q_a_proj.return_value = [
            torch.randn(attn_metadata.num_input_tokens, self.impl.qk_head_dim)
        ]
        self.impl.q_a_layernorm = MagicMock(side_effect=lambda x: x)

        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(attn_metadata.num_input_tokens,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        ]

        # q_proj return reshaped blob
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [
            torch.randn(attn_metadata.num_actual_tokens,
                        self.impl.num_heads * self.impl.qk_head_dim)
        ]

        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)

        # exec_kv_prefill should be called if cp_size==1, return (k_pe, kv_c_normed)
        num_prefill = attn_metadata.num_actual_tokens - attn_metadata.num_decode_tokens
        self.impl.exec_kv_prefill = MagicMock()
        self.impl.exec_kv_prefill.return_value = [
            torch.randn(num_prefill, self.impl.num_heads,
                        self.impl.qk_rope_head_dim),
            torch.randn(num_prefill, self.impl.num_heads,
                        self.impl.kv_lora_rank),
        ]

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(num_prefill, self.impl.num_heads,
                        self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        ]

        self.impl.kv_a_layernorm = MagicMock(side_effect=lambda x: x)

        kv_cache = [MagicMock(), MagicMock()]

        decode_res, prefill_res = self.impl._mla_preprocess(
            hidden_states=torch.randn(attn_metadata.num_input_tokens, 64),
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            need_gather_q_kv=False,  # enable_sp=True 会把它强制为 True
        )

        assert decode_res is None
        assert prefill_res is not None
        self.assertEqual(
            mock_get_tp_group.call_count,
            2)  # check all_gather called twice (q_c + kv_no_split)
        mock_npu_prefetch.assert_called()

    @patch("vllm_ascend.attention.mla_v1.npu_prefetch")
    @patch("torch_npu._npu_reshape_and_cache")
    @patch("vllm_ascend.attention.mla_v1.get_cp_group")
    def test_mla_preprocess_prefill_with_cp(self, mock_get_cp_group,
                                            mock_npu_reshape_and_cache,
                                            mock_npu_prefetch):
        mock_group = MagicMock()
        mock_group.all_gather.side_effect = lambda x, dim: x
        mock_get_cp_group.return_value = mock_group
        mock_npu_reshape_and_cache.return_value = None
        mock_npu_prefetch.return_value = None

        # config impl
        self.impl.cp_size = 2  # trigger CP
        self.impl.enable_sp = False
        self.impl.num_kv_heads = 1  # using num_kv_heads in setup will raise error

        # mock metadata
        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_prefills = 1
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_actual_tokens = 2
        attn_metadata.num_input_tokens = 2
        attn_metadata.slot_mapping = torch.arange(2)
        prefill = MagicMock()
        prefill.cos = torch.randn(2, self.impl.qk_rope_head_dim)
        prefill.sin = torch.randn(2, self.impl.qk_rope_head_dim)
        prefill.cp_kv_recover_idx = torch.arange(
            attn_metadata.num_actual_tokens)
        attn_metadata.prefill = prefill

        # mock module/ops
        self.impl.q_a_proj = MagicMock()
        self.impl.q_a_proj.weight = torch.randn(1)
        self.impl.q_a_proj.return_value = [
            torch.randn(attn_metadata.num_input_tokens, self.impl.qk_head_dim)
        ]
        self.impl.q_a_layernorm = MagicMock(side_effect=lambda x: x)

        # kv_a_proj_with_mqa return last-dim = kv_lora + qk_rope
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(attn_metadata.num_input_tokens,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        ]

        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [
            torch.randn(attn_metadata.num_actual_tokens,
                        self.impl.num_heads * self.impl.qk_head_dim)
        ]

        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.kv_a_layernorm = MagicMock(side_effect=lambda x: x)

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(attn_metadata.num_actual_tokens, self.impl.num_heads,
                        self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        ]

        kv_cache = [MagicMock(), MagicMock()]

        decode_res, prefill_res = self.impl._mla_preprocess(
            hidden_states=torch.randn(attn_metadata.num_input_tokens, 64),
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            need_gather_q_kv=False,
        )

        assert decode_res is None
        assert prefill_res is not None
        mock_get_cp_group.assert_called_once()
        mock_npu_reshape_and_cache.assert_called_once()
        mock_npu_prefetch.assert_called()

    @patch("torch_npu.atb.npu_ring_mla")
    def test_attention_with_mask_and_nomask(self, mock_ring):
        num_tokens = 2
        num_heads = self.impl.num_heads
        v_head_dim = self.impl.v_head_dim

        q_nope = torch.randn(num_tokens, num_heads, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, num_heads, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(4, num_heads, self.impl.qk_nope_head_dim)
        k_pe = torch.randn(4, num_heads, self.impl.qk_rope_head_dim)
        value = torch.randn(4, num_heads, v_head_dim)

        kv_mask_idx = torch.tensor([0, 1])
        kv_nomask_idx = torch.tensor([2, 3])
        attn_mask_seqlens = torch.tensor([2])
        attn_nomask_seqlens = torch.tensor([2])
        mask = torch.ones(2, 2)

        mock_ring.side_effect = lambda **kwargs: None

        out = self.impl._attention_with_mask_and_nomask(
            q_nope, q_pe, k_nope, k_pe, value, kv_mask_idx, kv_nomask_idx,
            attn_mask_seqlens, attn_nomask_seqlens, mask)

        # check output shape
        self.assertEqual(out.shape, (num_tokens, num_heads, v_head_dim))
        # should be called twice (mask + nomask)
        self.assertEqual(mock_ring.call_count, 2)

    @patch("torch_npu.atb.npu_ring_mla")
    def test_forward_prefill_cp(self, mock_ring):
        num_tokens = 4
        num_heads = self.impl.num_heads
        v_head_dim = self.impl.v_head_dim

        q_nope = torch.randn(num_tokens, num_heads, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, num_heads, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(num_tokens, num_heads, self.impl.qk_nope_head_dim)
        k_pe = torch.randn(num_tokens, num_heads, self.impl.qk_rope_head_dim)
        value = torch.randn(num_tokens, num_heads, v_head_dim)

        # fake attn_metadata.prefill
        prefill = MagicMock()
        prefill.q_head_idx = torch.tensor([0, 1])
        prefill.q_tail_idx = torch.tensor([2, 3])
        prefill.kv_with_q_head_nomask_idx = torch.tensor([], dtype=torch.long)
        prefill.kv_with_q_head_mask_idx = torch.tensor([0, 1])
        prefill.kv_with_q_tail_nomask_idx = torch.tensor([], dtype=torch.long)
        prefill.kv_with_q_tail_mask_idx = torch.tensor([2, 3])
        prefill.attn_mask_seqlens = torch.tensor([2, 2])
        prefill.head_attn_nomask_seqlens = torch.tensor([0])
        prefill.tail_attn_nomask_seqlens = torch.tensor([0])
        prefill.cp_prefill_mask = torch.ones(2, 2)
        prefill.q_full_idx = torch.tensor([0, 1, 2, 3])

        attn_metadata = MagicMock()
        attn_metadata.prefill = prefill

        # mock npu_ring_mla do not run
        mock_ring.side_effect = lambda **kwargs: None

        output = self.impl._forward_prefill_cp(q_nope, q_pe, k_nope, k_pe,
                                               value, (k_nope, k_pe),
                                               attn_metadata)

        # check output shape
        self.assertEqual(output.shape, (num_tokens, num_heads * v_head_dim))

        # should call ring_mla for twice（head + tail）
        self.assertGreaterEqual(mock_ring.call_count, 2)

    @patch("torch.distributed.all_to_all_single")
    @patch("torch.distributed.all_gather")
    @patch("torch_npu.atb.npu_multi_head_latent_attention_with_lse")
    def test_forward_decode_sp(self, mock_mla_with_lse, mock_all_gather,
                               mock_all_to_all_single):
        mock_mla_with_lse.return_value = MagicMock()

        # mock 分布式 all_to_all_single，直接复制输入到输出
        def fake_all_to_all_single(output, input, group=None):
            output.copy_(input)

        mock_all_to_all_single.side_effect = fake_all_to_all_single

        # mock all_gather，简单模拟单卡收集
        def fake_all_gather(tensor_list, tensor, group=None):
            for i in range(len(tensor_list)):
                tensor_list[i].copy_(tensor)

        mock_all_gather.side_effect = fake_all_gather

        num_tokens = 100
        block_size = 4
        num_blocks = 100

        self.impl.enable_sp = True
        self.impl.num_heads = 8
        self.impl.num_kv_heads = self.impl.num_heads
        self.impl.tp_size = 2
        self.impl.sp_size = 2
        self.impl.cp_size = 2
        self.impl.sp_rank = 0
        self.impl.cp_rank = 0
        self.impl.sp_group = MagicMock()  # 避免 dist 默认 group
        self.impl.cp_group = MagicMock()
        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(
            num_tokens, self.impl.v_head_dim)

        meta_data = MagicMock()
        meta_data.decode = MagicMock()
        meta_data.decode.num_computed_tokens_of_cp_sp = np.array([[[2, 2],
                                                                   [2, 2]]])
        meta_data.decode.block_table = torch.arange(num_blocks)

        q_nope = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, self.impl.num_heads,
                           self.impl.qk_rope_head_dim)
        k_nope = torch.randn(num_blocks, self.impl.num_heads, block_size,
                             self.impl.kv_lora_rank)
        k_pe = torch.randn(num_blocks, self.impl.num_heads, block_size,
                           self.impl.qk_rope_head_dim)

        result = self.impl._forward_decode_sp(q_nope, q_pe, k_nope, k_pe,
                                              block_size, meta_data)

        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.v_head_dim)

        mock_all_to_all_single.assert_called()
        mock_all_gather.assert_called()
        mock_mla_with_lse.assert_called()
