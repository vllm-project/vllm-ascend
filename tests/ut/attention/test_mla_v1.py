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
        runner = MagicMock()
        runner.scheduler_config = MagicMock()
        runner.model_config = MagicMock()
        runner.scheduler_config.max_num_seqs = 4
        runner.model_config.max_model_len = 1024
        runner.model_config.get_head_size.return_value = 64
        runner.model_config.dtype = torch.float16
        runner.chunked_prefill_enabled = False
        runner.device = "cpu"
        runner.block_size = 16

        ascend_config = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

            self.assertEqual(builder.runner, runner)
            self.assertEqual(builder.block_size, runner.block_size)
            self.assertEqual(builder.chunked_prefill_enabled,
                             runner.chunked_prefill_enabled)
            self.assertEqual(builder.torchair_graph_enabled, True)

    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_reorder_batch_with_torchair_graph(self, ascend_config):
        runner = MagicMock()
        runner.chunked_prefill_enabled = False
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True

        builder = AscendMLAMetadataBuilder(runner)

        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 2, 1: 1, 2: 3, 3: 1}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [1],
            1: [],
            2: [1, 1],
            3: []
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        self.assertFalse(modified)
        self.assertEqual(builder._num_decodes, 4)
        self.assertEqual(builder._num_prefills, 0)
        self.assertEqual(builder._num_decode_tokens, 7)
        self.assertEqual(builder._num_prefill_tokens, 0)
        input_batch.swap_states.assert_not_called()

    def test_reorder_batch_without_torchair_graph(self):
        ascend_config = MagicMock()
        runner = MagicMock()
        runner.chunked_prefill_enabled = False
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = False
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

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
        self.assertEqual(builder._num_decodes, 2)
        self.assertEqual(builder._num_prefills, 2)
        self.assertEqual(builder._num_decode_tokens, 2)
        self.assertEqual(builder._num_prefill_tokens, 5)
        input_batch.swap_states.assert_called_once_with(1, 2)

    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_get_graph_runner_block_tables_normal(self, mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros((8, 64), dtype=torch.int32)
        runner.chunked_prefill_enabled = False
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertTrue(torch.equal(result[:, :10], block_tables))

    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_get_graph_runner_block_tables_truncated(self, mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros((8, 4), dtype=torch.int32)
        runner.chunked_prefill_enabled = False
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 4)
        self.assertTrue(torch.equal(result, block_tables[:, :4]))

    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_get_graph_runner_block_tables_from_numpy(self,
                                                      mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        runner = MagicMock()
        runner.graph_block_tables = np.zeros((8, 64), dtype=np.int32)
        runner.chunked_prefill_enabled = False
        builder = AscendMLAMetadataBuilder(runner=runner)

        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)

        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertTrue(torch.equal(result[:, :10], block_tables))

    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_build_dummy(self, mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        runner = MagicMock()
        runner.model_config = MagicMock()
        runner.device = "cpu"
        runner.graph_block_tables = torch.zeros((8, 64), dtype=torch.int32)
        runner.model_config.get_head_size.return_value = 64
        runner.chunked_prefill_enabled = False
        runner.attn_mask = torch.zeros((1, 1), dtype=torch.bool)
        runner.spec_attn_mask = torch.zeros((1, 1), dtype=torch.bool)
        runner.dtype = torch.float16

        builder = AscendMLAMetadataBuilder(runner=runner,
                                           metadata_cls=AscendMLAMetadata)
        builder.rope_dim = 64

        with patch.object(builder,
                          "_get_graph_runner_block_tables",
                          side_effect=lambda x, y: y):
            metadata = builder.build_torchair_graph_dummy(3, 3)

        sin_golden = torch.ones(3,
                                1,
                                1,
                                64,
                                dtype=runner.dtype,
                                device=runner.device)
        cos_golden = torch.ones(3,
                                1,
                                1,
                                64,
                                dtype=runner.dtype,
                                device=runner.device)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_input_tokens, 3)
        self.assertEqual(metadata.num_actual_tokens, 3)
        self.assertEqual(metadata.num_decodes, 1)
        self.assertEqual(metadata.num_decode_tokens, 1)
        self.assertEqual(metadata.num_prefills, 0)
        self.assertEqual(metadata.attn_state, AscendAttentionState.DecodeOnly)
        self.assertIsNone(metadata.prefill)
        self.assertIsInstance(metadata.decode, AscendMLADecodeMetadata)
        self.assertEqual(metadata.block_tables.shape[0], 3)
        self.assertEqual(metadata.block_tables.shape[1], 64)
        self.assertEqual(metadata.seq_lens.shape[0], 3)
        self.assertEqual(metadata.slot_mapping.shape[0], 3)
        self.assertEqual(metadata.query_start_loc.shape[0], 3)
        assert torch.equal(sin_golden, metadata.decode.sin)
        assert torch.equal(cos_golden, metadata.decode.cos)


class TestAscendMLAImpl(TestBase):

    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
           return_value=2)
    @patch("vllm.config.get_current_vllm_config")
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def setUp(self, ascend_config, vllm_config, mock_get_tp_size, mock_tp):
        mock_tp.world_size = 2
        ascend_config.torchair_graph_config.enabled = True
        ascend_config.torchair_graph_config.enable_kv_nz = False
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config

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
        self.assertTrue(self.impl.torchair_graph_enabled)

    def test_v_up_proj_and_o_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads,
                        self.impl.kv_lora_rank)

        self.impl.o_proj.return_value = (torch.randn(
            batch_size, self.impl.num_heads * self.impl.v_head_dim), )
        if not hasattr(self.impl, 'W_UV') or self.impl.W_UV is None:
            self.impl.W_UV = torch.randn(self.impl.num_heads,
                                         self.impl.kv_lora_rank,
                                         self.impl.v_head_dim)
        result = self.impl._v_up_proj_and_o_proj(x)

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
        out, lse = self.impl._compute_prefill_context(query, kv_cache, 32,
                                                      metadata, prefix_out,
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

        out, lse = self.impl._compute_prefill_context(query, kv_cache, 32,
                                                      meta, prefix_out,
                                                      prefix_lse)

        mock_load.assert_called_once()
        mock_ring.assert_called_once()

        self.assertEqual(out.shape, prefix_out.shape)
        self.assertEqual(lse.shape, prefix_lse.shape)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv(self, mock_kv_cache):
        batch_size = 2
        hidden = torch.randn(batch_size, 128)
        cos = torch.randn(batch_size, 32)
        sin = torch.randn(batch_size, 32)
        kv_cache = (torch.randn(
            4, 8, self.impl.kv_lora_rank + self.impl.qk_rope_head_dim),
                    torch.randn(
                        4, 8,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))
        slots = torch.arange(batch_size, dtype=torch.long)

        proj_out = torch.randn(
            batch_size, self.impl.num_kv_heads, 1,
            self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa.return_value = (proj_out, )

        mock_kv_cache.return_value = (torch.randn(batch_size,
                                                  self.impl.num_kv_heads, 1,
                                                  self.impl.qk_rope_head_dim),
                                      torch.randn(batch_size,
                                                  self.impl.num_kv_heads, 1,
                                                  self.impl.kv_lora_rank),
                                      None, None)

        k_pe, k_nope, kv = self.impl.exec_kv(hidden, cos, sin, kv_cache, slots)

        self.impl.kv_a_proj_with_mqa.assert_called_once_with(hidden)
        mock_kv_cache.assert_called_once()
        self.assertEqual(k_pe.shape, (batch_size, self.impl.num_kv_heads, 1,
                                      self.impl.qk_rope_head_dim))
        self.assertEqual(
            k_nope.shape,
            (batch_size, self.impl.num_kv_heads, 1, self.impl.kv_lora_rank))
        self.assertEqual(kv.shape,
                         (batch_size, self.impl.num_kv_heads, 1,
                          self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill(self, mock_kv):
        B, N, S, H = 2, self.impl.num_kv_heads, 1, 128
        hidden_states = torch.randn(B, N, S, H)
        cos = torch.randn(B, S, 32)
        sin = torch.randn(B, S, 32)
        kv_cache = (
            torch.randn(100, 8,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim),
            torch.randn(100, 8,
                        self.impl.kv_lora_rank + self.impl.qk_rope_head_dim),
        )

        slots = torch.arange(B * S, dtype=torch.long)

        proj_out = torch.randn(
            B, N, S, self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa.return_value = (proj_out, )

        mock_kv.return_value = (None, None,
                                torch.randn(B, self.impl.num_kv_heads, S,
                                            self.impl.qk_rope_head_dim),
                                torch.randn(B, self.impl.num_kv_heads, S,
                                            self.impl.kv_lora_rank))

        k_pe, k_nope = self.impl.exec_kv_prefill(hidden_states, cos, sin,
                                                 kv_cache, slots)

        self.impl.kv_a_proj_with_mqa.assert_called_once_with(hidden_states)
        mock_kv.assert_called_once()

        self.assertEqual(
            k_pe.shape,
            (B, self.impl.num_kv_heads, S, self.impl.qk_rope_head_dim))
        self.assertEqual(
            k_nope.shape,
            (B, self.impl.num_kv_heads, S, self.impl.kv_lora_rank))

    @patch("torch_npu.npu_interleave_rope")
    def test_rope_single(self, mock_rope):
        B, N, D = 2, 16, 1024
        x = torch.randn(B, N, D)
        cos = torch.randn(B, N, 1, D)
        sin = torch.randn(B, N, 1, D)
        mock_rope.return_value = x.view(B, N, 1, D)
        result = self.impl.rope_single(x, cos, sin)
        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], D)
        mock_rope.assert_called_once()

    @patch("vllm_ascend.attention.mla_v1.AscendMLAImpl._v_up_proj_and_o_proj")
    @patch("torch_npu._npu_paged_attention_mla")
    def test_forward_decode_without_graph(self, mock_page_attention_mla,
                                          mock_up_proj):
        self.impl.running_in_graph = False
        num_tokens = 100
        num_blocks = 256
        block_size = 4
        q_nope = torch.randn(num_tokens, self.impl.num_heads,
                             self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, self.impl.num_heads,
                           self.impl.qk_rope_head_dim)
        kv_c_and_k_pe_cache = torch.randn(num_blocks, block_size,
                                          self.impl.num_heads,
                                          self.impl.kv_lora_rank)
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.block_table = MagicMock()
        metadata.decode.seq_lens = 10
        mock_page_attention_mla.return_value = torch.randn(
            num_tokens, self.impl.num_heads, self.impl.kv_lora_rank)
        mock_up_proj.return_value = torch.randn(num_tokens,
                                                self.impl.num_heads,
                                                self.impl.v_head_dim)
        result = self.impl._forward_decode(q_nope, q_pe, None, None,
                                           kv_c_and_k_pe_cache, metadata)
        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.num_heads)
        self.assertEqual(result.shape[2], self.impl.v_head_dim)
        mock_up_proj.assert_called_once()
        mock_page_attention_mla.assert_called_once()
