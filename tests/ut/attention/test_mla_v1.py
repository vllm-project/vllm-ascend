from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (AscendMLABackend,
                                                AscendMLAMetadata,
                                                AscendMLAMetadataBuilder,
                                                AscendMLAImpl,
                                                AscendMLAPrefillMetadata,
                                                AscendMLADecodeMetadata)


class TestAscendMLABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendMLABackend.get_name(), "ASCEND")

    def test_get_impl_cls(self):
        self.assertEqual(AscendMLABackend.get_impl_cls(),
                         AscendMLAImpl)

    def test_get_metadata_cls(self):
        self.assertEqual(AscendMLABackend.get_metadata_cls(),
                         AscendMLAMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(AscendMLABackend.get_builder_cls(),
                         AscendMLAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendMLABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

class TestAscendMLAPrefillMetadata(TestBase):
    def test_ascend_mla_prefill_metadata_default(self):
    # 构造最小必要输入
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendMLAPrefillMetadata(
            attn_mask=attn_mask,
            query_lens=query_lens,
            seq_lens=seq_lens,
            context_lens=context_lens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens
        )
        self.assertEqual(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.query_lens, query_lens)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.context_lens, context_lens)
        self.assertEqual(metadata.input_positions, input_positions)
        self.assertEqual(metadata.query_start_loc, query_start_loc)
        self.assertEqual(metadata.block_table, block_table)
        self.assertEqual(metadata.max_query_len, max_query_len)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertIsNone(metadata.chunked_context)
        
    def test_ascend_mla_prefill_metadata_with_chunked_context(self):
    # 构造嵌套的 ChunkedContextMetadata
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
            chunk_seq_lens=chunk_seq_lens
        )

        # 构造主 metadata
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
            chunked_context=chunked_context
        )

        self.assertIsNotNone(metadata.chunked_context)
        self.assertEqual(metadata.chunked_context.cu_seq_lens, cu_seq_lens)
        self.assertEqual(metadata.chunked_context.starts, starts)
        self.assertEqual(metadata.chunked_context.seq_tot, seq_tot)
        self.assertEqual(metadata.chunked_context.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.chunked_context.workspace, workspace)
        self.assertEqual(metadata.chunked_context.chunk_seq_lens, chunk_seq_lens)


class TestAscendMLADecodeMetadata(TestBase):
    def test_ascend_mla_decode_metadata_default(self):
        input_positions = torch.tensor([[1,2,3,4],[1,2,3,4]])
        block_table = torch.tensor([[0,3,2,1],[0,2,1,3]])
        seq_lens = torch.tensor([[2],[3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None 

        metadata = AscendMLADecodeMetadata(
            input_positions,
            block_table,
            seq_lens,
            max_seq_lens,
            seq_lens_list,
            attn_mask)
        
        self.assertEqual(metadata.input_positions, input_positions)
        self.assertEqual(metadata.block_table, block_table)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)

class TestAscendMLAMetadata(TestBase):
    def test_ascend_mla_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        query_start_loc = torch.tensor([1, 2, 3, 4])
        seq_lens = [30, 50]
        block_tables = torch.randint(0,100,(100,4))

        num_decodes = 4
        num_decode_tokens = 8
        num_prefills = 8

        num_input_tokens = 2

        max_num_tokens_across_dp = 2
        with_prefill_across_dp = False
        query_lens = None
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        decode = None
        prefill = None

        metadata = AscendMLAMetadata(
            num_actual_tokens,
            slot_mapping,
            query_start_loc,
            seq_lens,
            block_tables,
            num_decodes,
            num_decode_tokens,
            num_prefills,
            num_input_tokens,
            max_num_tokens_across_dp,
            with_prefill_across_dp,
            query_lens,
            head_dim,
            attn_mask,
            attn_state,
            decode,
            prefill
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertEqual(metadata.slot_mapping, slot_mapping)
        self.assertEqual(metadata.query_start_loc, query_start_loc)
        self.assertEqual(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.block_tables, block_tables)
        self.assertEqual(metadata.num_decodes, num_decodes)
        self.assertEqual(metadata.num_decode_tokens, num_decode_tokens)
        self.assertEqual(metadata.num_prefills, num_prefills)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertEqual(metadata.max_num_tokens_across_dp, max_num_tokens_across_dp)
        self.assertEqual(metadata.with_prefill_across_dp, with_prefill_across_dp)
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
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

            self.assertEqual(builder.runner, runner)
            self.assertEqual(builder.scheduler_config, runner.scheduler_config)
            self.assertEqual(builder.model_config, runner.model_config)
            self.assertEqual(builder.block_size, runner.block_size)
            self.assertEqual(builder.chunked_prefill_enabled, runner.chunked_prefill_enabled)
            self.assertEqual(builder.torchair_graph_enabled, True)


    def test_reorder_batch_with_torchair_graph(self):
        ascend_config = MagicMock()
        runner = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True # 在使能torchair的情况下进行测试
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

        # 模拟 input_batch
        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        # 模拟 scheduler_output
        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 2, 1: 1, 2: 3, 3: 1}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [1],  # 2 - 1 = 1 → decode
            1: [],   # 1 - 0 = 1 → decode
            2: [1, 1],  # 3 - 2 = 1 → decode
            3: []    # 1 - 0 = 1 → decode
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        # 所有请求都是 decode，不需要重排
        self.assertFalse(modified)
        self.assertEqual(builder._num_decodes, 4)
        self.assertEqual(builder._num_prefills, 0)
        self.assertEqual(builder._num_decode_tokens, 7)
        self.assertEqual(builder._num_prefill_tokens, 0)
        input_batch.swap_states.assert_not_called()

    def test_reorder_batch_without_torchair_graph(self):
        ascend_config = MagicMock()
        runner = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = False # 在使能torchair的情况下进行测试
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(runner)

        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 1, 1: 3, 2: 1, 3: 2}
        scheduler_output.scheduled_spec_decode_tokens = {
            0: [],   # 1 → decode
            1: [1],  # 3 → prefill
            2: [],   # 1 → decode
            3: []    # 2 → prefill
        }

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)
        
        self.assertTrue(modified)
        self.assertEqual(builder._num_decodes, 2)
        self.assertEqual(builder._num_prefills, 2)
        self.assertEqual(builder._num_decode_tokens, 2)
        self.assertEqual(builder._num_prefill_tokens, 5)
        input_batch.swap_states.assert_called_once_with(1, 2)

    def test_get_graph_runner_block_tables_normal(self):
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros((8, 64), dtype=torch.int32) # 用于构建builder的参数
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, 10)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertEqual(result[:,:10], block_tables)

    def test_get_graph_runner_block_tables_truncated(self):
        runner = MagicMock()
        runner.graph_block_tables = torch.zeros((8, 4), dtype=torch.int32) # 用于构建builder的参数
        builder = AscendMLAMetadataBuilder(runner=runner)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, 10)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result, block_tables[:, :4])
    
    def test_get_graph_runner_block_tables_from_numpy(self):
        runner = MagicMock()
        runner.graph_block_tables = np.zeros((8, 64), dtype=np.int32)
        builder = AscendMLAMetadataBuilder(runner=runner)

        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)

        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertEqual(result[:, :10], block_tables)

    
    def test_build_dummy(self):
        # 需要首先构建一个builder 构建builder需要一个runner
        runner = MagicMock()
        runner.model_config = MagicMock()
        runner.device = "cpu"
        runner.graph_block_tables = torch.zeros((8, 64), dtype=torch.int32)
        runner.model_config.get_head_size.return_value = 64

        runner.attn_mask = torch.zeros((1, 1), dtype=torch.bool)
        runner.spec_attn_mask = torch.zeros((1, 1), dtype=torch.bool)

        builder = AscendMLAMetadataBuilder(runner=runner, metadata_cls=AscendMLAMetadata) # 构建一个builder

        with patch.object(builder, "_get_graph_runner_block_tables", side_effect=lambda x, y: y):
            metadata = builder.build_dummy(3, 3)
        
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

    