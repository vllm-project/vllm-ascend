from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.torchair.torchair_sfa import (
    AscendSFATorchairBackend, AscendSFATorchairDecodeMetadata,
    AscendSFATorchairImpl, AscendSFATorchairMetadata,
    AscendSFATorchairMetadataBuilder, AscendSFATorchairPrefillMetadata)


class TestAscendSFATorchairBackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFATorchairBackend.get_name(),
                         "ASCEND_SFA_TORCHAIR")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFATorchairBackend.get_builder_cls(),
                         AscendSFATorchairMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFATorchairBackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFATorchairBackend.get_impl_cls()
        self.assertEqual(result, AscendSFATorchairImpl)


class TestAscendSFATorchairPrefillMetadata(TestBase):

    def test_ascend_sfa_prefill_metadata_default(self):
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendSFATorchairPrefillMetadata(
            attn_mask=attn_mask,
            query_lens=query_lens,
            seq_lens=seq_lens,
            context_lens=context_lens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            max_query_len=max_query_len,
            sin=None,
            cos=None,
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

    def test_ascend_sfa_prefill_metadata_with_chunked_context(self):
        cu_seq_lens = torch.tensor([0, 2, 4])
        starts = torch.tensor([0, 2])
        seq_tot = [2, 2]
        max_seq_lens = [2, 2]
        workspace = torch.randn(2, 4)
        chunk_seq_lens = torch.tensor([2, 2])

        chunked_context = AscendSFATorchairPrefillMetadata.TorchairChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens,
            starts=starts,
            seq_tot=seq_tot,
            max_seq_lens=max_seq_lens,
            workspace=workspace,
            chunk_seq_lens=chunk_seq_lens)

        metadata = AscendSFATorchairPrefillMetadata(
            attn_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
            query_lens=[1, 2],
            seq_lens=[2, 2],
            context_lens=torch.tensor([1, 2]),
            input_positions=torch.tensor([0, 1, 0, 1]),
            query_start_loc=torch.tensor([0, 1, 3]),
            block_table=torch.tensor([[0, 1], [2, 3]]),
            max_query_len=2,
            max_seq_lens=2,
            sin=None,
            cos=None,
            chunked_context=chunked_context)

        self.assertIsNotNone(metadata.chunked_context)
        self.assertIs(metadata.chunked_context.cu_seq_lens, cu_seq_lens)
        self.assertIs(metadata.chunked_context.starts, starts)
        self.assertEqual(metadata.chunked_context.seq_tot, seq_tot)
        self.assertEqual(metadata.chunked_context.max_seq_lens, max_seq_lens)
        self.assertIs(metadata.chunked_context.workspace, workspace)
        self.assertIs(metadata.chunked_context.chunk_seq_lens, chunk_seq_lens)


class TestAscendSFATorchairDecodeMetadata(TestBase):

    def test_ascend_sfa_decode_metadata_default(self):
        input_positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        block_table = torch.tensor([[0, 3, 2, 1], [0, 2, 1, 3]])
        seq_lens = torch.tensor([[2], [3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None

        metadata = AscendSFATorchairDecodeMetadata(input_positions,
                                                   block_table, seq_lens,
                                                   max_seq_lens, seq_lens_list,
                                                   None, None, attn_mask)

        self.assertIs(metadata.input_positions, input_positions)
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)


class TestAscendSFATorchairMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
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

        metadata = AscendSFATorchairMetadata(
            num_actual_tokens, slot_mapping, query_start_loc, seq_lens,
            block_tables, num_decodes, num_decode_tokens, num_prefills,
            num_input_tokens, query_lens, head_dim, attn_mask, attn_state,
            decode, prefill)

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


class TestAscendSFATorchairMetadataBuilder(TestBase):

    def test_ascend_sfa_metadata_builder_default(self):
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_model_config.get_head_size.return_value = 64
        mock_model_config.dtype = torch.float16

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = MagicMock(block_size=16)
        mock_vllm_config.scheduler_config = MagicMock(
            max_num_seqs=4, enable_chunked_prefill=False)
        mock_vllm_config.speculative_config = None

        mock_device = torch.device('cpu')
        ascend_config = MagicMock()
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True
        with patch("vllm_ascend.torchair.torchair_sfa.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendSFATorchairMetadataBuilder(None, None,
                                                       mock_vllm_config,
                                                       mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.enable_chunked_prefill)
            self.assertEqual(builder.torchair_graph_enabled, True)
            self.assertEqual(builder.max_blocks, (mock_vllm_config.model_config.max_model_len +
                                                  mock_vllm_config.cache_config.block_size - 1) \
                                                  // mock_vllm_config.cache_config.block_size)

    @patch("vllm_ascend.torchair.torchair_sfa.get_ascend_config")
    def test_reorder_batch_with_torchair_graph(self, ascend_config):
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_model_config.get_head_size.return_value = 64
        mock_model_config.dtype = torch.float16

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = MagicMock(block_size=16)
        mock_vllm_config.scheduler_config = MagicMock(
            max_num_seqs=4, enable_chunked_prefill=False)
        mock_vllm_config.speculative_config = None

        mock_device = torch.device('cpu')
        ascend_config.torchair_graph_config = MagicMock()
        ascend_config.torchair_graph_config.enabled = True

        builder = AscendSFATorchairMetadataBuilder(None, None,
                                                   mock_vllm_config,
                                                   mock_device)

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
        input_batch.swap_states.assert_not_called()

    @patch("vllm_ascend.torchair.torchair_sfa.get_ascend_config")
    def test_get_graph_runner_block_tables_normal(self, mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_model_config.get_head_size.return_value = 64
        mock_model_config.dtype = torch.float16

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = MagicMock(block_size=16)
        mock_vllm_config.scheduler_config = MagicMock(
            max_num_seqs=4, enable_chunked_prefill=False)
        mock_vllm_config.speculative_config = None
        mock_device = torch.device('cpu')

        builder = AscendSFATorchairMetadataBuilder(None, None,
                                                   mock_vllm_config,
                                                   mock_device)
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertTrue(torch.equal(result[:, :10], block_tables))

    @patch("vllm_ascend.torchair.torchair_sfa.get_ascend_config")
    def test_ge_graph_runner_block_tables_truncated(self, mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_model_config.get_head_size.return_value = 64
        mock_model_config.dtype = torch.float16

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = MagicMock(block_size=16)
        mock_vllm_config.scheduler_config = MagicMock(
            max_num_seqs=4, enable_chunked_prefill=False)
        mock_vllm_config.speculative_config = None

        mock_device = torch.device('cpu')

        builder = AscendSFATorchairMetadataBuilder(None, None,
                                                   mock_vllm_config,
                                                   mock_device)

        builder.max_blocks = 4
        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 4)
        self.assertTrue(torch.equal(result, block_tables[:, :4]))

    @patch("vllm_ascend.torchair.torchair_sfa.get_ascend_config")
    def test_get_graph_runner_block_tables_from_numpy(self,
                                                      mock_ascend_config):
        ascend_config = MagicMock()
        mock_ascend_config.return_value = ascend_config
        ascend_config.torchair_graph_config.enabled = False
        mock_model_config = MagicMock()
        mock_model_config.max_model_len = 1024
        mock_model_config.get_head_size.return_value = 64
        mock_model_config.dtype = torch.float16

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config = mock_model_config
        mock_vllm_config.cache_config = MagicMock(block_size=16)
        mock_vllm_config.scheduler_config = MagicMock(
            max_num_seqs=4, enable_chunked_prefill=False)
        mock_vllm_config.speculative_config = None

        mock_device = torch.device('cpu')
        builder = AscendSFATorchairMetadataBuilder(None, None,
                                                   mock_vllm_config,
                                                   mock_device)

        block_tables = torch.randint(0, 100, (3, 10), dtype=torch.int32)

        result = builder._get_graph_runner_block_tables(3, block_tables)

        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.shape[1], 64)
        self.assertTrue(torch.equal(result[:, :10], block_tables))
