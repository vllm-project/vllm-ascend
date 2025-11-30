from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (AscendMLABackend,
                                          AscendMLADecodeMetadata,
                                          AscendMLAImpl, AscendMLAMetadata,
                                          AscendMLAMetadataBuilder,
                                          AscendMLAPrefillMetadata)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


class TestAscendMLABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendMLABackend.get_name(), "ASCEND_MLA")

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
            chunk_seq_lens=chunk_seq_lens,
            chunk_seq_lens_npu=chunk_seq_lens)

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
        self.assertIs(metadata.chunked_context.chunk_seq_lens_npu,
                      chunk_seq_lens)


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
        num_actual_tokens_pcp_padded = 100
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

        metadata = AscendMLAMetadata(
            num_actual_tokens_pcp_padded, num_actual_tokens, slot_mapping,
            query_start_loc, seq_lens, block_tables, num_decodes,
            num_decode_tokens, num_prefills, num_input_tokens, query_lens,
            head_dim, attn_mask, attn_state, decode, prefill)

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

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_ascend_mla_metadata_builder_default(self, mock_get_dcp_size,
                                                 mock_dcp, mock_get_dcp_group):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_vllm_config.speculative_config = None

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                               mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.chunked_prefill_enabled)

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_ascend_mla_metadata_builder_spec_decode(self, mock_get_dcp_size,
                                                     mock_dcp,
                                                     mock_get_dcp_group):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_spec_config = MagicMock()
        mock_spec_config.num_speculative_tokens = 3
        mock_vllm_config.speculative_config = mock_spec_config

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                               mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.chunked_prefill_enabled)

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_ascend_mla_metadata_builder_build_full_graph(
            self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_spec_config = MagicMock()
        mock_spec_config.num_speculative_tokens = 1
        mock_spec_config.disable_padded_drafter_batch = True
        mock_vllm_config.speculative_config = mock_spec_config

        builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                           mock_device)
        common_metadata = MagicMock()
        model = MagicMock()
        common_metadata.graph_pad_size = 8
        common_metadata.num_reqs = 4
        common_metadata.num_actual_tokens = 5
        common_metadata.max_query_len = 5
        common_metadata.seq_lens_cpu = torch.Tensor([9, 10, 8, 8]).int()
        common_metadata.query_start_loc = torch.Tensor([0, 1, 2, 4, 5]).int()
        common_metadata.query_start_loc_cpu = torch.Tensor([0, 1, 2, 4,
                                                            5]).int()
        common_metadata.positions = torch.Tensor([1, 2, 3, 4, 5, 6]).int()
        block_table = torch.Tensor([[1, 0], [2, 0], [3, 0], [4, 0]]).int()
        common_metadata.block_table_tensor = block_table
        common_metadata.prefill_context_parallel_metadata = None
        metadata = builder.build(0, common_metadata, model)

        self.assertEqual(metadata.decode.actual_seq_lengths_q,
                         [1, 2, 4, 5, 6, 6, 7, 8])
        self.assertEqual(metadata.decode.block_table.shape[0], 8)

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_reorder_batch(self, mock_get_dcp_size, mock_dcp,
                           mock_get_dcp_group):
        ascend_config = MagicMock()

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_vllm_config.speculative_config = None

        with patch("vllm_ascend.attention.mla_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                               mock_device)
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

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_pad_actual_seq_lens_q_mtp_disable_pad(self, mock_get_dcp_size,
                                                   mock_dcp,
                                                   mock_get_dcp_group):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'
        mock_vllm_config.speculative_config = None

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                           mock_device)
        input_seq_lens = [1, 2, 4, 5]
        expect_output = [1, 2, 4, 5, 6, 6, 7, 8]
        num_reqs = 4
        num_reqs_pad_size = 4
        output_seq_lens = builder.pad_actual_seq_len_q_mtp_disable_pad(
            num_reqs_pad_size, num_reqs, input_seq_lens)
        self.assertEqual(output_seq_lens, expect_output)

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def test_pad_actual_seq_lens_q_mtp_enable_pad(self, mock_get_dcp_size,
                                                  mock_dcp,
                                                  mock_get_dcp_group):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.decode_max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'
        mock_vllm_config.speculative_config = None

        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group
        common_metadata = MagicMock()
        common_metadata.actual_seq_lengths_q = [2, 4, 6, 8]

        builder = AscendMLAMetadataBuilder(None, None, mock_vllm_config,
                                           mock_device)
        input_seq_lens = [2, 4, 6]
        expect_output = [2, 4, 6, 8]
        num_reqs = 3
        num_reqs_pad_size = 1
        output_seq_lens = builder.pad_actual_seq_len_q_mtp_enable_pad(
            num_reqs_pad_size, num_reqs, input_seq_lens, common_metadata)
        self.assertEqual(output_seq_lens, expect_output)


class TestAscendMLAMetadataBuilderBuild(TestBase):

    def setUp(self):
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.model_config = ModelConfig(max_model_len=2048)
        self.mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 32
        self.mock_vllm_config.cache_config = CacheConfig(block_size=32)
        self.mock_vllm_config.scheduler_config = SchedulerConfig(
            max_num_seqs=8, chunked_prefill_enabled=True)
        self.mock_vllm_config.speculative_config = None
        self.mock_device = torch.device("cpu")

        self.kv_cache_spec = MagicMock()
        self.kv_cache_spec.num_layers = 32
        self.kv_cache_spec.head_size = 128
        self.kv_cache_spec.num_heads = 32

    @patch(
        "vllm_ascend.attention.mla_v1.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    @patch("vllm_ascend.attention.mla_v1.torch.zeros", wraps=torch.zeros)
    @patch("torch.Tensor.npu", new=lambda self: self)
    @patch("torch.npu.is_available")
    def test_build_prefix_no_cache_metadata(self, mock_npu_available,
                                            mock_zeros, mock_get_ascend_config,
                                            mock_dcp_world_size):
        mock_npu_available.return_value = False
        mock_dcp_world_size.return_value = 1

        def zeros_override(*args, **kwargs):
            kwargs.pop('pin_memory', None)
            return mock_zeros._mock_wraps(*args, **kwargs)

        mock_zeros.side_effect = zeros_override
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3, 7]),
            query_start_loc_cpu=torch.tensor([0, 3, 7]),
            seq_lens_cpu=torch.tensor([5, 6]),
            num_reqs=2,
            num_actual_tokens=10,
            max_query_len=5,
            decode_token_per_req=torch.tensor([1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((10, 10)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        base_inputs = {
            "num_actual_tokens": 10,
            "slot_mapping": torch.tensor(range(10)),
            "query_start_loc": torch.tensor([0, 3, 7]),
            "seq_lens": torch.tensor([5, 6]),
            "block_tables": torch.zeros((10, 10)),
            "num_prefills": 2,
        }

        builder = AscendMLAMetadataBuilder(kv_cache_spec=self.kv_cache_spec,
                                           layer_names=["layer_0", "layer_1"],
                                           vllm_config=self.mock_vllm_config,
                                           device=self.mock_device)

        mock_model = MagicMock()
        metadata = builder.build(1, common_attn_metadata, mock_model)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens,
                         base_inputs["num_actual_tokens"])
        self.assertTrue(
            torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch(
        "vllm_ascend.attention.mla_v1.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    @patch("vllm_ascend.attention.mla_v1.torch.zeros", wraps=torch.zeros)
    @patch("torch.Tensor.npu", new=lambda self: self)
    @patch("torch.npu.is_available")
    def test_build_chunked_prefix_metadata(self, mock_npu_available,
                                           mock_zeros, mock_get_ascend_config,
                                           mock_dcp_world_size):
        mock_npu_available.return_value = False
        mock_dcp_world_size.return_value = 1

        def zeros_override(*args, **kwargs):
            kwargs.pop('pin_memory', None)
            return mock_zeros._mock_wraps(*args, **kwargs)

        mock_zeros.side_effect = zeros_override

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=15,
            max_query_len=6,
            decode_token_per_req=torch.tensor([1, 1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((15, 15)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        base_inputs = {
            "num_actual_tokens": 15,
            "slot_mapping": torch.tensor(range(15)),
            "query_start_loc": torch.tensor([0, 2, 5, 9]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "block_tables": torch.zeros((10, 10)),
            "num_prefills": 3,
        }

        builder = AscendMLAMetadataBuilder(kv_cache_spec=self.kv_cache_spec,
                                           layer_names=["layer_0", "layer_1"],
                                           vllm_config=self.mock_vllm_config,
                                           device=self.mock_device)

        mock_model = MagicMock()
        metadata = builder.build(1, common_attn_metadata, mock_model)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens,
                         base_inputs["num_actual_tokens"])
        self.assertTrue(
            torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch(
        "vllm_ascend.attention.mla_v1.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_build_decode_only_metadata(self, mock_get_ascend_config,
                                        mock_dcp_world_size):
        mock_dcp_world_size.return_value = 1

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2, 3]),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(3)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            decode_token_per_req=torch.tensor([1, 1, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((3, 3)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.DecodeOnly,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        base_inputs = {
            "num_actual_tokens": 3,
            "slot_mapping": torch.tensor(range(3)),
            "query_start_loc": torch.tensor([0, 1, 2, 3]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "num_decodes": 3,
        }

        builder = AscendMLAMetadataBuilder(kv_cache_spec=self.kv_cache_spec,
                                           layer_names=["layer_0", "layer_1"],
                                           vllm_config=self.mock_vllm_config,
                                           device=self.mock_device)

        mock_model = MagicMock()
        metadata = builder.build(1, common_attn_metadata, mock_model)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens,
                         base_inputs["num_actual_tokens"])
        self.assertTrue(
            torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch(
        "vllm_ascend.attention.mla_v1.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_build_for_graph_capture_decode_only(self, mock_get_ascend_config,
                                                 mock_dcp_world_size):
        mock_dcp_world_size.return_value = 1

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2, 3]),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(3)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            decode_token_per_req=torch.tensor([1, 1, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((3, 3)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.DecodeOnly,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        base_inputs = {
            "num_actual_tokens": 3,
            "slot_mapping": torch.tensor(range(3)),
            "query_start_loc": torch.tensor([0, 1, 2, 3]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "num_decodes": 3,
        }

        builder = AscendMLAMetadataBuilder(kv_cache_spec=self.kv_cache_spec,
                                           layer_names=["layer_0", "layer_1"],
                                           vllm_config=self.mock_vllm_config,
                                           device=self.mock_device)

        mock_model = MagicMock()
        metadata = builder.build_for_graph_capture(
            common_attn_metadata, AscendAttentionState.DecodeOnly, mock_model)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens,
                         base_inputs["num_actual_tokens"])
        self.assertTrue(
            torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch(
        "vllm_ascend.attention.mla_v1.get_decode_context_model_parallel_world_size"
    )
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def test_build_for_graph_capture_prefill(self, mock_get_ascend_config,
                                             mock_dcp_world_size):
        mock_dcp_world_size.return_value = 1

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3, 7]),
            query_start_loc_cpu=torch.tensor([0, 3, 7]),
            seq_lens_cpu=torch.tensor([5, 6]),
            num_reqs=2,
            num_actual_tokens=10,
            max_query_len=5,
            decode_token_per_req=torch.tensor([1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((10, 10)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        builder = AscendMLAMetadataBuilder(kv_cache_spec=self.kv_cache_spec,
                                           layer_names=["layer_0", "layer_1"],
                                           vllm_config=self.mock_vllm_config,
                                           device=self.mock_device)

        mock_model = MagicMock()

        with self.assertRaises(NotImplementedError) as ctx:
            builder.build_for_graph_capture(
                common_attn_metadata, AscendAttentionState.PrefillNoCache,
                mock_model)
        self.assertIn(
            "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state",
            str(ctx.exception))


class TestAscendMLAImpl(TestBase):

    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
           return_value=2)
    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.mla_v1.get_ascend_config")
    def setUp(self, ascend_config, get_current_vllm_config, mock_get_tp_size,
              mock_tp, mock_get_dcp_size, mock_dcp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()
        mock_dcp.world_size = 1
        mock_dcp.rank_in_group = MagicMock()
        mock_dcp.device_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        vllm_config.model_config = model_config
        get_current_vllm_config.return_value = vllm_config

        num_heads = 256
        head_size = 1024
        scale = 0.1
        num_kv_heads = 8
        kv_cache_dtype = "auto"

        kv_a_layernorm = MagicMock()
        kv_a_layernorm.weight = torch.randn(96)
        kv_a_layernorm.variance_epsilon = 1e-6
        kwargs = {
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "qk_head_dim": 96,
            "v_head_dim": 128,
            "q_lora_rank": 64,
            "q_proj": MagicMock(),
            "q_b_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "fused_qkv_a_proj": MagicMock(),
            "kv_a_layernorm": kv_a_layernorm,
            "rotary_emb": MagicMock(),
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
        self.assertEqual(self.impl.kv_lora_rank, 32)
        self.assertEqual(self.impl.qk_nope_head_dim, 64)
        self.assertEqual(self.impl.qk_rope_head_dim, 32)
        self.assertEqual(self.impl.qk_head_dim, 96)
        self.assertEqual(self.impl.v_head_dim, 128)
        self.assertIsNotNone(self.impl.q_proj)
        self.assertIsNotNone(self.impl.kv_b_proj)
        self.assertIsNotNone(self.impl.o_proj)
        self.assertIsNotNone(self.impl.kv_a_proj_with_mqa)
        self.assertIsNotNone(self.impl.kv_a_layernorm)
        self.assertEqual(self.impl.num_queries_per_kv, 32)
        self.assertEqual(self.impl.tp_size, 2)

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

    @patch('torch_npu.npu_format_cast')
    def test_process_weights_after_loading(self, mock_format_cast):
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
        mock_format_cast.return_value = layer.weight
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
        chunk_ctx.chunk_seq_lens_npu = [torch.tensor([8])]
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

    @patch('vllm_ascend.attention.mla_v1.get_forward_context')
    @patch("vllm_ascend.attention.mla_v1.AscendMLAImpl._v_up_proj")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode_without_graph(self,
                                          mock_npu_fused_infer_attention_score,
                                          mock_up_proj,
                                          mock_get_forward_context):
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
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe,
                                           block_size, metadata)
        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.num_heads)
        self.assertEqual(result.shape[2], self.impl.v_head_dim)
        mock_up_proj.assert_called_once()
        mock_npu_fused_infer_attention_score.assert_called_once()

    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("vllm_ascend.attention.mla_v1.maybe_npu_prefetch")
    def test_mla_preprocess(self, magic_npu_fetch,
                            mock_maybe_all_gather_and_maybe_unpad):
        magic_npu_fetch.return_value = MagicMock()
        mock_maybe_all_gather_and_maybe_unpad.side_effect = lambda x, label: x
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

        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.num_heads,
            self.impl.qk_rope_head_dim)
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_rope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.return_value = [
            torch.randn(
                num_prefill_tokens, self.impl.num_heads,
                self.impl.qk_rope_head_dim + self.impl.kv_lora_rank +
                self.impl.q_lora_rank)
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
            "mock_layer",
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv=False)

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

    @patch('vllm_ascend.attention.mla_v1.get_forward_context')
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode(self, mock_npu_fused_infer_attention_score,
                            mock_get_forward_context):
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
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS,
                                           attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)
