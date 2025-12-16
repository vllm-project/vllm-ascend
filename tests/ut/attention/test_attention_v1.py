from typing import List
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_cp import AscendAttentionCPImpl
from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata,
                                                AscendMetadataForPrefill)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.utils import AscendDeviceType


class TestAscendAttentionBackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(),
                         AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(),
                         AscendAttentionMetadataBuilder)

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    def test_get_kv_cache_shape_not_310p(self, mock_soc_version):
        result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_get_bsh_kv_cache_shape(self):
        result = AscendAttentionBackend.get_bsh_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30 * 40))

    def test_swap_blocks(self):
        src_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        dst_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dst = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache,
                                           src_to_dst)
        self.assertTrue(torch.all(dst_kv_cache[0][1] == src_kv_cache[0][0]))
        self.assertTrue(torch.all(dst_kv_cache[1][3] == src_kv_cache[1][2]))

    def test_copy_blocks(self):
        kv_caches = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dists = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.copy_blocks(kv_caches, src_to_dists)
        self.assertTrue(torch.all(kv_caches[0][1] == kv_caches[0][0]))
        self.assertTrue(torch.all(kv_caches[1][3] == kv_caches[1][2]))


class TestAscendAttentionMetadataBuilder(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.decode_max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        self.mock_device = 'cpu:0'
        torch.Tensor.pin_memory = lambda x: x  # noqa
        self.builder = AscendAttentionMetadataBuilder(None, None,
                                                      self.mock_vllm_config,
                                                      self.mock_device)

    def test_reorder_batch(self):
        mock_input_batch = MagicMock()
        mock_scheduler_output = MagicMock()

        result = self.builder.reorder_batch(mock_input_batch,
                                            mock_scheduler_output)

        self.assertFalse(result)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    def test_build_non_310p(self, mock_soc_version, mock_ascend_metadata):
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
        mock_model = MagicMock()

        self.builder.build(1, common_attn_metadata, mock_model)


class TestAscendAttentionBackendImpl(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"

        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"

        self.layer_no_quant = MagicMock(
            spec=['layer_name', '_k_scale_float', '_v_scale_float'])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0

        self.impl = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_192 = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl_error = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=192,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None)

        self.impl_swa = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=1024,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

    def test_forward_no_attn_metadata(self):
        """Test forward pass when attn_metadata is None"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        layer = self.layer_no_quant
        output = torch.empty_like(query)

        output = self.impl.forward(layer, query, key, value, kv_cache, None,
                                   output)

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    def test_forward_prefill(self, mock_get_forward_context,
                             mock_npu_fused_infer_attention_score,
                             mock_npu_reshape_and_cache):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.actual_seq_lengths_q = [10]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.num_decode_tokens = 0
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=False)
        mock_npu_fused_infer_attention_score.return_value = (torch.ones(
            10, 8, 64), torch.ones(10, 8, 64))
        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    def test_forward_decode_only(self, mock_get_forward_context,
                                 mock_npu_reshape_and_cache,
                                 mock_paged_attention):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_paged_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa(self, mock_npu_reshape_and_cache,
                                     mock_fused_infer_attention_score,
                                     mock_get_forward_context):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty(10, 8, 64)

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10] * 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8,
                                                                    64), 1)
        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_decode_only_swa_seq_len_mismatch(
            self, mock_npu_reshape_and_cache, mock_fused_infer_attention_score,
            mock_paged_attention, mock_get_forward_context):
        """Test forward pass in DecodeOnly state when seq)len_mismatch"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])  # len == 1 != query.size(0)==10
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        metadata.num_decodes = 10
        metadata.num_prefills = 0

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64),
                                                         torch.ones(10, 8, 64))

        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)

        mock_paged_attention.assert_called_once()
        mock_fused_infer_attention_score.assert_not_called()

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_raise_error(self, mock_paged_attention):
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        layer = self.layer_no_quant

        with self.assertRaises(NotImplementedError):
            self.impl_error.forward(layer, query, key, value, kv_cache,
                                    metadata, output)


class TestUpdateNpuAttnOutLse(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"

        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"

        self.layer_no_quant = MagicMock(
            spec=['layer_name', '_k_scale_float', '_v_scale_float'])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0

        self.impl = AscendAttentionCPImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

    @patch('torch_npu.npu_attention_update')
    def test_npu_attn_out_lse_update(self, mock_npu_attention_update):
        # Mock input data
        attn_lse_mask = torch.randn(8, 128, 1)
        attn_lse_nomask = torch.randn(8, 128, 1)
        attn_out_mask = torch.randn(8, 128, 128)
        attn_out_nomask = torch.randn(8, 128, 128)

        # Mock output
        mock_npu_attention_update.return_value = (torch.randn(8 * 128,
                                                              128), None)

        # Call the method under test
        output = self.impl._npu_attn_out_lse_update(attn_lse_mask,
                                                    attn_lse_nomask,
                                                    attn_out_mask,
                                                    attn_out_nomask)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (8, 128, 128))

        mock_npu_attention_update.assert_called_once()

    def test_update_out_and_lse(self):
        # Mock input data
        out_list = torch.randn(3, 2, 4,
                               8)  # [N, batch_size, num_heads, head_size]
        lse_list = torch.randn(3, 2, 4, 1)  # [N, batch_size, num_heads, 1]

        # Call the method under test
        out_final, lse_final = self.impl._update_out_and_lse(
            out_list, lse_list)

        # Assert the method call
        self.assertEqual(out_final.shape,
                         (2, 4, 8))  # [batch_size, num_heads, head_size]
        self.assertEqual(lse_final.shape,
                         (2, 4, 1))  # [batch_size, num_heads, 1]

        self.assertIsInstance(out_final, torch.Tensor)
        self.assertIsInstance(lse_final, torch.Tensor)

    @patch('torch.cat')
    @patch('torch.distributed.all_to_all_single')
    @patch('torch.distributed.all_gather')
    @patch('torch.stack')
    @patch('torch.split')
    def test_update_chunk_attn_out_lse_dcp_pcp_both_greater_than_1(
            self, mock_split, mock_stack, mock_all_gather,
            mock_all_to_all_single, mock_cat):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)
        self.impl.dcp_size = 2
        self.impl.pcp_size = 3

        # Mock output
        mock_cat.return_value = torch.randn(2, 4, 9)
        mock_all_to_all_single.return_value = torch.randn(4, 9, 2)
        mock_all_gather.return_value = [(2, 4, 9), (2, 4, 9), (2, 4, 9)]
        mock_stack.return_value = torch.randn(6, 2, 2, 9)
        mock_split.return_value = (torch.randn(6, 2, 2,
                                               8), torch.randn(6, 2, 2, 1))

        # Call the method under test
        output, lse = self.impl._update_chunk_attn_out_lse(
            prefix_chunk_output, prefix_chunk_lse)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 8))
        self.assertEqual(lse.shape, (2, 2, 1))

        self.assertEqual(mock_cat.call_count, 1)
        mock_all_to_all_single.assert_called_once()
        mock_stack.assert_called_once()
        mock_split.assert_called_once()
        self.assertEqual(mock_all_gather.call_count, 1)

    @patch('torch.cat')
    @patch('torch.chunk')
    @patch('torch.stack')
    @patch('torch.split')
    @patch('torch.distributed.all_to_all_single')
    @patch('torch.distributed.all_gather')
    def test_update_chunk_attn_out_lse_dcp_greater_than_1_only(
            self, mock_all_gather, mock_all_to_all_single, mock_split,
            mock_stack, mock_chunk, mock_cat):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)

        self.impl.dcp_size = 2
        self.impl.pcp_size = 1
        self.impl.head_size = 8

        # Mock output
        mock_cat.return_value = torch.randn(2, 4, 9)
        mock_all_to_all_single.return_value = torch.randn(2, 4, 9)
        mock_chunk.return_value = [torch.randn(2, 2, 9), torch.randn(2, 2, 9)]
        mock_stack.return_value = torch.randn(2, 2, 2, 9)
        mock_split.return_value = [
            torch.randn(2, 2, 2, 8),
            torch.randn(2, 2, 2, 1)
        ]

        # Call the method under test
        output, lse = self.impl._update_chunk_attn_out_lse(
            prefix_chunk_output, prefix_chunk_lse)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 8))
        self.assertEqual(lse.shape, (2, 2, 1))

        self.assertEqual(mock_cat.call_count, 1)
        mock_all_to_all_single.assert_called_once()
        mock_chunk.assert_called_once()
        mock_stack.assert_called_once()
        mock_split.assert_called_once()
        mock_all_gather.assert_not_called()

    @patch('torch.cat')
    @patch('torch.stack')
    @patch('torch.split')
    @patch('torch.distributed.all_to_all_single')
    @patch('torch.distributed.all_gather')
    @patch(
        'vllm_ascend.attention.attention_cp.AscendAttentionCPImpl._update_out_and_lse'
    )
    def test_update_chunk_attn_out_lse_pcp_greater_than_1_only(
            self, mock_update_out_and_lse, mock_all_gather,
            mock_all_to_all_single, mock_split, mock_stack, mock_cat):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)

        self.impl.dcp_size = 1
        self.impl.pcp_size = 2
        self.impl.head_size = 8

        # Mock output
        mock_cat.return_value = torch.randn(2, 4, 9)
        mock_all_gather.return_value = [(2, 4, 9), (2, 4, 9)]
        mock_stack.return_value = torch.randn(2, 2, 4, 9)
        mock_split.return_value = [
            torch.randn(2, 2, 4, 8),
            torch.randn(2, 2, 4, 1)
        ]
        mock_update_out_and_lse.return_value = torch.randn(2, 4,
                                                           8), torch.randn(
                                                               2, 4, 1)
        # Call the method under test
        output, lse = self.impl._update_chunk_attn_out_lse(
            prefix_chunk_output, prefix_chunk_lse)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(lse.shape, (2, 4, 1))
        self.impl._update_out_and_lse.assert_called_once()

        self.assertEqual(mock_cat.call_count, 1)
        mock_all_to_all_single.assert_not_called()
        mock_stack.assert_called_once()
        mock_split.assert_called_once()
        mock_all_gather.assert_called_once()


class TestAttentionWithNomaskAndMask(TestBase):

    @patch('vllm.distributed.parallel_state.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 1
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 1
        pcp_group.device_group = MagicMock()
        mock_get_pcp_group.return_value = pcp_group

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0

        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"

        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"

        self.layer_no_quant = MagicMock(
            spec=['layer_name', '_k_scale_float', '_v_scale_float'])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0

        self.impl = AscendAttentionCPImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None)

        self.impl.pcp_size = 1

        self.batch_size = 2

        # sequence length per batch
        self.q_lens_per_batch = [32, 64]
        self.kv_lens_nomask_per_batch = [32, 64]
        self.kv_lens_mask_per_batch = [32, 64]

        # TND layout requires cumulative sum computation.
        self.q_seqlens_cumsum = self._cumsum(self.q_lens_per_batch)  # [32, 96]
        self.kv_seqlens_nomask_cumsum = self._cumsum(
            self.kv_lens_nomask_per_batch)  # [32, 96]
        self.kv_seqlens_mask_cumsum = self._cumsum(
            self.kv_lens_mask_per_batch)  # [32, 96]

        # Compute T value in TND layout
        self.q_total_tokens = self.q_seqlens_cumsum[-1]
        self.kv_total_nomask = self.kv_seqlens_nomask_cumsum[-1]  #
        self.kv_total_mask = self.kv_seqlens_mask_cumsum[-1]

    def _cumsum(self, arr: List[int]) -> List[int]:
        result = []
        total = 0
        for val in arr:
            total += val
            result.append(total)
        return result

    def _build_attn_metadata(self, with_chunked_context=False):
        attn_metadata = AscendMetadata()
        attn_metadata.num_prefills = self.batch_size
        attn_metadata.num_decodes = 0
        attn_metadata.num_actual_tokens = self.q_total_tokens

        prefill_metadata = AscendMetadataForPrefill()
        pcp_metadata = AscendMetadataForPrefill.AscendPCPMetadata()
        pcp_metadata.attn_mask_seqlens = self.kv_seqlens_mask_cumsum
        pcp_metadata.head_attn_nomask_seqlens = self.kv_seqlens_nomask_cumsum
        pcp_metadata.tail_attn_nomask_seqlens = self.kv_seqlens_nomask_cumsum
        prefill_metadata.pcp_metadata = pcp_metadata

        prefill_metadata.actual_seq_lengths_q = torch.tensor(
            self.q_seqlens_cumsum)

        if with_chunked_context:
            chunked_context = AscendMetadataForPrefill.ChunkedContextMetadata(
                actual_chunk_seq_lengths=self.kv_seqlens_mask_cumsum,
                actual_seq_lengths_kv=self.kv_seqlens_mask_cumsum,
                starts=None,
                chunk_seq_mask_filtered_indices=None)
            prefill_metadata.chunked_context = chunked_context
        else:
            prefill_metadata.chunked_context = None

        attn_metadata.prefill = prefill_metadata
        attn_metadata.decode_meta = None
        return attn_metadata

    @patch('torch.ops.npu.npu_fused_infer_attention_score')
    def test_attention_with_nomask_none(self, mock_npu_attention):
        # Mock input data
        q = torch.randn(self.q_total_tokens, self.impl.num_heads,
                        self.impl.head_size)
        q_seqlens = self.q_seqlens_cumsum
        k_nomask = None
        v_nomask = None
        kv_seqlens_nomask = self.kv_seqlens_nomask_cumsum
        k_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)
        v_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)
        kv_seqlens_mask = self.kv_seqlens_mask_cumsum
        mask = torch.randn(self.q_total_tokens, self.kv_total_mask)
        attn_metadata = self._build_attn_metadata(with_chunked_context=False)
        # Mock output
        mock_npu_attention.return_value = torch.randn(96, 8, 64), torch.randn(
            96, 8, 1)

        # Call the method under test
        output, attn_lse = self.impl._attention_with_nomask_and_mask(
            q, q_seqlens, k_nomask, v_nomask, kv_seqlens_nomask, k_mask,
            v_mask, kv_seqlens_mask, mask, attn_metadata)

        # Verify only mask attention was invoked
        mock_npu_attention.assert_called_with(
            q,
            k_mask,
            v_mask,
            num_heads=self.impl.num_heads,
            num_key_value_heads=self.impl.num_kv_heads,
            input_layout="TND",
            atten_mask=mask,
            scale=self.impl.scale,
            sparse_mode=3,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens)
        # Assert the method call
        self.assertEqual(mock_npu_attention.call_count, 1)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(attn_lse, torch.Tensor)
        self.assertEqual(output.shape, (96, 8, 64))
        self.assertEqual(attn_lse.shape, (96, 8, 1))

    @patch('torch.ops.npu.npu_fused_infer_attention_score')
    @patch(
        'vllm_ascend.attention.attention_cp.AscendAttentionCPImpl._update_out_and_lse'
    )
    def test_attention_with_nomask_and_mask_chunk(
            self, mock_update_out_and_lse,
            mock_npu_fused_infer_attention_score):
        # Mock input data
        q = torch.randn(self.q_total_tokens, self.impl.num_heads,
                        self.impl.head_size)
        k_nomask = torch.randn(self.kv_total_nomask, self.impl.num_kv_heads,
                               self.impl.head_size)
        v_nomask = torch.randn(self.kv_total_nomask, self.impl.num_kv_heads,
                               self.impl.head_size)
        k_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)
        v_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)

        mask = torch.randn(self.q_total_tokens, self.kv_total_mask)
        attn_metadata = self._build_attn_metadata(with_chunked_context=True)

        # Mock output
        mock_npu_fused_infer_attention_score.return_value = torch.randn(
            self.q_total_tokens, self.impl.num_heads,
            self.impl.head_size), torch.randn(self.q_total_tokens,
                                              self.impl.num_heads, 1)
        mock_update_out_and_lse.return_value = torch.randn(
            self.q_total_tokens, self.impl.num_heads,
            self.impl.head_size), torch.randn(self.q_total_tokens,
                                              self.impl.num_heads, 1)
        # Call the method under test
        output, attn_lse = self.impl._attention_with_nomask_and_mask(
            q=q,
            q_seqlens=self.q_seqlens_cumsum,
            k_nomask=k_nomask,
            v_nomask=v_nomask,
            kv_seqlens_nomask=self.kv_seqlens_nomask_cumsum,
            k_mask=k_mask,
            v_mask=v_mask,
            kv_seqlens_mask=self.kv_seqlens_mask_cumsum,
            mask=mask,
            attn_metadata=attn_metadata)
        # Assert the method call
        self.assertEqual(mock_npu_fused_infer_attention_score.call_count, 2)
        self.assertIsNotNone(output)
        self.assertIsNotNone(attn_lse)

    @patch('torch.ops.npu.npu_fused_infer_attention_score')
    @patch(
        'vllm_ascend.attention.attention_cp.AscendAttentionCPImpl._update_out_and_lse'
    )
    def test_attention_with_nomask_and_mask_nochunk(
            self, mock_update_out_and_lse,
            mock_npu_fused_infer_attention_score):
        self.impl._npu_attn_out_lse_update = MagicMock()
        # Mock input data
        q = torch.randn(self.q_total_tokens, self.impl.num_heads,
                        self.impl.head_size)
        k_nomask = torch.randn(self.kv_total_nomask, self.impl.num_kv_heads,
                               self.impl.head_size)
        v_nomask = torch.randn(self.kv_total_nomask, self.impl.num_kv_heads,
                               self.impl.head_size)
        k_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)
        v_mask = torch.randn(self.kv_total_mask, self.impl.num_kv_heads,
                             self.impl.head_size)
        mask = torch.randn(self.q_total_tokens, self.kv_total_mask)

        attn_metadata = self._build_attn_metadata(with_chunked_context=True)
        attn_metadata.prefill.chunked_context = None

        # Mock output
        mock_npu_fused_infer_attention_score.return_value = torch.randn(
            self.q_total_tokens, self.impl.num_heads,
            self.impl.head_size), torch.randn(self.q_total_tokens,
                                              self.impl.num_heads, 1)
        mock_update_out_and_lse.return_value = torch.randn(
            self.q_total_tokens, self.impl.num_heads, self.impl.head_size)

        # Call the method under test
        output, attn_lse = self.impl._attention_with_nomask_and_mask(
            q=q,
            q_seqlens=self.q_seqlens_cumsum,
            k_nomask=k_nomask,
            v_nomask=v_nomask,
            kv_seqlens_nomask=self.kv_seqlens_nomask_cumsum,
            k_mask=k_mask,
            v_mask=v_mask,
            kv_seqlens_mask=self.kv_seqlens_mask_cumsum,
            mask=mask,
            attn_metadata=attn_metadata)
        # Assert the method call
        self.assertEqual(mock_npu_fused_infer_attention_score.call_count, 2)
        self.assertIsNotNone(output)
        self.assertEqual(attn_lse, None)

    @patch(
        'vllm_ascend.attention.attention_cp.AscendAttentionCPImpl._npu_attn_out_lse_update'
    )
    def test_update_chunk_attn_out_lse_with_current_attn_out_lse(
            self, mock_npu_attn_out_lse_update):
        # Mock input data
        current_attn_output_prefill = torch.randn(32764, 8, 128)
        current_attn_lse_prefill = torch.randn(32764, 8, 1)
        attn_output_full_chunk = torch.randn(65528, 8, 128)
        attn_lse_full_chunk = torch.randn(65528, 8, 1)
        prefill_query = torch.randn(32764, 8, 128)
        # mock attn_metadata
        attn_metadata = self._build_attn_metadata(with_chunked_context=True)
        attn_metadata.prefill.chunked_context.chunk_seq_mask_filtered_indices = torch.arange(
            32764, dtype=torch.int32)
        attn_metadata.prefill.chunked_context.kv_inverse_idx_for_chunk = torch.arange(
            32764, dtype=torch.int32)
        # Mock output
        mock_npu_attn_out_lse_update.return_value = torch.randn(32764, 8, 128)
        # test pcp_size > 1
        self.impl.pcp_size = 2
        self.impl.pcp_rank = 0
        self.impl.dcp_group = None
        self.impl.pcp_group = None
        # Call the method under test
        self.impl._update_chunk_attn_out_lse_with_current_attn_out_lse(
            current_attn_output_prefill, current_attn_lse_prefill,
            attn_output_full_chunk, attn_lse_full_chunk, prefill_query,
            attn_metadata)
        # Assert the method call
        self.impl._npu_attn_out_lse_update.assert_called_once()
        # test pcp_size = 1
        self.impl.pcp_size = 1
        self.impl._update_chunk_attn_out_lse_with_current_attn_out_lse(
            current_attn_output_prefill, current_attn_lse_prefill,
            attn_output_full_chunk, attn_lse_full_chunk, prefill_query,
            attn_metadata)
        self.assertEqual(self.impl._npu_attn_out_lse_update.call_count, 2)
