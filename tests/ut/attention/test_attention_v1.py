from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm.distributed.parallel_state import GroupCoordinator
from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata,
                                                CommonAttentionState)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


class TestAscendAttentionBackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "ASCEND")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(),
                         AscendAttentionBackendImpl)

    def test_get_metadata_cls(self):
        self.assertEqual(AscendAttentionBackend.get_metadata_cls(),
                         AscendMetadata)

    def test_get_state_cls(self):
        self.assertEqual(AscendAttentionBackend.get_state_cls(),
                         CommonAttentionState)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(),
                         AscendAttentionMetadataBuilder)

    @patch('vllm_ascend.attention.attention_v1.is_310p')
    def test_get_kv_cache_shape_310p(self, mock_is_310p):
        mock_is_310p.return_value = True
        result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 30 * 40 // 16, 20, 16))

    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=False)
    def test_get_kv_cache_shape_not_310p(self, mock_is_310p):
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

    def setUp(self):
        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_device = 'cpu:0'
        self.builder = AscendAttentionMetadataBuilder(self.mock_vllm_config,
                                                      self.mock_device)

    def test_reorder_batch(self):
        mock_input_batch = MagicMock()
        mock_scheduler_output = MagicMock()

        result = self.builder.reorder_batch(mock_input_batch,
                                            mock_scheduler_output)

        self.assertFalse(result)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    @patch('torch_npu.npu_format_cast')
    @patch('vllm_ascend.utils.nd_to_nz_2d')
    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=True)
    def test_build_prefill_no_cache(self, mock_is_310p, mock_nd_to_nz_2d,
                                    mock_npu_format_cast,
                                    mock_ascend_metadata):
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3, 7]),
            query_start_loc_cpu=torch.tensor([0, 3, 7]),
            seq_lens_cpu=torch.tensor([5, 6]),
            num_reqs=2,
            num_actual_tokens=10,
            max_query_len=5,
            decode_token_per_req=torch.tensor([1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping_cpu=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((10, 10)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.PrefillNoCache)

        mock_nz_tensor = MagicMock()
        mock_model = MagicMock()
        mock_nd_to_nz_2d.return_value = mock_nz_tensor
        mock_npu_format_cast.return_value = mock_nz_tensor

        self.builder.build(common_attn_metadata, mock_model)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    @patch('torch_npu.npu_format_cast')
    @patch('vllm_ascend.utils.nd_to_nz_spec')
    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=True)
    @patch('vllm_ascend.attention.attention_v1.AscendAttentionState')
    def test_build_chunked_prefill(self, mock_ascend_attention_state,
                                   mock_is_310p, mock_nd_to_nz_spec,
                                   mock_npu_format_cast, mock_ascend_metadata):
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=15,
            max_query_len=6,
            decode_token_per_req=torch.tensor([1, 1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping_cpu=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((15, 15)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill)

        mock_ascend_attention_state = MagicMock()
        mock_ascend_attention_state.PrefillNoCache = 0

        mock_nz_tensor = MagicMock()
        mock_model = MagicMock()
        mock_nd_to_nz_spec.return_value = mock_nz_tensor
        mock_npu_format_cast.return_value = mock_nz_tensor

        self.builder.build(common_attn_metadata, mock_model)

    @patch('vllm_ascend.attention.attention_v1.AscendMetadata')
    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=False)
    def test_build_non_310p(self, mock_is_310p, mock_ascend_metadata):
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=15,
            max_query_len=6,
            decode_token_per_req=torch.tensor([1, 1, 1]),
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping_cpu=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((15, 15)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill)
        mock_model = MagicMock()

        self.builder.build(common_attn_metadata, mock_model)


class TestAscendAttentionBackendImpl(TestBase):

    @patch('vllm.distributed.parallel_state._CP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_context_model_parallel_world_size",
           return_value=1)
    def setUp(self):
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

    @patch('torch.ops.vllm.unified_ascend_attention_with_output')
    def test_forward_trace_flag_true(self, mock_unified_attention):
        """Test forward pass when trace_flag is True"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        metadata = self.attn_metadata
        layer = self.layer

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=True)

        mock_unified_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_paged_attention_splitfuse')
    def test_forward_with_quant_method(self, mock_paged_attention):
        """Test forward pass when layer has quant_method"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        k_cache = torch.ones(1, 10, 8, 64, dtype=torch.int8)
        v_cache = torch.ones(1, 10, 8, 64, dtype=torch.int8)
        kv_cache = [k_cache, v_cache]
        ret_value = torch.ones(1, 1, 10, 8, 64, dtype=torch.int8)

        metadata = MagicMock()
        metadata.num_actual_tokens = torch.randn(10, 8 * 64)
        metadata.block_tables = torch.randn(10, 8 * 64)
        metadata.seq_lens = torch.randn(10, 8 * 64)
        metadata.attn_mask = torch.randn(10, 8 * 64)
        metadata.query_lens = torch.randn(10, 8 * 64)
        layer = self.layer
        layer.quant_method = MagicMock()
        layer.quant_method.apply.return_value = ret_value

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        layer.quant_method.apply.assert_called_once()
        assert output.shape == (10, 8 * 64)

    def test_forward_no_attn_metadata(self):
        """Test forward pass when attn_metadata is None"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        layer = self.layer_no_quant

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   None,
                                   trace_flag=False)

        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_flash_attention')
    def test_forward_prefill_no_cache(self, mock_flash_attention,
                                      mock_reshape_cache):
        """Test forward pass in PrefillNoCache state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.seq_lens = torch.tensor([10])
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        # layer.quant_method.apply.return_value = metadata
        print(self.layer_no_quant._v_scale_float)
        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        mock_reshape_cache.assert_called_once()
        mock_flash_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_flash_attention_qlens')
    def test_forward_prefill_cache_hit(self, mock_flash_attention_qlens,
                                       mock_npu_reshape_and_cache):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        mock_flash_attention_qlens.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention')
    def test_forward_decode_only(self, mock_paged_attention,
                                 mock_npu_reshape_and_cache):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        mock_paged_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_forward_decode_only_swa(self, mock_fused_infer_attention_score,
                                     mock_npu_reshape_and_cache):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10] * 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8,
                                                                    64), 1)
        output = self.impl_swa.forward(layer,
                                       query,
                                       key,
                                       value,
                                       kv_cache,
                                       metadata,
                                       trace_flag=False)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=False)
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('vllm_ascend.attention.attention_v1.vanilla_chunked_prefill')
    def test_forward_head_size_192(self, mock_vanilla_prefill,
                                   mock_npu_reshape_and_cache, mock_is_310p):
        """Test forward pass when head_size is 192"""

        self.impl.head_size = 192
        query = torch.randn(10, 8 * 192)
        key = torch.randn(10, 8 * 192)
        value = torch.randn(10, 8 * 192)
        kv_cache = torch.empty(2, 5, 128, 8, 192)
        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        mock_vanilla_prefill.return_value = MagicMock()

        output = self.impl_192.forward(layer,
                                       query,
                                       key,
                                       value,
                                       kv_cache,
                                       metadata,
                                       trace_flag=False)

        mock_vanilla_prefill.assert_called_once()
        assert output.shape == (10, 8 * 192)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention_splitfuse')
    def test_forward_normal_v1_situation(self, mock_paged_attention,
                                         mock_npu_reshape_and_cache):
        """Test forward pass in normal V1 situation"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        mock_paged_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention_splitfuse')
    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=True)
    def test_forward_310p_device(self, mock_is_310p, mock_paged_attention,
                                 mock_npu_reshape_and_cache,
                                 mock_npu_format_cast):
        """Test forward pass on 310P device"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        mock_npu_format_cast.return_value = metadata.attn_mask
        output = self.impl.forward(layer,
                                   query,
                                   key,
                                   value,
                                   kv_cache,
                                   metadata,
                                   trace_flag=False)

        mock_paged_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    def test_forward_raise_error(self, mock_paged_attention):
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant

        with self.assertRaises(NotImplementedError):
            self.impl_error.forward(layer,
                                    query,
                                    key,
                                    value,
                                    kv_cache,
                                    metadata,
                                    trace_flag=False)
