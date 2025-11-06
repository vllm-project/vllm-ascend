from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (AscendAttentionBackend,
                                                AscendAttentionBackendImpl,
                                                AscendAttentionMetadataBuilder,
                                                AscendAttentionState,
                                                AscendMetadata)
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

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        self.mock_vllm_config = MagicMock()
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.decode_max_num_seqs = 10
        self.mock_device = 'cpu:0'
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
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((10, 10)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        mock_nz_tensor = MagicMock()
        mock_model = MagicMock()
        mock_nd_to_nz_2d.return_value = mock_nz_tensor
        mock_npu_format_cast.return_value = mock_nz_tensor

        self.builder.build(1, common_attn_metadata, mock_model)

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
            slot_mapping=torch.tensor(range(20)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            positions=torch.tensor([10, 10]),
            attn_mask=torch.ones((15, 15)),
            spec_attn_mask=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_computed_tokens_cpu=None,
            seq_lens=None)

        mock_ascend_attention_state = MagicMock()
        mock_ascend_attention_state.PrefillNoCache = 0

        mock_nz_tensor = MagicMock()
        mock_model = MagicMock()
        mock_nd_to_nz_spec.return_value = mock_nz_tensor
        mock_npu_format_cast.return_value = mock_nz_tensor

        self.builder.build(1, common_attn_metadata, mock_model)

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

    @patch('vllm.distributed.parallel_state.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group):
        mock_dcp.world_size = 1
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 1
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

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
    @patch('torch_npu._npu_flash_attention')
    def test_forward_prefill_no_cache(self, mock_flash_attention,
                                      mock_reshape_cache):
        """Test forward pass in PrefillNoCache state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.seq_lens = torch.tensor([10])
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        layer = self.layer_no_quant

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_reshape_cache.assert_called_once()
        mock_flash_attention.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_forward_prefill_cache_hit(self,
                                       mock_npu_fused_infer_attention_score,
                                       mock_npu_reshape_and_cache):
        """Test forward pass in PrefillCacheHit state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        mock_npu_fused_infer_attention_score.return_value = (output, 1)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        layer = self.layer_no_quant

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('vllm_ascend.attention.attention_v1.get_forward_context')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention')
    def test_forward_decode_only(self, mock_paged_attention,
                                 mock_npu_reshape_and_cache,
                                 mock_get_forward_context):
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
    @patch('vllm_ascend.attention.attention_v1.get_graph_params')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch.npu.graph_task_group_end')
    @patch('torch.npu.graph_task_group_begin')
    @patch('torch.npu.ExternalEvent')
    @patch('torch_npu.npu.current_stream')
    @patch('vllm_ascend.attention.attention_v1.weak_ref_tensors')
    def test_paged_attention_with_existing_workspace(
        self,
        mock_get_forward_context,
        mock_get_graph_params,
        mock_npu_reshape_and_cache,
        mock_paged_attention,
        mock_graph_begin,
        mock_graph_end,
        mock_external_event_class,
        mock_current_stream,
        mock_weak_ref_tensors,
    ):
        graph_params = MagicMock()
        attn_metadata = MagicMock()
        num_tokens = 10

        graph_params.workspaces = {num_tokens: 10}
        graph_params.events = {num_tokens: []}
        graph_params.attn_params = {num_tokens: []}
        graph_params.handles = {num_tokens: []}

        query = torch.randn(2, 5, 8)  # [batch_size, seq_len, hidden_size]
        key_cache = MagicMock()
        value_cache = MagicMock()
        num_kv_heads = 4
        num_heads = 8
        scale = 0.1
        output = torch.randn(2, 5, 8)

        self_obj = MagicMock()
        self_obj.key_cache = key_cache
        self_obj.value_cache = value_cache
        self_obj.num_kv_heads = num_kv_heads
        self_obj.num_heads = num_heads
        self_obj.scale = scale

        mock_stream = MagicMock()
        mock_current_stream.return_value = mock_stream
        mock_event_instance = MagicMock()
        mock_external_event_class.return_value = mock_event_instance

        mock_handle = MagicMock()
        mock_graph_end.return_value = mock_handle

        workspace = graph_params.workspaces.get(num_tokens)
        self.assertEqual(workspace, 10)

        weak_ref_tensors = MagicMock(side_effect=lambda x: x)

        # 2. Handle graph capturing mode
        stream = mock_current_stream()
        event = mock_external_event_class()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append((
            weak_ref_tensors(query),
            weak_ref_tensors(self_obj.key_cache),
            weak_ref_tensors(self_obj.value_cache),
            self_obj.num_kv_heads,
            self_obj.num_heads,
            self_obj.scale,
            weak_ref_tensors(attn_metadata.block_tables),
            attn_metadata.seq_lens,
            output,
        ))

        mock_event_instance.wait.assert_called_once_with(mock_stream)
        mock_event_instance.reset.assert_called_once_with(mock_stream)
        self.assertEqual(len(graph_params.events[num_tokens]), 1)
        self.assertEqual(len(graph_params.attn_params[num_tokens]), 1)

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
        metadata.num_decodes = 0
        metadata.num_prefills = 10
        layer = self.layer_no_quant

        mock_get_forward_context.return_value = MagicMock(capturing=True)
        mock_get_graph_params.return_value = graph_params

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_paged_attention.assert_called_once()
        self.assertEqual(len(graph_params.handles[num_tokens]), 0)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_forward_decode_only_swa(self, mock_fused_infer_attention_score,
                                     mock_npu_reshape_and_cache):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty(10, 8, 64)

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
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu._npu_paged_attention')
    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_forward_decode_only_swa_seq_len_mismatch(
            self, mock_fused_infer_attention_score, mock_paged_attention,
            mock_npu_reshape_and_cache, mock_get_forward_context):
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

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8,
                                                                    64), 1)

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl_swa.forward(layer, query, key, value, kv_cache,
                                       metadata, output)

        mock_paged_attention.assert_called_once()
        mock_fused_infer_attention_score.assert_not_called()

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
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.query_lens = torch.tensor([10])
        metadata.seq_lens = torch.tensor([10])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_vanilla_prefill.return_value = MagicMock()

        output = self.impl_192.forward(layer, query, key, value, kv_cache,
                                       metadata, output)

        mock_vanilla_prefill.assert_called_once()
        assert output.shape == (10, 8 * 192)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    def test_forward_normal_v1_situation(self,
                                         mock_npu_fused_infer_attention_score,
                                         mock_npu_reshape_and_cache):
        """Test forward pass in normal V1 situation"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        mock_npu_fused_infer_attention_score.return_value = (output, 1)

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

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8 * 64)

    @patch('torch_npu.npu_format_cast')
    @patch('torch_npu._npu_reshape_and_cache')
    @patch('torch_npu.npu_fused_infer_attention_score')
    @patch('vllm_ascend.attention.attention_v1.is_310p', return_value=True)
    def test_forward_310p_device(self, mock_is_310p,
                                 mock_npu_fused_infer_attention_score,
                                 mock_npu_reshape_and_cache,
                                 mock_npu_format_cast):
        """Test forward pass on 310P device"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        mock_npu_fused_infer_attention_score.return_value = (output, 1)

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

        mock_npu_format_cast.return_value = metadata.attn_mask

        output = self.impl.forward(layer, query, key, value, kv_cache,
                                   metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
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
