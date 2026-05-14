from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.attention.kvcomp_attn.attention_utils import get_kvcomp_decode_params, reshape_and_cache_kvcomp
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


class TestAscendAttentionBackend(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()

        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1

        self.mock_config.parallel_config = mock_parallel_config

        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()

        from vllm_ascend.attention.utils import enable_cp

        enable_cp.cache_clear()

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(), AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(), AscendAttentionMetadataBuilder)

    def test_get_kv_cache_shape_not(self):
        result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_swap_blocks(self):
        src_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        dst_kv_cache = [torch.zeros((10, 20)), torch.zeros((10, 20))]
        src_to_dst = torch.tensor([[0, 1], [2, 3]])
        AscendAttentionBackend.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)
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
        self.mock_vllm_config.speculative_config = None
        self.mock_vllm_config.model_config.max_model_len = 640
        self.mock_vllm_config.model_config.hf_text_config.sliding_window = None
        self.mock_vllm_config.cache_config.block_size = 64
        self.mock_vllm_config.compilation_config.cudagraph_mode = None
        self.mock_vllm_config.scheduler_config.max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.decode_max_num_seqs = 10
        self.mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        self.mock_device = "cpu:0"
        torch.Tensor.pin_memory = lambda x: x  # noqa
        self.builder = AscendAttentionMetadataBuilder(None, None, self.mock_vllm_config, self.mock_device)

    def test_reorder_batch(self):
        mock_input_batch = MagicMock()
        mock_scheduler_output = MagicMock()

        result = self.builder.reorder_batch(mock_input_batch, mock_scheduler_output)

        self.assertFalse(result)

    def test_unpadded_preserves_internal_seq_lens_cpu(self):
        internal_seq_lens_cpu = torch.tensor([4, 5, 6], dtype=torch.int32)
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 9]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 9]),
            seq_lens=torch.tensor([4, 5, 6], dtype=torch.int32),
            _seq_lens_cpu=internal_seq_lens_cpu,
            seq_lens_cpu=None,
            num_computed_tokens_cpu=None,
            num_reqs=3,
            num_actual_tokens=9,
            max_query_len=4,
            block_table_tensor=torch.zeros((3, 1), dtype=torch.int32),
            slot_mapping=torch.arange(9, dtype=torch.int32),
            causal=True,
            actual_seq_lengths_q=[2, 3, 4],
            positions=torch.arange(9),
            attn_state=AscendAttentionState.ChunkedPrefill,
            max_seq_len=6,
        )

        unpadded_metadata = common_attn_metadata.unpadded(num_actual_tokens=5, num_actual_reqs=2)

        self.assertTrue(torch.equal(unpadded_metadata._seq_lens_cpu, internal_seq_lens_cpu[:2]))
        self.assertIsNone(unpadded_metadata.seq_lens_cpu)

    @patch("vllm_ascend.attention.attention_v1.AscendMetadata")
    def test_build(self, mock_ascend_metadata):
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
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6,
        )
        mock_model = MagicMock()

        self.builder.build(1, common_attn_metadata, mock_model)


class TestAscendAttentionBackendImpl(TestBase):
    def setUp(self):
        self.mock_event = MagicMock()
        self.mock_event.record.return_value = None
        self.mock_event.wait.return_value = None

        self.mock_stream = MagicMock()
        self.event_patcher = patch("torch_npu.npu.Event", return_value=self.mock_event)
        self.stream_patcher = patch("torch_npu.npu.current_stream", return_value=self.mock_stream)

        self.event_patcher.start()
        self.stream_patcher.start()

        self.layer = MagicMock()
        self.layer.layer_name = "test_layer"
        self.layer._k_scale_float = 1.0
        self.layer._v_scale_float = 1.0
        self.attention_type = MagicMock()
        self.attention_type.DECODER = "decoder"
        self.attention_type.ENCODER = "encoder"
        self.attn_metadata = MagicMock()
        self.attn_metadata.return_value = "1"
        self.layer_no_quant = MagicMock(spec=["layer_name", "_k_scale_float", "_v_scale_float"])
        self.layer_no_quant.layer_name = "test_layer"
        self.layer_no_quant._k_scale_float = 1.0
        self.layer_no_quant._v_scale_float = 1.0
        self.mock_vllm_config = MagicMock()
        self.config_patcher = patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.config_patcher.start()

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
            kv_sharing_target_layer_name=None,
        )

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
            kv_sharing_target_layer_name=None,
        )

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
            kv_sharing_target_layer_name=None,
        )

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
            kv_sharing_target_layer_name=None,
        )

    def test_forward_no_attn_metadata(self):
        """Test forward pass when attn_metadata is None"""
        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 0, 0, 8, 64)
        layer = self.layer_no_quant
        output = torch.empty_like(query)

        output = self.impl.forward(layer, query, key, value, kv_cache, None, output)

        assert output.shape == (10, 8 * 64)

    @patch("torch_npu._npu_reshape_and_cache")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward_fused_infer_attention(
        self, mock_get_forward_context, mock_npu_fused_infer_attention_score, mock_npu_reshape_and_cache
    ):
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
        mock_npu_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64), torch.ones(10, 8, 64))
        output = self.impl.forward(layer, query, key, value, kv_cache, metadata, output)

        mock_npu_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch("vllm_ascend.attention.attention_v1.using_paged_attention")
    @patch("torch_npu._npu_paged_attention")
    @patch("torch_npu._npu_reshape_and_cache")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward_paged_attention(
        self, mock_get_forward_context, mock_npu_reshape_and_cache, mock_paged_attention, mock_using_paged_attention
    ):
        """Test forward pass in DecodeOnly state"""
        query = torch.randn(4, 8 * 64)
        key = torch.randn(4, 8 * 64)
        value = torch.randn(4, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([4])
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 4
        metadata.slot_mapping = torch.zeros(4, dtype=torch.long)
        metadata.num_decodes = 4
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_using_paged_attention.return_value = True

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        output = self.impl.forward(layer, query, key, value, kv_cache, metadata, output)

        mock_paged_attention.assert_called_once()
        assert output.shape == (4, 8 * 64)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("torch_npu._npu_reshape_and_cache")
    def test_forward_decode_only_swa(
        self, mock_npu_reshape_and_cache, mock_fused_infer_attention_score, mock_get_forward_context
    ):
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
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64), 1)
        output = self.impl_swa.forward(layer, query, key, value, kv_cache, metadata, output)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu._npu_paged_attention")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("torch_npu._npu_reshape_and_cache")
    def test_forward_decode_only_swa_seq_len_mismatch(
        self,
        mock_npu_reshape_and_cache,
        mock_fused_infer_attention_score,
        mock_paged_attention,
        mock_get_forward_context,
    ):
        """Test forward pass in DecodeOnly state when seq)len_mismatch"""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
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
        metadata.actual_seq_lengths_q = [10]

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64), torch.ones(10, 8, 64))

        output = self.impl_swa.forward(layer, query, key, value, kv_cache, metadata, output)

        mock_paged_attention.assert_not_called()
        mock_fused_infer_attention_score.assert_called_once()

        assert output.shape == (10, 8, 64)

    def test_get_kvcomp_params_early_exit(self):
        """
        Test that get_kvcomp_decode_params returns original values
        when kvcomp is disabled or hashk_cache is missing.
        """
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        block_table = torch.zeros(1, 5, dtype=torch.long)
        actual_seq_lengths_kv = [10]

        metadata = MagicMock()
        # Mocking the case where hashk_caches is not properly initialized
        kvcomp_metadata = MagicMock()
        kvcomp_metadata.hashk_caches = [None]
        metadata.kvcomp_metadata = kvcomp_metadata

        self.impl.enable_hamming_sparse = True
        self.impl.layerIndex = 0

        res_bt, res_sl = get_kvcomp_decode_params(0, kvcomp_metadata, query, key, block_table, actual_seq_lengths_kv)

        self.assertIs(res_bt, block_table)
        self.assertEqual(res_sl, actual_seq_lengths_kv)

    def test_get_kvcomp_params_reuse(self):
        """
        Test that in DecodeOnly state, if the current layer is a skip layer,
        it correctly reuses the Hamming results from a previous layer.
        """
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        block_table = torch.zeros(1, 5, dtype=torch.long)
        actual_seq_lengths_kv = [10]

        self.impl.enable_hamming_sparse = True
        self.impl.layerIndex = 1

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        expected_bt = torch.ones(1, 5)
        expected_sl = torch.tensor([5])

        # Construct kvcomp_metadata
        kvcomp_metadata = MagicMock()
        kvcomp_metadata.hashk_caches = [MagicMock(), MagicMock()]
        kvcomp_metadata.hamming_output = expected_bt
        kvcomp_metadata.seq_lens_from_hamming = expected_sl

        kvcomp_config = MagicMock()
        kvcomp_config.vllm_hash_attention_skip_layers = [False, True]
        kvcomp_config.top_k_index_reuse = [0, 0]
        kvcomp_metadata.kvcomp_config = kvcomp_config
        metadata.kvcomp_metadata = kvcomp_metadata

        metadata.hamming_output_records = [{"new_block_table": expected_bt, "new_seq_lens_list": expected_sl}, None]

        res_bt, res_sl = get_kvcomp_decode_params(1, kvcomp_metadata, query, key, block_table, actual_seq_lengths_kv)

        self.assertTrue(torch.equal(res_bt, expected_bt))
        self.assertTrue(torch.equal(res_sl, expected_sl))

    @patch("torch.ops._C_ascend.npu_reshape_and_cache_bnsd", create=True)
    def test_get_kvcomp_params_prefill(self, mock_reshape_and_cache):
        """
        Test that in non-DecodeOnly state (e.g., Prefill), only Hash compute
        and Cache update are performed, and original params are returned.
        """
        key = torch.randn(2, 8, 64)

        self.impl.enable_hamming_sparse = True
        self.impl.layerIndex = 0

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.PrefillCacheHit
        metadata.slot_mapping = torch.zeros(2)
        metadata.actual_seq_lengths_q_device = torch.tensor([1, 1])
        metadata.num_actual_tokens = 2
        metadata.actual_query_lens = torch.tensor([1, 1], dtype=torch.int32)
        metadata.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)

        kvcomp_metadata = MagicMock()
        kvcomp_metadata.hashk_caches = [MagicMock()]
        kvcomp_config = MagicMock()
        kvcomp_config.vllm_hash_attention_skip_layers = [False]
        kvcomp_metadata.kvcomp_config = kvcomp_config

        # Mock HashEncoder
        hash_encoder = MagicMock()
        hash_encoder.compute_hash.return_value = torch.ones(2, 8, 8)
        kvcomp_metadata.hash_encoder = hash_encoder

        metadata.kvcomp_metadata = kvcomp_metadata

        reshape_and_cache_kvcomp(kvcomp_metadata, 0, key)

        # Ensure cache update was called but Hamming was bypassed
        self.assertTrue(mock_reshape_and_cache.called)

    @patch("torch.ops._C_ascend.npu_reshape_and_cache_bnsd", create=True)
    @patch("torch.ops._C_ascend.npu_hamming_dist_top_k", create=True)
    def test_get_kvcomp_params_decode_hamming(self, mock_hamming, mock_reshape):
        """
        Test that in DecodeOnly state, the full flow including Hash computation
        and Hamming Distance Top-K operation is executed.
        """
        query = torch.randn(2, 8, 64)
        key = torch.randn(2, 8, 64)
        block_table = torch.zeros(2, 5, dtype=torch.long)
        actual_seq_lengths_kv = [10, 10]

        self.impl.enable_hamming_sparse = True
        self.impl.layerIndex = 0

        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10, 10])
        metadata.actual_seq_lengths_q_device = torch.tensor([1, 1])
        metadata.slot_mapping = torch.zeros(2)
        metadata.block_tables = block_table
        metadata.hamming_output_records = [None]
        metadata.num_actual_tokens = 2
        metadata.actual_query_lens = torch.tensor([1, 1], dtype=torch.int32)
        metadata.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
        metadata.chunk_sizes_for_hamming = torch.tensor([64, 64], dtype=torch.int32)
        metadata.max_seq_len_for_hamming = 1024
        metadata.block_tables_for_hamming = torch.zeros(2, 10, dtype=torch.int32)
        metadata.new_seq_lens_list = torch.tensor([5, 5], dtype=torch.int32)

        kvcomp_metadata = MagicMock()
        kvcomp_metadata.hashk_caches = [MagicMock()]

        kvcomp_config = MagicMock()
        kvcomp_config.vllm_hash_attention_skip_layers = [False]
        kvcomp_config.chunk_size = 64
        kvcomp_metadata.kvcomp_config = kvcomp_config

        # Mock necessary Hamming parameters
        kvcomp_metadata.chunk_sizes_for_hamming_full = torch.tensor([1, 1])
        kvcomp_metadata.topk_for_hamming_full = torch.tensor([1, 1])
        kvcomp_metadata.topk_for_hamming_full_cpu = torch.tensor([1, 1])
        kvcomp_metadata.hamming_output = torch.zeros(2, 1)

        # Mock HashEncoder
        hash_encoder = MagicMock()
        hash_encoder.compute_hash.return_value = torch.ones(2, 8, 8)
        kvcomp_metadata.hash_encoder = hash_encoder

        metadata.kvcomp_metadata = kvcomp_metadata

        # Mock npu_hamming_dist_top_k output; note the squeeze(1) in implementation
        mock_hamming.return_value = torch.ones(2, 1, 5)

        res_bt, res_sl = get_kvcomp_decode_params(0, kvcomp_metadata, query, key, block_table, actual_seq_lengths_kv)

        self.assertTrue(mock_reshape.called)
        self.assertTrue(mock_hamming.called)
        # Verify shape after squeeze(1) becomes (2, 5)
        self.assertEqual(res_bt.shape, (2, 5))
        self.assertTrue(torch.equal(res_bt, torch.ones(2, 5)))
        # Verify the result is recorded in hamming_output_records
        self.assertTrue(torch.equal(kvcomp_metadata.hamming_output, torch.ones(2, 5)))
        self.assertIsNotNone(kvcomp_metadata.seq_lens_for_hamming)
