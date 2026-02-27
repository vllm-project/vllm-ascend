from typing import List
from unittest.mock import MagicMock, patch

import torch

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.attention.context_parallel.attention_cp import \
    AscendAttentionCPImpl
from vllm_ascend.attention.context_parallel.common_cp import (
    AscendMetadataForPrefill, AscendPCPMetadata)


class TestAscendAttentionCPImpl(TestBase):

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
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
        self.mock_vllm_config = MagicMock()
        self.config_patcher = patch(
            'vllm_ascend.attention.attention_v1.get_current_vllm_config',
            return_value=self.mock_vllm_config)
        self.config_patcher.start()

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

    def test_init(self):
        self.assertEqual(self.impl.pcp_size, 2)
        self.assertEqual(self.impl.pcp_rank, 0)
        self.assertEqual(self.impl.dcp_size, 2)
        self.assertEqual(self.impl.dcp_rank, 0)

    def test_forward_prefill_cp(self):
        query = torch.randn(2, 4, 128)
        key = torch.randn(4, 1, 128)
        value = torch.randn(4, 1, 128)

        def mock_attention_with_nomask_and_mask(q, k_mask, **kwargs):
            mock_output = torch.randn_like(q)
            mock_lse = torch.randn_like(k_mask)
            return mock_output, mock_lse

        self.impl._attention_with_nomask_and_mask = MagicMock()
        self.impl._attention_with_nomask_and_mask.side_effect = mock_attention_with_nomask_and_mask

        attn_metadata = MagicMock()
        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.pcp_metadata.q_head_idx = torch.tensor([0])
        attn_metadata.prefill.pcp_metadata.q_tail_idx = torch.tensor([1])
        attn_metadata.prefill.pcp_metadata.q_full_idx = torch.tensor([0, 1])
        attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx = torch.tensor(
            [0])
        attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx = torch.tensor(
            [0])
        attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx = torch.tensor(
            [0])

        output, attn_lse = self.impl._forward_prefill_cp(
            query, key, value, attn_metadata)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 4)
        self.assertEqual(output.shape[2], 128)

    @patch('torch_npu.npu_attention_update')
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch(
        'vllm_ascend.attention.context_parallel.attention_cp.get_forward_context'
    )
    @patch_distributed_groups(dcp_size=2, pcp_size=2)
    def test_forward_decode_pcp_dcp(self, mock_all2all, mock_dcp, mock_pcp,
                                    mock_get_forward_context,
                                    mock_npu_fused_infer_attention_score,
                                    mock_npu_attention_update):
        query = torch.randn(2, 4, 64)
        self.impl.key_cache = torch.randn(100, 64, 1, 64)
        self.impl.value_cache = torch.randn(100, 64, 1, 64)

        # Mock output
        mock_npu_attention_update.return_value = (torch.randn(2 * 4, 64), None)

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        def mock_npu_fused_infer_attention_score_func(query, k_nope, value,
                                                      **common_kwargs):
            mock_output = torch.randn_like(query)
            mock_lse = torch.randn(query.shape[0], query.shape[1], 1)
            return mock_output, mock_lse

        mock_npu_fused_infer_attention_score.side_effect = mock_npu_fused_infer_attention_score_func

        attn_metadata = MagicMock()
        attn_metadata.decode_meta = MagicMock()
        attn_metadata.num_decodes_flatten = 5
        attn_metadata.decode_meta.batch_seq_mask = torch.tensor(
            [1, 0], dtype=torch.bool)
        output = self.impl._forward_decode_pcp_dcp(query, attn_metadata)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 4)
        self.assertEqual(output.shape[2], 64)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_prefill_query_all_gather(self):
        query = torch.randn(2, 4, 128)

        attn_metadata = MagicMock()
        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.chunked_context = MagicMock()
        attn_metadata.prefill.chunked_context.cp_kv_recover_idx_for_chunk = torch.tensor(
            [1, 2, 3, 0])
        output = self.impl._prefill_query_all_gather(attn_metadata, query)

        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 8)
        self.assertEqual(output.shape[2], 128)

    @patch('torch.ops.npu.npu_fused_infer_attention_score')
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_compute_prefill_context(self, mock_npu_attention):

        block_num = 100
        block_size = 128
        kv_num_heads = 1
        head_size = 128
        kv_cache = (torch.randn(block_num, block_size, kv_num_heads,
                                head_size),
                    torch.randn(block_num, block_size, kv_num_heads,
                                head_size))

        batch_size = 1024
        self.impl.head_size = head_size
        self.impl.num_heads = 4
        num_heads = self.impl.num_heads * self.impl.dcp_size
        query = torch.randn(batch_size, num_heads, head_size)

        attn_metadata = MagicMock()
        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.chunked_context = MagicMock()
        local_context_lens_allranks = torch.tensor([[[256, 256], [256, 256]]])
        attn_metadata.prefill.chunked_context.local_context_lens_allranks = local_context_lens_allranks
        attn_metadata.prefill.chunked_context.batch_chunk_seq_mask = torch.randint(
            0, 2, (1024, ), dtype=torch.bool)
        attn_metadata.prefill.chunked_context.local_total_toks = local_context_lens_allranks[:,
                                                                                             0,
                                                                                             0].sum(
                                                                                             )

        def mock_load_kv_for_chunk(attn_metadata, kv_cache,
                                   local_chunked_kv_lens_rank, query,
                                   total_toks):
            return torch.randn(total_toks, kv_num_heads,
                               head_size), torch.randn(total_toks,
                                                       kv_num_heads, head_size)

        self.impl._load_kv_for_chunk = MagicMock()
        self.impl._load_kv_for_chunk.side_effect = mock_load_kv_for_chunk

        mock_npu_attention.return_value = torch.randn(batch_size, num_heads,
                                                      head_size), torch.randn(
                                                          batch_size,
                                                          num_heads, 1)

        context_output = self.impl._compute_prefill_context(
            query, kv_cache, attn_metadata)
        local_context_output = torch.cat(context_output,
                                         dim=-1).permute([1, 2,
                                                          0]).contiguous()
        global_context_output = self.impl._gather_global_context_output(
            local_context_output)
        global_context_output = global_context_output.permute([2, 0, 1
                                                               ]).contiguous()
        result_output, result_lse = self.impl._update_global_context_output(
            global_context_output)

        self.assertEqual(result_output.shape[0], batch_size)
        self.assertEqual(result_output.shape[1], self.impl.num_heads)
        self.assertEqual(result_output.shape[2], head_size)
        self.assertEqual(result_lse.shape[0], batch_size)
        self.assertEqual(result_lse.shape[1], self.impl.num_heads)
        self.assertEqual(result_lse.shape[2], 1)

    @patch('torch_npu.atb.npu_paged_cache_load')
    def test_load_kv_for_chunk(self, mock_npu_paged_cache_load):
        block_num = 100
        block_size = 128
        num_heads = 1
        head_size = 128

        kv_cache = (torch.randn(block_num, block_size, num_heads, head_size),
                    torch.randn(block_num, block_size, num_heads, head_size))
        query = torch.randn(4, 8, 128)
        total_toks = 256
        local_chunked_kv_lens_rank = torch.randn(total_toks)

        attn_metadata = MagicMock()

        key, value = self.impl._load_kv_for_chunk(attn_metadata, kv_cache,
                                                  local_chunked_kv_lens_rank,
                                                  query, total_toks)

        self.assertEqual(key.shape[0], total_toks)
        self.assertEqual(key.shape[1], num_heads)
        self.assertEqual(key.shape[2], head_size)
        self.assertEqual(value.shape[0], total_toks)
        self.assertEqual(value.shape[1], num_heads)
        self.assertEqual(value.shape[2], head_size)

    @patch('torch_npu.Event', create=True)
    @patch('torch_npu._npu_reshape_and_cache')
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_reshape_and_cache(self, mock_event_class, mock_npu_reshape_and_cache):
        num_tokens = 4
        block_num = 100
        block_size = 128
        num_heads = 1
        head_size = 128
        self.impl.head_size = head_size
        self.impl.is_kv_producer = False

        kv_cache = (torch.randn(block_num, block_size, num_heads, head_size),
                    torch.randn(block_num, block_size, num_heads, head_size))

        attn_metadata = MagicMock()
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_decodes = 1
        attn_metadata.num_prefills = 1
        attn_metadata.slot_mapping = torch.randn(2)
        attn_metadata.num_actual_tokens_pcp_padded = num_tokens * self.impl.pcp_size
        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.pcp_metadata.pcp_allgather_restore_idx = torch.tensor(
            [0, 3, 1, 2, 0, 0, 0, 0])

        key = torch.randn(num_tokens, num_heads, head_size)
        value = torch.randn(num_tokens, num_heads, head_size)

        key, value = self.impl.reshape_and_cache(key, value, kv_cache,
                                                 attn_metadata)
        self.assertEqual(key.shape[0], num_tokens * self.impl.pcp_size)
        self.assertEqual(key.shape[1], num_heads)
        self.assertEqual(key.shape[2], head_size)
        self.assertEqual(value.shape[0], num_tokens * self.impl.pcp_size)
        self.assertEqual(value.shape[1], num_heads)
        self.assertEqual(value.shape[2], head_size)


class TestUpdateNpuAttnOutLse(TestBase):

    @patch_distributed_groups(needs_mocks=False)
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
        self.impl.dcp_size = 1

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
        pcp_metadata = AscendPCPMetadata()
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
        'vllm_ascend.attention.context_parallel.attention_cp.AscendAttentionCPImpl._update_out_and_lse'
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
        'vllm_ascend.attention.context_parallel.attention_cp.AscendAttentionCPImpl._npu_attn_out_lse_update'
    )
    def test_attention_with_nomask_and_mask_nochunk(
            self, mock_npu_attn_out_lse_update,
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
        attn_metadata.prefill.chunked_context = None

        # Mock output
        mock_npu_fused_infer_attention_score.return_value = torch.randn(
            self.q_total_tokens, self.impl.num_heads,
            self.impl.head_size), torch.randn(self.q_total_tokens,
                                              self.impl.num_heads, 1)
        mock_npu_attn_out_lse_update.return_value = torch.randn(
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
        mock_npu_attn_out_lse_update.assert_called_once()
        self.assertEqual(mock_npu_fused_infer_attention_score.call_count, 2)
        self.assertIsNotNone(output)
        self.assertEqual(attn_lse, None)

    @patch(
        'vllm_ascend.attention.context_parallel.attention_cp.AscendAttentionCPImpl._npu_attn_out_lse_update'
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

    @patch_distributed_groups(dcp_size=2, pcp_size=3)
    def test_update_chunk_attn_out_lse_dcp2_pcp3(self, mock_all_to_all_single,
                                                 mock_dcp, mock_pcp):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)
        self.impl.dcp_size = 2
        self.impl.pcp_size = 3
        self.impl.head_size = 8

        # Call the method under test
        chunk_data = torch.cat([prefix_chunk_output, prefix_chunk_lse],
                               dim=-1).permute([1, 2, 0]).contiguous()
        global_context_output = self.impl._gather_global_context_output(
            chunk_data)
        global_context_output = global_context_output.permute([2, 0, 1
                                                               ]).contiguous()
        output, lse = self.impl._update_global_context_output(
            global_context_output)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 8))
        self.assertEqual(lse.shape, (2, 2, 1))

        mock_all_to_all_single.assert_called_once()
        mock_pcp.all_gather.assert_called_once()

    @patch_distributed_groups(dcp_size=2)
    def test_update_chunk_attn_out_lse_dcp2_pcp1(self, mock_all_to_all_single,
                                                 mock_dcp, mock_pcp):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)

        self.impl.dcp_size = 2
        self.impl.pcp_size = 1
        self.impl.head_size = 8

        # Call the method under test
        chunk_data = torch.cat([prefix_chunk_output, prefix_chunk_lse],
                               dim=-1).permute([1, 2, 0]).contiguous()
        global_context_output = self.impl._gather_global_context_output(
            chunk_data)
        global_context_output = global_context_output.permute([2, 0, 1
                                                               ]).contiguous()
        output, lse = self.impl._update_global_context_output(
            global_context_output)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 8))
        self.assertEqual(lse.shape, (2, 2, 1))

        mock_all_to_all_single.assert_called_once()
        mock_pcp.all_gather.assert_not_called()

    @patch_distributed_groups(pcp_size=2)
    def test_update_chunk_attn_out_lse_dcp1_pcp2(self, mock_all_to_all_single,
                                                 mock_dcp, mock_pcp):
        # Mock input data
        prefix_chunk_output = torch.randn(2, 4, 8)
        prefix_chunk_lse = torch.randn(2, 4, 1)

        self.impl.dcp_size = 1
        self.impl.pcp_size = 2
        self.impl.head_size = 8

        # Call the method under test
        chunk_data = torch.cat([prefix_chunk_output, prefix_chunk_lse],
                               dim=-1).permute([1, 2, 0]).contiguous()
        global_context_output = self.impl._gather_global_context_output(
            chunk_data)
        global_context_output = global_context_output.permute([2, 0, 1
                                                               ]).contiguous()
        output, lse = self.impl._update_global_context_output(
            global_context_output)

        # Assert the method call
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(lse.shape, (2, 4, 1))

        mock_all_to_all_single.assert_not_called()
        mock_pcp.all_gather.assert_called_once()
