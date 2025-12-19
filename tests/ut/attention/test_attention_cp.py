from typing import List
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_cp import AscendAttentionCPImpl
from vllm_ascend.attention.attention_v1 import (AscendMetadata,
                                                AscendMetadataForPrefill)


class TestAscendAttentionCPImpl(TestBase):

    @patch('vllm_ascend.attention.attention_cp.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm_ascend.attention.attention_cp.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    def setUp(self, mock_get_dcp_size, mock_dcp, mock_get_dcp_group, mock_pcp,
              mock_get_pcp_group):
        mock_dcp.world_size = 2
        mock_dcp.rank_in_group = 0
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 2
        dcp_group.device_group = MagicMock()
        mock_get_dcp_group.return_value = dcp_group

        mock_pcp.world_size = 2
        mock_pcp.rank_in_group = 0
        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.rank_in_group = 0
        pcp_group.world_size = 2
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

    @patch('vllm_ascend.attention.attention_cp.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP')
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.all_to_all_single")
    @patch('vllm_ascend.attention.attention_cp.get_forward_context')
    def test_forward_decode_pcp_dcp(self, mock_get_forward_context,
                                    mock_all_to_all_single, mock_all_gather,
                                    mock_npu_fused_infer_attention_score,
                                    mock_dcp, mock_get_dcp_group):

        def mock_dcp_all_gather_func(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        mock_dcp.world_size = 2
        mock_dcp.rank_in_group = 0
        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.rank_in_group = 0
        dcp_group.world_size = 2
        dcp_group.device_group = MagicMock()
        dcp_group.all_gather = mock_dcp_all_gather_func
        mock_get_dcp_group.return_value = dcp_group

        query = torch.randn(2, 4, 128)
        self.impl.key_cache = torch.randn(100, 128, 1, 128)
        self.impl.value_cache = torch.randn(100, 128, 1, 128)

        def mock_npu_attention_update(attn_out_lse_list):
            mock_output = torch.randn(attn_out_lse_list[0].shape[0],
                                      attn_out_lse_list[0].shape[1],
                                      attn_out_lse_list[0].shape[2] - 1)
            return mock_output

        self.impl._npu_attention_update = MagicMock()
        self.impl._npu_attention_update.side_effect = mock_npu_attention_update

        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_all_to_all_single.side_effect = lambda output, input, *args, **kwargs: output.copy_(
            input)

        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        def mock_npu_fused_infer_attention_score_func(query, k_nope, value,
                                                      **common_kwargs):
            mock_output = torch.randn_like(query)
            mock_lse = torch.randn(query.shape[0], query.shape[1], 1)
            return mock_output, mock_lse

        mock_npu_fused_infer_attention_score.side_effect = mock_npu_fused_infer_attention_score_func

        attn_metadata = MagicMock()
        attn_metadata.decode_meta = MagicMock()
        attn_metadata.decode_meta.batch_seq_mask = torch.tensor(
            [1, 0], dtype=torch.bool)

        output = self.impl._forward_decode_pcp_dcp(query, attn_metadata)

        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 4)
        self.assertEqual(output.shape[2], 128)

    @patch('vllm_ascend.attention.attention_cp.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP')
    @patch('vllm_ascend.attention.attention_cp.get_dcp_group')
    @patch('vllm.distributed.parallel_state._DCP')
    def test_prefill_query_all_gather(self, mock_dcp, mock_get_dcp_group,
                                      mock_pcp, mock_get_pcp_group):
        query = torch.randn(2, 4, 128)

        def mock_all_gather_func(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        dcp_group = MagicMock(spec=GroupCoordinator)
        dcp_group.all_gather = mock_all_gather_func
        mock_get_dcp_group.return_value = dcp_group

        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.all_gather = mock_all_gather_func
        mock_get_pcp_group.return_value = pcp_group

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
        attn_metadata.prefill.chunked_context.local_context_lens_allranks = torch.tensor(
            [[[256, 256], [256, 256]]])
        attn_metadata.prefill.chunked_context.batch_chunk_seq_mask = torch.randint(
            0, 2, (1024, ), dtype=torch.bool)

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

        result_output, result_lse = self.impl._compute_prefill_context(
            query, kv_cache, attn_metadata)

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

    @patch('vllm_ascend.attention.attention_cp.get_pcp_group')
    @patch('vllm.distributed.parallel_state._PCP')
    @patch('torch_npu._npu_reshape_and_cache')
    def test_reshape_and_cache(self, mock_npu_reshape_and_cache, mock_pcp,
                               mock_get_pcp_group):
        num_tokens = 4
        block_num = 100
        block_size = 128
        num_heads = 1
        head_size = 128
        self.impl.head_size = head_size

        kv_cache = (torch.randn(block_num, block_size, num_heads, head_size),
                    torch.randn(block_num, block_size, num_heads, head_size))

        attn_metadata = MagicMock()
        attn_metadata.num_decode_tokens = 1
        attn_metadata.num_decodes = 1
        attn_metadata.num_prefills = 1
        attn_metadata.slot_mapping = torch.randn(2)
        attn_metadata.num_actual_tokens_pcp_padded = num_tokens * self.impl.pcp_size
        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.pcp_allgather_restore_idx = torch.tensor(
            [0, 3, 1, 2, 0, 0, 0, 0])

        key = torch.randn(num_tokens, num_heads, head_size)
        value = torch.randn(num_tokens, num_heads, head_size)

        def mock_all_gather_func(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        pcp_group = MagicMock(spec=GroupCoordinator)
        pcp_group.all_gather = mock_all_gather_func
        mock_get_pcp_group.return_value = pcp_group

        key, value = self.impl.reshape_and_cache(key, value, kv_cache,
                                                 attn_metadata)
        self.assertEqual(key.shape[0], num_tokens * self.impl.pcp_size)
        self.assertEqual(key.shape[1], num_heads)
        self.assertEqual(key.shape[2], head_size)
        self.assertEqual(value.shape[0], num_tokens * self.impl.pcp_size)
        self.assertEqual(value.shape[1], num_heads)
        self.assertEqual(value.shape[2], head_size)


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
        'vllm_ascend.attention.attention_cp.AscendAttentionCPImpl._npu_attn_out_lse_update'
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
        self.impl.head_size = 8
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
