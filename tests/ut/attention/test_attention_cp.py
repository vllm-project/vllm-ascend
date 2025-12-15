from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_cp import AscendAttentionCPImpl


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
