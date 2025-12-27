from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.common_cp import CPChunkedContextMetadata
from vllm_ascend.attention.mla_cp import AscendMlaCPImpl
from vllm_ascend.attention.mla_v1 import ChunkedContextMetadata


def get_pcp_split_info(pcp_rank, pcp_size, seq_lens):
    q_head_idx, q_tail_idx = [], []
    kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx = [], []
    kv_with_q_tail_nomask_idx, kv_with_q_tail_mask_idx = [], []
    chunk_seqlens = []
    kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = [], []
    q_req_offset = 0
    kv_req_offset = 0
    q_head_chunk_id = pcp_rank
    q_tail_chunk_id = pcp_size * 2 - 1 - pcp_rank
    for i, seq_len in enumerate(seq_lens):
        chunk_len = seq_len // 2
        chunk_seqlens.append(chunk_len)
        q_head_idx.extend(list(range(q_req_offset, q_req_offset + chunk_len)))
        kv_with_q_head_nomask_idx.extend(
            list(
                range(kv_req_offset,
                      kv_req_offset + chunk_len * q_head_chunk_id)))
        kv_with_q_head_mask_idx.extend(
            list(
                range(kv_req_offset + chunk_len * q_head_chunk_id,
                      kv_req_offset + chunk_len * (q_head_chunk_id + 1))))
        kv_with_q_head_nomask_seqlens.append(chunk_len * q_head_chunk_id)

        q_tail_idx.extend(
            list(range(q_req_offset + chunk_len,
                       q_req_offset + chunk_len * 2)))
        kv_with_q_tail_nomask_idx.extend(
            list(
                range(kv_req_offset,
                      kv_req_offset + chunk_len * q_tail_chunk_id)))
        kv_with_q_tail_mask_idx.extend(
            list(
                range(kv_req_offset + chunk_len * q_tail_chunk_id,
                      kv_req_offset + chunk_len * (q_tail_chunk_id + 1))))
        kv_with_q_tail_nomask_seqlens.append(chunk_len * q_tail_chunk_id)

        q_req_offset += seq_len
        kv_req_offset += seq_len * pcp_size
    return (
        torch.tensor(q_head_idx),
        torch.tensor(q_tail_idx),
        torch.tensor(kv_with_q_head_nomask_idx),
        torch.tensor(kv_with_q_head_mask_idx),
        torch.tensor(kv_with_q_tail_nomask_idx),
        torch.tensor(kv_with_q_tail_mask_idx),
        chunk_seqlens,
        kv_with_q_head_nomask_seqlens,
        kv_with_q_tail_nomask_seqlens,
    )


def get_chunk_metadata(pcp_size, dcp_size, num_prefills, num_decodes,
                       block_size, num_computed_tokens_cpu, num_reqs,
                       chunked_prefill_workspace_size,
                       num_computed_tokens_of_pcp_dcp, cp_local_block_size):
    reqs_start = num_decodes
    context_lens_cpu = num_computed_tokens_cpu[reqs_start:num_reqs]
    max_context_len_cpu = context_lens_cpu.max().item()
    num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
    max_context_chunk = (chunked_prefill_workspace_size //
                         num_prefills_with_context_cpu)
    max_context_chunk = max_context_chunk // block_size * block_size

    assert max_context_chunk > 0
    num_chunks = (max_context_len_cpu + max_context_chunk -
                  1) // max_context_chunk
    chunk_starts = torch.arange(num_chunks, dtype=torch.int32) \
                       .unsqueeze(1).expand(-1, num_prefills) * max_context_chunk
    chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                           chunk_starts + max_context_chunk)
    chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
    cu_seq_lens_cpu = torch.zeros(num_chunks,
                                  num_prefills + 1,
                                  dtype=torch.int32)
    torch.cumsum(chunk_seq_lens,
                 dim=1,
                 out=cu_seq_lens_cpu[:, 1:],
                 dtype=torch.int32)

    def cdiv(a, b):
        return (a + b - 1) // b

    if dcp_size * pcp_size > 1:
        if num_computed_tokens_of_pcp_dcp is not None:
            local_context_lens_allranks = torch.tensor(
                num_computed_tokens_of_pcp_dcp[reqs_start:num_reqs]).reshape(
                    -1, dcp_size * pcp_size)
        # Note(qcs): The max local context lengths
        # padded to `cp_local_block_size`.
        padded_local_context_lens_cpu = (cdiv(
            context_lens_cpu,
            cp_local_block_size * pcp_size * dcp_size,
        ) * cp_local_block_size)
        padded_local_max_context_chunk_across_ranks = (cdiv(
            max_context_chunk,
            cp_local_block_size * pcp_size * dcp_size,
        ) * cp_local_block_size)
        local_chunk_starts = (
            torch.arange(num_chunks, dtype=torch.int32).unsqueeze(1).expand(
                -1, num_prefills) *
            padded_local_max_context_chunk_across_ranks)
        local_chunk_ends = torch.min(
            padded_local_context_lens_cpu.unsqueeze(0),
            local_chunk_starts + padded_local_max_context_chunk_across_ranks,
        )
        padded_local_chunk_seq_lens = (local_chunk_ends -
                                       local_chunk_starts).clamp(min=0)
        padded_local_cu_chunk_seq_lens_cpu = torch.zeros(num_chunks,
                                                         num_prefills + 1,
                                                         dtype=torch.int32)
        torch.cumsum(
            padded_local_chunk_seq_lens,
            dim=1,
            out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
            dtype=torch.int32,
        )
        chunked_context_metadata = CPChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(non_blocking=True),
            starts=local_chunk_starts.to(non_blocking=True),
            seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            chunk_seq_lens=chunk_seq_lens,
            chunk_seq_lens_npu=chunk_seq_lens,
            workspace=None,
            padded_chunk_seq_lens_npu=padded_local_chunk_seq_lens,
            padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
            local_context_lens_allranks=local_context_lens_allranks.tolist(),
            padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                non_blocking=True),
            cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
            chunk_size=padded_local_max_context_chunk_across_ranks,
        )
    else:
        chunked_context_metadata = (ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens_cpu.to(non_blocking=True),
            starts=chunk_starts.to(non_blocking=True),
            seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
            max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
            chunk_seq_lens=chunk_seq_lens,
            chunk_seq_lens_npu=chunk_seq_lens,
            workspace=None,
        ))
    return chunked_context_metadata


class TestAscendMLAImpl(TestBase):

    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
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
              mock_tp, mock_get_dcp_size, mock_dcp, mock_pcp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()
        mock_dcp.world_size = 2
        mock_dcp.rank_in_group = MagicMock()
        mock_dcp.device_group = MagicMock()
        mock_pcp.world_size = 2
        mock_pcp.rank_in_group = MagicMock()
        mock_pcp.device_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        vllm_config.model_config = model_config
        get_current_vllm_config.return_value = vllm_config
        vllm_config.additional_config = {"refresh": True}
        init_ascend_config(vllm_config)

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

        self.impl = AscendMlaCPImpl(num_heads=num_heads,
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
        self.assertEqual(self.impl.pcp_size, 2)
        self.assertEqual(self.impl.dcp_size, 2)

    @patch('vllm_ascend.attention.mla_cp.get_dcp_group')
    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("vllm_ascend.attention.mla_cp.maybe_npu_prefetch")
    def test_mla_preprocess_dcp(self, magic_npu_fetch,
                                mock_maybe_all_gather_and_maybe_unpad,
                                mock_get_dcp_group):

        self.impl.num_kv_heads = 1
        self.impl.num_heads = 16
        self.impl.qk_rope_head_dim = 64
        self.impl.kv_lora_rank = 512
        self.impl.q_lora_rank = 1536
        self.impl.dcp_size = 2
        self.impl.pcp_size = 2
        block_num = 10
        block_size = 128
        batch_size = 2
        hidden_size = 1024
        hidden_states = torch.randn(batch_size, hidden_size)

        kv_cache0 = torch.randn(block_num, block_size, self.impl.num_kv_heads,
                                self.impl.kv_lora_rank)
        kv_cache1 = torch.randn(block_num, block_size, self.impl.num_kv_heads,
                                self.impl.qk_rope_head_dim)
        kv_cache = (kv_cache0, kv_cache1)

        mock_dcp_group = MagicMock()

        def mock_all_gather_func(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        mock_dcp_group.all_gather = mock_all_gather_func
        mock_get_dcp_group.return_value = mock_dcp_group

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_prefills = 0
        attn_metadata.num_prefill_tokens = 0
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_actual_tokens = 2
        attn_metadata.slot_mapping = torch.arange(4)
        attn_metadata.decode.cos = torch.randn(2, 64)
        attn_metadata.decode.sin = torch.randn(2, 64)

        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.q_lora_rank)
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(batch_size, self.impl.num_heads,
                        self.impl.qk_rope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.return_value = [
            torch.randn(
                attn_metadata.num_actual_tokens, self.impl.qk_rope_head_dim +
                self.impl.kv_lora_rank + self.impl.q_lora_rank)
        ]

        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]

        self.impl._q_proj_and_k_up_proj = MagicMock()
        self.impl._q_proj_and_k_up_proj.return_value = [
            torch.randn(attn_metadata.num_decodes, self.impl.num_heads,
                        self.impl.kv_lora_rank),
            torch.randn(attn_metadata.num_decodes, self.impl.num_heads,
                        self.impl.qk_rope_head_dim)
        ]

        magic_npu_fetch.return_value = MagicMock()
        mock_maybe_all_gather_and_maybe_unpad.side_effect = lambda x, label: x

        decode_res, prefill_res = self.impl._mla_preprocess(
            "mock_layer",
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv=False)

        self.assertIsNotNone(decode_res)
        self.assertIsNone(prefill_res)

    @patch('torch_npu._npu_reshape_and_cache')
    @patch('vllm_ascend.attention.mla_cp.get_pcp_group')
    @patch("torch.ops.vllm.maybe_all_gather_and_maybe_unpad")
    @patch("vllm_ascend.attention.mla_cp.maybe_npu_prefetch")
    def test_mla_preprocess_pcp(self, magic_npu_fetch,
                                mock_maybe_all_gather_and_maybe_unpad,
                                mock_get_pcp_group,
                                mock_npu_reshape_and_cache):
        self.impl.num_kv_heads = 1
        self.impl.num_heads = 16
        self.impl.qk_rope_head_dim = 64
        self.impl.kv_lora_rank = 512
        self.impl.q_lora_rank = 1536
        self.impl.dcp_size = 2
        self.impl.pcp_size = 2
        block_num = 10
        block_size = 128
        batch_size = 2
        hidden_size = 1024
        hidden_states = torch.randn(batch_size, hidden_size)

        kv_cache0 = torch.randn(block_num, block_size, self.impl.num_kv_heads,
                                self.impl.kv_lora_rank)
        kv_cache1 = torch.randn(block_num, block_size, self.impl.num_kv_heads,
                                self.impl.qk_rope_head_dim)
        kv_cache = (kv_cache0, kv_cache1)

        mock_pcp_group = MagicMock()

        def mock_all_gather_func(tensor, dim):
            return torch.cat([tensor, tensor], dim=dim)

        mock_pcp_group.all_gather = mock_all_gather_func
        mock_get_pcp_group.return_value = mock_pcp_group

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 0
        attn_metadata.num_prefills = 2
        attn_metadata.num_prefill_tokens = 2
        attn_metadata.num_decode_tokens = 0
        attn_metadata.num_actual_tokens = 2
        attn_metadata.num_actual_tokens_pcp_padded = 4
        attn_metadata.prefill.pcp_metadata = MagicMock()
        attn_metadata.prefill.pcp_metadata.pcp_allgather_restore_idx = torch.arange(
            4)
        attn_metadata.slot_mapping = torch.arange(4)
        attn_metadata.prefill.cos = torch.randn(2, 64)
        attn_metadata.prefill.sin = torch.randn(2, 64)

        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.q_lora_rank)
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(batch_size, self.impl.num_heads,
                        self.impl.qk_rope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.return_value = [
            torch.randn(
                attn_metadata.num_actual_tokens, self.impl.qk_rope_head_dim +
                self.impl.kv_lora_rank + self.impl.q_lora_rank)
        ]

        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]

        self.impl._q_proj_and_k_up_proj = MagicMock()
        self.impl._q_proj_and_k_up_proj.return_value = [
            torch.randn(attn_metadata.num_decodes, self.impl.num_heads,
                        self.impl.kv_lora_rank),
            torch.randn(attn_metadata.num_decodes, self.impl.num_heads,
                        self.impl.qk_rope_head_dim)
        ]

        magic_npu_fetch.return_value = MagicMock()
        mock_maybe_all_gather_and_maybe_unpad.side_effect = lambda x, label: x

        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.return_value = torch.randn(
            attn_metadata.num_prefill_tokens, self.impl.num_kv_heads,
            self.impl.kv_lora_rank)

        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [
            torch.randn(attn_metadata.num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_head_dim)
        ]
        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(attn_metadata.num_prefill_tokens * self.impl.pcp_size,
                        self.impl.num_heads,
                        self.impl.v_head_dim + self.impl.qk_nope_head_dim)
        ]
        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]
        self.impl.exec_kv_prefill = MagicMock()
        self.impl.exec_kv_prefill.return_value = [
            torch.randn(attn_metadata.num_prefill_tokens, self.impl.num_heads,
                        self.impl.qk_rope_head_dim),
            torch.randn(attn_metadata.num_prefill_tokens, self.impl.num_heads,
                        self.impl.kv_lora_rank)
        ]

        decode_res, prefill_res = self.impl._mla_preprocess(
            "mock_layer",
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv=False)
        self.assertIsNone(decode_res)
        self.assertIsNotNone(prefill_res)

    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("torch.distributed.all_to_all_single")
    def test_process_attn_out_lse(self, mock_all_to_all_single, mock_pcp):
        self.impl.dcp_size = 2
        self.impl.pcp_size = 2

        B = 2
        N = self.impl.num_heads
        self.impl.kv_lora_rank = 512

        attn_output = torch.randn(B, N, self.impl.kv_lora_rank)
        softmax_lse = torch.randn(B, N, 1)

        mock_all_to_all_single.side_effect = lambda output, input, *args, **kwargs: output.copy_(
            input)

        def make_all_gather(ws):
            return lambda tensor, dim: torch.cat([tensor] * ws, dim=dim)

        mock_pcp.all_gather = MagicMock(side_effect=make_all_gather(2))

        decode_metadata = MagicMock()
        decode_metadata.actual_seq_lengths_q = MagicMock()
        decode_metadata.seq_lens_list = MagicMock()
        decode_metadata.batch_seq_mask = torch.tensor([True, False],
                                                      dtype=torch.bool)

        result = self.impl._process_attn_out_lse(attn_output, softmax_lse,
                                                 decode_metadata)

        self.assertEqual(result.shape[0], B * self.impl.pcp_size)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], self.impl.kv_lora_rank + 1)

    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("torch.distributed.all_to_all_single")
    @patch('vllm_ascend.attention.mla_cp.get_forward_context')
    @patch("torch_npu.atb.npu_multi_head_latent_attention")
    @patch('torch_npu.npu_attention_update')
    def test_forward_decode_pcp_dcp(self, mock_npu_attention_update,
                                    mock_npu_multi_head_latent_attention,
                                    mock_get_forward_context,
                                    mock_all_to_all_single, mock_pcp):
        self.impl.dcp_size = 2
        self.impl.pcp_size = 2
        self.impl.num_kv_heads = 1
        self.impl.num_heads = 16
        self.impl.kv_lora_rank = 64
        self.impl.qk_nope_head_dim = 64
        self.impl.spec_token_num = 1
        B = 2
        N = self.impl.num_heads * self.impl.dcp_size
        BS = 128
        NB = 100

        q_nope = torch.randn(B, N, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(B, N, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(NB, BS, 1, self.impl.kv_lora_rank)
        k_pe = torch.randn(NB, BS, 1, self.impl.qk_rope_head_dim)

        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_lengths_q = MagicMock()
        attn_metadata.decode.seq_lens_list = MagicMock()
        attn_metadata.decode.batch_seq_mask = torch.tensor([False, False],
                                                           dtype=torch.bool)

        self.impl.enable_kv_nz = True

        mock_npu_attention_update.return_value = (torch.randn(
            B, self.impl.num_heads, self.impl.kv_lora_rank), None)
        mock_npu_multi_head_latent_attention.return_value = [
            torch.randn(B, N, self.impl.kv_lora_rank),
            torch.randn(B, N, 1)
        ]
        mock_get_forward_context.return_value = MagicMock(capturing=False)

        mock_all_to_all_single.side_effect = lambda output, input, *args, **kwargs: output.copy_(
            input)

        def make_all_gather(ws):
            return lambda tensor, dim: torch.cat([tensor] * ws, dim=dim)

        mock_pcp.all_gather = MagicMock(side_effect=make_all_gather(2))

        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(
            B, self.impl.v_head_dim)

        result = self.impl._forward_decode_pcp_dcp(q_nope, q_pe, k_nope, k_pe,
                                                   BS, attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], self.impl.v_head_dim)

    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("torch_npu.atb.npu_paged_cache_load")
    @patch("torch_npu.atb.npu_ring_mla")
    def test_compute_prefill_context_with_dcp_pcp(self, mock_ring, mock_load,
                                                  mock_dcp, mock_pcp):

        def mock_all_gather(ws):
            return lambda tensor, dim: torch.cat([tensor] * ws, dim=dim)

        def mock_ring_attn(q_nope, q_rope, k_nope, k_rope, value, mask, seqlen,
                           head_num, kv_head_num, pre_out, prev_lse, qk_scale,
                           kernel_type, mask_type, input_layout, calc_type,
                           output, softmax_lse):
            return torch.randn(q_rope.shape[0], value.shape[1], value.shape[2])

        mock_ring.side_effect = mock_ring_attn

        def mock_kv_b_proj(kv_c_normed):
            return (torch.randn(kv_c_normed.shape[0],
                                self.impl.num_heads,
                                self.impl.v_head_dim +
                                self.impl.qk_nope_head_dim,
                                dtype=torch.float16), )

        def mock_reorg_kvcache(allgatered_kv_c_normed: torch.Tensor,
                               allgatered_k_pe: torch.Tensor,
                               padded_local_chunk_seq_lens_lst: list[int],
                               local_context_lens_allranks: list[list[int]],
                               sum_seq_len: int, max_seq_len: int,
                               chunk_size: int, chunk_idx: int, toks: int):
            return torch.randn(sum_seq_len, allgatered_kv_c_normed.shape[1],
                               allgatered_kv_c_normed.shape[2]), torch.randn(
                                   sum_seq_len, allgatered_k_pe.shape[1],
                                   allgatered_k_pe.shape[2])

        # mock proj
        self.impl.kv_b_proj.side_effect = mock_kv_b_proj
        NUM_BLOCKS, BLOCK_SIZE = 10, 32  # fixed
        USED_BLOCKS = 3
        # pcp_size, dcp_size, nums_tokens_per_rank, nums_all_rank_context, num_prefills, num_decodes, num_seqs, cp_local_block_size, num_computed_tokens, num_computed_tokens_of_pcp_dcp
        test_cases = [
            (2, 2, [4], [128], 1, 0, 1, 1, [[[32, 32], [32, 32]]]),
            (1, 2, [4], [128], 1, 0, 1, 1, [[[64, 64]]]),
            (2, 1, [4], [128], 1, 0, 1, 1, [[[64], [64]]]),
            (2, 2, [4, 7], [128, 128], 2, 0, 2, 1, [[[32, 32], [32, 32]],
                                                    [[32, 32], [32, 32]]]),
        ]
        # kv cache tensor
        kv_cache_0 = torch.randn(NUM_BLOCKS,
                                 BLOCK_SIZE,
                                 self.impl.num_heads,
                                 self.impl.kv_lora_rank,
                                 dtype=torch.float16)
        kv_cache_1 = torch.randn(NUM_BLOCKS,
                                 BLOCK_SIZE,
                                 self.impl.num_heads,
                                 self.impl.v_head_dim,
                                 dtype=torch.float16)
        kv_cache = [kv_cache_0, kv_cache_1]
        max_model_len = 4096
        max_num_seqs = 25
        # create chunk context
        chunked_prefill_workspace_size = min(
            max(8 * max_model_len, 4 * max_num_seqs * BLOCK_SIZE), 128 * 1024)
        self.impl.prefill_mask = torch.triu(
            torch.ones(10, 10, dtype=torch.float16), 1)
        for test_case in test_cases:
            pcp_size, dcp_size, nums_tokens_per_rank, nums_all_rank_context, num_prefills, num_decodes, num_seqs, cp_local_block_size, num_computed_tokens_of_pcp_dcp = test_case
            mock_dcp.all_gather = MagicMock(
                side_effect=mock_all_gather(dcp_size))
            mock_pcp.all_gather = MagicMock(
                side_effect=mock_all_gather(pcp_size))
            assert len(nums_tokens_per_rank) == len(nums_all_rank_context)
            nums_context_per_rank = []
            for num_all_rank_context in nums_all_rank_context:
                assert num_all_rank_context % (pcp_size * dcp_size) == 0
                nums_context_per_rank.append(num_all_rank_context //
                                             (pcp_size * dcp_size))
            self.impl.dcp_size = dcp_size
            self.impl.pcp_size = pcp_size
            # create input
            query = torch.randn(sum(nums_tokens_per_rank),
                                self.impl.num_heads,
                                self.impl.qk_head_dim,
                                dtype=torch.float16)
            q_nope = query[..., :self.impl.qk_nope_head_dim]
            q_pe = query[..., self.impl.qk_nope_head_dim:]
            prefix_out = torch.randn(sum(nums_tokens_per_rank),
                                     self.impl.num_heads,
                                     self.impl.v_head_dim,
                                     dtype=torch.float16)
            prefix_lse = torch.randn(sum(nums_tokens_per_rank),
                                     self.impl.num_heads,
                                     dtype=torch.float16)
            chunk_ctx = get_chunk_metadata(
                pcp_size,
                dcp_size,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
                block_size=BLOCK_SIZE,
                num_computed_tokens_cpu=torch.tensor(nums_all_rank_context),
                num_reqs=num_seqs,
                chunked_prefill_workspace_size=chunked_prefill_workspace_size,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp,
                cp_local_block_size=cp_local_block_size)
            meta = MagicMock()
            prefill_meta = MagicMock()
            prefill_meta.query_lens = nums_tokens_per_rank
            prefill_meta.block_table = torch.randint(
                0, USED_BLOCKS, (1, 64))  # (batch, max_blocks)
            prefill_meta.chunked_context = chunk_ctx
            meta.prefill = prefill_meta

            with patch.object(self.impl, '_reorg_kvcache') as mock_reorg:
                mock_reorg.side_effect = mock_reorg_kvcache

                out, lse = self.impl._compute_prefill_context(
                    q_nope, q_pe, kv_cache, self.impl.qk_rope_head_dim, meta,
                    prefix_out, prefix_lse)

            iters = len(chunk_ctx.seq_tot)
            self.impl.dcp_size = 1
            self.impl.pcp_size = 1
            self.assertEqual(mock_reorg.call_count,
                             iters * (1 if dcp_size * pcp_size > 1 else 0))
            self.assertEqual(mock_load.call_count, iters)
            self.assertEqual(mock_ring.call_count, iters)
            self.assertEqual(mock_dcp.all_gather.call_count,
                             (1 if dcp_size > 1 else 0))
            self.assertEqual(mock_pcp.all_gather.call_count,
                             iters * (1 if pcp_size > 1 else 0))
            mock_reorg.reset_mock()
            mock_load.reset_mock()
            mock_ring.reset_mock()
            mock_dcp.reset_mock()
            mock_pcp.reset_mock()
            self.assertEqual(out.shape, prefix_out.shape)
            self.assertEqual(lse.shape, prefix_lse.shape)

    def test_reorg_kvcache_with_dcp_pcp(self):
        BLOCK_SIZE = 128  # fixed
        max_model_len = 4096
        max_num_seqs = 25
        test_cases = [
            (2, 2, [4], [128], 1, 0, 1, 1, [[[32, 32], [32, 32]]]),
            (1, 2, [4], [128], 1, 0, 1, 1, [[[64, 64]]]),
            (2, 1, [4], [128], 1, 0, 1, 1, [[[64], [64]]]),
            (2, 2, [4, 7], [128, 128], 2, 0, 2, 1, [[[32, 32], [32, 32]],
                                                    [[32, 32], [32, 32]]]),
        ]
        for test_case in test_cases:
            pcp_size, dcp_size, nums_tokens_per_rank, nums_all_rank_context, num_prefills, num_decodes, num_seqs, cp_local_block_size, num_computed_tokens_of_pcp_dcp = test_case
            if pcp_size * dcp_size == 1:
                continue
            chunked_prefill_workspace_size = min(
                max(8 * max_model_len, 4 * max_num_seqs * BLOCK_SIZE),
                128 * 1024)
            chunked_context = get_chunk_metadata(
                pcp_size,
                dcp_size,
                num_prefills=num_prefills,
                num_decodes=num_decodes,
                block_size=BLOCK_SIZE,
                num_computed_tokens_cpu=torch.tensor(nums_all_rank_context),
                num_reqs=num_seqs,
                chunked_prefill_workspace_size=chunked_prefill_workspace_size,
                num_computed_tokens_of_pcp_dcp=num_computed_tokens_of_pcp_dcp,
                cp_local_block_size=cp_local_block_size)

            for i in range(len(chunked_context.seq_tot)):
                allgatered_kv_c_normed = torch.randn(
                    chunked_context.seq_tot[i] * pcp_size * dcp_size,
                    self.impl.num_heads, self.impl.v_head_dim)
                allgatered_k_pe = torch.randn(
                    chunked_context.seq_tot[i] * pcp_size * dcp_size,
                    self.impl.num_heads, self.impl.qk_rope_head_dim)
                result_kv, result_k_pe = self.impl._reorg_kvcache(
                    allgatered_kv_c_normed,
                    allgatered_k_pe,
                    padded_local_chunk_seq_lens_lst=chunked_context.
                    padded_local_chunk_seq_lens[i],
                    local_context_lens_allranks=chunked_context.
                    local_context_lens_allranks,
                    sum_seq_len=chunked_context.cu_seq_lens_lst[i][-1],
                    max_seq_len=chunked_context.max_seq_lens[i],
                    chunk_size=chunked_context.chunk_size,
                    chunk_idx=i,
                    toks=chunked_context.seq_tot[i],
                )
                self.assertEqual(result_kv.shape,
                                 (chunked_context.cu_seq_lens_lst[i][-1],
                                  self.impl.num_heads, self.impl.v_head_dim))
                self.assertEqual(
                    result_k_pe.shape,
                    (chunked_context.cu_seq_lens_lst[i][-1],
                     self.impl.num_heads, self.impl.qk_rope_head_dim))

                self.assertEqual(result_kv.shape[0],
                                 chunked_context.cu_seq_lens_lst[i][-1])
                self.assertEqual(result_k_pe.shape[0],
                                 chunked_context.cu_seq_lens_lst[i][-1])

    def test_out_lse_reshape(self):
        test_cases = [10, 1, 128, 512]
        for test_case in test_cases:
            num_tokens = test_case
            num_heads, head_dim = self.impl.num_heads, self.impl.v_head_dim
            attn_out = torch.randn(num_tokens, num_heads, head_dim)
            attn_lse = torch.randn(num_tokens, num_heads, 1)

            out, lse = self.impl._out_lse_reshape(attn_out, attn_lse)

            assert out.shape == (num_tokens * num_heads, head_dim)
            assert out.is_contiguous()

            assert lse.shape == (num_tokens * num_heads, )
            assert lse.is_contiguous()

            expected_out = attn_out.contiguous().view(-1, head_dim)
            expected_lse = attn_lse.contiguous().view(-1)

            assert torch.allclose(out, expected_out)
            assert torch.allclose(lse, expected_lse)

    @patch('torch_npu.npu_attention_update')
    def test_npu_attention_update_with_dcp_pcp(self,
                                               mock_npu_attention_update):
        NUM_TOKENS = 10  # fixed
        test_cases = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)]
        for test_case in test_cases:
            self.impl.dcp_size = test_case[0]
            self.impl.pcp_size = test_case[1]
            num_heads, head_dim = self.impl.num_heads, self.impl.kv_lora_rank + 1

            def mock_out_lse_reshape(attn_out, attn_lse):
                attn_out = attn_out.contiguous().view(
                    attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
                attn_lse = attn_lse.contiguous().view(
                    attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
                return attn_out, attn_lse

            self.impl._out_lse_reshape = MagicMock()
            self.impl._out_lse_reshape.side_effect = mock_out_lse_reshape

            def mock_npu_attention_update_effect(attn_lse_split_cp,
                                                 attn_out_split_cp,
                                                 update_type):
                return torch.randn_like(
                    attn_out_split_cp[0]), torch.randn_like(
                        attn_lse_split_cp[0])

            mock_npu_attention_update.side_effect = mock_npu_attention_update_effect
            attn_out_lse = torch.randn(self.impl.pcp_size * NUM_TOKENS,
                                       self.impl.dcp_size * num_heads,
                                       head_dim)
            out = self.impl._npu_attention_update(attn_out_lse)
            self.impl.dcp_size = 1
            self.impl.pcp_size = 1
            assert out.shape == (NUM_TOKENS, num_heads, self.impl.kv_lora_rank)

    @patch('torch_npu.atb.npu_ring_mla')
    def test_attention_with_mask_and_nomask_with_dcp_pcp(
            self, mock_npu_ring_mla):
        num_heads = self.impl.num_heads
        v_head_dim = self.impl.v_head_dim
        qk_nope_head_dim = self.impl.qk_nope_head_dim
        qk_rope_head_dim = self.impl.qk_rope_head_dim

        def mock_npu_ring_mla_effect(q_nope, q_rope, k_nope, k_rope, value,
                                     mask, seqlen, head_num, kv_head_num,
                                     pre_out, prev_lse, qk_scale, kernel_type,
                                     mask_type, input_layout, calc_type,
                                     output, softmax_lse):

            return torch.randn(q_nope.shape[0], value.shape[1],
                               value.shape[-1])

        mock_npu_ring_mla.side_effect = mock_npu_ring_mla_effect
        test_cases = [([8], 2, 2), ([8], 2, 1), ([8], 1, 2), ([8], 2, 2),
                      ([8, 12], 2, 2)]
        for test_case in test_cases:
            scheduled_tokens, pcp_size, dcp_size = test_case
            nums_tokens_per_rank = []
            for num_tokens in scheduled_tokens:
                assert num_tokens % (2 * pcp_size) == 0
                nums_tokens_per_rank.append(num_tokens // pcp_size)
            seq_len_q, seq_len_k = sum(nums_tokens_per_rank), sum(
                scheduled_tokens)
            q_nope = torch.randn(seq_len_q,
                                 num_heads,
                                 qk_nope_head_dim,
                                 dtype=torch.float16)
            q_pe = torch.randn(seq_len_q,
                               num_heads,
                               qk_rope_head_dim,
                               dtype=torch.float16)
            k_nope = torch.randn(seq_len_k,
                                 num_heads,
                                 qk_nope_head_dim,
                                 dtype=torch.float16)
            k_pe = torch.randn(seq_len_k,
                               num_heads,
                               qk_rope_head_dim,
                               dtype=torch.float16)
            value = torch.randn(seq_len_k,
                                num_heads,
                                v_head_dim,
                                dtype=torch.float16)
            mask = torch.triu(torch.ones(10, 10, dtype=torch.float16), 1)
            for rank in range(pcp_size):
                q_head_idx, q_tail_idx, kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx, kv_with_q_tail_nomask_idx, \
                    kv_with_q_tail_mask_idx, chunk_seqlens, kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = get_pcp_split_info(
                    rank, pcp_size, nums_tokens_per_rank)

                output_head, lse_head = self.impl._attention_with_mask_and_nomask(
                    q_nope=torch.index_select(q_nope, 0, q_head_idx),
                    q_pe=torch.index_select(q_pe, 0, q_head_idx),
                    k_nope=k_nope,
                    k_pe=k_pe,
                    value=value,
                    kv_mask_idx=kv_with_q_head_mask_idx,
                    kv_nomask_idx=kv_with_q_head_nomask_idx,
                    attn_mask_seqlens=torch.tensor(
                        [chunk_seqlens, chunk_seqlens], dtype=torch.int32),
                    attn_nomask_seqlens=kv_with_q_head_nomask_seqlens,
                    mask=mask)
                self.assertEqual(output_head.shape,
                                 (q_head_idx.shape[0], num_heads, v_head_dim))
                self.assertEqual(lse_head.shape,
                                 (num_heads, q_head_idx.shape[0]))
                self.assertEqual(mock_npu_ring_mla.call_count,
                                 1 + (kv_with_q_head_nomask_idx.shape[0] != 0))
                mock_npu_ring_mla.reset_mock()
                output_tail, lse_tail = self.impl._attention_with_mask_and_nomask(
                    q_nope=torch.index_select(q_nope, 0, q_tail_idx),
                    q_pe=torch.index_select(q_pe, 0, q_tail_idx),
                    k_nope=k_nope,
                    k_pe=k_pe,
                    value=value,
                    kv_mask_idx=kv_with_q_tail_mask_idx,
                    kv_nomask_idx=kv_with_q_tail_nomask_idx,
                    attn_mask_seqlens=torch.tensor(
                        [chunk_seqlens, chunk_seqlens], dtype=torch.int32),
                    attn_nomask_seqlens=kv_with_q_tail_nomask_seqlens,
                    mask=mask)

                self.assertEqual(output_tail.shape,
                                 (q_tail_idx.shape[0], num_heads, v_head_dim))
                self.assertEqual(lse_tail.shape,
                                 (num_heads, q_tail_idx.shape[0]))
                self.assertEqual(mock_npu_ring_mla.call_count,
                                 1 + (kv_with_q_tail_nomask_idx.shape[0] != 0))
                mock_npu_ring_mla.reset_mock()

    @patch("torch.distributed.all_to_all_single")
    @patch('vllm.distributed.parallel_state._PCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_process_attn_out_lse_with_dcp_pcp(self, mock_pcp,
                                               mock_all_to_all):
        B, H, D = 4, self.impl.num_heads, self.impl.v_head_dim  # total: [4, 4, 8]
        test_cases = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4)]
        for test_case in test_cases:
            print(test_case)
            self.impl.dcp_size = test_case[0]
            self.impl.pcp_size = test_case[1]
            # Inputs
            attn_output = torch.randn(B, H, D)
            softmax_lse = torch.randn(B, H, 1)
            batch_seq_mask = torch.tensor([False, True, False, False])  # [B]
            decode_meta = MagicMock()
            decode_meta.batch_seq_mask = batch_seq_mask

            def mock_all_to_all_side_effect(output, input, group=None):
                output.copy_(input)

            mock_all_to_all.side_effect = mock_all_to_all_side_effect

            def mock_all_gather(ws):
                return lambda tensor, dim: torch.cat([tensor] * ws, dim=dim)

            mock_pcp.all_gather = MagicMock(
                side_effect=mock_all_gather(self.impl.pcp_size))

            result = self.impl._process_attn_out_lse(attn_output, softmax_lse,
                                                     decode_meta)
            # [PCP * S, DCP * H, D + 1]
            self.assertIsInstance(result, torch.Tensor)
            assert result.shape == (B * self.impl.pcp_size, H, D + 1)
            self.impl.dcp_size = 1
            self.impl.pcp_size = 1

    @patch('torch_npu.atb.npu_ring_mla')
    def test_forward_prefill_cp_with_dcp_pcp(self, mock_npu_ring_mla):

        def mock_attention_with_nomask_and_mask(
                q_nope: torch.Tensor, q_pe: torch.Tensor, k_nope: torch.Tensor,
                k_pe: torch.Tensor, value: torch.Tensor,
                kv_mask_idx: torch.Tensor, kv_nomask_idx: torch.Tensor,
                attn_mask_seqlens: torch.Tensor,
                attn_nomask_seqlens: torch.Tensor, mask: torch.Tensor):
            mock_output = torch.randn(q_nope.shape[0],
                                      self.impl.num_heads,
                                      self.impl.v_head_dim,
                                      dtype=k_pe.dtype,
                                      device=k_pe.device)
            mock_lse = torch.randn(self.impl.num_heads,
                                   q_pe.shape[0],
                                   dtype=torch.float32,
                                   device=k_pe.device)
            return mock_output, mock_lse

        def mock_compute_prefill_context(q_nope, q_pe, kv_c_and_k_pe_cache,
                                         rope_dim, attn_metadata,
                                         prefix_output, prefix_lse):
            mock_output = torch.randn_like(prefix_output)
            mock_lse = torch.randn_like(prefix_lse)
            return mock_output, mock_lse

        def mock_npu_ring_mla_effect(q_nope, q_rope, k_nope, k_rope, value,
                                     mask, seqlen, head_num, kv_head_num,
                                     pre_out, prev_lse, qk_scale, kernel_type,
                                     mask_type, input_layout, calc_type,
                                     output, softmax_lse):
            return torch.randn(q_nope.shape[0], value.shape[1],
                               value.shape[-1])

        self.impl._attention_with_mask_and_nomask = MagicMock()
        self.impl._attention_with_mask_and_nomask.side_effect = mock_attention_with_nomask_and_mask
        self.impl._compute_prefill_context = MagicMock()
        self.impl._compute_prefill_context.side_effect = mock_compute_prefill_context
        mock_npu_ring_mla.side_effect = mock_npu_ring_mla_effect
        block_num = 10
        block_size = 32
        kv_c_and_k_pe_cache = (torch.randn(block_num,
                                           block_size,
                                           1,
                                           self.impl.q_lora_rank,
                                           dtype=torch.float16),
                               torch.randn(block_num,
                                           block_size,
                                           1,
                                           self.impl.qk_rope_head_dim,
                                           dtype=torch.float16))
        test_cases = [([8], 2, 2), ([8], 2, 1), ([8], 1, 2), ([8], 2, 2),
                      ([8, 16], 2, 2)]
        for test_case in test_cases:
            scheduled_tokens, pcp_size, dcp_size = test_case
            nums_tokens_per_rank = []
            for num_tokens in scheduled_tokens:
                assert num_tokens % (
                    2 * pcp_size) == 0  # padded head&tail compute balance
                nums_tokens_per_rank.append(num_tokens // pcp_size)
            seq_len_q, seq_len_k = sum(nums_tokens_per_rank), sum(
                scheduled_tokens)

            q_nope = torch.randn(seq_len_q,
                                 self.impl.num_heads,
                                 self.impl.qk_nope_head_dim,
                                 dtype=torch.float16)
            q_pe = torch.randn(seq_len_q,
                               self.impl.num_heads,
                               self.impl.qk_rope_head_dim,
                               dtype=torch.float16)
            k_nope = torch.randn(seq_len_k,
                                 self.impl.num_heads,
                                 self.impl.qk_nope_head_dim,
                                 dtype=torch.float16)
            k_pe = torch.randn(seq_len_k,
                               self.impl.num_heads,
                               self.impl.qk_rope_head_dim,
                               dtype=torch.float16)
            value = torch.randn(seq_len_k,
                                self.impl.num_heads,
                                self.impl.v_head_dim,
                                dtype=torch.float16)
            # only test one rank
            for rank in range(pcp_size):
                q_head_idx, q_tail_idx, kv_with_q_head_nomask_idx, kv_with_q_head_mask_idx, kv_with_q_tail_nomask_idx, \
                    kv_with_q_tail_mask_idx, chunk_seqlens, kv_with_q_head_nomask_seqlens, kv_with_q_tail_nomask_seqlens = get_pcp_split_info(
                    rank, pcp_size, nums_tokens_per_rank)
                attn_metadata = MagicMock()
                attn_metadata.prefill = MagicMock()
                attn_metadata.prefill.pcp_metadata.q_head_idx = q_head_idx
                attn_metadata.prefill.pcp_metadata.q_tail_idx = q_tail_idx
                attn_metadata.prefill.pcp_metadata.q_full_idx = torch.cat([
                    attn_metadata.prefill.pcp_metadata.q_head_idx,
                    attn_metadata.prefill.pcp_metadata.q_tail_idx
                ])
                attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx = kv_with_q_head_nomask_idx

                attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx = kv_with_q_head_mask_idx
                attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx = kv_with_q_tail_nomask_idx
                attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx = kv_with_q_tail_mask_idx
                attn_metadata.prefill.pcp_metadata.attn_mask_seqlens = torch.tensor(
                    [chunk_seqlens, chunk_seqlens], dtype=torch.int32)
                attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens = kv_with_q_head_nomask_seqlens
                attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens = kv_with_q_tail_nomask_seqlens
                attn_metadata.prefill.pcp_metadata.pcp_prefill_mask = torch.triu(
                    torch.ones(10, 10, dtype=torch.float16), 1)

                output = self.impl._forward_prefill_cp(q_nope, q_pe, k_nope,
                                                       k_pe, value,
                                                       kv_c_and_k_pe_cache,
                                                       attn_metadata)
                self.assertEqual(
                    output.shape,
                    (seq_len_q, self.impl.num_heads * self.impl.v_head_dim))
