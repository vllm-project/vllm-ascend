from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_cp import AscendMlaCPImpl


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

    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.all_to_all_single")
    def test_process_attn_out_lse(self, mock_all_to_all_single,
                                  mock_all_gather):
        self.impl.dcp_size = 2
        self.impl.pcp_size = 2

        B = 2
        N = self.impl.num_heads
        self.impl.kv_lora_rank = 512

        attn_output = torch.randn(B, N, self.impl.kv_lora_rank)
        softmax_lse = torch.randn(B, N, 1)

        mock_all_to_all_single.side_effect = lambda output, input, *args, **kwargs: output.copy_(
            input)

        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        decode_metadata = MagicMock()
        decode_metadata.actual_seq_lengths_q = MagicMock()
        decode_metadata.seq_lens_list = MagicMock()
        decode_metadata.batch_seq_mask = torch.tensor([True, False],
                                                      dtype=torch.bool)

        result = self.impl._process_attn_out_lse(attn_output, softmax_lse,
                                                 decode_metadata)

        self.assertEqual(result[0].shape[0], B)
        self.assertEqual(result[0].shape[1], N / self.impl.dcp_size)
        self.assertEqual(result[0].shape[2], self.impl.kv_lora_rank + 1)

    @patch("torch.distributed.all_gather")
    @patch("torch.distributed.all_to_all_single")
    @patch('vllm_ascend.attention.mla_cp.get_forward_context')
    @patch("torch_npu.atb.npu_multi_head_latent_attention")
    @patch('torch_npu.npu_attention_update')
    def test_forward_decode_pcp_dcp(self, mock_npu_attention_update,
                                    mock_npu_multi_head_latent_attention,
                                    mock_get_forward_context,
                                    mock_all_to_all_single, mock_all_gather):
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

        def mock_all_gather_func(tensor_list, tensor, group=None):
            tensor_list[0] = tensor
            tensor_list[1] = tensor.clone()

        mock_all_gather.side_effect = mock_all_gather_func

        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(
            B, self.impl.v_head_dim)

        result = self.impl._forward_decode_pcp_dcp(q_nope, q_pe, k_nope, k_pe,
                                                   BS, attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], self.impl.v_head_dim)
