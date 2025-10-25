from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase


from tests.ut.base import TestBase
from types import SimpleNamespace
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import (AscendSFABackend,
                                          AscendSFADecodeMetadata,
                                          AscendSFAImpl, AscendSFAMetadata,
                                          AscendSFAMetadataBuilder,
                                          AscendSFAPrefillMetadata)


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(TestAscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_metadata_cls(self):
        self.assertEqual(TestAscendSFABackend.get_metadata_cls(),
                         AscendSFAMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(TestAscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = TestAscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = TestAscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class AscendSFAPrefillMetadata(TestBase):

    def test_ascend_sfa_prefill_metadata_default(self):
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendSFAPrefillMetadata(attn_mask=attn_mask,
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

    def test_ascend_sfa_prefill_metadata_with_chunked_context(self):
        cu_seq_lens = torch.tensor([0, 2, 4])
        starts = torch.tensor([0, 2])
        seq_tot = [2, 2]
        max_seq_lens = [2, 2]
        workspace = torch.randn(2, 4)
        chunk_seq_lens = torch.tensor([2, 2])

        chunked_context = AscendSFAPrefillMetadata.ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens,
            starts=starts,
            seq_tot=seq_tot,
            max_seq_lens=max_seq_lens,
            workspace=workspace,
            chunk_seq_lens=chunk_seq_lens)

        metadata = AscendSFAPrefillMetadata(
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


class TestAscendSFADecodeMetadata(TestBase):

    def test_ascend_sfa_decode_metadata_default(self):
        input_positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        block_table = torch.tensor([[0, 3, 2, 1], [0, 2, 1, 3]])
        seq_lens = torch.tensor([[2], [3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None

        metadata = AscendSFADecodeMetadata(input_positions, block_table,
                                           seq_lens, max_seq_lens,
                                           seq_lens_list, attn_mask)

        self.assertIs(metadata.input_positions, input_positions)
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
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
        enable_dbo_across_dp = False

        metadata = AscendSFAMetadata(
            num_actual_tokens, slot_mapping,
            query_start_loc, seq_lens, block_tables, num_decodes,
            num_decode_tokens, num_prefills, num_input_tokens, 
            query_lens, head_dim, attn_mask, attn_state, decode, 
            prefill,enable_dbo_across_dp)

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
        self.assertEqual(metadata.enable_dbo_across_dp, enable_dbo_across_dp)


class TestAscendSFAMetadataBuilder(TestBase):

    def test_ascend_sfa_metadata_builder_default(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_vllm_config.speculative_config = None

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.sfa_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendSFAMetadataBuilder(None, None, mock_vllm_config,
                                               mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.chunked_prefill_enabled)

    def test_ascend_sfa_metadata_builder_spec_decode(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_spec_config = MagicMock()
        mock_spec_config.num_speculative_tokens = 3
        mock_vllm_config.speculative_config = mock_spec_config

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.sfa_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendSFAMetadataBuilder(None, None, mock_vllm_config,
                                               mock_device)

            self.assertEqual(builder.block_size,
                             mock_vllm_config.cache_config.block_size)
            self.assertEqual(
                builder.chunked_prefill_enabled,
                mock_vllm_config.scheduler_config.chunked_prefill_enabled)

    def test_reorder_batch(self):
        ascend_config = MagicMock()

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_device = 'cpu'

        mock_vllm_config.speculative_config = None

        with patch("vllm_ascend.attention.sfa_v1.get_ascend_config",
                   return_value=ascend_config):
            builder = AscendSFAMetadataBuilder(None, None, mock_vllm_config,
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


class TestAscendSFAImpl(TestBase):

    @patch('vllm.distributed.parallel_state._DCP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_decode_context_model_parallel_world_size",
           return_value=1)
    @patch('vllm.distributed.parallel_state._TP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm.distributed.get_tensor_model_parallel_world_size",
           return_value=2)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_ascend_config")
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

        self.impl = AscendSFAImpl(num_heads=num_heads,
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

    @patch("torch_npu.npu_interleave_rope", side_effect=lambda x, cos, sin: x)
    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_sfa_preprocess(self, mock_kv_rmsnorm_rope_cache, _mock_interleave):
        self.impl.num_heads = 8
        self.impl.num_kv_heads = 8
        self.impl.q_lora_rank = 64
        self.impl.kv_lora_rank = 32
        self.impl.qk_nope_head_dim = 64
        self.impl.qk_rope_head_dim = 64
        self.impl.qk_head_dim = self.impl.qk_nope_head_dim + self.impl.qk_rope_head_dim  # 128
        self.impl.v_head_dim = 128

        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight = torch.randn(self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.kv_a_layernorm.variance_epsilon = 1e-6

        def fake_q_b_proj(x):
            T = x.shape[0]
            return (torch.randn(T, self.impl.num_heads * self.impl.qk_head_dim),)
        self.impl.q_b_proj = MagicMock(side_effect=fake_q_b_proj)

        self.impl.kv_b_proj_w_k = torch.randn(self.impl.qk_nope_head_dim, self.impl.kv_lora_rank)

        def fake_fused_qkv(x):
            T = x.shape[0]
            last = self.impl.q_lora_rank + (self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
            return [torch.randn(T, self.impl.num_heads, last)]
        self.impl.fused_qkv_a_proj = MagicMock(side_effect=fake_fused_qkv)

        self.impl.q_a_layernorm = MagicMock(side_effect=lambda x: x)

        self.impl.indexer_select = MagicMock(return_value=torch.tensor([0], dtype=torch.int64))

        def fake_kv_rmsnorm_rope_cache(kv_no_split, weight, cos, sin, slots, v_cache, k_cache, **kw):
            B = kv_no_split.shape[0]
            k_rope = torch.randn(B, 1, 1, self.impl.qk_rope_head_dim)
            k_nope = torch.randn(B, 1, 1, self.impl.kv_lora_rank)
            return (k_rope, k_nope, None, None)
        mock_kv_rmsnorm_rope_cache.side_effect = fake_kv_rmsnorm_rope_cache

        batch_size, seq_len, hidden_size = 4, 8, 1024
        hidden_states = torch.randn(batch_size * seq_len, hidden_size)

        kv_cache = (torch.empty(1), torch.empty(1))

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_prefills = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_actual_tokens = 4
        attn_metadata.slot_mapping = torch.arange(4, dtype=torch.long)
        
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.cos=torch.randn(
                                attn_metadata.num_decode_tokens, 
                                self.impl.qk_rope_head_dim)
        attn_metadata.decode.sin=torch.randn(attn_metadata.num_decode_tokens, self.impl.qk_rope_head_dim)

        attn_metadata.prefill = MagicMock()
        attn_metadata.prefill.cos=torch.randn(attn_metadata.num_prefills, self.impl.qk_rope_head_dim)
        attn_metadata.prefill.sin=torch.randn(attn_metadata.num_prefills, self.impl.qk_rope_head_dim)

        decode_res, prefill_res = self.impl._sfa_preprocess(
            hidden_states, kv_cache, attn_metadata, need_gather_q_kv=False
        )

        self.assertIsNotNone(decode_res)
        self.assertIsNotNone(prefill_res)
        self.assertTrue(hasattr(decode_res, "q_nope") and hasattr(decode_res, "q_pe"))
        self.assertTrue(hasattr(prefill_res, "k_nope") and hasattr(prefill_res, "k_pe"))
   
    def _mk_prefill_meta(self, T: int):
        return SimpleNamespace(
            num_actual_tokens=T,
            num_decode_tokens=0,
            num_decodes=0,
            num_prefills=T,
            decode=None,
            prefill=SimpleNamespace(
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                query_lens=[1] * T,
                seq_lens=[1] * T,
            ),
        )

    def _mk_decode_meta(self, T: int):
        return SimpleNamespace(
            num_actual_tokens=T,
            num_decode_tokens=T,
            num_decodes=T,
            num_prefills=0,
            prefill=None,
            decode=SimpleNamespace(
                block_table=torch.zeros(1, 1, dtype=torch.int32),
                actual_seq_lengths_q=torch.tensor([1] * T, dtype=torch.int32),
                seq_lens=torch.tensor([1] * T, dtype=torch.int32),
            ),
        )

    def _mk_preprocess_res(self, T: int):
        N = self.impl.num_heads
        L = 16  
        q_nope = torch.randn(T, N, L)
        q_pe   = torch.randn(T, N, 64)
        k_nope = torch.randn(T, N, L)
        k_rope = torch.randn(T, N, 64)
        topk   = torch.zeros(T, 1, dtype=torch.int32)
        return SimpleNamespace(
            query_states=(q_nope, q_pe),
            key_states=(k_nope, k_rope),
            topk_indices=topk,
        )

    @patch.object(AscendSFAImpl, "apply_attention_fusion")
    @patch.object(AscendSFAImpl, "_sfa_preprocess")
    def test_forward_prefill_only(self, mock_preproc, mock_fuse):
        T = 6
        mock_preproc.return_value = (None, self._mk_preprocess_res(T))
        mock_fuse.side_effect = lambda **kwargs: torch.randn(
            T, self.impl.num_heads * self.impl.v_head_dim
        )
        self.impl.mla_epilog = lambda x, absorb=True: x

        hidden_states = torch.randn(T, 64)
        kv_cache = (torch.empty(1), torch.empty(1), torch.empty(1))
        output = torch.empty(T, self.impl.num_heads * self.impl.v_head_dim,
                             dtype=torch.float16)

        out = self.impl.forward(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=self._mk_prefill_meta(T),
            need_gather_q_kv=False,
            output=output,
        )

        self.assertEqual(out.shape,
                         (T, self.impl.num_heads * self.impl.v_head_dim))
        self.assertTrue(mock_preproc.called)
        self.assertTrue(mock_fuse.called)

    @patch.object(AscendSFAImpl, "apply_attention_fusion")
    @patch.object(AscendSFAImpl, "_sfa_preprocess")
    def test_forward_decode_only(self, mock_preproc, mock_fuse):
        T = 5
        mock_preproc.return_value = (self._mk_preprocess_res(T), None)
        mock_fuse.side_effect = lambda **kwargs: torch.randn(
            T, self.impl.num_heads * self.impl.v_head_dim
        )
        self.impl.mla_epilog = lambda x, absorb=True: x

        hidden_states = torch.randn(T, 64)
        kv_cache = (torch.empty(1), torch.empty(1), torch.empty(1))
        output = torch.empty(T, self.impl.num_heads * self.impl.v_head_dim,
                             dtype=torch.float16)

        out = self.impl.forward(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=self._mk_decode_meta(T),
            need_gather_q_kv=False,
            output=output,
        )

        self.assertEqual(out.shape,
                         (T, self.impl.num_heads * self.impl.v_head_dim))
        self.assertTrue(mock_preproc.called)
        self.assertTrue(mock_fuse.called)
