from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.v1.attention.backends.utils import AttentionCGSupport

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import (AscendSFABackend, AscendSFAImpl,
                                          AscendSFAMetadata,
                                          AscendSFAMetadataBuilder)


class TestAscendSFABackend(TestBase):

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_metadata_cls(self):
        self.assertEqual(AscendSFABackend.get_metadata_cls(),
                         AscendSFAMetadata)

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(),
                         AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)


class TestAscendSFAMetadata(TestBase):

    def test_ascend_sfa_metadata_default(self):
        has_prefill = True
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_tables = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            has_prefill=has_prefill,
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            cum_query_lens=cum_query_lens,
            block_tables=block_tables,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.has_prefill, has_prefill)
        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_tables, block_tables)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)


class TestAscendSFAMetadataBuilder(TestBase):

    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        device = torch.device("cuda")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        assert builder.aclgraph_support == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        assert builder.device == device
        assert builder.vllm_config == vllm_config

    def test_ascend_sfa_metadata_builder_build(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        device = torch.device("cuda")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)

        model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        model.model.start_layer = 0

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
            model=model,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    def test_ascend_sfa_metadata_builder_build_for_graph_capture(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        device = torch.device("cuda")

        builder = AscendSFAMetadataBuilder(kv_cache_spec=kv_cache_spec,
                                           layer_names=layer_names,
                                           vllm_config=vllm_config,
                                           device=device)

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc_cpu = torch.tensor(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)

        model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(10)]
        model.model.start_layer = 0

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
            model=model,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly


class TestAscendSFAImpl(TestBase):

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
              mock_tp, mock_get_dcp_size, mock_dcp):
        # Setup mock responses
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

        # Initialization parameters for AscendSFAImpl
        self.num_heads = 8
        self.head_size = 128
        self.scale = 0.1
        self.num_kv_heads = 8
        self.kv_cache_dtype = "auto"
        self.kwargs = {
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 128,
            "v_head_dim": 128,
            "rotary_emb": MagicMock(),
            "q_proj": MagicMock(),
            "q_b_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "indexer": MagicMock(),
            "fused_qkv_a_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "kv_a_layernorm": MagicMock(),
        }

        # Initialize the implementation once
        self.impl = AscendSFAImpl(num_heads=self.num_heads,
                                  head_size=self.head_size,
                                  scale=self.scale,
                                  num_kv_heads=self.num_kv_heads,
                                  kv_cache_dtype=self.kv_cache_dtype,
                                  **self.kwargs)

    def test_sfa_forward(self):
        # mock data for test
        hidden_states = torch.randn(10, 128)
        kv_cache = (torch.randn(10, 128), torch.randn(10, 128))
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 10
        attn_metadata.has_prefill = True
        attn_metadata.slot_mapping = torch.randn(10, 4, 1024)
        attn_metadata.cos = torch.randn(10, 1, 1, 128)
        attn_metadata.sin = torch.randn(10, 1, 1, 128)

        output = torch.zeros_like(hidden_states)

        result = self.impl.forward(
            layer_name="layer1",
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            need_gather_q_kv=False,
            output=output,
        )

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertTrue((result == output).all())

    def test_sfa_process_weights_after_loading(self):
        num_heads = 8
        head_size = 128
        scale = 0.1
        num_kv_heads = 8
        kv_cache_dtype = "auto"

        kwargs = {
            "q_lora_rank": 64,
            "kv_lora_rank": 32,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 128,
            "v_head_dim": 128,
            "rotary_emb": MagicMock(),
            "q_proj": MagicMock(),
            "q_b_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "indexer": MagicMock(),
            "fused_qkv_a_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "kv_a_layernorm": MagicMock(),
        }

        impl = AscendSFAImpl(num_heads=num_heads,
                             head_size=head_size,
                             scale=scale,
                             num_kv_heads=num_kv_heads,
                             kv_cache_dtype=kv_cache_dtype,
                             **kwargs)

        # Simulate the weights loading process
        impl.process_weights_after_loading(torch.float32)
        self.assertIsNotNone(impl.W_UV)
        self.assertIsNotNone(impl.W_UK_T)

    def test_sfa_rope_single(self):

        # Mock cos and sin
        B, N, D = 4, 8, 128  # B = batch size, N = num_heads, D = head_dim
        cos = torch.randn(B, N, D)
        sin = torch.randn(B, N, D)

        # Mock input tensor
        x = torch.randn(B, N, D)

        # Call rope_single function
        result = self.impl.rope_single(x, cos, sin)

        # Assert result shape
        self.assertEqual(result.shape, (B, N, D))
        self.assertTrue(torch.allclose(
            result, x))  # Expecting the result to match some expected behavior

    def test_sfa_q_proj_and_k_up_proj(self):

        num_kv_heads = 8

        # Mock input tensor
        B, S = 4, 10  # Batch size and sequence length
        x = torch.randn(B, S, num_kv_heads)

        # Call _q_proj_and_k_up_proj function
        ql_nope, q_pe = self.impl._q_proj_and_k_up_proj(x)

        # Assert the shape of the result
        self.assertEqual(ql_nope.shape, (B, S, num_kv_heads))
        self.assertEqual(q_pe.shape, (B, S, num_kv_heads))

    def test_sfa_exec_kv(self):
        num_kv_heads = 8
        # Mock kv_no_split, cos, sin, and kv_cache
        kv_no_split = torch.randn(4, num_kv_heads, 1,
                                  128)  # Example shape for kv_no_split
        cos = torch.randn(4, 1, 1, 128)
        sin = torch.randn(4, 1, 1, 128)
        kv_cache = (torch.randn(4, 128), torch.randn(4, 128))

        # Mock slots tensor
        slots = torch.tensor([0, 1, 2, 3])

        # Call exec_kv function
        k_pe, k_nope = self.impl.exec_kv(kv_no_split, cos, sin, kv_cache,
                                         slots)

        # Assert the output shapes
        self.assertEqual(k_pe.shape, (4, num_kv_heads, 1, 128))
        self.assertEqual(k_nope.shape, (4, num_kv_heads, 1, 128))

    def test_sfa_forward_with_prefill(self):
        # Mock data
        hidden_states = torch.randn(10, 128)
        kv_cache = (torch.randn(10, 128), torch.randn(10, 128))
        attn_metadata = MagicMock()
        attn_metadata.num_actual_tokens = 10
        attn_metadata.has_prefill = True
        attn_metadata.slot_mapping = torch.randn(10, 4, 1024)
        attn_metadata.cos = torch.randn(10, 1, 1, 128)
        attn_metadata.sin = torch.randn(10, 1, 1, 128)

        output = torch.zeros_like(hidden_states)

        # Call forward with prefill
        result = self.impl.forward(
            layer_name="layer1",
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            need_gather_q_kv=False,
            output=output,
        )

        self.assertEqual(result.shape, hidden_states.shape)
        self.assertTrue((result == output).all())

    def test_sfa_process_weights_after_loading_with_nz(self):
        # Simulate NZ enablement and weight loading
        self.impl.process_weights_after_loading(torch.float32)
        self.assertIsNotNone(self.impl.W_UV)
        self.assertIsNotNone(self.impl.W_UK_T)

    def test_sfa_indexer_select(self):
        num_kv_heads = 8

        # Mock input tensors
        B, S = 4, 10  # Batch size and sequence length
        x = torch.randn(B, S, num_kv_heads)
        qr = torch.randn(B, S, num_kv_heads)
        kv_cache = (torch.randn(4, 128), torch.randn(4,
                                                     128), torch.randn(4, 128))
        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(4, 1, 1, 128)
        attn_metadata.sin = torch.randn(4, 1, 1, 128)
        attn_metadata.slot_mapping = torch.randint(0, 100, (B, S))

        # Call indexer_select method
        topk_indices = self.impl.indexer_select(x, qr, kv_cache, attn_metadata)

        # Assert that the result is a tensor with the expected shape
        self.assertEqual(topk_indices.shape, torch.Size([B, S, num_kv_heads]))
