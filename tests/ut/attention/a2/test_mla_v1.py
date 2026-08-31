import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import (
    AscendMLABackend,
    AscendMLADecodeMetadata,
    AscendMLAImpl,
    AscendMLAMetadata,
    AscendMLAMetadataBuilder,
    AscendMLAPCPImpl,
    AscendMLAPCPMetadata,
    AscendMLAPCPMetadataBuilder,
    AscendMLAPrefillMetadata,
    ChunkedContextMetadata,
    DecodeMLAPreprocessResult,
    PrefillMLAPreprocessResult,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


class TestAscendMLABackend(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()

        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1
        self.mock_parallel_config = mock_parallel_config

        self.mock_config.parallel_config = mock_parallel_config

        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()
        self.mla_config_patcher = patch(
            "vllm_ascend.attention.mla_v1.get_current_vllm_config",
            return_value=self.mock_config,
        )
        self.mla_config_patcher.start()

        from vllm_ascend.attention.utils import enable_dcp

        enable_dcp.cache_clear()

    def test_get_name(self):
        self.assertEqual(AscendMLABackend.get_name(), "ASCEND_MLA")

    @patch("vllm_ascend.attention.mla_v1.enable_pcp")
    def test_get_builder_cls(self, mock_enable_pcp):
        mock_enable_pcp.return_value = False
        self.assertEqual(AscendMLABackend.get_builder_cls(), AscendMLAMetadataBuilder)
        mock_enable_pcp.return_value = True
        self.assertIs(AscendMLABackend.get_builder_cls(), AscendMLAPCPMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendMLABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    @patch("vllm_ascend.attention.mla_v1.enable_pcp")
    def test_get_impl_cls(self, mock_enable_pcp):
        mock_enable_pcp.return_value = False
        self.assertEqual(AscendMLABackend.get_impl_cls(), AscendMLAImpl)
        mock_enable_pcp.return_value = True
        self.assertIs(AscendMLABackend.get_impl_cls(), AscendMLAPCPImpl)

    def test_get_supported_kernel_block_sizes(self):
        result = AscendMLABackend.get_supported_kernel_block_sizes()
        self.assertEqual(result, [128])

    @patch("vllm_ascend.attention.mla_v1.enable_dcp")
    def test_get_builder_cls_with_dcp(self, mock_enable_dcp):
        mock_enable_dcp.return_value = True
        builder_cls = AscendMLABackend.get_builder_cls()
        self.assertIsNotNone(builder_cls)

    @patch("vllm_ascend.attention.mla_v1.enable_dcp")
    def test_get_impl_cls_with_dcp(self, mock_enable_dcp):
        mock_enable_dcp.return_value = True
        impl_cls = AscendMLABackend.get_impl_cls()
        self.assertIsNotNone(impl_cls)

    @patch("vllm_ascend.attention.mla_v1.enable_dcp")
    @patch("vllm_ascend.attention.mla_v1.enable_pcp")
    def test_pcp_and_dcp_are_rejected(self, mock_enable_pcp, mock_enable_dcp):
        mock_enable_dcp.return_value = True
        mock_enable_pcp.return_value = True

        with self.assertRaisesRegex(NotImplementedError, "does not support PCP and DCP"):
            AscendMLABackend.get_builder_cls()
        with self.assertRaisesRegex(NotImplementedError, "does not support PCP and DCP"):
            AscendMLABackend.get_impl_cls()


def _make_pcp_metadata(
    *,
    num_actual_tokens: int,
    num_decode_tokens: int,
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill,
) -> AscendMLAPCPMetadata:
    return AscendMLAPCPMetadata(
        num_actual_tokens=num_actual_tokens,
        slot_mapping=torch.empty(0, dtype=torch.int64),
        query_start_loc=torch.tensor([0, num_actual_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([num_actual_tokens], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_actual_tokens], dtype=torch.int32),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        num_decodes=int(num_decode_tokens > 0),
        num_decode_tokens=num_decode_tokens,
        num_prefills=int(num_actual_tokens > num_decode_tokens),
        attn_state=attn_state,
    )


def test_mla_pcp_metadata_keeps_expanded_slot_mapping() -> None:
    expanded_slots = torch.tensor(
        [5, 10, 11, -1, -1, 20, 21, -1],
        dtype=torch.int64,
    )
    common_metadata = SimpleNamespace(slot_mapping=expanded_slots)
    metadata = _make_pcp_metadata(
        num_actual_tokens=3,
        num_decode_tokens=1,
        attn_state=AscendAttentionState.PrefillCacheHit,
    )
    builder = AscendMLAPCPMetadataBuilder.__new__(AscendMLAPCPMetadataBuilder)
    builder.pcp_size = 2
    builder.pcp_rank = 1

    with patch.object(AscendMLAMetadataBuilder, "build", return_value=metadata):
        result = builder.build(0, common_metadata)

    assert result.slot_mapping is expanded_slots
    assert result.pcp_local_num_input_tokens == 4
    assert result.pcp_local_prefill_start == 3
    assert result.pcp_local_prefill_end == 5
    assert result.attn_state == AscendAttentionState.ChunkedPrefill


def test_mla_pcp_prefill_gathers_padded_cache_inputs() -> None:
    impl = AscendMLAPCPImpl.__new__(AscendMLAPCPImpl)
    captured: dict[str, torch.Tensor] = {}

    def fake_gather(tensors, slot_mapping, num_decode_tokens):
        assert num_decode_tokens == 0
        captured["local_kv"] = tensors[0]
        captured["local_cos"] = tensors[1]
        captured["slots"] = slot_mapping
        return (
            tuple(torch.cat((tensor + 100, tensor), dim=0) for tensor in tensors),
            slot_mapping,
        )

    def fake_exec_kv_prefill(self, kv, cos, sin, kv_cache, slots):
        captured["gathered_kv"] = kv
        captured["cache_slots"] = slots
        return cos, kv[:, :2]

    pcp_group = SimpleNamespace(world_size=2, rank_in_group=1)
    metadata = _make_pcp_metadata(num_actual_tokens=3, num_decode_tokens=1)
    metadata.slot_mapping = torch.tensor(
        [5, 10, 11, -1, -1, 20, 21, -1],
        dtype=torch.int64,
    )
    metadata.pcp_local_num_input_tokens = 4
    metadata.pcp_local_prefill_start = 3
    metadata.pcp_local_prefill_end = 5
    kv = torch.arange(9, dtype=torch.float32).view(3, 3)
    cos = torch.tensor([[1.0], [2.0], [1.0]])
    sin = torch.tensor([[3.0], [4.0], [0.0]])

    with (
        patch(
            "vllm_ascend.attention.mla_v1.get_pcp_group",
            return_value=pcp_group,
        ),
        patch(
            "vllm_ascend.attention.mla_v1._gather_prefill_cache_inputs",
            side_effect=fake_gather,
        ),
        patch.object(
            AscendMLAImpl,
            "exec_kv_prefill",
            fake_exec_kv_prefill,
        ),
    ):
        k_pe, k_nope = impl.exec_kv_prefill(
            kv,
            cos,
            sin,
            (torch.empty(0), torch.empty(0)),
            torch.empty(0, dtype=torch.int64),
            attn_metadata=metadata,
        )

    expected_slots = torch.tensor([10, 11, -1, 20, 21, -1])
    torch.testing.assert_close(captured["slots"], expected_slots)
    assert captured["local_kv"] is kv
    assert captured["local_cos"] is cos
    assert captured["gathered_kv"].shape[0] == 6
    torch.testing.assert_close(captured["cache_slots"], expected_slots)
    torch.testing.assert_close(k_pe, cos[:2])
    torch.testing.assert_close(k_nope, kv[:2, :2])


class TestDecodeMLAPreprocessResult(TestBase):
    def test_decode_mla_preprocess_result_default(self):
        result = DecodeMLAPreprocessResult()
        self.assertIsNone(result.ql_nope)
        self.assertIsNone(result.q_pe)
        self.assertIsNone(result.k_nope)
        self.assertIsNone(result.k_pe)
        self.assertIsNone(result.decode_q_wo_k_up)
        self.assertIsNone(result.dequant_scale_q_nope)

    def test_decode_mla_preprocess_result_with_values(self):
        ql_nope = torch.randn(2, 4, 8)
        q_pe = torch.randn(2, 4, 8)
        k_nope = torch.randn(2, 4, 8)
        k_pe = torch.randn(2, 4, 8)
        decode_q_wo_k_up = torch.randn(2, 4, 8)
        dequant_scale_q_nope = torch.randn(2, 4, 8)

        result = DecodeMLAPreprocessResult(
            ql_nope=ql_nope,
            q_pe=q_pe,
            k_nope=k_nope,
            k_pe=k_pe,
            decode_q_wo_k_up=decode_q_wo_k_up,
            dequant_scale_q_nope=dequant_scale_q_nope,
        )

        self.assertIs(result.ql_nope, ql_nope)
        self.assertIs(result.q_pe, q_pe)
        self.assertIs(result.k_nope, k_nope)
        self.assertIs(result.k_pe, k_pe)
        self.assertIs(result.decode_q_wo_k_up, decode_q_wo_k_up)
        self.assertIs(result.dequant_scale_q_nope, dequant_scale_q_nope)


class TestPrefillMLAPreprocessResult(TestBase):
    def test_prefill_mla_preprocess_result_default(self):
        result = PrefillMLAPreprocessResult()
        self.assertIsNone(result.q_nope)
        self.assertIsNone(result.q_pe)
        self.assertIsNone(result.k_nope)
        self.assertIsNone(result.k_pe)
        self.assertIsNone(result.value)

    def test_prefill_mla_preprocess_result_with_values(self):
        q_nope = torch.randn(2, 4, 8)
        q_pe = torch.randn(2, 4, 8)
        k_nope = torch.randn(2, 4, 8)
        k_pe = torch.randn(2, 4, 8)
        value = torch.randn(2, 4, 8)

        result = PrefillMLAPreprocessResult(q_nope=q_nope, q_pe=q_pe, k_nope=k_nope, k_pe=k_pe, value=value)

        self.assertIs(result.q_nope, q_nope)
        self.assertIs(result.q_pe, q_pe)
        self.assertIs(result.k_nope, k_nope)
        self.assertIs(result.k_pe, k_pe)
        self.assertIs(result.value, value)


class TestAscendMLAPrefillMetadata(TestBase):
    def test_ascend_mla_prefill_metadata_default(self):
        attn_mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        query_lens = [1, 2]
        seq_lens = [2, 2]
        context_lens = torch.tensor([1, 2])
        input_positions = torch.tensor([0, 1, 0, 1])
        query_start_loc = torch.tensor([0, 1, 3])
        block_table = torch.tensor([[0, 1], [2, 3]])
        max_query_len = 2
        max_seq_lens = 2

        metadata = AscendMLAPrefillMetadata(
            attn_mask=attn_mask,
            query_lens=query_lens,
            seq_lens=seq_lens,
            context_lens=context_lens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            block_table=block_table,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
        )
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

    def test_ascend_mla_prefill_metadata_with_chunked_context(self):
        cu_seq_lens = torch.tensor([0, 2, 4])
        starts = torch.tensor([0, 2])
        seq_tot = [2, 2]
        max_seq_lens = [2, 2]
        workspace = torch.randn(2, 4)
        chunk_seq_lens = torch.tensor([2, 2])

        chunked_context = ChunkedContextMetadata(
            cu_seq_lens=cu_seq_lens,
            starts=starts,
            seq_tot=seq_tot,
            max_seq_lens=max_seq_lens,
            workspace=workspace,
            chunk_seq_lens=chunk_seq_lens,
            chunk_seq_lens_npu=chunk_seq_lens,
            chunk_actual_seq_lengths_kv_list=[[2, 4]],
        )

        metadata = AscendMLAPrefillMetadata(
            attn_mask=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
            query_lens=[1, 2],
            seq_lens=[2, 2],
            context_lens=torch.tensor([1, 2]),
            input_positions=torch.tensor([0, 1, 0, 1]),
            query_start_loc=torch.tensor([0, 1, 3]),
            block_table=torch.tensor([[0, 1], [2, 3]]),
            max_query_len=2,
            max_seq_lens=2,
            chunked_context=chunked_context,
        )

        self.assertIsNotNone(metadata.chunked_context)
        self.assertIs(metadata.chunked_context.cu_seq_lens, cu_seq_lens)
        self.assertIs(metadata.chunked_context.starts, starts)
        self.assertEqual(metadata.chunked_context.seq_tot, seq_tot)
        self.assertEqual(metadata.chunked_context.max_seq_lens, max_seq_lens)
        self.assertIs(metadata.chunked_context.workspace, workspace)
        self.assertIs(metadata.chunked_context.chunk_seq_lens, chunk_seq_lens)
        self.assertIs(metadata.chunked_context.chunk_seq_lens_npu, chunk_seq_lens)


class TestAscendMLADecodeMetadata(TestBase):
    def test_ascend_mla_decode_metadata_default(self):
        input_positions = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        block_table = torch.tensor([[0, 3, 2, 1], [0, 2, 1, 3]])
        seq_lens = torch.tensor([[2], [3]])
        max_seq_lens = 4
        seq_lens_list = [2, 3]
        attn_mask = None

        metadata = AscendMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_lens=max_seq_lens,
            seq_lens_list=seq_lens_list,
            attn_mask=attn_mask,
        )

        self.assertIs(metadata.input_positions, input_positions)
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.seq_lens, seq_lens)
        self.assertEqual(metadata.max_seq_lens, max_seq_lens)
        self.assertEqual(metadata.seq_lens_list, seq_lens_list)
        self.assertIsNone(attn_mask)


class TestAscendMLAMetadata(TestBase):
    def test_ascend_mla_metadata_default(self):
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

        metadata = AscendMLAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            block_tables=block_tables,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_input_tokens=num_input_tokens,
            query_lens=query_lens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
            decode=decode,
            prefill=prefill,
        )

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
        self.assertTrue(metadata.causal)
        self.assertEqual(metadata.decode, decode)
        self.assertEqual(metadata.prefill, prefill)


class TestAscendMLAMetadataBuilder(TestBase):
    def setUp(self):
        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

    def tearDown(self):
        self.parent_init_patcher.stop()

    def test_ascend_mla_metadata_builder_default(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.model_config.hf_text_config.mla_use_nope = False
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"

        mock_vllm_config.speculative_config = None

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)

            self.assertEqual(builder.block_size, mock_vllm_config.cache_config.block_size)
            self.assertEqual(builder.chunked_prefill_enabled, mock_vllm_config.scheduler_config.enable_chunked_prefill)

    def test_metadata_builder_uses_layer_rope_mode(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_vllm_config.speculative_config = None

        for layer_uses_rope in (True, False):
            with self.subTest(layer_uses_rope=layer_uses_rope):
                # Deliberately disagree with the target config: the layer wins.
                mock_vllm_config.model_config.hf_text_config = SimpleNamespace(
                    qk_rope_head_dim=64,
                    mla_use_nope=layer_uses_rope,
                )
                mock_vllm_config.compilation_config.static_forward_context = {
                    "self_attn": SimpleNamespace(impl=SimpleNamespace(use_mla_rope=layer_uses_rope)),
                }
                with patch("vllm_ascend.attention.mla_v1.get_ascend_config", return_value=MagicMock()):
                    builder = AscendMLAMetadataBuilder(None, ["self_attn"], mock_vllm_config, "cpu")

                self.assertEqual(builder.use_mla_rope, layer_uses_rope)

    def test_ascend_mla_metadata_builder_spec_decode(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"

        mock_spec_config = MagicMock()
        mock_spec_config.num_speculative_tokens = 3
        mock_vllm_config.speculative_config = mock_spec_config

        ascend_config = MagicMock()
        with patch("vllm_ascend.attention.mla_v1.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)

            self.assertEqual(builder.block_size, mock_vllm_config.cache_config.block_size)
            self.assertEqual(builder.chunked_prefill_enabled, mock_vllm_config.scheduler_config.enable_chunked_prefill)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_ascend_mla_metadata_builder_build_full_graph(self, mock_get_cos_and_sin_mla):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.model_config.hf_text_config.mla_use_nope = False
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        mock_vllm_config.parallel_config.decode_context_parallel_size = 1
        mock_device = "cpu"
        torch.Tensor.pin_memory = lambda x: x  # noqa

        mock_spec_config = MagicMock()
        mock_spec_config.num_speculative_tokens = 1
        mock_spec_config.disable_padded_drafter_batch = True
        mock_vllm_config.speculative_config = mock_spec_config

        builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
        common_metadata = MagicMock()
        common_metadata.graph_pad_size = 8
        common_metadata.num_reqs = 4
        common_metadata.num_actual_tokens = 5
        common_metadata.max_query_len = 5
        common_metadata.context_parallel_metadata = None
        common_metadata.seq_lens_cpu = torch.Tensor([9, 10, 8, 8]).int()
        common_metadata.query_start_loc = torch.Tensor([0, 1, 2, 4, 5]).int()
        common_metadata.query_start_loc_cpu = torch.Tensor([0, 1, 2, 4, 5]).int()
        common_metadata.positions = torch.Tensor([1, 2, 3, 4, 5, 6]).int()
        common_metadata.causal = False
        block_table = torch.Tensor([[1, 0], [2, 0], [3, 0], [4, 0]]).int()
        common_metadata.block_table_tensor = block_table
        mock_get_cos_and_sin_mla.return_value = (torch.tensor([6, 6]), torch.Tensor([6, 6]))
        metadata = builder.build(0, common_metadata)
        self.assertFalse(metadata.causal)

        self.assertEqual(metadata.decode.actual_seq_lengths_q, [1, 2, 4, 5, 6, 6, 7, 8])
        self.assertEqual(metadata.decode.block_table.shape[0], 8)

    def test_reorder_batch(self):
        ascend_config = MagicMock()

        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"

        mock_vllm_config.speculative_config = None

        with patch("vllm_ascend.attention.mla_v1.get_ascend_config", return_value=ascend_config):
            builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
            builder.decode_threshold = 1

        input_batch = MagicMock()
        input_batch.req_ids = [0, 1, 2, 3]

        scheduler_output = MagicMock()
        scheduler_output.num_scheduled_tokens = {0: 1, 1: 3, 2: 1, 3: 2}
        scheduler_output.scheduled_spec_decode_tokens = {0: [], 1: [1], 2: [], 3: []}

        input_batch.swap_states = MagicMock()

        modified = builder.reorder_batch(input_batch, scheduler_output)

        self.assertTrue(modified)
        input_batch.swap_states.assert_called_once_with(1, 2)

    def test_determine_chunked_prefill_workspace_size(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.scheduler_config.enable_chunked_prefill = True
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 128
        mock_vllm_config.scheduler_config.max_num_batched_tokens = 4096
        mock_vllm_config.model_config.max_model_len = 4096

        result = AscendMLAMetadataBuilder.determine_chunked_prefill_workspace_size(mock_vllm_config)
        self.assertGreater(result, 0)

    def test_get_cudagraph_support(self):
        mock_vllm_config = MagicMock()
        mock_kv_cache_spec = MagicMock()

        result = AscendMLAMetadataBuilder.get_cudagraph_support(mock_vllm_config, mock_kv_cache_spec)
        from vllm.v1.attention.backend import AttentionCGSupport

        self.assertEqual(result, AttentionCGSupport.UNIFORM_BATCH)

    def test_set_num_actual_tokens(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"

        mock_vllm_config.speculative_config = None

        builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_actual_tokens = 100

        builder.set_num_actual_tokens(common_attn_metadata)
        self.assertEqual(builder.num_actual_tokens, 100)

    def test_pad_actual_seq_lens_q_mtp_disable_pad(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"
        mock_vllm_config.speculative_config = None

        builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
        input_seq_lens = [1, 2, 4, 5]
        expect_output = [1, 2, 4, 5, 6, 6, 7, 8]
        num_reqs = 4
        num_reqs_pad_size = 4
        output_seq_lens = builder.pad_actual_seq_len_q_mtp_disable_pad(num_reqs_pad_size, num_reqs, input_seq_lens)
        self.assertEqual(output_seq_lens, expect_output)

    def test_pad_actual_seq_lens_q_mtp_enable_pad(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"
        mock_vllm_config.speculative_config = None

        common_metadata = MagicMock()
        common_metadata.actual_seq_lengths_q = [2, 4, 6, 8]

        builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
        input_seq_lens = [2, 4, 6]
        expect_output = [2, 4, 6, 8]
        num_reqs = 3
        num_reqs_pad_size = 1
        output_seq_lens = builder.pad_actual_seq_len_q_mtp_enable_pad(
            num_reqs_pad_size, num_reqs, input_seq_lens, common_metadata
        )
        self.assertEqual(output_seq_lens, expect_output)

    def test_pad_actual_seq_lens_q_mtp_enable_pad_with_padding(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.max_model_len = 1024
        mock_vllm_config.model_config.get_head_size.return_value = 64
        mock_vllm_config.model_config.dtype = torch.float16
        mock_vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        mock_vllm_config.cache_config.block_size = 16
        mock_vllm_config.scheduler_config.max_num_seqs = 4
        mock_vllm_config.scheduler_config.chunked_prefill_enabled = False
        mock_vllm_config.scheduler_config.enable_chunked_prefill = False
        mock_device = "cpu"
        mock_vllm_config.speculative_config = None

        common_metadata = MagicMock()
        common_metadata.actual_seq_lengths_q = [2, 4, 6, 100]

        builder = AscendMLAMetadataBuilder(None, [], mock_vllm_config, mock_device)
        input_seq_lens = [2, 4, 6]
        num_reqs = 3
        num_reqs_pad_size = 1
        output_seq_lens = builder.pad_actual_seq_len_q_mtp_enable_pad(
            num_reqs_pad_size, num_reqs, input_seq_lens, common_metadata
        )
        self.assertEqual(len(output_seq_lens), 4)
        self.assertEqual(output_seq_lens[:3], [2, 4, 6])
        self.assertEqual(output_seq_lens[-1], 100)


class TestAscendMLAMetadataBuilderBuild(TestBase):
    def setUp(self):
        # Mock parent class __init__ to avoid complex initialization,
        # but still set the essential attributes that child class needs
        def mock_parent_init(
            self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen
        ):
            self.metadata_cls = metadata_cls
            self.kv_cache_spec = kv_cache_spec
            self.model_config = vllm_config.model_config
            self.vllm_config = vllm_config
            self.device = device
            self.chunked_prefill_workspace_size = 128 * 1024
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size, vllm_config.model_config.get_head_size()),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )

        self.parent_init_patcher = patch(
            "vllm.model_executor.layers.attention.mla_attention.MLACommonMetadataBuilder.__init__", mock_parent_init
        )
        self.parent_init_patcher.start()

        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.cache_config = CacheConfig(block_size=32)
        mock_scheduler_config = MagicMock(spec=SchedulerConfig)
        mock_scheduler_config.max_num_seqs = 8
        mock_scheduler_config.chunked_prefill_enabled = True
        mock_scheduler_config.enable_chunked_prefill = True
        self.mock_vllm_config.scheduler_config = mock_scheduler_config
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.parallel_config.decode_context_parallel_size = 1
        self.mock_vllm_config.speculative_config = None
        self.mock_device = torch.device("cpu")
        fake_weight_path = os.path.join(os.path.dirname(__file__), "..", "..", "_fake_weight")
        model_config = ModelConfig(
            model=fake_weight_path,
            skip_tokenizer_init=True,
        )
        model_config.hf_text_config.head_dim = 128
        model_config.hf_text_config.qk_rope_head_dim = 32
        self.mock_vllm_config.model_config = model_config
        self.kv_cache_spec = MagicMock()
        self.kv_cache_spec.num_layers = 32
        self.kv_cache_spec.head_size = 64
        self.kv_cache_spec.num_heads = 32

    def tearDown(self):
        self.parent_init_patcher.stop()

    def test_flash_attn_metadata_contract_and_stable_buffer(self):
        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        builder._flash_attn_impl = SimpleNamespace(num_heads=8)
        builder._flash_attn_head_dim = 256
        cu_seqlens = torch.tensor([0, 5, 12], dtype=torch.int32)

        def return_output_buffer(*args, **kwargs):
            return kwargs["output_buffer"]

        with patch(
            "vllm_ascend.attention.mla_v1.build_flash_attn_metadata",
            side_effect=return_output_buffer,
        ) as mock_build:
            first = builder._build_flash_attn_metadata(
                buffer_key=("prefill", 0, 2),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                causal=True,
            )
            second = builder._build_flash_attn_metadata(
                buffer_key=("prefill", 0, 2),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                causal=True,
            )

        self.assertIs(first, second)
        args = mock_build.call_args.args
        kwargs = mock_build.call_args.kwargs
        self.assertEqual(args[:3], (8, 8, 256))
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_kv"], "TND")
        self.assertEqual(kwargs["max_seqlen_q"], -1)
        self.assertEqual(kwargs["max_seqlen_kv"], -1)

    def test_flash_mla_metadata_contract_and_stable_buffers(self):
        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=[],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        builder._flash_mla_impl = SimpleNamespace(
            num_heads=8,
            num_kv_heads=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

        def return_output_buffer(*args, **kwargs):
            return kwargs["output_buffer"]

        with patch(
            "vllm_ascend.attention.mla_v1.build_flash_mla_metadata",
            side_effect=return_output_buffer,
        ) as mock_build:
            first = builder._build_flash_mla_metadata(
                actual_seq_lengths_q=[2, 5],
                seq_lens=[9, 0],
                causal=True,
            )
            second = builder._build_flash_mla_metadata(
                actual_seq_lengths_q=[2, 5],
                seq_lens=[11, 0],
                causal=True,
            )

        first_cu_seqlens, first_cache_seqlens, first_metadata = first
        second_cu_seqlens, second_cache_seqlens, second_metadata = second
        self.assertIs(first_cu_seqlens, second_cu_seqlens)
        self.assertIs(first_cache_seqlens, second_cache_seqlens)
        self.assertIs(first_metadata, second_metadata)
        torch.testing.assert_close(first_cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
        torch.testing.assert_close(first_cache_seqlens, torch.tensor([11, 1], dtype=torch.int32))

        args = mock_build.call_args.args
        kwargs = mock_build.call_args.kwargs
        self.assertIs(args[0], first_cache_seqlens)
        self.assertEqual(args[1:3], (8, 1))
        self.assertIs(kwargs["cu_seqlens_q"], first_cu_seqlens)
        self.assertEqual(kwargs["head_dim_qk"], 576)
        self.assertEqual(kwargs["head_dim_v"], 512)
        self.assertEqual(kwargs["mask_mode"], 3)
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["max_seqlen_q"], -1)
        self.assertEqual(kwargs["max_seqlen_kv"], -1)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.mla_v1.torch.zeros", wraps=torch.zeros)
    @patch("torch.Tensor.npu", new=lambda self: self)
    @patch("torch.npu.is_available")
    def test_build_prefix_no_cache_metadata(self, mock_npu_available, mock_zeros, mock_get_cos_and_sin_mla):
        mock_npu_available.return_value = False
        torch.Tensor.pin_memory = lambda x: x  # noqa

        def zeros_override(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return mock_zeros._mock_wraps(*args, **kwargs)

        mock_zeros.side_effect = zeros_override
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
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6,
        )

        base_inputs = {
            "num_actual_tokens": 10,
            "slot_mapping": torch.tensor(range(10)),
            "query_start_loc": torch.tensor([0, 3, 7]),
            "seq_lens": torch.tensor([5, 6]),
            "block_tables": torch.zeros((10, 10)),
            "num_prefills": 2,
        }

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.tensor(10), torch.Tensor(10))
        metadata = builder.build(1, common_attn_metadata)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens, base_inputs["num_actual_tokens"])
        self.assertTrue(torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.mla_v1.torch.zeros", wraps=torch.zeros)
    @patch("torch.Tensor.npu", new=lambda self: self)
    @patch("torch.npu.is_available")
    def test_build_chunked_prefix_metadata(self, mock_npu_available, mock_zeros, mock_get_cos_and_sin_mla):
        mock_npu_available.return_value = False
        torch.Tensor.pin_memory = lambda x: x  # noqa

        def zeros_override(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return mock_zeros._mock_wraps(*args, **kwargs)

        mock_zeros.side_effect = zeros_override

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

        base_inputs = {
            "num_actual_tokens": 15,
            "slot_mapping": torch.tensor(range(15)),
            "query_start_loc": torch.tensor([0, 2, 5, 9]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "block_tables": torch.zeros((10, 10)),
            "num_prefills": 3,
        }

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.tensor(10), torch.Tensor(10))
        metadata = builder.build(1, common_attn_metadata)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens, base_inputs["num_actual_tokens"])
        self.assertTrue(torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_build_decode_only_metadata(self, mock_get_cos_and_sin_mla):
        torch.Tensor.pin_memory = lambda x: x  # noqa

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2, 3]),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(3)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            decode_token_per_req=torch.tensor([1, 1, 1]),
            positions=torch.tensor([10, 10]),
            attn_state=AscendAttentionState.DecodeOnly,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6,
        )

        base_inputs = {
            "num_actual_tokens": 3,
            "slot_mapping": torch.tensor(range(3)),
            "query_start_loc": torch.tensor([0, 1, 2, 3]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "num_decodes": 3,
        }

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.tensor([10, 10]), torch.Tensor([10, 10]))
        metadata = builder.build(1, common_attn_metadata)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens, base_inputs["num_actual_tokens"])
        self.assertTrue(torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_build_decode_metadata_without_disable_padded_drafter_batch(self, mock_get_cos_and_sin_mla):
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 3
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 1, 2, 3])
        common_attn_metadata.positions = torch.tensor([10, 10, 10])
        common_attn_metadata.decode_token_per_req = 1
        common_attn_metadata.actual_seq_lengths_q = [1, 2, 3, 4, 5]

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )

        builder.num_actual_tokens = 3
        builder.num_decode_tokens = 3
        builder.num_decodes = 3
        builder.graph_pad_size = 5  # > num_reqs
        builder.seq_lens = torch.tensor([4, 5, 6])
        builder.slot_mapping = torch.tensor(range(3))
        builder.block_table = torch.zeros((3, 10))

        mock_speculative_config = MagicMock()
        mock_speculative_config.disable_padded_drafter_batch = False
        builder.speculative_config = mock_speculative_config

        builder.attn_mask_builder = MagicMock()
        builder.attn_mask_builder.get_splitfuse_attn_mask.return_value = torch.randn(1, 1, 5, 5)

        mock_get_cos_and_sin_mla.return_value = (torch.randn(5, 32), torch.randn(5, 32))

        metadata = builder.build_decode_metadata(0, common_attn_metadata)

        self.assertIsInstance(metadata, AscendMLADecodeMetadata)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_build_for_graph_capture_decode_only(self, mock_get_cos_and_sin_mla):
        torch.Tensor.pin_memory = lambda x: x  # noqa

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2, 3]),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([4, 5, 6]),
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(3)),
            actual_seq_lengths_q=torch.tensor([0, 1, 2]),
            decode_token_per_req=torch.tensor([1, 1, 1]),
            positions=torch.tensor([10, 10]),
            attn_state=AscendAttentionState.DecodeOnly,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6,
        )

        base_inputs = {
            "num_actual_tokens": 3,
            "slot_mapping": torch.tensor(range(3)),
            "query_start_loc": torch.tensor([0, 1, 2, 3]),
            "seq_lens": torch.tensor([4, 5, 6]),
            "num_decodes": 3,
        }

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.tensor([10, 10]), torch.Tensor([10, 10]))
        metadata = builder.build_for_graph_capture(common_attn_metadata, AscendAttentionState.DecodeOnly)

        self.assertIsInstance(metadata, AscendMLAMetadata)
        self.assertEqual(metadata.num_actual_tokens, base_inputs["num_actual_tokens"])
        self.assertTrue(torch.all(metadata.slot_mapping == base_inputs["slot_mapping"]))
        self.assertEqual(metadata.head_dim, self.kv_cache_spec.head_size)

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_build_for_graph_capture_prefill(self, mock_get_cos_and_sin_mla):
        torch.Tensor.pin_memory = lambda x: x  # noqa
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
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=None,
            max_seq_len=6,
        )

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.tensor(10), torch.Tensor(10))
        with self.assertRaises(NotImplementedError) as ctx:
            builder.build_for_graph_capture(common_attn_metadata, AscendAttentionState.PrefillNoCache)
        self.assertIn(
            "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state",
            str(ctx.exception),
        )

    @patch("vllm_ascend.attention.mla_v1.get_cos_and_sin_mla")
    def test_build_with_seq_lens_only(self, mock_get_cos_and_sin_mla):
        torch.Tensor.pin_memory = lambda x: x  # noqa

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2, 5, 8]),
            query_start_loc_cpu=torch.tensor([0, 2, 5, 8]),
            seq_lens_cpu=None,
            num_reqs=3,
            num_actual_tokens=8,
            max_query_len=3,
            block_table_tensor=torch.zeros((10, 10)),
            slot_mapping=torch.tensor(range(8)),
            actual_seq_lengths_q=torch.tensor([2, 3, 3]),
            decode_token_per_req=torch.tensor([0, 0, 0]),
            positions=torch.tensor([0, 1, 0, 1, 2, 0, 1, 2]),
            attn_state=AscendAttentionState.PrefillNoCache,
            num_computed_tokens_cpu=None,
            seq_lens=torch.tensor([2, 3, 3]),
            max_seq_len=3,
        )
        common_attn_metadata._seq_lens_cpu = None

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        mock_get_cos_and_sin_mla.return_value = (torch.randn(3, 32), torch.randn(3, 32))
        metadata = builder.build(0, common_attn_metadata)

        self.assertIsInstance(metadata, AscendMLAMetadata)

    def test_build_chunked_metadata_without_chunked_prefill(self):
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 3

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        builder.chunked_prefill_enabled = False

        result = builder.build_chunked_metadata(0, common_attn_metadata)

        self.assertIsNone(result)

    def test_build_chunked_metadata_with_no_context(self):
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 3

        builder = AscendMLAMetadataBuilder(
            kv_cache_spec=self.kv_cache_spec,
            layer_names=["layer_0", "layer_1"],
            vllm_config=self.mock_vllm_config,
            device=self.mock_device,
        )
        builder.chunked_prefill_enabled = True
        builder.seq_lens = torch.tensor([2, 2, 2])
        builder.query_lens = torch.tensor([2, 2, 2])
        builder.num_decodes = 0

        result = builder.build_chunked_metadata(0, common_attn_metadata)

        self.assertIsNone(result)


class TestAscendMLAImpl(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    def setUp(self, get_current_vllm_config, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 1
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        vllm_config.model_config = model_config
        get_current_vllm_config.return_value = vllm_config
        vllm_config.additional_config = {"refresh": True}
        vllm_config.parallel_config = parallel_config
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
            "g_proj": None,
            "use_mla_rope": True,
        }

        self.impl = AscendMLAImpl(
            num_heads=num_heads,
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
            **kwargs,
        )
        self.impl.fa_quant_layer = False

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
        # 256 is power of 2, so padding should be 0
        self.assertEqual(self.impl.num_heads_padded, 256)
        self.assertEqual(self.impl.head_padding, 0)

    def test_flash_attn_qkv_padding_for_mla_192(self):
        q_nope = torch.randn(3, 8, 128)
        q_pe = torch.randn(3, 8, 64)
        k_nope = torch.randn(3, 8, 128)
        k_pe = torch.randn(3, 8, 64)
        value = torch.randn(3, 8, 128)

        query, key, padded_value = self.impl._prepare_flash_attn_qkv(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            value,
            256,
        )

        self.assertEqual(query.shape[-1], 256)
        self.assertEqual(key.shape[-1], 256)
        self.assertEqual(padded_value.shape[-1], 256)
        self.assertTrue(torch.equal(query[..., :192], torch.cat((q_nope, q_pe), dim=-1)))
        self.assertTrue(torch.equal(key[..., :192], torch.cat((k_nope, k_pe), dim=-1)))
        self.assertTrue(torch.count_nonzero(query[..., 192:]) == 0)
        self.assertTrue(torch.count_nonzero(key[..., 192:]) == 0)
        self.assertTrue(torch.count_nonzero(padded_value[..., 128:]) == 0)

    @patch("vllm_ascend.attention.mla_v1.run_flash_attn")
    def test_flash_attn_prefill_main_contract(self, mock_flash_attn):
        query = torch.randn(3, 8, 256)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
        metadata = torch.empty(4096, dtype=torch.int32)
        expected = (torch.randn_like(query), torch.randn(3, 8))
        mock_flash_attn.return_value = expected

        result = self.impl._run_flash_attn_prefill(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            metadata=metadata,
            causal=True,
            attn_mask=torch.ones(4, 4, dtype=torch.int8),
        )

        self.assertIs(result, expected)
        kwargs = mock_flash_attn.call_args.kwargs
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_kv"], "TND")
        self.assertEqual(kwargs["max_seqlen_q"], -1)
        self.assertEqual(kwargs["max_seqlen_kv"], -1)
        self.assertTrue(kwargs["return_softmax_lse"])

    @patch("vllm_ascend.attention.mla_v1.get_ascend_device_type")
    @patch("vllm_ascend.attention.mla_v1.run_flash_mla")
    def test_flash_mla_decode_main_contract(self, mock_flash_mla, mock_device_type):
        from vllm_ascend.utils import AscendDeviceType

        mock_device_type.return_value = AscendDeviceType.A5
        num_tokens = 2
        num_heads = self.impl.num_heads
        self.impl.kv_lora_rank = 512
        self.impl.qk_rope_head_dim = 64
        self.impl.num_kv_heads = 1
        self.impl.enable_kv_nz = False

        q_nope = torch.randn(num_tokens, num_heads, 512)
        q_pe = torch.randn(num_tokens, num_heads, 64)
        unified_cache = torch.randn(4, 128, 1, 576)
        block_table = torch.zeros(2, 4, dtype=torch.int32)
        cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
        cache_seqlens = torch.tensor([16, 32], dtype=torch.int32)
        metadata_buffer = torch.empty(4096, dtype=torch.int32)
        attn_mask = torch.ones(2048, 2048, dtype=torch.int8)
        decode = AscendMLADecodeMetadata(
            input_positions=torch.empty(0),
            block_table=block_table,
            seq_lens=torch.tensor([16, 32], dtype=torch.int32),
            max_seq_lens=32,
            seq_lens_list=[16, 32],
            flash_mla_cu_seqlens_q=cu_seqlens_q,
            flash_mla_cache_seqlens=cache_seqlens,
            flash_mla_metadata=metadata_buffer,
            attn_mask=attn_mask,
        )
        attn_metadata = MagicMock()
        attn_metadata.decode = decode
        attn_metadata.causal = True
        partial_output = torch.randn(num_tokens, num_heads, 512)
        mock_flash_mla.return_value = (partial_output, torch.empty(1))
        expected = torch.randn(num_tokens, num_heads * self.impl.v_head_dim)

        with patch.object(
            self.impl,
            "_v_up_proj_batch_major",
            return_value=expected,
        ) as mock_up_proj:
            result = self.impl._forward_decode(
                q_nope,
                q_pe,
                torch.empty(0),
                torch.empty(0),
                128,
                attn_metadata,
                unified_kv_cache=unified_cache,
            )

        self.assertIs(result, expected)
        mock_up_proj.assert_called_once_with(partial_output)
        args = mock_flash_mla.call_args.args
        kwargs = mock_flash_mla.call_args.kwargs
        torch.testing.assert_close(args[0], torch.cat((q_nope, q_pe), dim=-1))
        self.assertIs(args[1], unified_cache)
        self.assertIs(kwargs["block_table"], block_table)
        self.assertIs(kwargs["cache_seqlens"], cache_seqlens)
        self.assertIs(kwargs["cu_seqlens_q"], cu_seqlens_q)
        self.assertIs(kwargs["attn_mask"], attn_mask)
        self.assertIs(kwargs["metadata"], metadata_buffer)
        self.assertEqual(kwargs["head_dim_v"], 512)
        self.assertEqual(kwargs["softmax_scale"], self.impl.scale)
        self.assertEqual(kwargs["mask_mode"], 3)
        self.assertEqual(kwargs["layout_q"], "TND")
        self.assertEqual(kwargs["layout_kv"], "PA_BBND")
        self.assertEqual(kwargs["layout_out"], "TND")
        self.assertFalse(kwargs["return_softmax_lse"])

    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    def test_init_head_padding_for_non_power_of_two(self, mock_get_current_vllm_config):
        """Test head padding computation for num_heads that are not power of 2 (e.g. GLM-4.7-Flash with 20 heads)."""
        mock_get_current_vllm_config.return_value = MagicMock()
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
            "kv_a_layernorm": MagicMock(),
            "rotary_emb": MagicMock(),
            "g_proj": None,
            "use_mla_rope": True,
        }
        impl = AscendMLAImpl(
            num_heads=20,
            head_size=1024,
            scale=0.1,
            num_kv_heads=20,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        self.assertEqual(impl.num_heads, 20)
        self.assertEqual(impl.num_heads_padded, 32)  # next power of 2
        self.assertEqual(impl.head_padding, 12)  # 32 - 20

    def test_q_proj_and_k_up_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)
        q_proj_output = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)
        self.impl.q_proj.return_value = (q_proj_output,)
        if not hasattr(self.impl, "W_UK_T") or self.impl.W_UK_T is None:
            self.impl.W_UK_T = torch.randn(self.impl.num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank)
        result = self.impl._q_proj_and_k_up_proj(x)
        ql_nope, q_pe = result
        self.assertEqual(ql_nope.shape[0], batch_size)
        self.assertEqual(ql_nope.shape[1], self.impl.num_heads)
        self.assertEqual(ql_nope.shape[2], self.impl.kv_lora_rank)
        self.assertEqual(q_pe.shape[0], batch_size)
        self.assertEqual(q_pe.shape[1], self.impl.num_heads)
        self.assertEqual(q_pe.shape[2], self.impl.qk_rope_head_dim)

    @patch("torch_npu.npu_interleave_rope")
    def test_rope_single(self, mock_npu_interleave_rope):
        batch_size = 2
        seq_len = 10
        dim = 32

        x = torch.randn(batch_size, seq_len, dim)
        cos = torch.randn(seq_len, dim)
        sin = torch.randn(seq_len, dim)

        mock_npu_interleave_rope.return_value = torch.randn(batch_size, seq_len, 1, dim)

        result = self.impl.rope_single(x, cos, sin)

        self.assertEqual(result.shape, (batch_size, seq_len, dim))
        mock_npu_interleave_rope.assert_called_once()

    def test_forward_mha_not_implemented(self):
        layer_name = "layer_0"
        hidden_states = torch.randn(2, 10, 768)
        kv_cache = [torch.randn(10, 1, 1, 768), torch.randn(10, 1, 1, 768)]
        attn_metadata = MagicMock()

        with self.assertRaises(NotImplementedError) as ctx:
            self.impl.forward_mha(layer_name, hidden_states, kv_cache, attn_metadata)
        self.assertIn(
            "forward_mha is not supported for MLA attention. Use forward() instead.",
            str(ctx.exception),
        )

    def test_forward_mqa_not_implemented(self):
        layer_name = "layer_0"
        hidden_states = torch.randn(2, 10, 768)
        kv_cache = [torch.randn(10, 1, 1, 768), torch.randn(10, 1, 1, 768)]
        attn_metadata = MagicMock()

        with self.assertRaises(NotImplementedError) as ctx:
            self.impl.forward_mqa(layer_name, hidden_states, kv_cache, attn_metadata)
        self.assertIn(
            "forward_mqa is not supported for MLA attention. Use forward() instead.",
            str(ctx.exception),
        )

    @patch("vllm_ascend.attention.mla_v1.torch_npu")
    def test_v_up_proj(self, mock_torch_npu):
        batch_size = 4
        x = torch.randn(self.impl.num_heads, batch_size, self.impl.kv_lora_rank)
        if not hasattr(self.impl, "W_UV") or self.impl.W_UV is None:
            self.impl.W_UV = torch.randn(self.impl.num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim)

        expected_shape = (batch_size, self.impl.num_heads * self.impl.v_head_dim)
        mock_torch_npu.npu_transpose_batchmatmul.return_value = torch.randn(*expected_shape)

        result = self.impl._v_up_proj(x)
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], self.impl.num_heads * self.impl.v_head_dim)

    @patch("vllm_ascend.attention.mla_v1.get_draft_graph_params")
    @patch("vllm_ascend.attention.mla_v1.get_graph_params")
    @patch("torch.npu.stream")
    @patch("torch.npu.graph_task_update_begin")
    @patch("torch.npu.graph_task_update_end")
    @patch("torch_npu.npu_fused_infer_attention_score_v2.out")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_update_graph_params(
        self,
        mock_get_forward_context,
        mock_fia,
        mock_update_end,
        mock_update_begin,
        mock_stream,
        mock_get_graph_params,
        mock_get_draft_graph_params,
    ):
        mock_update_stream = MagicMock()
        mock_forward_context = MagicMock()
        mock_attn_metadata = MagicMock()
        mock_forward_context.attn_metadata = {"layer_0": mock_attn_metadata}
        mock_attn_metadata.decode.seq_lens_list = [10, 20, 30]
        mock_attn_metadata.decode.actual_seq_lengths_q = [10, 20, 30]
        mock_attn_metadata.decode.block_table = torch.randint(0, 100, (3, 4))
        mock_graph_params = MagicMock()

        mock_graph_params.attn_params = {
            100: [
                (
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ),
            ]
        }
        mock_graph_params.handles = {100: [MagicMock()]}
        mock_graph_params.events = {100: [MagicMock()]}
        mock_graph_params.workspaces = {100: MagicMock()}

        mock_get_graph_params.return_value = mock_graph_params
        mock_get_draft_graph_params.return_value = mock_graph_params

        # forward context
        mock_ctx = MagicMock()
        mock_get_forward_context.return_value = mock_ctx

        # speculative_config
        mock_speculative_config = MagicMock()
        mock_speculative_config.disable_padded_drafter_batch = False

        mock_ctx.is_draft_model = True
        mock_ctx.is_draft_model_prefill = False
        AscendMLAImpl.update_graph_params(
            mock_update_stream,
            mock_forward_context,
            100,
            speculative_config=mock_speculative_config,
            draft_attn_metadatas=[{"layer_0": mock_attn_metadata}],
        )

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_update_graph_params_empty_layers(self, mock_get_forward_context):
        # if num_layers == 0
        mock_update_stream = MagicMock()
        mock_forward_context = MagicMock()
        mock_forward_context.attn_metadata = {}

        mock_ctx = MagicMock()
        mock_ctx.is_draft_model = False
        mock_get_forward_context.return_value = mock_ctx

        AscendMLAImpl.update_graph_params(mock_update_stream, mock_forward_context, 100)

    @patch("vllm_ascend.attention.mla_v1.get_graph_params")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    @patch("torch.npu.graph_task_update_end")
    @patch("torch.npu.graph_task_update_begin")
    @patch("torch.npu.stream")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_update_graph_params_with_mtp(
        self,
        mock_get_forward_context,
        mock_npu_stream,
        mock_graph_task_update_begin,
        mock_graph_task_update_end,
        mock_npu_fused_infer,
        mock_get_graph_params,
    ):
        mock_update_stream = MagicMock()
        mock_forward_context = MagicMock()

        mock_attn_metadata = MagicMock()
        mock_attn_metadata.decode = MagicMock()
        mock_attn_metadata.decode.seq_lens_list = [10, 20, 30]
        mock_attn_metadata.decode.actual_seq_lengths_q = [10, 20, 30]
        mock_forward_context.attn_metadata = {"layer_0": mock_attn_metadata}

        # forward context
        mock_ctx = MagicMock()
        mock_ctx.is_draft_model = False
        mock_get_forward_context.return_value = mock_ctx

        mock_stream_context = MagicMock()
        mock_npu_stream.return_value = mock_stream_context

        mock_out = MagicMock()
        mock_npu_fused_infer.out = mock_out

        mock_graph_params = MagicMock()

        mock_graph_params.attn_params = {
            100: [
                (
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                ),
            ]
        }
        mock_graph_params.handles = {100: [MagicMock()]}
        mock_graph_params.events = {100: [MagicMock()]}
        mock_get_graph_params.return_value = mock_graph_params

        mock_speculative_config = MagicMock()
        mock_speculative_config.method = "mtp"
        mock_speculative_config.num_speculative_tokens = 4

        AscendMLAImpl.update_graph_params(
            mock_update_stream, mock_forward_context, 100, speculative_config=mock_speculative_config
        )

    def test_get_context_seq_len_npu(self):
        mock_attn_metadata = MagicMock()
        mock_prefill_metadata = MagicMock()
        mock_chunked_context = MagicMock()
        mock_chunked_context.chunk_seq_lens_npu = torch.tensor([10, 20, 30])
        mock_chunked_context.seq_tot = [10, 30, 60]
        mock_prefill_metadata.chunked_context = mock_chunked_context
        mock_attn_metadata.prefill = mock_prefill_metadata

        result = self.impl.get_context_seq_len_npu(1, mock_attn_metadata)
        self.assertEqual(result, 20)

    def test_reorg_kvcache(self):
        kv_c_normed = torch.randn(2, 4, 8)
        k_pe = torch.randn(2, 4, 8)
        mock_chunked_context = MagicMock()
        result_kv, result_k_pe = self.impl._reorg_kvcache(kv_c_normed, k_pe, mock_chunked_context, 0, 10)
        self.assertIs(result_kv, kv_c_normed)
        self.assertIs(result_k_pe, k_pe)

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_for_fused_fa_quant(self, mock_format_cast):
        mock_format_cast.return_value = torch.randn(128, 128)

        self.impl.fa_quant_layer = True
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.weight.data = torch.randn(128, 128)
        self.impl.q_proj.weight_scale.data = torch.randn(128, 128)
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, 128, 64)
        self.impl.fused_qkv_a_proj.weight_scale = torch.randn(64)

        mock_layer = MagicMock()
        mock_layer.quant_kscale = torch.randn(128)
        mock_layer.fak_descale_float = torch.randn(1)
        self.impl.vllm_config = MagicMock()
        self.impl.vllm_config.compilation_config = MagicMock()
        self.impl.vllm_config.compilation_config.static_forward_context = {"layer_0": mock_layer}
        self.impl.layer_name = "layer_0"

        self.impl.enable_mlapo = False
        self.impl._process_weights_for_fused(torch.float16)
        self.assertTrue(hasattr(self.impl, "weight_uq_qr"))
        self.assertTrue(hasattr(self.impl, "weight_dq"))
        self.assertTrue(hasattr(self.impl, "weight_dkv_kr"))

    def _setup_mlapo_weight_prep_mocks(self):
        self.impl.enable_mlapo = True
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.weight.data = torch.randn(128, 128)
        self.impl.q_proj.weight_scale.data = torch.randn(128, 128)
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, 128, 64)
        self.impl.fused_qkv_a_proj.weight_scale = torch.randn(64)
        self.impl.q_lora_rank = 32
        self.impl.vllm_config.scheduler_config.max_num_batched_tokens = 4096
        self.impl.vllm_config.kv_transfer_config = MagicMock()
        self.impl.vllm_config.kv_transfer_config.is_kv_consumer = False

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_for_fused_mlapo(self, mock_format_cast):
        mock_format_cast.return_value = torch.randn(128, 128)

        self.impl.fa_quant_layer = False
        self._setup_mlapo_weight_prep_mocks()

        self.impl._process_weights_for_fused(torch.float16)
        self.assertTrue(hasattr(self.impl, "weight_uq_qr"))
        self.assertTrue(hasattr(self.impl, "weight_dq"))
        self.assertTrue(hasattr(self.impl, "weight_dkv_kr"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dq"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_uq_qr"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dkv_kr"))
        self.assertFalse(hasattr(self.impl, "quant_kscale"))

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_for_fused_mlapo_with_fa_quant(self, mock_format_cast):
        mock_format_cast.return_value = torch.randn(128, 128)

        self.impl.fa_quant_layer = True
        self._setup_mlapo_weight_prep_mocks()

        mock_layer = MagicMock()
        mock_layer.quant_kscale = torch.randn(1, 512)
        mock_layer.fak_descale_float = torch.randn(1)
        self.impl.vllm_config.compilation_config = MagicMock()
        self.impl.vllm_config.compilation_config.static_forward_context = {"layer_0": mock_layer}
        self.impl.layer_name = "layer_0"

        self.impl._process_weights_for_fused(torch.float16)
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dq"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dkv_kr"))
        self.assertTrue(hasattr(self.impl, "quant_kscale"))
        self.assertTrue(hasattr(self.impl, "fak_descale_float"))

    @patch("vllm_ascend.attention.mla_v1.get_current_hardware_profile")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_for_fused_mlapo_a5(self, mock_format_cast, mock_get_ascend_device_type):
        mock_format_cast.return_value = torch.randn(128, 128)

        self.impl.enable_mlapo = True
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, 128, 64)
        self.impl.fused_qkv_a_proj.weight_scale = torch.randn(64, 128, 128)
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.weight.data = torch.randn(128, 128)
        self.impl.q_proj.weight_scale.data = torch.randn(128, 128, 128)
        self.impl.q_lora_rank = 32
        self.impl._mlapo_quant_type = object
        self.impl._mlapo_uses_native_weights = False

        mock_get_ascend_device_type.return_value = get_hardware_profile(AscendDeviceType.A5)
        self.impl._process_weights_for_fused(torch.float16)
        self.assertTrue(hasattr(self.impl, "weight_dq"))
        self.assertTrue(hasattr(self.impl, "weight_uq_qr"))
        self.assertTrue(hasattr(self.impl, "weight_dkv_kr"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dq"))
        self.assertTrue(hasattr(self.impl, "dequant_scale_w_dkv_kr"))

    @patch("vllm_ascend.attention.mla_v1.DeviceOperator")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("torch_npu.npu_attention_update")
    def test__forward_prefill(self, mock_npu_attention_update, mock_fia, mock_device_operator):
        batch_size = 2

        # create input tensors
        q_nope = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_nope_head_dim)
        k_pe = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_rope_head_dim)
        value = torch.randn(batch_size, self.impl.num_heads, self.impl.v_head_dim)

        kv_c_and_k_pe_cache = [torch.randn(10, 1, 1, 192), torch.randn(10, 1, 1, 32)]

        attn_metadata = MagicMock()
        prefill_metadata = MagicMock()
        prefill_metadata.actual_seq_lengths_q = [10, 20]
        prefill_metadata.attn_mask = torch.randn(1, 1, 20, 20)
        prefill_metadata.chunked_context = MagicMock()
        prefill_metadata.chunked_context.seq_tot = [10, 10]
        prefill_metadata.chunked_context.starts = [0, 10]
        prefill_metadata.chunked_context.chunk_seq_lens_npu = [10, 10]
        prefill_metadata.chunked_context.chunk_actual_seq_lengths_kv_list = [[10], [10]]
        prefill_metadata.block_table = torch.randint(0, 100, (2, 4))
        attn_metadata.prefill = prefill_metadata

        mock_device_operator.kv_cache_load = MagicMock()

        mock_fia.return_value = (
            torch.randn(batch_size, self.impl.num_heads, self.impl.v_head_dim),
            torch.randn(self.impl.num_heads, batch_size),
        )

        mock_npu_attention_update.return_value = (
            torch.randn(batch_size, self.impl.num_heads, self.impl.v_head_dim),
            None,
        )

        mock_kv_b_proj = MagicMock()
        # create [toks, num_heads, qk_nope_head_dim + v_head_dim] tensor
        toks = 10
        kv_nope_shape = (toks, self.impl.num_heads, self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        mock_kv_b_proj.return_value = (torch.randn(kv_nope_shape), None)
        self.impl.kv_b_proj = mock_kv_b_proj

        result = self.impl._forward_prefill(q_nope, q_pe, k_nope, k_pe, value, kv_c_and_k_pe_cache, attn_metadata)

        # verify result shape
        self.assertEqual(result.shape[0], batch_size)
        self.assertEqual(result.shape[1], self.impl.num_heads * self.impl.v_head_dim)

    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.mla_v1.DeviceOperator")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_forward_prefill_non_power_of_two_heads(self, mock_fia, mock_device_operator, mock_get_current_vllm_config):
        """Test prefill with non-power-of-2 heads uses concat instead of query_rope/key_rope kwargs."""
        mock_get_current_vllm_config.return_value = MagicMock()
        num_heads = 20
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
            "kv_a_layernorm": MagicMock(),
            "rotary_emb": MagicMock(),
            "g_proj": None,
            "use_mla_rope": True,
        }
        impl = AscendMLAImpl(
            num_heads=num_heads,
            head_size=1024,
            scale=0.1,
            num_kv_heads=num_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        batch_size = 2
        q_nope = torch.randn(batch_size, num_heads, impl.qk_nope_head_dim)
        q_pe = torch.randn(batch_size, num_heads, impl.qk_rope_head_dim)
        k_nope = torch.randn(batch_size, num_heads, impl.qk_nope_head_dim)
        k_pe = torch.randn(batch_size, num_heads, impl.qk_rope_head_dim)
        value = torch.randn(batch_size, num_heads, impl.v_head_dim)
        kv_c_and_k_pe_cache = [torch.randn(10, 1, 1, 192), torch.randn(10, 1, 1, 32)]

        attn_metadata = MagicMock()
        prefill_metadata = MagicMock()
        prefill_metadata.actual_seq_lengths_q = [10, 20]
        prefill_metadata.attn_mask = torch.randn(1, 1, 20, 20)
        prefill_metadata.chunked_context = None
        attn_metadata.prefill = prefill_metadata

        mock_device_operator.kv_cache_load = MagicMock()
        mock_fia.return_value = (
            torch.randn(batch_size, num_heads, impl.v_head_dim),
            torch.randn(num_heads, batch_size),
        )

        result = impl._forward_prefill(q_nope, q_pe, k_nope, k_pe, value, kv_c_and_k_pe_cache, attn_metadata)

        # FIA should be called without query_rope/key_rope when head_padding > 0
        mock_fia.assert_called_once()
        call_kwargs = mock_fia.call_args.kwargs
        self.assertNotIn("query_rope", call_kwargs)
        self.assertNotIn("key_rope", call_kwargs)
        self.assertEqual(call_kwargs.get("num_heads"), num_heads)
        self.assertEqual(result.shape, (batch_size, num_heads * impl.v_head_dim))

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading(self, mock_format_cast):
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight
        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_mlapo(self, mock_format_cast):
        # test with enable_mlapo=True
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight

        self.impl.enable_mlapo = True

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    @patch("vllm_ascend.attention.mla_v1.get_current_hardware_profile")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_mlapo_a5(self, mock_format_cast, mock_get_ascend_device_type):
        # test with enable_mlapo=True and device_type=A5
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight
        self.impl.enable_mlapo = True
        mock_fused_qkv_a_proj = MagicMock()
        mock_quant_method = MagicMock()

        from vllm_ascend.attention.mla_v1 import AscendW8A8LinearMethod

        mock_quant_method.quant_method = MagicMock(spec=AscendW8A8LinearMethod)
        mock_fused_qkv_a_proj.quant_method = mock_quant_method
        self.impl.fused_qkv_a_proj = mock_fused_qkv_a_proj

        # set device_type=A5

        mock_get_ascend_device_type.return_value = get_hardware_profile(AscendDeviceType.A5)

        self.impl._process_weights_for_fused = MagicMock()

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.impl._process_weights_for_fused.assert_called_once_with(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    @patch("vllm_ascend.attention.mla_v1.get_current_hardware_profile")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_mlapo_non_a5(self, mock_format_cast, mock_get_ascend_device_type):
        # test with enable_mlapo=True and device_type!=A5
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight

        self.impl.enable_mlapo = True

        mock_fused_qkv_a_proj = MagicMock()
        mock_quant_method = MagicMock()
        from vllm_ascend.attention.mla_v1 import AscendW8A8LinearMethod

        mock_quant_method.quant_method = MagicMock(spec=AscendW8A8LinearMethod)
        mock_fused_qkv_a_proj.quant_method = mock_quant_method
        self.impl.fused_qkv_a_proj = mock_fused_qkv_a_proj

        mock_get_ascend_device_type.return_value = get_hardware_profile(AscendDeviceType.A2)

        self.impl._process_weights_for_fused = MagicMock()

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.impl._process_weights_for_fused.assert_called_once_with(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    @patch("vllm_ascend.attention.mla_v1.maybe_trans_nz")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_fa_quant(self, mock_format_cast, mock_maybe_trans_nz):
        # test with enable_mlapo=False and fa_quant_layer=True
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight

        self.impl.enable_mlapo = False
        self.impl.fa_quant_layer = True

        self.impl._process_weights_for_fused = MagicMock()

        mock_maybe_trans_nz.return_value = torch.randn(1, 2, 3)

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.impl._process_weights_for_fused.assert_called_once()

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_kv_b_proj(self, mock_format_cast):
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        mock_format_cast.return_value = layer.weight

        self.impl.enable_mlapo = False
        self.impl.fa_quant_layer = False

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

    def test_compute_prefill_context_none(self):
        batch_size = 4
        kv_cache = torch.randn(10, 1, 1, 192)
        query = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)
        metadata = MagicMock()
        metadata.prefill = None
        prefix_out = torch.randn(2, 16, 128)
        prefix_lse = torch.randn(2, 16, 8)
        q_pe = query[..., self.impl.qk_nope_head_dim :]
        q_nope = query[..., : self.impl.qk_nope_head_dim]

        out, lse = self.impl._compute_prefill_context(q_nope, q_pe, kv_cache, 32, metadata, prefix_out, prefix_lse)

        self.assertTrue(torch.equal(prefix_out, out))
        self.assertTrue(torch.equal(prefix_lse, lse))

    def test_compute_prefill_context_empty_iters(self):
        # test compute_prefill_context with iters == 0
        batch_size = 4
        kv_cache = [torch.randn(10, 1, 1, 192), torch.randn(10, 1, 1, 32)]
        query = torch.randn(batch_size, self.impl.num_heads, self.impl.qk_head_dim)

        # Create a mock metadata where prefill_metadata.chunked_context.seq_tot is an empty list, so that iters == 0
        metadata = MagicMock()
        prefill_metadata = MagicMock()
        chunked_context = MagicMock()
        chunked_context.seq_tot = []  # iters == 0
        prefill_metadata.chunked_context = chunked_context
        metadata.prefill = prefill_metadata

        prefix_out = torch.randn(2, 16, 128)
        prefix_lse = torch.randn(2, 16, 8)
        q_pe = query[..., self.impl.qk_nope_head_dim :]
        q_nope = query[..., : self.impl.qk_nope_head_dim]

        out, lse = self.impl._compute_prefill_context(q_nope, q_pe, kv_cache, 32, metadata, prefix_out, prefix_lse)

        self.assertTrue(torch.equal(prefix_out, out))
        self.assertTrue(torch.equal(prefix_lse, lse))

    @patch("torch_npu.npu_gather_pa_kv_cache")
    @patch("torch_npu.npu_attention_update")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_compute_prefill_context(self, mock_fia, mock_update, mock_load):
        S, N, D, VD = 2, self.impl.num_heads, self.impl.qk_head_dim, self.impl.v_head_dim
        _, AND = self.impl.qk_rope_head_dim, self.impl.qk_nope_head_dim
        latent_kv_dim = self.impl.kv_lora_rank
        num_blocks, block_size = 100, 20
        query = torch.randn(S, N, D)
        q_nope = query[..., : self.impl.qk_nope_head_dim]
        q_pe = query[..., self.impl.qk_nope_head_dim :]
        kv_cache_0 = torch.randn(num_blocks, block_size, N, latent_kv_dim)
        kv_cache_1 = torch.randn(num_blocks, block_size, N, D)
        kv_cache = [kv_cache_0, kv_cache_1]
        prefix_out = torch.randn(S, N, VD)
        prefix_lse = torch.randn(N, S)

        self.impl.kv_b_proj.return_value = (torch.randn(8, N, VD + AND),)

        # Mock FIA to return output and lse
        mock_fia.return_value = (torch.randn(S, N, VD), torch.randn(N, S))
        # Mock attention_update to return merged output
        mock_update.return_value = (torch.randn(S * N, VD), None)

        chunk_ctx = MagicMock()
        chunk_ctx.seq_tot = [8]
        chunk_ctx.chunk_seq_lens = [torch.tensor([8])]
        chunk_ctx.chunk_seq_lens_npu = [torch.tensor([8])]
        chunk_ctx.starts = [torch.tensor([0])]

        prefill_meta = MagicMock()
        prefill_meta.chunked_context = chunk_ctx
        prefill_meta.query_lens = torch.tensor([S])
        prefill_meta.block_table = torch.randint(0, 100, (S, 4))

        meta = MagicMock()
        meta.prefill = prefill_meta
        self.impl.prefill_mask = torch.triu(torch.ones(512, 512, device=q_nope.device, dtype=q_nope.dtype), 1)

        out, lse = self.impl._compute_prefill_context(q_nope, q_pe, kv_cache, 32, meta, prefix_out, prefix_lse)

        mock_load.assert_called_once()
        mock_fia.assert_called_once()
        mock_update.assert_called_once()

        self.assertEqual(out.shape, prefix_out.shape)

    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("torch_npu.npu_gather_pa_kv_cache")
    @patch("torch_npu.npu_attention_update")
    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_compute_prefill_context_non_power_of_two_heads(
        self, mock_fia, mock_update, mock_load, mock_get_current_vllm_config
    ):
        """Test prefill context with non-power-of-2 heads uses concat for query and key."""
        mock_get_current_vllm_config.return_value = MagicMock()
        num_heads = 20
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
            "kv_a_layernorm": MagicMock(),
            "rotary_emb": MagicMock(),
            "g_proj": None,
            "use_mla_rope": True,
        }
        impl = AscendMLAImpl(
            num_heads=num_heads,
            head_size=1024,
            scale=0.1,
            num_kv_heads=num_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        S, N, D, VD = 2, num_heads, impl.qk_head_dim, impl.v_head_dim
        latent_kv_dim = impl.kv_lora_rank
        num_blocks, block_size = 100, 20
        query = torch.randn(S, N, D)
        q_nope = query[..., : impl.qk_nope_head_dim]
        q_pe = query[..., impl.qk_nope_head_dim :]
        kv_cache_0 = torch.randn(num_blocks, block_size, N, latent_kv_dim)
        kv_cache_1 = torch.randn(num_blocks, block_size, N, D)
        kv_cache = [kv_cache_0, kv_cache_1]
        prefix_out = torch.randn(S, N, VD)
        prefix_lse = torch.randn(N, S)

        impl.kv_b_proj.return_value = (torch.randn(8, N, VD + impl.qk_nope_head_dim),)

        mock_fia.return_value = (torch.randn(S, N, VD), torch.randn(N, S))
        mock_update.return_value = (torch.randn(S * N, VD), None)

        chunk_ctx = MagicMock()
        chunk_ctx.seq_tot = [8]
        chunk_ctx.chunk_seq_lens = [torch.tensor([8])]
        chunk_ctx.chunk_seq_lens_npu = [torch.tensor([8])]
        chunk_ctx.starts = [torch.tensor([0])]
        chunk_ctx.chunk_actual_seq_lengths_kv_list = [[8]]

        prefill_meta = MagicMock()
        prefill_meta.chunked_context = chunk_ctx
        prefill_meta.query_lens = torch.tensor([S])
        prefill_meta.block_table = torch.randint(0, 100, (S, 4))

        meta = MagicMock()
        meta.prefill = prefill_meta

        out, lse = impl._compute_prefill_context(q_nope, q_pe, kv_cache, 32, meta, prefix_out, prefix_lse)

        mock_fia.assert_called_once()
        call_kwargs = mock_fia.call_args.kwargs
        self.assertNotIn("query_rope", call_kwargs)
        self.assertNotIn("key_rope", call_kwargs)
        self.assertEqual(out.shape, prefix_out.shape)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.attention.mla_v1.AscendMLAImpl._v_up_proj")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode_without_graph(
        self, mock_npu_fused_infer_attention_score_v2, mock_up_proj, mock_get_forward_context
    ):
        num_tokens = 100
        block_size = 4
        q_nope = torch.randn(num_tokens, self.impl.num_heads, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(num_tokens, self.impl.num_heads, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(num_tokens, self.impl.num_heads, self.impl.qk_nope_head_dim)
        k_pe = torch.randn(num_tokens, self.impl.num_heads, self.impl.qk_rope_head_dim)
        metadata = MagicMock()
        metadata.decode = MagicMock()
        metadata.decode.block_table = MagicMock()
        metadata.decode.actual_seq_lengths = 10
        mock_npu_fused_infer_attention_score_v2.return_value = [
            torch.randn(num_tokens, self.impl.num_heads, self.impl.kv_lora_rank),
            None,
        ]
        mock_up_proj.return_value = torch.randn(num_tokens, self.impl.num_heads, self.impl.v_head_dim)
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, block_size, metadata)
        self.assertEqual(result.shape[0], num_tokens)
        self.assertEqual(result.shape[1], self.impl.num_heads)
        self.assertEqual(result.shape[2], self.impl.v_head_dim)
        mock_up_proj.assert_called_once()
        mock_npu_fused_infer_attention_score_v2.assert_called_once()

    def test_mla_preprocess(self):
        batch_size = 4
        seq_len = 8
        hidden_size = 1024
        hidden_states = torch.randn(batch_size * seq_len, hidden_size)

        kv_cache = MagicMock()

        attn_metadata = MagicMock()
        attn_metadata.num_decodes = 2
        attn_metadata.num_prefills = 2
        attn_metadata.num_decode_tokens = 2
        attn_metadata.num_actual_tokens = 4
        num_prefill_tokens = 2
        attn_metadata.slot_mapping = torch.arange(4)
        attn_metadata.decode.cos = torch.randn(2, 64)
        attn_metadata.decode.sin = torch.randn(2, 64)
        attn_metadata.prefill.cos = torch.randn(2, 64)
        attn_metadata.prefill.sin = torch.randn(2, 64)

        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.return_value = torch.randn(
            attn_metadata.num_actual_tokens, self.impl.num_heads, self.impl.qk_rope_head_dim
        )
        self.impl.kv_a_proj_with_mqa = MagicMock()
        self.impl.kv_a_proj_with_mqa.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads, self.impl.qk_rope_head_dim + self.impl.kv_lora_rank)
        ]
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.return_value = [
            torch.randn(
                num_prefill_tokens,
                self.impl.num_heads,
                self.impl.qk_rope_head_dim + self.impl.kv_lora_rank + self.impl.q_lora_rank,
            )
        ]
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.return_value = [torch.randn(num_prefill_tokens, self.impl.num_heads, self.impl.qk_head_dim)]
        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads, self.impl.v_head_dim + self.impl.qk_nope_head_dim)
        ]
        self.impl.rope_single = MagicMock(side_effect=lambda x, cos, sin: x)
        self.impl.exec_kv_decode = MagicMock()
        self.impl.exec_kv_decode.return_value = [MagicMock(), MagicMock()]
        self.impl.exec_kv_prefill = MagicMock()
        self.impl.exec_kv_prefill.return_value = [
            torch.randn(num_prefill_tokens, self.impl.num_heads, self.impl.qk_rope_head_dim),
            torch.randn(num_prefill_tokens, self.impl.num_heads, self.impl.kv_lora_rank),
        ]
        self.impl._q_proj_and_k_up_proj = MagicMock()
        self.impl._q_proj_and_k_up_proj.return_value = [MagicMock(), MagicMock()]
        self.impl.num_kv_heads = self.impl.num_heads

        decode_res, prefill_res = self.impl._mla_preprocess("mock_layer", hidden_states, kv_cache, attn_metadata)

        self.assertIsNotNone(decode_res)
        self.assertIsNotNone(prefill_res)

    @patch("vllm_ascend.attention.mla_v1.DeviceOperator.reshape_and_cache")
    @patch("torch_npu.npu_interleave_rope")
    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_mla_preprocess_prefill_without_rope(self, mock_rope_cache, mock_rope, mock_cache):
        self.impl.use_mla_rope = False
        self.impl.num_heads = self.impl.num_kv_heads = 1
        self.impl.qk_nope_head_dim = self.impl.qk_rope_head_dim = 2
        self.impl.qk_head_dim = 4
        self.impl.kv_lora_rank = self.impl.v_head_dim = 2
        self.impl.q_proj.side_effect = lambda x: (x,)
        self.impl.kv_a_layernorm = torch.nn.Identity()
        self.impl.kv_b_proj.side_effect = lambda x: (torch.cat((x, x), dim=-1),)

        q_c = torch.arange(16, dtype=torch.float32).view(4, 4)
        kv_no_split = q_c + 100
        kv_cache = (torch.empty(4, 1, 2), torch.empty(4, 1, 2))
        for num_decode_tokens in (0, 1):
            with self.subTest(num_decode_tokens=num_decode_tokens):
                num_actual_tokens = num_decode_tokens + 2
                metadata = SimpleNamespace(
                    num_decode_tokens=num_decode_tokens,
                    num_actual_tokens=num_actual_tokens,
                    slot_mapping=torch.arange(4),
                    prefill=SimpleNamespace(cos=None, sin=None),
                )
                result = self.impl.mla_preprocess_prefill(q_c, kv_no_split, kv_cache, metadata)

                q = q_c[num_decode_tokens:num_actual_tokens].unsqueeze(1)
                kv = kv_no_split[num_decode_tokens:num_actual_tokens].unsqueeze(1)
                torch.testing.assert_close(result.q_nope, q[..., :2])
                torch.testing.assert_close(result.q_pe, q[..., 2:])
                torch.testing.assert_close(result.k_nope, kv[..., :2])
                torch.testing.assert_close(result.k_pe, kv[..., 2:])
                torch.testing.assert_close(result.value, kv[..., :2])
                torch.testing.assert_close(mock_cache.call_args.kwargs["key"], kv[..., :2])
                torch.testing.assert_close(mock_cache.call_args.kwargs["value"], kv[..., 2:])
                torch.testing.assert_close(
                    mock_cache.call_args.kwargs["slot_mapping"],
                    metadata.slot_mapping[num_decode_tokens:num_actual_tokens],
                )
        mock_rope.assert_not_called()
        mock_rope_cache.assert_not_called()

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            None,
            None,
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank),
        ]

        k_pe, k_nope = self.impl.exec_kv_prefill(kv_no_split, cos, sin, kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_prefill_with_fa_quant(self, mock_kv_rmsnorm_rope_cache):
        # if fa_quant_layer is True
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.fa_quant_layer = True
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        block_size = 1

        mock_kv_rmsnorm_rope_cache.return_value = [
            None,
            None,
            torch.randn(B, N, block_size, self.impl.qk_rope_head_dim),
            torch.randn(B, N, block_size, self.impl.kv_lora_rank),
        ]

        k_pe, k_nope = self.impl.exec_kv_prefill(kv_no_split, cos, sin, kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_decode(self, mock_kv_rmsnorm_rope_cache):
        B = 2
        N = self.impl.num_kv_heads
        D = self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        kv_no_split = torch.randn(B, N, D)
        self.impl.enable_kv_nz = None
        self.impl.kv_a_layernorm.weight = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = MagicMock()
        cos = MagicMock()
        sin = MagicMock()
        slots = MagicMock()
        kv_cache = [MagicMock(), MagicMock()]

        mock_kv_rmsnorm_rope_cache.return_value = [
            torch.randn(B, N, 1, self.impl.qk_rope_head_dim),
            torch.randn(B, N, 1, self.impl.kv_lora_rank),
            None,
            None,
        ]

        k_pe, k_nope = self.impl.exec_kv_decode(kv_no_split, cos, sin, kv_cache, slots)

        self.assertEqual(k_pe.shape[-1], self.impl.qk_rope_head_dim)
        self.assertEqual(k_nope.shape[-1], self.impl.kv_lora_rank)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode(self, mock_npu_fused_infer_attention_score_v2, mock_get_forward_context):
        B = 2
        N = self.impl.num_kv_heads
        BS = 100
        HD = self.impl.v_head_dim
        self.impl.kv_lora_rank = 256
        self.impl.spec_token_num = 1
        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(B, N, HD)
        q_nope = torch.randn(B, N, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(B, N, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, N, self.impl.kv_lora_rank)
        k_pe = torch.randn(BS, N, self.impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_qlen = MagicMock()
        attn_metadata.decode.actual_seq_kvlen = MagicMock()
        self.impl.enable_kv_nz = True

        mock_npu_fused_infer_attention_score_v2.return_value = [torch.randn(B, N, self.impl.kv_lora_rank), None]
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS, attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], N)
        self.assertEqual(result.shape[2], HD)

    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode_non_power_of_two_heads(
        self, mock_npu_fused_infer_attention_score_v2, mock_get_forward_context, mock_get_current_vllm_config
    ):
        """Test decode with non-power-of-2 heads pads to next power of 2 and slices output."""
        mock_get_current_vllm_config.return_value = MagicMock()
        num_heads = 20
        kwargs = {
            "kv_lora_rank": 256,
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
            "kv_a_layernorm": MagicMock(),
            "rotary_emb": MagicMock(),
            "g_proj": None,
            "use_mla_rope": True,
        }
        impl = AscendMLAImpl(
            num_heads=num_heads,
            head_size=1024,
            scale=0.1,
            num_kv_heads=num_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        B = 2
        BS = 100
        HD = impl.v_head_dim
        impl.spec_token_num = 1
        impl._v_up_proj = MagicMock()
        impl._v_up_proj.return_value = torch.randn(B, num_heads, HD)
        q_nope = torch.randn(B, num_heads, impl.qk_nope_head_dim)
        q_pe = torch.randn(B, num_heads, impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, num_heads, impl.kv_lora_rank)
        k_pe = torch.randn(BS, num_heads, impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_qlen = MagicMock()
        attn_metadata.decode.actual_seq_kvlen = MagicMock()
        impl.enable_kv_nz = True
        impl.fa_quant_layer = False

        # Return padded output so slice logic works
        mock_npu_fused_infer_attention_score_v2.return_value = [
            torch.randn(impl.num_heads_padded, B, impl.kv_lora_rank),
            None,
        ]
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS, attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], num_heads)
        self.assertEqual(result.shape[2], HD)

        # Verify num_query_heads passed to FIA is padded
        mock_npu_fused_infer_attention_score_v2.assert_called_once()
        call_kwargs = mock_npu_fused_infer_attention_score_v2.call_args.kwargs
        self.assertEqual(call_kwargs.get("num_query_heads"), impl.num_heads_padded)

    @patch("vllm_ascend.attention.mla_v1.get_current_vllm_config")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode_non_power_of_two_heads_normal(
        self, mock_npu_fused_infer_attention_score_v2, mock_get_forward_context, mock_get_current_vllm_config
    ):
        """Test normal decode (BNSD_NBSD) with non-power-of-2 heads pads q and slices output."""
        mock_get_current_vllm_config.return_value = MagicMock()
        num_heads = 20
        kwargs = {
            "kv_lora_rank": 256,
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
            "kv_a_layernorm": MagicMock(),
            "rotary_emb": MagicMock(),
            "g_proj": None,
            "use_mla_rope": True,
        }
        impl = AscendMLAImpl(
            num_heads=num_heads,
            head_size=1024,
            scale=0.1,
            num_kv_heads=num_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        B = 2
        BS = 100
        HD = impl.v_head_dim
        impl.spec_token_num = 1
        impl._v_up_proj = MagicMock()
        impl._v_up_proj.return_value = torch.randn(B, num_heads, HD)
        q_nope = torch.randn(B, num_heads, impl.qk_nope_head_dim)
        q_pe = torch.randn(B, num_heads, impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, num_heads, impl.kv_lora_rank)
        k_pe = torch.randn(BS, num_heads, impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_qlen = MagicMock()
        attn_metadata.decode.actual_seq_kvlen = MagicMock()
        attn_metadata.decode.block_table = MagicMock()
        impl.enable_kv_nz = False
        impl.fa_quant_layer = False
        impl.speculative_config = None

        mock_npu_fused_infer_attention_score_v2.return_value = [
            torch.randn(impl.num_heads_padded, B, 1, impl.kv_lora_rank),
            None,
        ]
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        result = impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS, attn_metadata)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], num_heads)
        self.assertEqual(result.shape[2], HD)

        mock_npu_fused_infer_attention_score_v2.assert_called_once()
        call_kwargs = mock_npu_fused_infer_attention_score_v2.call_args.kwargs
        self.assertEqual(call_kwargs.get("num_query_heads"), impl.num_heads_padded)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode_with_fa_quant(self, mock_npu_fused_infer_attention_score_v2, mock_get_forward_context):
        # test fa_quant_layer is True
        B = 2
        N = self.impl.num_heads  # use num_heads instead of num_kv_heads
        BS = 100
        HD = self.impl.v_head_dim
        self.impl.kv_lora_rank = 256
        self.impl.spec_token_num = 1
        self.impl._v_up_proj = MagicMock()
        self.impl._v_up_proj.return_value = torch.randn(B, self.impl.num_kv_heads, HD)
        q_nope = torch.randn(B, N, self.impl.qk_nope_head_dim)
        q_pe = torch.randn(B, N, self.impl.qk_rope_head_dim)
        k_nope = torch.randn(BS, self.impl.num_kv_heads, self.impl.kv_lora_rank)
        k_pe = torch.randn(BS, self.impl.num_kv_heads, self.impl.qk_rope_head_dim)
        attn_metadata = MagicMock()
        attn_metadata.attn_state = AscendAttentionState.SpecDecoding
        attn_metadata.decode = MagicMock()
        attn_metadata.decode.actual_seq_qlen = MagicMock()
        attn_metadata.decode.actual_seq_kvlen = MagicMock()
        attn_metadata.decode.actual_seq_lengths_q = [10, 20]
        attn_metadata.decode.attn_mask = MagicMock()
        self.impl.fa_quant_layer = True
        self.impl.speculative_config = MagicMock()
        self.impl.fak_descale_float = torch.randn(1)  # add fak_descale_float attribute

        mock_npu_fused_infer_attention_score_v2.return_value = [
            torch.randn(B, self.impl.num_kv_heads, self.impl.kv_lora_rank),
            None,
        ]
        mock_get_forward_context.return_value = MagicMock(capturing=False)
        dequant_scale_q_nope = torch.randn(B, N)  # shape is [B, num_heads]
        result = self.impl._forward_decode(q_nope, q_pe, k_nope, k_pe, BS, attn_metadata, dequant_scale_q_nope)

        self.assertEqual(result.shape[0], B)
        self.assertEqual(result.shape[1], self.impl.num_kv_heads)
        self.assertEqual(result.shape[2], HD)
