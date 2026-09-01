from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionPCPImpl,
    AscendAttentionPCPMetadata,
    AscendAttentionPCPMetadataBuilder,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    PagedAttentionGraphParam,
    cache_graph_workspace,
    needs_layer_aware_fia_graph_replay,
    using_paged_attention,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
from vllm_ascend.utils import AscendDeviceType

LARGE_HEAD_PREFILL_PATH = "vllm_ascend.device.utils.npu_large_head_prefill_attention"


class TestAttentionGraphHelpers(TestBase):
    def test_cache_graph_workspace_keeps_first_workspace_by_default(self):
        graph_params = SimpleNamespace(workspaces={1: torch.empty(4)})
        candidate_workspace = torch.empty(8)

        result = cache_graph_workspace(graph_params, 1, candidate_workspace, use_max_workspace=False)

        self.assertEqual(result.numel(), 4)
        self.assertEqual(graph_params.workspaces[1].numel(), 4)

    def test_cache_graph_workspace_updates_to_larger_workspace(self):
        graph_params = SimpleNamespace(workspaces={1: torch.empty(4)})
        candidate_workspace = torch.empty(8)

        result = cache_graph_workspace(graph_params, 1, candidate_workspace, use_max_workspace=True)

        self.assertEqual(result.numel(), 8)
        self.assertEqual(graph_params.workspaces[1].numel(), 8)

    def test_large_head_uses_paged_attention_on_a2(self):
        vllm_config = MagicMock()
        vllm_config.speculative_config = None
        with patch(
            "vllm_ascend.attention.utils.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A2),
        ):
            self.assertTrue(using_paged_attention(1, vllm_config, head_size=FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE))


class TestAscendAttentionBackend(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()

        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1

        self.mock_config.parallel_config = mock_parallel_config

        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()

        from vllm_ascend.attention.utils import enable_dcp, enable_pcp

        self.enable_dcp = enable_dcp
        self.enable_pcp = enable_pcp
        enable_dcp.cache_clear()
        enable_pcp.cache_clear()
        self.addCleanup(self.utils_patcher.stop)
        self.addCleanup(enable_dcp.cache_clear)
        self.addCleanup(enable_pcp.cache_clear)

    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        self.assertEqual(AscendAttentionBackend.get_impl_cls(), AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        self.assertEqual(AscendAttentionBackend.get_builder_cls(), AscendAttentionMetadataBuilder)

    def test_get_impl_cls_with_pcp(self):
        self.mock_config.parallel_config.prefill_context_parallel_size = 2
        self.enable_pcp.cache_clear()

        self.assertIs(
            AscendAttentionBackend.get_impl_cls(),
            AscendAttentionPCPImpl,
        )

    def test_get_builder_cls_with_pcp(self):
        self.mock_config.parallel_config.prefill_context_parallel_size = 2
        self.enable_pcp.cache_clear()

        self.assertIs(
            AscendAttentionBackend.get_builder_cls(),
            AscendAttentionPCPMetadataBuilder,
        )

    def test_pcp_and_dcp_are_rejected_together(self):
        self.mock_config.parallel_config.prefill_context_parallel_size = 2
        self.mock_config.parallel_config.decode_context_parallel_size = 2
        self.enable_pcp.cache_clear()
        self.enable_dcp.cache_clear()

        with self.assertRaisesRegex(NotImplementedError, "PCP and DCP"):
            AscendAttentionBackend.get_impl_cls()
        with self.assertRaisesRegex(NotImplementedError, "PCP and DCP"):
            AscendAttentionBackend.get_builder_cls()

    def test_get_kv_cache_shape(self):
        with patch.object(attn_module.envs, "VLLM_KV_CACHE_LAYOUT", None):
            result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_get_kv_cache_shape_uses_bnsd_for_hnd_layouts(self):
        for layout in ("LBHNC", "HND"):
            with self.subTest(layout=layout), patch.object(attn_module.envs, "VLLM_KV_CACHE_LAYOUT", layout):
                result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
            self.assertEqual(result, (2, 10, 30, 20, 40))

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

    @patch.object(AscendAttentionMetadataBuilder, "metadata_cls")
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


def test_pcp_metadata_keeps_expanded_slot_mapping() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.pcp_size = 2
    common_metadata = SimpleNamespace(
        slot_mapping=torch.tensor(
            [10, 11, -1, -1, 20, 21, -1, -1],
            dtype=torch.int64,
        )
    )
    metadata = AscendAttentionPCPMetadata(
        num_actual_tokens=3,
        num_decode_tokens=1,
        num_prefills=1,
        attn_state=AscendAttentionState.PrefillCacheHit,
    )

    with patch.object(
        AscendAttentionMetadataBuilder,
        "build",
        return_value=metadata,
    ):
        result = builder.build(0, common_metadata)

    assert result.slot_mapping is common_metadata.slot_mapping
    assert result.pcp_local_num_input_tokens == 4
    assert result.attn_state == AscendAttentionState.ChunkedPrefill


def test_pcp_cache_write_uses_gathered_inputs() -> None:
    impl = AscendAttentionPCPImpl.__new__(AscendAttentionPCPImpl)
    impl.attn_type = attn_module.AttentionType.DECODER
    impl.key_cache = None
    impl.value_cache = None
    impl.kv_sharing_target_layer_name = None
    impl.is_kv_producer = True
    impl.use_bnsd_kv_cache = False

    query = torch.empty((4, 2, 1))
    output = torch.empty((4, 2, 1))
    key = torch.arange(4).reshape(4, 1, 1)
    value = key + 10
    gathered_key = torch.arange(20, 27).reshape(7, 1, 1)
    gathered_value = torch.arange(30, 37).reshape(7, 1, 1)
    gathered_slots = torch.tensor([10, 11, -1, -1, 21, -1, -1])
    slot_mapping = torch.tensor([10, 11, -1, -1, 20, 21, -1, -1])
    metadata = AscendAttentionPCPMetadata(
        num_actual_tokens=3,
        num_decode_tokens=1,
        pcp_local_num_input_tokens=4,
        slot_mapping=slot_mapping,
    )
    key_cache = torch.empty((8, 1, 1))
    value_cache = torch.empty((8, 1, 1))

    with (
        patch(
            "vllm_ascend.attention.attention_v1._gather_prefill_cache_inputs",
            return_value=(
                (gathered_key, gathered_value),
                gathered_slots,
            ),
        ) as gather_inputs,
        patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache") as reshape_and_cache,
        patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written"),
    ):
        result = impl.reshape_and_cache(
            query,
            key,
            value,
            (key_cache, value_cache),
            metadata,
            output,
        )

    local_inputs, actual_slots, num_decode_tokens = gather_inputs.call_args.args
    torch.testing.assert_close(local_inputs[0], key)
    torch.testing.assert_close(local_inputs[1], value)
    assert actual_slots is slot_mapping
    assert num_decode_tokens == 1

    cache_args = reshape_and_cache.call_args.kwargs
    torch.testing.assert_close(cache_args["key"], gathered_key)
    torch.testing.assert_close(cache_args["value"], gathered_value)
    torch.testing.assert_close(cache_args["slot_mapping"], gathered_slots)
    assert metadata.slot_mapping is slot_mapping
    assert metadata.num_actual_tokens == 3
    assert result[0] is query
    assert result[1] is key
    assert result[2] is value
    assert result[3] is output


def test_pcp_builder_keeps_short_extend_in_prefill() -> None:
    builder = AscendAttentionPCPMetadataBuilder.__new__(AscendAttentionPCPMetadataBuilder)
    builder.decode_threshold = 1
    common_metadata = SimpleNamespace(
        context_parallel_metadata=None,
        max_query_len=4,
        num_reqs=2,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 1, 5], dtype=torch.int32),
        is_prefilling=torch.tensor([True, True], dtype=torch.bool),
    )

    assert builder._split_decodes_and_prefills(common_metadata) == (0, 2, 0, 5)


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
        self.utils_config_patcher = patch(
            "vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_vllm_config
        )
        self.config_patcher.start()
        self.utils_config_patcher.start()
        needs_layer_aware_fia_graph_replay.cache_clear()
        self.addCleanup(needs_layer_aware_fia_graph_replay.cache_clear)
        self.addCleanup(self.utils_config_patcher.stop)
        self.addCleanup(self.config_patcher.stop)

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

        self.impl_swa_sink = AscendAttentionBackendImpl(
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
            sinks=torch.tensor([-3.4062], dtype=torch.bfloat16),
        )

        self.impl_large_head = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name=None,
        )

        self.impl_kv_share = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name="producer_layer",
        )

        self.impl_c8_kv_share = AscendC8AttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type=self.attention_type.DECODER,
            kv_sharing_target_layer_name="producer_layer",
        )

    def test_hnd_layout_is_recorded_during_initialization(self):
        with patch.object(attn_module.envs, "VLLM_KV_CACHE_LAYOUT", "HND"):
            impl = AscendAttentionBackendImpl(
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

        self.assertEqual(impl.kv_cache_layout, "HND")
        self.assertTrue(impl.use_bnsd_kv_cache)

    def test_hnd_reshape_and_cache_uses_custom_scatter(self):
        self.impl.use_bnsd_kv_cache = True
        query = torch.empty(2, 8, 64)
        key = torch.randn(2, 8, 64)
        value = torch.randn(2, 8, 64)
        kv_cache = (torch.empty(4, 8, 128, 64), torch.empty(4, 8, 128, 64))
        output = torch.empty_like(query)
        metadata = MagicMock()
        metadata.slot_mapping = torch.arange(2)
        metadata.num_actual_tokens = 2

        with (
            patch.object(
                torch.ops._C_ascend,
                "npu_scatter_pa_kv_cache",
                create=True,
            ) as scatter,
            patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache") as reshape_and_cache,
            patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written"),
        ):
            self.impl.reshape_and_cache(query, key, value, kv_cache, metadata, output)

        scatter.assert_called_once()
        reshape_and_cache.assert_not_called()
        self.assertEqual(scatter.call_args.kwargs["cache_mode"], "Norm")
        self.assertEqual(scatter.call_args.kwargs["scatter_mode"], "NHSD")

    def test_get_fia_params_uses_layout_specific_cache_view(self):
        metadata = MagicMock()
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.block_tables = torch.zeros(1, 1)
        metadata.seq_lens_list = [1]
        current_key = torch.empty(1, 8, 64)
        current_value = torch.empty_like(current_key)

        self.impl.use_bnsd_kv_cache = False
        self.impl.key_cache = torch.empty(4, 128, 8, 64)
        self.impl.value_cache = torch.empty_like(self.impl.key_cache)
        key, value, block_size, _, _ = self.impl._get_fia_params(current_key, current_value, metadata)
        self.assertEqual(key.shape, (4, 128, 512))
        self.assertEqual(value.shape, (4, 128, 512))
        self.assertEqual(block_size, 128)

        self.impl.use_bnsd_kv_cache = True
        self.impl.key_cache = torch.empty(4, 8, 128, 64)
        self.impl.value_cache = torch.empty_like(self.impl.key_cache)
        key, value, block_size, _, _ = self.impl._get_fia_params(current_key, current_value, metadata)
        self.assertEqual(key.shape, (4, 8, 128, 64))
        self.assertEqual(value.shape, (4, 8, 128, 64))
        self.assertEqual(block_size, 128)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_large_head_prefill_uses_device_operator_fallback(self, mock_get_forward_context):
        query = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        key = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        value = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.actual_seq_lengths_q = [2]
        metadata.causal = True
        metadata.attn_mask = None
        mock_get_forward_context.return_value = MagicMock(capturing=False)

        with patch(LARGE_HEAD_PREFILL_PATH, return_value=(torch.ones_like(query), None)) as mock_forward:
            result = self.impl_large_head.forward_impl(query, key, value, (), metadata, output)

        mock_forward.assert_called_once()
        self.assertIs(result, output)
        self.assertTrue(torch.equal(result, torch.ones_like(query)))

    def test_supported_head_prefill_uses_fia(self):
        query = torch.randn(2, 8, 64)
        key = torch.randn(2, 8, 64)
        value = torch.randn(2, 8, 64)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.actual_seq_lengths_q = [2]

        self.impl.forward_fused_infer_attention = MagicMock(return_value=output)
        with patch(LARGE_HEAD_PREFILL_PATH, return_value=(torch.empty_like(query), None)) as mock_forward:
            result = self.impl.forward_impl(query, key, value, (), metadata, output)

        mock_forward.assert_not_called()
        self.impl.forward_fused_infer_attention.assert_called_once()
        self.assertIs(result, output)

    @patch("vllm_ascend.attention.attention_v1.using_paged_attention", return_value=True)
    def test_decode_uses_paged_attention(self, mock_using_pa):
        query = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        output = torch.empty_like(query)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly

        self.impl_large_head.forward_paged_attention = MagicMock(return_value=output)
        self.impl_large_head.forward_fused_infer_attention = MagicMock(return_value=output)

        result = self.impl_large_head.forward_impl(query, None, None, (), metadata, output)

        self.impl_large_head.forward_paged_attention.assert_called_once()
        self.impl_large_head.forward_fused_infer_attention.assert_not_called()
        self.assertIs(result, output)
        mock_using_pa.assert_called_once()

    @patch("torch_npu.npu_fused_infer_attention_score")
    def test_a5_device_operator_uses_fia_for_large_head(self, mock_fia):
        query = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        key = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        value = torch.randn(2, 8, FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE)
        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.PrefillNoCache
        metadata.actual_seq_lengths_q = [2]

        mock_fia.return_value = (torch.ones_like(query), None)
        with patch(LARGE_HEAD_PREFILL_PATH, return_value=(torch.empty_like(query), None)) as mock_forward:
            result = A5DeviceAdaptor.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                attn_metadata=metadata,
                key_cache=None,
                value_cache=None,
                current_key=key,
                current_value=value,
                num_heads=8,
                num_key_value_heads=8,
                head_size=FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
                scale=1.0,
                is_prefill_no_cache=True,
                block_table=None,
                input_layout="TND",
                block_size=128,
                actual_seq_lengths=[2],
                actual_seq_lengths_kv=[2],
                sparse_mode=3,
            )

        mock_forward.assert_not_called()
        mock_fia.assert_called_once()
        self.assertEqual(result[0].shape, query.shape)

    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    def test_kv_sharing_target_skips_cache_write(self, mock_reshape_and_cache):
        query = torch.randn(2, 8, 64)
        key = torch.randn(2, 8, 64)
        value = torch.randn(2, 8, 64)
        kv_cache = (
            torch.empty(4, 128, 8, 64),
            torch.empty(4, 128, 8, 64),
        )
        output = torch.empty_like(query)
        metadata = MagicMock()
        metadata.slot_mapping = torch.arange(2)
        metadata.num_actual_tokens = 2
        self.impl_kv_share.is_kv_producer = False

        returned = self.impl_kv_share.reshape_and_cache(query, key, value, kv_cache, metadata, output)

        mock_reshape_and_cache.assert_not_called()
        self.assertIs(self.impl_kv_share.key_cache, kv_cache[0])
        self.assertIs(self.impl_kv_share.value_cache, kv_cache[1])
        self.assertIs(returned[0], query)
        self.assertIs(returned[1], key)
        self.assertIs(returned[2], value)
        self.assertIs(returned[3], output)

    @patch("torch_npu.npu_scatter_pa_kv_cache", create=True)
    def test_c8_kv_sharing_target_skips_nz_cache_write(self, mock_scatter_pa_kv_cache):
        query = torch.randn(2, 8, 64)
        key = torch.randn(2, 8, 64)
        value = torch.randn(2, 8, 64)
        kv_cache = (
            torch.empty(4, 128, 8, 64),
            torch.empty(4, 128, 8, 64),
        )
        output = torch.empty_like(query)
        metadata = MagicMock()
        metadata.slot_mapping = torch.arange(2)
        metadata.num_actual_tokens = 2
        self.impl_c8_kv_share.is_kv_producer = False

        returned = self.impl_c8_kv_share._reshape_and_cache(query, key, value, kv_cache, metadata, output)

        mock_scatter_pa_kv_cache.assert_not_called()
        self.assertIs(self.impl_c8_kv_share.key_cache, kv_cache[0])
        self.assertIs(self.impl_c8_kv_share.value_cache, kv_cache[1])
        self.assertIs(returned[0], query)
        self.assertIs(returned[1], key)
        self.assertIs(returned[2], value)
        self.assertIs(returned[3], output)

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

    @patch("torch_npu.npu_scatter_pa_kv_cache")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward_fused_infer_attention(
        self, mock_get_forward_context, mock_npu_fused_infer_attention_score, mock_npu_scatter_pa_kv_cache
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
    @patch("torch_npu.npu_scatter_pa_kv_cache")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward_paged_attention(
        self, mock_get_forward_context, mock_npu_scatter_pa_kv_cache, mock_paged_attention, mock_using_paged_attention
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
    @patch("torch_npu.npu_scatter_pa_kv_cache")
    def test_forward_decode_only_swa(
        self, mock_npu_scatter_pa_kv_cache, mock_fused_infer_attention_score, mock_get_forward_context
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
        metadata.actual_seq_lengths_q = [10]
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
    @patch("torch_npu.npu_fused_infer_attention_score_v2")
    @patch("torch_npu.npu_scatter_pa_kv_cache")
    def test_forward_decode_only_swa_sink(
        self, mock_npu_scatter_pa_kv_cache, mock_fused_infer_attention_score, mock_get_forward_context
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
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        layer = self.layer_no_quant
        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64), 1)
        output = self.impl_swa_sink.forward(layer, query, key, value, kv_cache, metadata, output)
        print(output.shape)
        mock_fused_infer_attention_score.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("torch_npu._npu_paged_attention")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("torch_npu.npu_scatter_pa_kv_cache")
    def test_forward_decode_only_swa_seq_len_mismatch(
        self,
        mock_npu_scatter_pa_kv_cache,
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

    @patch("torch.npu.stream")
    @patch("torch.npu.graph_task_update_begin")
    @patch("torch.npu.graph_task_update_end")
    @patch("torch_npu.npu_fused_infer_attention_score")
    @patch("vllm_ascend.attention.attention_v1.get_graph_params")
    @patch("vllm_ascend.attention.attention_v1._EXTRA_CTX")
    @patch("vllm_ascend.attention.attention_v1.using_paged_attention", return_value=False)
    @patch("vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", return_value=False)
    @patch("vllm_ascend.attention.attention_v1._ATTN_KEYS_BUFFER", new=[])
    def test_update_graph_params(
        self,
        mock_needs_layer_aware_fia_graph_replay,
        mock_using_paged_attention,
        mock_EXTRA_CTX,
        mock_get_graph_params,
        mock_fia,
        mock_graph_task_update_end,
        mock_graph_task_update_begin,
        mock_stream,
    ):
        """Test behavior when _ATTN_KEYS_BUFFER is [] after dummy_run."""

        mock_EXTRA_CTX.sinks = False
        mock_EXTRA_CTX.is_draft_model = False

        param: list[MagicMock | None] = [MagicMock()] * 22
        param[16] = None  # sliding_window
        param[17] = None  # c8_k_aq_scale
        param[21] = None  # layer_name

        mock_get_graph_params.return_value.attn_params = {1: [tuple(param)] * 3}
        mock_get_graph_params.return_value.handles = {1: [MagicMock()] * 3}
        mock_get_graph_params.return_value.events = {1: [MagicMock()] * 3}

        attn_metadata_keys = [
            "model.layers.10.self_attn.attn",
            "model.layers.2.self_attn.attn",
            "model.layers.5.self_attn.attn",
        ]
        forward_context = MagicMock()
        forward_context.attn_metadata = {key: MagicMock() for key in attn_metadata_keys}
        # breakpoint()
        self.impl.update_graph_params(self.mock_stream, forward_context, 1, self.mock_vllm_config)

        expected = [
            "model.layers.2.self_attn.attn",
            "model.layers.5.self_attn.attn",
            "model.layers.10.self_attn.attn",
        ]
        self.assertEqual(attn_module._ATTN_KEYS_BUFFER, expected)
        self.assertEqual(mock_fia.out.call_count, 3)

    @patch("torch.npu.stream")
    @patch("torch.npu.graph_task_update_begin")
    @patch("torch.npu.graph_task_update_end")
    @patch("torch_npu._npu_paged_attention")
    @patch("torch_npu._npu_paged_attention_get_workspace", return_value=MagicMock())
    @patch("vllm_ascend.attention.attention_v1.get_graph_params")
    @patch("vllm_ascend.attention.attention_v1._EXTRA_CTX")
    @patch("vllm_ascend.attention.attention_v1.using_paged_attention", return_value=True)
    @patch("vllm_ascend.attention.attention_v1.needs_layer_aware_fia_graph_replay", return_value=False)
    @patch("vllm_ascend.attention.attention_v1._ATTN_KEYS_BUFFER", new=[])
    def test_update_graph_params_handles_captured_paged_attention_params(
        self,
        mock_needs_layer_aware_fia_graph_replay,
        mock_using_paged_attention,
        mock_EXTRA_CTX,
        mock_get_graph_params,
        mock_get_workspace,
        mock_paged_attention,
        mock_graph_task_update_end,
        mock_graph_task_update_begin,
        mock_stream,
    ):
        mock_EXTRA_CTX.sinks = False
        mock_EXTRA_CTX.is_draft_model = False

        query = MagicMock()
        key_cache = MagicMock()
        value_cache = MagicMock()
        block_table = MagicMock()
        output = MagicMock()
        captured_seq_lens = MagicMock()
        current_seq_lens = MagicMock()
        pa_param = PagedAttentionGraphParam(
            (
                query,
                key_cache,
                value_cache,
                8,
                8,
                1.0,
                block_table,
                captured_seq_lens,
                output,
            ),
            "model.layers.0.self_attn.attn",
        )

        mock_get_graph_params.return_value.attn_params = {1: [pa_param]}
        mock_get_graph_params.return_value.handles = {1: [MagicMock()]}
        mock_get_graph_params.return_value.events = {1: [MagicMock()]}

        forward_context = MagicMock()
        forward_context.attn_metadata = {
            "model.layers.0.self_attn.attn": MagicMock(seq_lens=current_seq_lens),
        }

        self.impl.update_graph_params(self.mock_stream, forward_context, 1, self.mock_vllm_config)

        mock_get_workspace.assert_called_once()
        mock_paged_attention.assert_called_once()
        self.assertEqual(mock_paged_attention.call_args.kwargs["context_lens"], current_seq_lens)
        mock_graph_task_update_begin.assert_called_once()
        mock_graph_task_update_end.assert_called_once()
