from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import BlasstConfig, clear_ascend_config, init_ascend_config
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
    BlasSTParamProvider,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadataBuilder,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    cache_graph_workspace,
    needs_layer_aware_fia_graph_replay,
    using_paged_attention,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.device.utils import FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
from vllm_ascend.utils import AscendDeviceType

LARGE_HEAD_PREFILL_PATH = "vllm_ascend.device.utils.npu_large_head_prefill_attention"
BLASST_GET_WS_PATH = "vllm_ascend.attention.attention_v1.torch.ops._C_ascend.npu_blasst_attention_score_get_workspace"


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
    def test_get_name(self):
        self.assertEqual(AscendAttentionBackend.get_name(), "CUSTOM")

    def test_get_impl_cls(self):
        with patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=False):
            self.assertEqual(AscendAttentionBackend.get_impl_cls(), AscendAttentionBackendImpl)

    def test_get_builder_cls(self):
        with patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=False):
            self.assertEqual(AscendAttentionBackend.get_builder_cls(), AscendAttentionMetadataBuilder)

    def test_supports_pcp_only_for_main_implementation(self):
        with patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=False):
            self.assertTrue(AscendAttentionBackend.supports_pcp())

        class OtherAttentionBackend(AscendAttentionBackend):
            @staticmethod
            def get_impl_cls():
                return AscendC8AttentionBackendImpl

        self.assertFalse(OtherAttentionBackend.supports_pcp())

    def test_get_impl_cls_with_dcp(self):
        with patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=True):
            self.assertIs(
                AscendAttentionBackend.get_impl_cls(),
                AscendAttentionDCPImpl,
            )

    def test_get_builder_cls_with_dcp(self):
        with patch("vllm_ascend.attention.attention_v1.enable_dcp", return_value=True):
            self.assertIs(
                AscendAttentionBackend.get_builder_cls(),
                AscendAttentionDCPMetadataBuilder,
            )

    def test_get_kv_cache_shape(self):
        with patch.object(attn_module.envs_vllm, "VLLM_KV_CACHE_LAYOUT", None):
            result = AscendAttentionBackend.get_kv_cache_shape(10, 20, 30, 40)
        self.assertEqual(result, (2, 10, 20, 30, 40))

    def test_get_kv_cache_shape_uses_bnsd_for_hnd_layouts(self):
        for layout in ("LBHNC", "HND"):
            with self.subTest(layout=layout), patch.object(attn_module.envs_vllm, "VLLM_KV_CACHE_LAYOUT", layout):
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
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
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

    def test_pcp_mode_is_initialized_from_config(self):
        self.assertFalse(self.builder.pcp_enabled)
        self.assertIs(self.builder.metadata_cls, AscendMetadata)

        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 2
        pcp_builder = AscendAttentionMetadataBuilder(None, None, self.mock_vllm_config, self.mock_device)

        self.assertTrue(pcp_builder.pcp_enabled)
        self.assertIs(pcp_builder.metadata_cls, AscendMetadata)

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
    builder = AscendAttentionMetadataBuilder.__new__(AscendAttentionMetadataBuilder)
    builder.pcp_size = 2
    expanded_slot_mapping = torch.tensor(
        [10, 11, -1, -1, 20, 21, -1, -1],
        dtype=torch.int64,
    )
    metadata = AscendMetadata(
        num_actual_tokens=3,
        num_decode_tokens=1,
        num_prefills=1,
        attn_state=AscendAttentionState.PrefillCacheHit,
    )

    builder._finalize_pcp_metadata(metadata, expanded_slot_mapping)

    assert metadata.slot_mapping is expanded_slot_mapping
    assert metadata.pcp_local_num_input_tokens == 4
    assert metadata.attn_state == AscendAttentionState.ChunkedPrefill


def test_pcp_cache_write_uses_gathered_inputs() -> None:
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
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
    metadata = AscendMetadata(
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
        result = impl._reshape_and_cache_pcp(
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
    builder = AscendAttentionMetadataBuilder.__new__(AscendAttentionMetadataBuilder)
    builder.decode_threshold = 1
    builder.pcp_enabled = True
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
        self.mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.mock_vllm_config.cache_config.cache_dtype = "float16"
        # __init__ reads get_ascend_config().blasst_config; same pattern as
        # TestAscendMLAImpl: init a default (blasst disabled) config first.
        self.mock_vllm_config.additional_config = {"refresh": True}
        init_ascend_config(self.mock_vllm_config)
        self.addCleanup(clear_ascend_config)

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
        with patch.object(attn_module.envs_vllm, "VLLM_KV_CACHE_LAYOUT", "HND"):
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

    def test_hnd_reshape_and_cache_passes_bnsd_to_device_operator(self):
        self.impl.use_bnsd_kv_cache = True
        query = torch.empty(2, 8, 64)
        key = torch.randn(2, 8, 64)
        value = torch.randn(2, 8, 64)
        key_cache = torch.empty(4, 8, 128, 64)
        value_cache = torch.empty_like(key_cache)
        output = torch.empty_like(query)
        metadata = MagicMock()
        metadata.slot_mapping = torch.arange(2)
        metadata.num_actual_tokens = 2

        with (
            patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache") as reshape_and_cache,
            patch("vllm_ascend.attention.attention_v1.notify_kv_cache_written"),
        ):
            self.impl.reshape_and_cache(
                query,
                key,
                value,
                (key_cache, value_cache),
                metadata,
                output,
            )

        reshape_and_cache.assert_called_once()
        call_kwargs = reshape_and_cache.call_args.kwargs
        self.assertIs(call_kwargs["key_cache"], key_cache)
        self.assertIs(call_kwargs["value_cache"], value_cache)
        self.assertTrue(call_kwargs["use_bnsd"])

    def test_hnd_do_kv_cache_update_passes_bnsd_to_device_operator(self):
        self.impl.use_bnsd_kv_cache = True
        self.impl.key_cache = None
        self.impl.value_cache = None
        key = torch.randn(2, 8, 64)
        value = torch.randn_like(key)
        key_cache = torch.empty(4, 8, 128, 64)
        value_cache = torch.empty_like(key_cache)
        slot_mapping = torch.arange(2)

        with patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache") as reshape_and_cache:
            self.impl.do_kv_cache_update(
                MagicMock(),
                key,
                value,
                [key_cache, value_cache],
                slot_mapping,
            )

        reshape_and_cache.assert_called_once()
        call_kwargs = reshape_and_cache.call_args.kwargs
        self.assertIs(call_kwargs["key_cache"], key_cache)
        self.assertIs(call_kwargs["value_cache"], value_cache)
        self.assertTrue(call_kwargs["use_bnsd"])

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

    @patch("vllm_ascend.attention.attention_v1._EXTRA_CTX")
    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    @patch("vllm_ascend.attention.attention_v1.torch_npu._npu_paged_attention")
    @patch("vllm_ascend.attention.attention_v1.using_paged_attention", return_value=True)
    def test_forward_paged_attention(
        self,
        mock_using_paged_attention,
        mock_paged_attention,
        mock_reshape_and_cache,
        mock_extra_ctx,
    ):
        """Test forward pass in DecodeOnly state"""
        mock_extra_ctx.capturing = False

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
        metadata.causal = True
        metadata.model_runner_type = None
        layer = self.layer_no_quant

        output = self.impl.forward(layer, query, key, value, kv_cache, metadata, output)

        mock_paged_attention.assert_called_once()
        mock_reshape_and_cache.assert_called_once()
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

    @patch("vllm_ascend.attention.attention_v1._EXTRA_CTX")
    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score_v2")
    def test_forward_decode_only_swa_sink(
        self, mock_fused_infer_attention_score_v2, mock_reshape_and_cache, mock_extra_ctx
    ):
        """Test forward pass in DecodeOnly state"""
        mock_extra_ctx.capturing = False

        query = torch.randn(10, 8 * 64)
        key = torch.randn(10, 8 * 64)
        value = torch.randn(10, 8 * 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty(10, 8, 64)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10] * 10)
        metadata.seq_lens_list = [10] * 10
        metadata.actual_seq_lengths_q = [10]
        metadata.attn_mask = torch.randn(1, 1, 10, 10)
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 100
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        metadata.causal = True
        metadata.model_runner_type = None
        layer = self.layer_no_quant
        mock_fused_infer_attention_score_v2.return_value = (torch.ones(10, 8, 64), 1)
        output = self.impl_swa_sink.forward(layer, query, key, value, kv_cache, metadata, output)
        mock_fused_infer_attention_score_v2.assert_called_once()
        mock_reshape_and_cache.assert_called_once()
        assert output.shape == (10, 8, 64)

    @patch("vllm_ascend.attention.attention_v1._EXTRA_CTX")
    @patch("vllm_ascend.attention.attention_v1.DeviceOperator.reshape_and_cache")
    @patch("vllm_ascend.attention.attention_v1.torch_npu._npu_paged_attention")
    @patch("vllm_ascend.attention.attention_v1.torch_npu.npu_fused_infer_attention_score")
    def test_forward_decode_only_swa_seq_len_mismatch(
        self,
        mock_fused_infer_attention_score,
        mock_paged_attention,
        mock_reshape_and_cache,
        mock_extra_ctx,
    ):
        """Test forward pass in DecodeOnly state when seq)len_mismatch"""
        mock_extra_ctx.capturing = False

        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kv_cache = torch.empty(2, 5, 128, 8, 64)
        output = torch.empty_like(query)

        metadata = self.attn_metadata
        metadata.attn_state = AscendAttentionState.DecodeOnly
        metadata.seq_lens = torch.tensor([10])  # len == 1 != query.size(0)==10
        metadata.seq_lens_list = [10]
        metadata.block_tables = torch.zeros(1, 5, dtype=torch.long)
        metadata.num_actual_tokens = 10
        metadata.slot_mapping = torch.zeros(10, dtype=torch.long)
        layer = self.layer_no_quant
        metadata.num_decodes = 10
        metadata.num_prefills = 0
        metadata.actual_seq_lengths_q = [10]
        metadata.causal = True
        metadata.model_runner_type = None

        mock_fused_infer_attention_score.return_value = (torch.ones(10, 8, 64), torch.ones(10, 8, 64))

        output = self.impl_swa.forward(layer, query, key, value, kv_cache, metadata, output)

        mock_paged_attention.assert_not_called()
        mock_fused_infer_attention_score.assert_called_once()
        mock_reshape_and_cache.assert_called_once()

        assert output.shape == (10, 8, 64)


class TestBlasstGate(TestBase):
    """Gating of the BlasST dispatch.

    Every unsupported case must fall back to the baseline FIA path instead of
    reaching the op (which either fails loudly or, for non-causal batches,
    computes silently wrong output). Static limits are folded into
    ``_blasst_supported`` at init; per-call limits live in the gate.
    """

    @staticmethod
    def _impl(**overrides):
        """Partial instance covering exactly what _can_use_blasst reads."""
        impl = object.__new__(AscendAttentionBackendImpl)
        impl._blasst_supported = overrides.pop("supported", True)
        impl.key_cache = overrides.pop("key_cache", object())
        impl.value_cache = None
        impl._use_layer_aware_fia_graph_replay = overrides.pop("layer_aware", False)
        assert not overrides, overrides
        return impl

    @staticmethod
    def _meta(**overrides):
        meta = AscendMetadata()
        meta.causal = True
        meta.attn_state = AscendAttentionState.DecodeOnly
        meta.actual_seq_lengths_q = [1, 2]
        meta.num_decodes = 0
        meta.num_prefills = 0
        for key, value in overrides.items():
            setattr(meta, key, value)
        return meta

    def _gate(self, impl, meta, *, capturing=False, is_draft=False):
        # _EXTRA_CTX attributes are set dynamically at runtime, so replace the
        # module-level name with a stub instead of patching its attributes.
        extra_ctx = SimpleNamespace(capturing=capturing, is_draft_model=is_draft)
        with patch.object(attn_module, "_EXTRA_CTX", extra_ctx):
            return impl._can_use_blasst(meta)

    def test_supported_decode_batch_passes(self):
        self.assertTrue(self._gate(self._impl(), self._meta()))

    def test_static_unsupported_short_circuits(self):
        self.assertFalse(self._gate(self._impl(supported=False), self._meta()))

    def test_draft_model_falls_back(self):
        # Draft-model eager steps stay on the exact baseline path: spec decode
        # verifies the draft against the target model, so approximation is
        # only ever applied to the target forward.
        self.assertFalse(self._gate(self._impl(), self._meta(), is_draft=True))

    def test_non_causal_eager_routes_with_no_mask_mode(self):
        # Non-causal batches (attn_mask=None) are supported: the eager call
        # site selects sparse_mode=0 (no mask, full attention) for them.
        self.assertTrue(self._gate(self._impl(), self._meta(causal=False)))
        # ...but the graph path bakes sparse_mode=3, so capturing non-causal
        # decode buckets stays on the baseline.
        self.assertFalse(self._gate(self._impl(), self._meta(causal=False), capturing=True))

    def test_batch_over_host_seq_limit_falls_back(self):
        limit = attn_module.BLASST_MAX_HOST_SEQ
        self.assertFalse(self._gate(self._impl(), self._meta(actual_seq_lengths_q=list(range(1, limit + 2)))))
        # exactly at the limit still passes
        self.assertTrue(self._gate(self._impl(), self._meta(actual_seq_lengths_q=list(range(1, limit + 1)))))

    def test_missing_kv_cache_falls_back_for_cache_states(self):
        impl = self._impl(key_cache=None)
        self.assertFalse(self._gate(impl, self._meta()))
        # PrefillNoCache reads the new K/V directly and needs no cache handle.
        self.assertTrue(self._gate(impl, self._meta(attn_state=AscendAttentionState.PrefillNoCache)))

    def test_mixed_chunked_batch_routes_regardless_of_phase_split(self):
        # Mixed decode+prefill chunked batches run as one fused BlasST call on
        # all hardware: the baseline phase-split only avoids
        # batch-composition-dependent numerics, and the fused call was
        # validated against per-phase FIA (LongBench-v2, 50/50 agreement).
        impl = self._impl()
        mixed = dict(attn_state=AscendAttentionState.ChunkedPrefill, num_decodes=1, num_prefills=1)
        self.assertTrue(self._gate(impl, self._meta(**mixed)))

    def test_capture_routes_decode_only_buckets(self):
        # Capture follows the forward flow: decode buckets route through the
        # custom path, non-decode buckets never capture through it.
        impl = self._impl()
        self.assertTrue(self._gate(impl, self._meta(), capturing=True))
        self.assertFalse(self._gate(impl, self._meta(attn_state=AscendAttentionState.ChunkedPrefill), capturing=True))

    def test_capture_draft_model_and_layer_aware_fall_back(self):
        impl = self._impl()
        self.assertFalse(self._gate(impl, self._meta(), capturing=True, is_draft=True))
        self.assertFalse(self._gate(self._impl(layer_aware=True), self._meta(), capturing=True))

    def test_unsupported_states_fall_back(self):
        impl = self._impl()
        for state in (AscendAttentionState.PrefillCacheHit, AscendAttentionState.SpecDecoding):
            self.assertFalse(self._gate(impl, self._meta(attn_state=state)))

    def _init_impl(
        self,
        *,
        enabled=True,
        dtype=torch.float16,
        kv_dtype="auto",
        head_size=128,
        sliding_window=None,
        sinks=None,
        batch_invariant=False,
        op_registered=True,
        quant=None,
        kv_layout=None,
        layer_types=None,
        text_layer_types=None,
    ):
        hf_config = None
        if layer_types is not None or text_layer_types is not None:
            hf_config = SimpleNamespace(
                layer_types=layer_types,
                text_config=None if text_layer_types is None else SimpleNamespace(layer_types=text_layer_types),
            )
        vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
            kv_transfer_config=None,
            quant_config=quant,
            # cache_config.cache_dtype is what __init__ resolves
            # self.kv_cache_dtype from (the ctor arg no longer feeds it).
            cache_config=SimpleNamespace(cache_dtype=kv_dtype),
            # hf_config feeds the hybrid linear-attention gate in __init__.
            model_config=SimpleNamespace(dtype=dtype, hf_config=hf_config),
        )
        with (
            patch.object(attn_module, "get_current_vllm_config", return_value=vllm_config),
            patch.object(
                attn_module,
                "get_ascend_config",
                return_value=SimpleNamespace(blasst_config=BlasstConfig(enabled=enabled)),
            ),
            patch.object(attn_module, "needs_layer_aware_fia_graph_replay", return_value=False),
            patch.object(attn_module.envs_vllm, "VLLM_BATCH_INVARIANT", batch_invariant),
            patch.object(attn_module.envs_vllm, "VLLM_KV_CACHE_LAYOUT", kv_layout),
            patch.object(
                attn_module.torch.ops,
                "_C_ascend",
                SimpleNamespace(npu_blasst_attention_score=True) if op_registered else SimpleNamespace(),
            ),
        ):
            return AscendAttentionBackendImpl(
                num_heads=16,
                head_size=head_size,
                scale=0.088,
                num_kv_heads=4,
                alibi_slopes=None,
                sliding_window=sliding_window,
                kv_cache_dtype=kv_dtype,
                logits_soft_cap=None,
                attn_type="decoder",
                kv_sharing_target_layer_name=None,
                sinks=sinks,
            )

    def test_static_capability_matrix(self):
        self.assertTrue(self._init_impl()._blasst_supported)
        # head_dim 256 (e.g. Qwen3.5 full-attention layers) is supported in
        # addition to 128; other head sizes stay on the baseline path.
        self.assertTrue(self._init_impl(head_size=256)._blasst_supported)
        # Quantized KV dtypes only construct with a C8 quant config (the
        # __init__ dtype validation rejects them otherwise); C8 models must
        # stay on the baseline path.
        c8_quant = SimpleNamespace(enable_c8_quant=True)
        for kwargs in (
            dict(enabled=False),
            # Model-dtype axis in isolation: "auto" KV would follow the fp32
            # model dtype and trip the __init__ C8 validation, so pin fp16 KV.
            dict(dtype=torch.float32, kv_dtype="float16"),
            dict(kv_dtype="fp8", quant=c8_quant),
            dict(kv_dtype="int8", quant=c8_quant),
            dict(head_size=64),
            dict(head_size=192),
            dict(sliding_window=512),
            dict(sinks=torch.zeros(1)),
            dict(batch_invariant=True),
            # BNSD-family KV layouts (HND/LBHNC) flip use_bnsd_kv_cache; the
            # op's paged view assumes the interleaved NBL cache.
            dict(kv_layout="HND"),
            dict(kv_layout="LBHNC"),
        ):
            self.assertFalse(self._init_impl(**kwargs)._blasst_supported, kwargs)
        # Hybrid linear-attention stacks (linear_attention / mamba / gdn
        # anywhere in layer_types, top-level or under text_config) are excluded
        # as a whole; pure full-attention stacks stay supported.
        for tokens in (["linear_attention"], ["mamba"], ["gdn"], ["full_attention", "gdn"]):
            self.assertFalse(self._init_impl(layer_types=tokens)._blasst_supported, tokens)
        self.assertFalse(self._init_impl(text_layer_types=["mamba"])._blasst_supported)
        self.assertTrue(self._init_impl(layer_types=["full_attention"] * 4)._blasst_supported)

    def test_enabled_but_op_missing_raises(self):
        # Opt-in feature without the op in this build fails loudly (same
        # policy as sparse_kv_offload_manager), not a silent FIA fallback.
        with self.assertRaises(RuntimeError):
            self._init_impl(op_registered=False)
        # not enabled -> missing op is fine
        self.assertFalse(self._init_impl(enabled=False, op_registered=False)._blasst_supported)

    def test_host_seq_limit_constant_matches_op_tiling(self):
        # The Python gate and the C++ tiling array are one contract kept in
        # two languages; if FIA_MAX_HOST_SEQ_LIST (tiling.h) changes, this
        # test must be updated in lockstep (see the comment at
        # BLASST_MAX_HOST_SEQ in attention_v1.py).
        self.assertEqual(attn_module.BLASST_MAX_HOST_SEQ, 256)


class TestBlasstParamProvider(TestBase):
    """BlasSTParamProvider.resolve: per-replay rebinding of the captured task.

    resolve must hand the task the current step's host seq lists and block
    table, plus a workspace large enough for the current step (the
    flash-decode split area is non-monotonic in kv length, so a replay can
    need more than was probed at capture time).
    """

    def setUp(self):
        self._saved_grown = attn_module._blasst_grown_workspace
        attn_module._blasst_grown_workspace = None
        self.addCleanup(setattr, attn_module, "_blasst_grown_workspace", self._saved_grown)

    @staticmethod
    def _provider(workspace):
        return BlasSTParamProvider(
            layer_name="layer0",
            query=torch.empty(4, 2, 128),
            key=torch.empty(4, 2, 128),
            value=torch.empty(4, 2, 128),
            block_size=128,
            attn_mask=None,
            # The provider must stay hashable: UpdatableGraph keys its
            # provider_sizes dict by provider instance, so op_kwargs is a
            # tuple of items rather than a dict.
            op_kwargs=(("num_heads", 2), ("scale", 0.1), ("input_layout", "TND")),
            workspace=workspace,
        )

    @staticmethod
    def _metadata():
        return SimpleNamespace(
            actual_seq_lengths_q=[1, 3],
            seq_lens_list=[5, 7],
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
        )

    def _resolve(self, provider, need):
        metadata = self._metadata()
        context = {"layer0": metadata}
        # create=True: the UT environment has no registered _C_ascend ops, so
        # the namespace attribute only exists while mocked.
        with patch(BLASST_GET_WS_PATH, return_value=need, create=True) as get_ws:
            params = provider.resolve(context)
        return params, get_ws, metadata

    def test_provider_is_hashable(self):
        provider = self._provider(torch.empty(8, dtype=torch.uint8))
        self.assertEqual({provider: "task"}[provider], "task")

    def test_resolve_maps_current_metadata_into_task_params(self):
        captured = torch.empty(8, dtype=torch.uint8)
        provider = self._provider(captured)
        params, get_ws, metadata = self._resolve(provider, need=0)
        # The workspace probe sees the current step's host lists and block
        # table, and the op kwargs are re-expanded from the tuple.
        self.assertEqual(get_ws.call_args.kwargs["actual_seq_lengths"], [1, 3])
        self.assertEqual(get_ws.call_args.kwargs["actual_seq_lengths_kv"], [5, 7])
        self.assertIs(get_ws.call_args.kwargs["blocktable"], metadata.block_tables)
        self.assertEqual(get_ws.call_args.kwargs["num_heads"], 2)
        self.assertEqual(get_ws.call_args.kwargs["input_layout"], "TND")
        self.assertEqual(params["actual_seq_lengths"], [1, 3])
        self.assertEqual(params["actual_seq_lengths_kv"], [5, 7])
        self.assertIs(params["blocktable"], metadata.block_tables)
        self.assertIs(params["workspace"], captured)

    def test_resolve_prefers_larger_global_grown_workspace(self):
        captured = torch.empty(8, dtype=torch.uint8)
        grown = torch.empty(16, dtype=torch.uint8)
        attn_module._blasst_grown_workspace = grown
        params, _, _ = self._resolve(self._provider(captured), need=0)
        self.assertIs(params["workspace"], grown)

    def test_resolve_grows_workspace_when_current_step_needs_more(self):
        captured = torch.empty(8, dtype=torch.uint8)
        provider = self._provider(captured)
        params, _, _ = self._resolve(provider, need=32)
        grown = params["workspace"]
        self.assertIsNot(grown, captured)
        self.assertEqual(grown.numel(), 32)
        self.assertIs(attn_module._blasst_grown_workspace, grown)
        # A later step needing the same amount reuses the global buffer
        # instead of reallocating; a still-larger need replaces it.
        params_again, _, _ = self._resolve(self._provider(captured), need=32)
        self.assertIs(params_again["workspace"], grown)
        params_more, _, _ = self._resolve(self._provider(captured), need=64)
        self.assertEqual(params_more["workspace"].numel(), 64)
        self.assertIs(attn_module._blasst_grown_workspace, params_more["workspace"])
