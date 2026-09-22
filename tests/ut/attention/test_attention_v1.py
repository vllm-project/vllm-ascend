from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import vllm_ascend.attention.attention_v1 as attn_module
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
    AscendC8AttentionBackendImpl,
    AscendMetadata,
)
from vllm_ascend.attention.context_parallel.attention_cp import (
    AscendAttentionDCPImpl,
    AscendAttentionDCPMetadataBuilder,
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

    def _build_parallel_drafting_metadata(self, *, seq_lens_cpu_is_exact, seq_lens_cpu_is_approximate=False):
        """Run ``build`` for a parallel-drafting batch, recording every tolist.

        Returns ``(metadata, tolist_sources)``.
        """
        seq_lens_device = torch.tensor([4, 5, 6], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([4, 5, 6], dtype=torch.int32)
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            seq_lens=seq_lens_device,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_is_exact=seq_lens_cpu_is_exact,
            seq_lens_cpu_is_approximate=seq_lens_cpu_is_approximate,
            num_computed_tokens_cpu=None,
            num_reqs=3,
            num_actual_tokens=3,
            max_query_len=1,
            block_table_tensor=torch.zeros((3, 1), dtype=torch.int32),
            slot_mapping=torch.arange(3, dtype=torch.int32),
            causal=True,
            actual_seq_lengths_q=[1, 2, 3],
            positions=torch.arange(3),
            attn_state=AscendAttentionState.DecodeOnly,
            max_seq_len=6,
        )
        self.builder.speculative_config = SimpleNamespace(parallel_drafting=True)

        tolist_sources = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor, *args, **kwargs):
            tolist_sources.append(tensor)
            return original_tolist(tensor, *args, **kwargs)

        with (
            patch.object(torch.Tensor, "tolist", new=tracked_tolist),
            patch.object(
                AscendAttentionMetadataBuilder,
                "metadata_cls",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        ):
            metadata = self.builder.build(0, common_attn_metadata)
        return metadata, tolist_sources, seq_lens_device, seq_lens_cpu

    def test_parallel_drafting_uses_cpu_mirror_when_it_is_exact(self):
        """Target build: the exact host mirror feeds the Python list.

        The backend still receives the device ``seq_lens`` tensor, but no
        ``.tolist()`` is issued against it -- that call would block host
        dispatch on the compute stream. See issue #16271.
        """
        metadata, tolist_sources, seq_lens_device, seq_lens_cpu = self._build_parallel_drafting_metadata(
            seq_lens_cpu_is_exact=True
        )

        self.assertIs(metadata.seq_lens, seq_lens_device)
        self.assertEqual(metadata.seq_lens_list, [4, 5, 6])
        self.assertFalse(any(src is seq_lens_device for src in tolist_sources))
        self.assertTrue(any(src.data_ptr() == seq_lens_cpu.data_ptr() for src in tolist_sources))

    def test_parallel_drafting_falls_back_to_device_when_mirror_is_not_exact(self):
        """Draft build: the host only has an optimistic bound, so keep the D2H."""
        metadata, tolist_sources, seq_lens_device, _ = self._build_parallel_drafting_metadata(
            seq_lens_cpu_is_exact=False
        )

        self.assertIs(metadata.seq_lens, seq_lens_device)
        self.assertEqual(metadata.seq_lens_list, [4, 5, 6])
        self.assertTrue(any(src is seq_lens_device for src in tolist_sources))

    def test_seq_lens_cpu_is_exact_defaults_to_false(self):
        """Unaudited producers must keep the previous (device) behaviour."""
        self.assertFalse(AscendCommonAttentionMetadata.seq_lens_cpu_is_exact)

    def test_parallel_drafting_accepts_an_approximate_mirror_when_opted_in(self):
        """Draft build under ``enable_dspark_draft_kv_optimistic_bound``.

        The mirror is only an optimistic bound, so it is *not* exact -- but the
        producer has opted into the approximation, and the point of opting in is
        to stop issuing the blocking ``.tolist()`` against the device tensor.
        """
        metadata, tolist_sources, seq_lens_device, seq_lens_cpu = self._build_parallel_drafting_metadata(
            seq_lens_cpu_is_exact=False,
            seq_lens_cpu_is_approximate=True,
        )

        self.assertIs(metadata.seq_lens, seq_lens_device)
        self.assertEqual(metadata.seq_lens_list, [4, 5, 6])
        self.assertFalse(any(src is seq_lens_device for src in tolist_sources))
        self.assertTrue(any(src.data_ptr() == seq_lens_cpu.data_ptr() for src in tolist_sources))

    def test_seq_lens_cpu_is_approximate_defaults_to_false(self):
        """The approximation must never be on unless a producer asked for it."""
        self.assertFalse(AscendCommonAttentionMetadata.seq_lens_cpu_is_approximate)

    def test_approximate_mirror_arms_the_draft_tail_mask(self):
        """The producer's opt-in is the only gate on the device-side mask.

        ``build_attn_metadata`` publishes an approximate mirror only when
        ``enable_dspark_draft_kv_optimistic_bound`` is set, so an approximate
        mirror must by itself arm the tail mask -- otherwise the bound would go
        unmasked and cost acceptance rate. See issue #16271.
        """
        metadata, _, _, _ = self._build_parallel_drafting_metadata(
            seq_lens_cpu_is_exact=False,
            seq_lens_cpu_is_approximate=True,
        )

        self.assertTrue(metadata.draft_kv_upper_bound)

    def test_exact_mirror_leaves_the_draft_tail_mask_disarmed(self):
        """A target build needs no mask: its host mirror already is the truth."""
        metadata, _, _, _ = self._build_parallel_drafting_metadata(seq_lens_cpu_is_exact=True)

        self.assertFalse(metadata.draft_kv_upper_bound)

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

    @patch("vllm_ascend.attention.attention_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.attention_v1.torch.npu.graph_task_update_begin")
    @patch("vllm_ascend.attention.attention_v1.torch.npu.graph_task_update_end")
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

    @patch("vllm_ascend.attention.attention_v1.torch.npu.stream")
    @patch("vllm_ascend.attention.attention_v1.torch.npu.graph_task_update_begin")
    @patch("vllm_ascend.attention.attention_v1.torch.npu.graph_task_update_end")
    @patch("vllm_ascend.attention.attention_v1.torch_npu._npu_paged_attention")
    @patch("vllm_ascend.attention.attention_v1.torch_npu._npu_paged_attention_get_workspace", return_value=MagicMock())
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
            "model.layers.0.self_attn.attn": MagicMock(
                seq_lens=current_seq_lens,
                block_tables=block_table,
                seq_lens_list=[10],
            ),
        }

        self.impl.update_graph_params(self.mock_stream, forward_context, 1, self.mock_vllm_config)

        mock_get_workspace.assert_called_once()
        mock_paged_attention.assert_called_once()
        self.assertEqual(mock_paged_attention.call_args.kwargs["context_lens"], current_seq_lens)
        mock_graph_task_update_begin.assert_called_once()
        mock_graph_task_update_end.assert_called_once()


class TestBuildDraftTailMask(TestBase):
    """The mask is the whole correctness argument for #16271's draft path.

    Feeding the operator the optimistic bound U instead of the true length L is
    free on the host but makes the kernel read [L, U) -- the draft KV the
    previous step rolled back. These tests pin what the mask has to hide, and
    the one boundary that is easy to get wrong without anything failing loudly.
    """

    def test_masks_exactly_the_positions_past_each_query_token(self):
        # Two requests with different true lengths, so a mask that ignored the
        # per-request dimension could not pass.
        seq_lens = torch.tensor([5, 7], dtype=torch.int32)
        mask = attn_module.build_draft_tail_mask(seq_lens, num_reqs=2, query_len=3, kv_span=8, sliding_window=None)

        self.assertEqual(tuple(mask.shape), (2, 3, 8))
        self.assertEqual(mask.dtype, torch.int8)
        for req, length in enumerate([5, 7]):
            for j in range(3):
                last = length - 3 + j
                expected = torch.tensor([1 if c > last else 0 for c in range(8)], dtype=torch.int8)
                self.assertTrue(torch.equal(mask[req, j], expected), f"req={req} j={j}")

    def test_hides_the_rolled_back_tail_for_every_query_token(self):
        """[L, U) must be masked everywhere; that tail is what costs acceptance."""
        true_lens = torch.tensor([6, 4], dtype=torch.int32)
        rejected = [3, 8]  # what the previous step's draft lost
        mask = attn_module.build_draft_tail_mask(true_lens, num_reqs=2, query_len=4, kv_span=16, sliding_window=None)

        for req, length in enumerate([6, 4]):
            upper_bound = length + rejected[req]
            tail = mask[req, :, length:upper_bound]
            self.assertTrue(bool((tail == 1).all()), f"req={req} tail not fully masked")
            # The last position each query token *may* see must stay visible,
            # or the mask would be cutting real context instead of the tail.
            for j in range(4):
                self.assertEqual(int(mask[req, j, length - 4 + j]), 0)

    def test_sliding_window_keeps_pre_tokens_plus_one_positions(self):
        """Band mode with pre_tokens=W keeps W+1 positions, closed at both ends.

        Measured against the operator: writing the boundary as ``<=`` instead of
        ``<`` shifts the window by one and moves the result by 0.157 relative
        error -- 27x the bf16 noise floor, and nowhere near large enough to fail
        anything by itself.
        """
        window = 3
        seq_lens = torch.tensor([12], dtype=torch.int32)
        mask = attn_module.build_draft_tail_mask(seq_lens, num_reqs=1, query_len=2, kv_span=16, sliding_window=window)

        for j in range(2):
            visible = (mask[0, j] == 0).nonzero().flatten().tolist()
            last = 12 - 2 + j
            self.assertEqual(visible, list(range(last - window, last + 1)))
            self.assertEqual(len(visible), window + 1)

    def test_without_a_window_nothing_before_the_query_token_is_masked(self):
        seq_lens = torch.tensor([9], dtype=torch.int32)
        mask = attn_module.build_draft_tail_mask(seq_lens, num_reqs=1, query_len=2, kv_span=12, sliding_window=None)
        self.assertTrue(bool((mask[0, 0, :8] == 0).all()))

    def test_non_causal_keeps_the_bound_flat_across_the_query_block(self):
        """A DSpark query-block forward is non-causal: every query token sees
        the whole visible prefix, so the bound stays at ``L - 1`` instead of
        advancing with j. Advancing it here would hide real context; not
        masking past it would expose the rolled-back tail.
        """
        seq_lens = torch.tensor([5, 7], dtype=torch.int32)
        mask = attn_module.build_draft_tail_mask(
            seq_lens, num_reqs=2, query_len=3, kv_span=8, sliding_window=None, causal=False
        )

        for req, length in enumerate([5, 7]):
            for j in range(1, 3):
                self.assertTrue(torch.equal(mask[req, j], mask[req, 0]), f"req={req} j={j}")
            self.assertEqual(int(mask[req, 0, length - 1]), 0)
            self.assertEqual(int(mask[req, 0, length]), 1)


class TestForwardDraftTailMasked(TestBase):
    """Eligibility. Every rejected case must fall through, not compute wrongly."""

    def _impl(self, **overrides):
        impl = SimpleNamespace(sinks=None, sliding_window=None, num_heads=2, num_kv_heads=1, head_size=4, scale=0.5)
        impl.__dict__.update(overrides)
        return impl

    def _metadata(self, **overrides):
        meta = SimpleNamespace(
            draft_kv_upper_bound=True,
            draft_query_lens=[3, 3],
            draft_tail_mask_cache={},
            causal=True,
            seq_lens=torch.tensor([7, 9], dtype=torch.int32),
        )
        meta.__dict__.update(overrides)
        return meta

    def _call(self, impl, meta, num_tokens=6, block_table=None):
        block_table = torch.zeros((2, 2), dtype=torch.int32) if block_table is None else block_table
        query = torch.zeros(num_tokens, impl.num_heads, impl.head_size)
        output = torch.zeros(num_tokens, impl.num_heads, impl.head_size)
        return AscendAttentionBackendImpl._forward_draft_tail_masked(
            impl, query, torch.zeros(1), torch.zeros(1), meta, block_table, 4, [10, 12], num_tokens, output
        )

    def test_falls_through_when_the_build_is_not_a_bounded_draft(self):
        self.assertIsNone(self._call(self._impl(), self._metadata(draft_kv_upper_bound=False)))

    def test_falls_through_on_a_ragged_batch(self):
        """BSND needs a rectangular batch; a padded build must not be reshaped."""
        self.assertIsNone(self._call(self._impl(), self._metadata(draft_query_lens=[3, 2])))

    def test_falls_through_for_a_learnable_sink(self):
        """A sink changes what the mask would have to express, so it bails."""
        self.assertIsNone(self._call(self._impl(sinks=torch.zeros(1)), self._metadata()))

    def test_non_causal_build_is_eligible_and_is_not_aliased_to_the_causal_mask(self):
        """Every stock Qwen3 DSpark drafter is non-causal, so bailing here would
        drop exactly the models this path exists for -- and the two masks differ,
        so the cache key has to carry causality.
        """
        impl = self._impl()
        seq_lens = torch.tensor([5, 7], dtype=torch.int32)
        non_causal_meta = self._metadata(causal=False, seq_lens=seq_lens)
        causal_meta = self._metadata(seq_lens=seq_lens)
        fake = MagicMock(return_value=(torch.zeros(2, 3, 2, 4), None))
        with (
            patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", fake),
            patch.object(attn_module, "build_draft_tail_mask", wraps=attn_module.build_draft_tail_mask) as builder,
        ):
            self.assertIsNotNone(self._call(impl, non_causal_meta))
            non_causal_mask = fake.call_args.kwargs["atten_mask"]
            # Reusing the same metadata must not rebuild the mask.
            self._call(impl, non_causal_meta)
            # A causal build over the same lengths is a different mask.
            self._call(impl, causal_meta)

        self.assertEqual(builder.call_count, 2)
        for req in range(2):
            self.assertTrue(torch.equal(non_causal_mask[req, 0], non_causal_mask[req, 2]))

    def test_falls_through_when_the_token_count_does_not_match_the_batch(self):
        self.assertIsNone(self._call(self._impl(), self._metadata(), num_tokens=5))

    def test_eligible_build_uses_bsnd_with_a_per_request_mask(self):
        impl, meta = self._impl(), self._metadata()
        fake = MagicMock(return_value=(torch.zeros(2, 3, 2, 4), None))
        with patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", fake):
            result = self._call(impl, meta)

        self.assertIsNotNone(result)
        kwargs = fake.call_args.kwargs
        # TND would be routed to the split-fuse template, which rejects any mask
        # that is not 2048x2048 with sparse_mode 3 or 4.
        self.assertEqual(kwargs["input_layout"], "BSND")
        self.assertEqual(kwargs["sparse_mode"], 0)
        # BSND takes per-request query lengths, not TND's cumulative form.
        self.assertEqual(kwargs["actual_seq_lengths"], [3, 3])
        # The KV lengths handed to the operator stay the host-side bound.
        self.assertEqual(kwargs["actual_seq_lengths_kv"], [10, 12])
        self.assertEqual(tuple(kwargs["query"].shape), (2, 3, 2, 4))
        self.assertEqual(tuple(kwargs["atten_mask"].shape), (2, 3, 2 * 4))

    def test_mask_is_built_once_and_reused_across_layers(self):
        """One mask per step, shared by every layer in the attention group."""
        impl, meta = self._impl(), self._metadata()
        fake = MagicMock(return_value=(torch.zeros(2, 3, 2, 4), None))
        with (
            patch.object(attn_module.torch_npu, "npu_fused_infer_attention_score", fake),
            patch.object(attn_module, "build_draft_tail_mask", wraps=attn_module.build_draft_tail_mask) as builder,
        ):
            self._call(impl, meta)
            self._call(impl, meta)

        self.assertEqual(builder.call_count, 1)
        self.assertEqual(fake.call_count, 2)
