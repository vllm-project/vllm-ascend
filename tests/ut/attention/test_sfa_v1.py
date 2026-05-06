import sys
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    DSACPContext,
)
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):
    def setUp(self):
        # The backend's get_builder_cls/get_impl_cls call enable_cp(), which
        # depends on a vllm config being available. Patch it out for tests
        # that don't focus on the CP path.
        self.utils_patcher = patch("vllm_ascend.attention.sfa_v1.enable_cp", return_value=False)
        self.utils_patcher.start()

    def tearDown(self):
        self.utils_patcher.stop()

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_name_with_v2_runner(self):
        with patch("vllm_ascend.attention.sfa_v1.envs_vllm") as mock_envs:
            mock_envs.VLLM_USE_V2_MODEL_RUNNER = True
            self.assertEqual(AscendSFABackend.get_name(), "FLASH_ATTN")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_kv_cache_shape_with_cache_type(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128, cache_type="quant")
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)

    def test_get_supported_kernel_block_sizes(self):
        result = AscendSFABackend.get_supported_kernel_block_sizes()
        self.assertEqual(result, [128])

    def test_accept_output_buffer(self):
        self.assertTrue(AscendSFABackend.accept_output_buffer)

    @patch("vllm_ascend.attention.sfa_v1.enable_cp")
    def test_get_builder_cls_with_cp(self, mock_enable_cp):
        mock_enable_cp.return_value = True
        builder_cls = AscendSFABackend.get_builder_cls()
        self.assertIsNotNone(builder_cls)

    @patch("vllm_ascend.attention.sfa_v1.enable_cp")
    def test_get_impl_cls_with_cp(self, mock_enable_cp):
        mock_enable_cp.return_value = True
        impl_cls = AscendSFABackend.get_impl_cls()
        self.assertIsNotNone(impl_cls)


class TestDSACPContext(TestBase):
    def test_dsa_cp_context_default(self):
        slot_mapping_cp = torch.tensor([0, 1, 2, 3])
        actual_seq_lengths_query = torch.tensor([2, 4])
        actual_seq_lengths_key = torch.tensor([5, 10])
        ctx = DSACPContext(
            num_tokens=4,
            num_tokens_pad=8,
            local_start=0,
            local_end=4,
            local_end_with_pad=4,
            slot_mapping_cp=slot_mapping_cp,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )
        self.assertEqual(ctx.num_tokens, 4)
        self.assertEqual(ctx.num_tokens_pad, 8)
        self.assertEqual(ctx.local_start, 0)
        self.assertEqual(ctx.local_end, 4)
        self.assertEqual(ctx.local_end_with_pad, 4)
        self.assertIs(ctx.slot_mapping_cp, slot_mapping_cp)
        self.assertIs(ctx.actual_seq_lengths_query, actual_seq_lengths_query)
        self.assertIs(ctx.actual_seq_lengths_key, actual_seq_lengths_key)


class TestAscendSFAMetadata(TestBase):
    def test_ascend_sfa_metadata_default(self):
        num_actual_tokens = 100
        slot_mapping = torch.randn(100, 4, 1024)
        seq_lens = torch.tensor([30, 50])
        cum_query_lens = torch.tensor([0, 30, 80])
        block_table = torch.randint(0, 100, (100, 4))

        rope_dim = 32
        max_seq_len = int(seq_lens.max().item())
        sin = torch.randn(max_seq_len, rope_dim)
        cos = torch.randn(max_seq_len, rope_dim)

        num_input_tokens = 2
        head_dim = None
        attn_mask = None
        attn_state = AscendAttentionState.ChunkedPrefill

        metadata = AscendSFAMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            num_input_tokens=num_input_tokens,
            head_dim=head_dim,
            attn_mask=attn_mask,
            attn_state=attn_state,
        )

        self.assertEqual(metadata.num_actual_tokens, num_actual_tokens)
        self.assertIs(metadata.slot_mapping, slot_mapping)
        self.assertTrue(torch.equal(metadata.seq_lens, seq_lens))
        self.assertTrue(torch.equal(metadata.cum_query_lens, cum_query_lens))
        self.assertIs(metadata.block_table, block_table)
        self.assertIs(metadata.sin, sin)
        self.assertIs(metadata.cos, cos)
        self.assertEqual(metadata.num_input_tokens, num_input_tokens)
        self.assertIs(metadata.head_dim, head_dim)
        self.assertIs(metadata.attn_mask, attn_mask)
        self.assertEqual(metadata.attn_state, attn_state)
        self.assertIsNone(metadata.dsa_cp_context)
        self.assertEqual(metadata.num_decodes, 0)
        self.assertEqual(metadata.num_decode_tokens, 0)
        self.assertEqual(metadata.num_prefills, 0)


class TestAscendSFAMetadataBuilder(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def setUp(self, mock_tp):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        self.mock_cfg = MagicMock()

        self.mock_cfg.parallel_config = MagicMock()
        self.mock_cfg.parallel_config.tensor_parallel_size = 1
        self.mock_cfg.parallel_config.prefill_context_parallel_size = 1
        self.mock_cfg.parallel_config.decode_context_parallel_size = 1

        self.mock_cfg.compilation_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config = MagicMock()
        self.mock_cfg.compilation_config.pass_config.enable_sp = False

        self.mock_cfg.speculative_config.num_speculative_tokens = 0

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

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

        if hasattr(enable_dsa_cp, "cache_clear"):
            enable_dsa_cp.cache_clear()

    def tearDown(self):
        self.patcher.stop()
        self.parent_init_patcher.stop()

    def _make_vllm_config(self):
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        vllm_config.scheduler_config.max_num_seqs = 16
        return vllm_config

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = self._make_vllm_config()
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        assert builder.device == device
        assert builder.vllm_config == vllm_config

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_no_speculative_config(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1"]
        vllm_config = self._make_vllm_config()
        vllm_config.speculative_config = None
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        self.assertEqual(builder.decode_threshold, 1)
        self.assertEqual(builder.reorder_batch_threshold, 1)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_determine_chunked_prefill_workspace_size(self):
        with patch("vllm_ascend.attention.sfa_v1.ascend_chunked_prefill_workspace_size") as mock_workspace_size:
            mock_workspace_size.return_value = 4096
            vllm_config = MagicMock()
            result = AscendSFAMetadataBuilder.determine_chunked_prefill_workspace_size(vllm_config)
            self.assertEqual(result, 4096)
            mock_workspace_size.assert_called_once_with(vllm_config)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_get_cudagraph_support(self):
        from vllm.v1.attention.backend import AttentionCGSupport

        result = AscendSFAMetadataBuilder.get_cudagraph_support(MagicMock(), MagicMock())
        self.assertEqual(result, AttentionCGSupport.UNIFORM_BATCH)

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_reorder_batch(self):
        kv_cache_spec = MagicMock()
        layer_names = ["layer1"]
        vllm_config = self._make_vllm_config()
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )
        result = builder.reorder_batch(MagicMock(), MagicMock())
        self.assertFalse(result)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False

        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = self._make_vllm_config()
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        metadata = builder.build(
            common_prefix_len=10,
            common_attn_metadata=common_attn_metadata,
        )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_uses_seq_lens_cpu_when_private_is_none(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False
        mock_get_current_vllm_config.return_value = self._make_vllm_config()

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=MagicMock(),
            layer_names=["layer1"],
            vllm_config=self._make_vllm_config(),
            device=torch.device("cpu"),
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 4
        common_attn_metadata.num_input_tokens = 4
        common_attn_metadata.query_start_loc = torch.tensor([0, 2, 4])
        common_attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata.seq_lens_cpu = torch.tensor([2, 2])
        common_attn_metadata.seq_lens = torch.tensor([2, 2])
        common_attn_metadata.positions = torch.tensor([0, 1, 0, 1])
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.zeros((2, 4))

        mock_get_cos_and_sin_mla.return_value = (torch.randn(4), torch.randn(4))

        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertIsInstance(metadata, AscendSFAMetadata)
        self.assertTrue(torch.equal(metadata.seq_lens_cpu, torch.tensor([2, 2])))

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_falls_back_to_seq_lens_when_cpu_is_none(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        mock_enable_dsa_cp.return_value = False
        mock_get_current_vllm_config.return_value = self._make_vllm_config()

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=MagicMock(),
            layer_names=["layer1"],
            vllm_config=self._make_vllm_config(),
            device=torch.device("cpu"),
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = 4
        common_attn_metadata.num_input_tokens = 4
        common_attn_metadata.query_start_loc = torch.tensor([0, 2, 4])
        common_attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        common_attn_metadata._seq_lens_cpu = None
        common_attn_metadata.seq_lens_cpu = None
        common_attn_metadata.seq_lens = torch.tensor([2, 2])
        common_attn_metadata.positions = torch.tensor([0, 1, 0, 1])
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.zeros((2, 4))

        mock_get_cos_and_sin_mla.return_value = (torch.randn(4), torch.randn(4))

        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertIsInstance(metadata, AscendSFAMetadata)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        layer_names = ["layer1", "layer2"]
        vllm_config = self._make_vllm_config()
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        attn_metadata = builder.build_for_graph_capture(
            common_attn_metadata=common_attn_metadata,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        assert isinstance(attn_metadata, AscendSFAMetadata)
        assert attn_metadata.attn_state == AscendAttentionState.DecodeOnly

    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_with_dsa_cp(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
        mock_get_tp_group,
    ):
        mock_enable_dsa_cp.return_value = True
        mock_get_current_vllm_config.return_value = self._make_vllm_config()

        tp_group = MagicMock()
        tp_group.world_size = 2
        tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = tp_group

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=MagicMock(),
            layer_names=["layer1"],
            vllm_config=self._make_vllm_config(),
            device=torch.device("cpu"),
        )
        # Ensure the cached value matches the test's expectation.
        builder.enable_dsa_cp = True

        num_input_tokens = 8
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = num_input_tokens
        common_attn_metadata.num_input_tokens = num_input_tokens
        common_attn_metadata.query_start_loc = torch.tensor([0, 4, 8])
        common_attn_metadata.slot_mapping = torch.arange(num_input_tokens, dtype=torch.int64)
        common_attn_metadata._seq_lens_cpu = torch.tensor([4, 4])
        common_attn_metadata.seq_lens_cpu = torch.tensor([4, 4])
        common_attn_metadata.seq_lens = torch.tensor([4, 4])
        common_attn_metadata.positions = torch.arange(num_input_tokens, dtype=torch.int64)
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.zeros((2, 4))

        # cos/sin for DSA-CP: shape [num_input_tokens, ...]
        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(num_input_tokens, 1, 1, 64),
            torch.randn(num_input_tokens, 1, 1, 64),
        )

        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertIsInstance(metadata, AscendSFAMetadata)
        self.assertIsNotNone(metadata.dsa_cp_context)
        self.assertEqual(metadata.dsa_cp_context.num_tokens, num_input_tokens)

    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_with_dsa_cp_padding(
        self,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
        mock_get_tp_group,
    ):
        # Trigger the padding paths (cos/sin and slot_mapping pad)
        mock_enable_dsa_cp.return_value = True
        mock_get_current_vllm_config.return_value = self._make_vllm_config()

        tp_group = MagicMock()
        tp_group.world_size = 4
        tp_group.rank_in_group = 0
        mock_get_tp_group.return_value = tp_group

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=MagicMock(),
            layer_names=["layer1"],
            vllm_config=self._make_vllm_config(),
            device=torch.device("cpu"),
        )
        builder.enable_dsa_cp = True

        # 5 input tokens that need to be padded to 8 (round-up to multiple of 4)
        num_input_tokens = 5
        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 2
        common_attn_metadata.num_actual_tokens = num_input_tokens
        common_attn_metadata.num_input_tokens = num_input_tokens
        common_attn_metadata.query_start_loc = torch.tensor([0, 2, 5])
        common_attn_metadata.slot_mapping = torch.arange(num_input_tokens, dtype=torch.int64)
        common_attn_metadata._seq_lens_cpu = torch.tensor([2, 3])
        common_attn_metadata.seq_lens_cpu = torch.tensor([2, 3])
        common_attn_metadata.seq_lens = torch.tensor([2, 3])
        common_attn_metadata.positions = torch.arange(num_input_tokens, dtype=torch.int64)
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.zeros((2, 4))

        mock_get_cos_and_sin_mla.return_value = (
            torch.randn(num_input_tokens, 1, 1, 64),
            torch.randn(num_input_tokens, 1, 1, 64),
        )

        metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
        self.assertIsInstance(metadata, AscendSFAMetadata)
        self.assertIsNotNone(metadata.dsa_cp_context)

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_build_for_graph_capture_unsupported_state(
        self, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()
        mock_get_current_vllm_config.return_value = cfg

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=MagicMock(),
            layer_names=["layer1"],
            vllm_config=self._make_vllm_config(),
            device=torch.device("cpu"),
        )

        with self.assertRaises(NotImplementedError):
            builder.build_for_graph_capture(
                common_attn_metadata=MagicMock(),
                attn_state=AscendAttentionState.ChunkedPrefill,
            )


def _make_indexer_mock():
    indexer = MagicMock()
    indexer.n_head = 64
    indexer.head_dim = 128
    indexer.wq_b = MagicMock()
    indexer.wk_weights_proj = MagicMock()
    indexer.k_norm = MagicMock()
    return indexer


def _make_impl_kwargs(extra=None):
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
        "q_a_layernorm": MagicMock(),
        "rotary_emb": MagicMock(),
        "indexer": _make_indexer_mock(),
        "layer_name": "layer_0",
    }
    if extra:
        kwargs.update(extra)
    return kwargs


class TestAscendSFAImpl(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    def setUp(
        self,
        mock_get_current_vllm_config,
        _mock_enable_dsa_cp,
        _mock_enable_dsa_cp_with_layer_shard,
        _mock_enable_dsa_cp_with_o_proj_tp,
        mock_tp,
    ):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()

        vllm_config = MagicMock()
        speculative_config = MagicMock()
        model_config = MagicMock()
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        parallel_config.tensor_parallel_size = 2
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        model_config.dtype = torch.float16
        model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.model_config = model_config
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {"refresh": True}
        vllm_config.parallel_config = parallel_config
        mock_get_current_vllm_config.return_value = vllm_config
        init_ascend_config(vllm_config)

        self.kwargs = _make_impl_kwargs()
        self.impl = AscendSFAImpl(
            num_heads=256,
            head_size=1024,
            scale=0.1,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **self.kwargs,
        )
        # Reset class-level singletons across tests for isolation.
        AscendSFAImpl.o_proj_full_pool = None
        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None

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
        self.assertEqual(self.impl.num_queries_per_kv, 32)
        self.assertEqual(self.impl.n_head, 64)
        self.assertEqual(self.impl.head_dim, 128)
        self.assertTrue(self.impl.is_rope_neox_style)
        self.assertFalse(self.impl.use_torch_npu_lightning_indexer)

    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    def test_init_with_glm_moe_dsa_model(
        self,
        mock_get_current_vllm_config,
        _mock_enable_dsa_cp,
        _mock_enable_dsa_cp_with_layer_shard,
        _mock_enable_dsa_cp_with_o_proj_tp,
        mock_tp,
    ):
        mock_tp.world_size = 1
        mock_tp.rank_in_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 0
        vllm_config.speculative_config = speculative_config
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_config.model_type = "glm_moe_dsa"
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {"refresh": True}
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config = parallel_config
        mock_get_current_vllm_config.return_value = vllm_config
        init_ascend_config(vllm_config)

        impl = AscendSFAImpl(
            num_heads=256,
            head_size=1024,
            scale=0.1,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **_make_impl_kwargs(),
        )
        self.assertFalse(impl.is_rope_neox_style)
        self.assertTrue(impl.use_torch_npu_lightning_indexer)

    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.get_ascend_config")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard")
    @patch("vllm_ascend.attention.sfa_v1.register_all_layers_to_shard_weight_series")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    def test_init_with_dsa_cp_layer_shard(
        self,
        mock_get_current_vllm_config,
        mock_register,
        mock_enable_dsa_cp_with_layer_shard,
        mock_enable_dsa_cp,
        mock_get_ascend_config,
        mock_tp,
    ):
        mock_tp.world_size = 2
        mock_tp.rank_in_group = MagicMock()
        mock_enable_dsa_cp.return_value = True
        mock_enable_dsa_cp_with_layer_shard.return_value = True

        ascend_config = MagicMock()
        ascend_config.layer_sharding = ["layer_0", "missing_layer"]
        ascend_config.is_sparse_c8_layer.return_value = False
        ascend_config.enable_shared_expert_dp = False
        mock_get_ascend_config.return_value = ascend_config

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        vllm_config.model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.parallel_config.tensor_parallel_size = 2
        mock_get_current_vllm_config.return_value = vllm_config

        kwargs = _make_impl_kwargs(extra={"layer_0": "sharding_cfg_0"})
        impl = AscendSFAImpl(
            num_heads=4,
            head_size=128,
            scale=0.1,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )
        self.assertEqual(impl.local_num_heads, 4 * 2)
        self.assertEqual(impl.layer_sharding_kwargs, ["sharding_cfg_0"])
        mock_register.assert_called_once()

    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_ascend_config")
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    def test_init_with_sparse_c8(
        self,
        mock_get_current_vllm_config,
        mock_get_ascend_config,
        _mock_enable_dsa_cp,
        _mock_enable_dsa_cp_with_layer_shard,
        _mock_enable_dsa_cp_with_o_proj_tp,
        mock_tp,
    ):
        mock_tp.world_size = 1
        mock_tp.rank_in_group = MagicMock()

        ascend_config = MagicMock()
        ascend_config.is_sparse_c8_layer.return_value = True
        ascend_config.enable_shared_expert_dp = False
        ascend_config.layer_sharding = None
        mock_get_ascend_config.return_value = ascend_config

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        vllm_config.model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_get_current_vllm_config.return_value = vllm_config

        impl = AscendSFAImpl(
            num_heads=4,
            head_size=128,
            scale=0.1,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **_make_impl_kwargs(),
        )
        self.assertTrue(impl.use_sparse_c8_indexer)
        self.assertEqual(impl.c8_k_cache_dtype, torch.int8)
        self.assertEqual(impl.c8_k_scale_cache_dtype, torch.float16)

    def test_update_graph_params_no_op(self):
        # Just verify it doesn't raise. SFA's update_graph_params is a no-op.
        AscendSFAImpl.update_graph_params(
            update_stream=MagicMock(),
            forward_context=MagicMock(),
            num_tokens=128,
        )

    def test_forward_mha_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.impl.forward_mha(
                q=torch.randn(2, 4, 8),
                kv_c_normed=torch.randn(2, 4, 8),
                k_pe=torch.randn(2, 4, 8),
                kv_c_and_k_pe_cache=torch.randn(2, 4, 8),
                attn_metadata=MagicMock(),
                k_scale=torch.tensor(1.0),
                output=torch.randn(2, 4, 8),
            )
        self.assertIn("forward_mha is not supported", str(ctx.exception))

    def test_forward_mqa_not_implemented(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.impl.forward_mqa(
                q=torch.randn(2, 4, 8),
                kv_c_and_k_pe_cache=torch.randn(2, 4, 8),
                attn_metadata=MagicMock(),
                layer=MagicMock(),
            )
        self.assertIn("forward_mqa is not supported", str(ctx.exception))

    @patch("torch_npu.npu_interleave_rope")
    def test_rope_single(self, mock_npu_interleave_rope):
        batch_size = 2
        num_heads = 4
        dim = 32

        x = torch.randn(batch_size, num_heads, dim)
        cos = torch.randn(batch_size, 1, 1, dim)
        sin = torch.randn(batch_size, 1, 1, dim)

        mock_npu_interleave_rope.return_value = torch.randn(batch_size, num_heads, 1, dim)

        result = self.impl.rope_single(x, cos, sin)

        self.assertEqual(result.shape, (batch_size, num_heads, dim))
        mock_npu_interleave_rope.assert_called_once()

    def test_q_proj_and_k_up_proj(self):
        batch_size = 4
        x = torch.randn(batch_size, self.impl.local_num_heads, self.impl.qk_head_dim)
        q_proj_output = torch.randn(batch_size, self.impl.local_num_heads, self.impl.qk_head_dim)
        self.impl.q_proj.return_value = (q_proj_output,)
        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )

        ql_nope, q_pe = self.impl._q_proj_and_k_up_proj(x)
        self.assertEqual(ql_nope.shape, (batch_size, self.impl.local_num_heads, self.impl.kv_lora_rank))
        self.assertEqual(q_pe.shape, (batch_size, self.impl.local_num_heads, self.impl.qk_rope_head_dim))

    def test_v_up_proj_with_bmm_path(self):
        # Use small num_input_tokens to satisfy BMM_TRANS_MAX_SUPPORTED_TOKENS branch
        num_input_tokens = 8
        x = torch.randn(num_input_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank, dtype=torch.float16)
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim, dtype=torch.float16
        )

        with patch.object(torch.ops._C_ascend, "batch_matmul_transpose", create=True) as mock_bmm:
            def fake_bmm(a, b, out):
                out.copy_(torch.zeros_like(out))
            mock_bmm.side_effect = fake_bmm
            result = self.impl._v_up_proj(x)

        self.assertEqual(result.shape, (num_input_tokens, self.impl.local_num_heads * self.impl.v_head_dim))

    def test_v_up_proj_fallback_path(self):
        # Use float32 (not in the list) to force the fallback branch
        num_input_tokens = 4
        x = torch.randn(num_input_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank, dtype=torch.float32)
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim, dtype=torch.float32
        )

        result = self.impl._v_up_proj(x)
        self.assertEqual(result.shape, (num_input_tokens, self.impl.local_num_heads * self.impl.v_head_dim))

    def test_get_full_kv_returns_input(self):
        k = torch.randn(4, 8, 16)
        result = self.impl._get_full_kv(k, MagicMock())
        self.assertIs(result, k)

    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_exec_kv_dsa_cp_disabled(self, mock_torch_npu):
        self.impl.enable_dsa_cp = False
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight = torch.randn(64)
        self.impl.kv_a_layernorm.variance_epsilon = 1e-6
        kv_no_split = torch.randn(2, self.impl.num_kv_heads * (self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))
        cos = torch.randn(2, 32)
        sin = torch.randn(2, 32)
        kv_cache = (torch.randn(10, 1, 1, 64), torch.randn(10, 1, 1, 32))
        slots = torch.tensor([0, 1], dtype=torch.int32)

        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None, None, None)

        k_pe, k_nope = self.impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, MagicMock())
        self.assertIsNone(k_pe)
        self.assertIsNone(k_nope)
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.assert_called_once()

    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_exec_kv_dsa_cp_enabled(self, mock_torch_npu):
        self.impl.enable_dsa_cp = True
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight = torch.randn(64)
        self.impl.kv_a_layernorm.variance_epsilon = 1e-6
        kv_no_split = torch.randn(2, self.impl.num_kv_heads * (self.impl.kv_lora_rank + self.impl.qk_rope_head_dim))
        cos = torch.randn(2, 32)
        sin = torch.randn(2, 32)
        kv_cache = (torch.randn(10, 1, 1, 64), torch.randn(10, 1, 1, 32))
        slots = torch.tensor([0, 1], dtype=torch.int32)

        fake_k_pe = torch.randn(2, 1, 1, 32)
        fake_k_nope = torch.randn(2, 1, 1, 64)
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None, fake_k_pe, fake_k_nope)

        k_pe, k_nope = self.impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, MagicMock())
        self.assertIs(k_pe, fake_k_pe)
        self.assertIs(k_nope, fake_k_nope)

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False)
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_indexer_select_pre_process_no_triton(self, mock_torch_npu):
        self.impl.use_sparse_c8_indexer = False
        x = torch.randn(2, self.impl.qk_head_dim)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)

        self.impl.k_norm.return_value = torch.randn(2, self.impl.head_dim)
        mock_torch_npu.npu_rotary_mul.return_value = torch.randn(2, 1, 1, self.impl.qk_rope_head_dim)

        k_li, k_li_scale = self.impl.indexer_select_pre_process(x, cos, sin)
        self.assertIsNone(k_li_scale)
        self.assertEqual(k_li.shape[-1], self.impl.head_dim)

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", True)
    @patch("vllm_ascend.attention.sfa_v1.rope_forward_triton_siso")
    def test_indexer_select_pre_process_with_triton(self, mock_rope):
        self.impl.use_sparse_c8_indexer = False
        x = torch.randn(2, self.impl.qk_head_dim)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.k_norm.return_value = torch.randn(2, self.impl.head_dim)
        mock_rope.return_value = torch.randn(2, 1, self.impl.head_dim)

        k_li, k_li_scale = self.impl.indexer_select_pre_process(x, cos, sin)
        self.assertIsNone(k_li_scale)
        mock_rope.assert_called_once()

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", True)
    @patch("vllm_ascend.attention.sfa_v1.rope_forward_triton_siso")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_indexer_select_pre_process_sparse_c8(self, mock_torch_npu, mock_rope):
        self.impl.use_sparse_c8_indexer = True
        self.impl.c8_k_cache_dtype = torch.int8
        self.impl.c8_k_scale_cache_dtype = torch.float16
        AscendSFAImpl.k_hadamard = torch.randn(self.impl.head_dim, self.impl.head_dim)

        x = torch.randn(2, self.impl.qk_head_dim)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.k_norm.return_value = torch.randn(2, self.impl.head_dim)
        mock_rope.return_value = torch.randn(2, 1, self.impl.head_dim)

        mock_torch_npu.npu_dynamic_quant.return_value = (
            torch.randint(-128, 127, (2, self.impl.head_dim), dtype=torch.int8),
            torch.randn(2, dtype=torch.float32),
        )

        k_li, k_li_scale = self.impl.indexer_select_pre_process(x, cos, sin)
        self.assertIsNotNone(k_li_scale)
        self.assertEqual(k_li_scale.dtype, torch.float16)
        AscendSFAImpl.k_hadamard = None

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False)
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_indexer_select_post_process_default(self, mock_torch_npu):
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )

        mock_torch_npu.npu_rotary_mul.return_value = torch.randn(
            2, self.impl.n_head, 1, self.impl.qk_rope_head_dim
        )

        attn_metadata = MagicMock()
        attn_metadata.block_table = torch.tensor([[0]])
        actual_seq_lengths_query = torch.tensor([1, 2])
        actual_seq_lengths_key = torch.tensor([1, 2])

        with patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", True)
    @patch("vllm_ascend.attention.sfa_v1.rope_forward_triton_siso")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_indexer_select_post_process_torch_npu_indexer(self, mock_torch_npu, mock_rope):
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = True
        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )

        mock_rope.return_value = torch.randn(2, self.impl.n_head, self.impl.head_dim)
        mock_torch_npu.npu_lightning_indexer.return_value = (torch.tensor([[0]]), None)

        attn_metadata = MagicMock()
        attn_metadata.block_table = torch.tensor([[0]])
        actual_seq_lengths_query = torch.tensor([1, 2])
        actual_seq_lengths_key = torch.tensor([1, 2])

        result = self.impl.indexer_select_post_process(
            x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
        )
        self.assertIsNotNone(result)
        mock_torch_npu.npu_lightning_indexer.assert_called_once()

    @patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", True)
    @patch("vllm_ascend.attention.sfa_v1.rope_forward_triton_siso")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_indexer_select_post_process_sparse_c8(self, mock_torch_npu, mock_rope):
        self.impl.use_sparse_c8_indexer = True
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.c8_k_cache_dtype = torch.int8
        self.impl.c8_k_scale_cache_dtype = torch.float16
        AscendSFAImpl.q_hadamard = torch.randn(self.impl.head_dim, self.impl.head_dim)

        x = torch.randn(2, self.impl.qk_head_dim)
        q_c = torch.randn(2, self.impl.q_lora_rank)
        cos = torch.randn(2, self.impl.qk_rope_head_dim)
        sin = torch.randn(2, self.impl.qk_rope_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.zeros(2, 4, 1, self.impl.head_dim, dtype=torch.int8),
            torch.randn(2, 4, 1, 1, dtype=torch.float16),
        )

        kw_out = torch.randn(2, self.impl.head_dim * 2)
        self.impl.wk_weights_proj.return_value = (kw_out, None)
        self.impl.wq_b.return_value = (
            torch.randn(2, self.impl.n_head * self.impl.head_dim),
            None,
        )
        mock_rope.return_value = torch.randn(2, self.impl.n_head, self.impl.head_dim)
        mock_torch_npu.npu_dynamic_quant.return_value = (
            torch.randint(-128, 127, (2 * self.impl.n_head, self.impl.head_dim), dtype=torch.int8),
            torch.randn(2 * self.impl.n_head, dtype=torch.float32),
        )

        attn_metadata = MagicMock()
        attn_metadata.block_table = torch.tensor([[0]])
        actual_seq_lengths_query = torch.tensor([1, 2])
        actual_seq_lengths_key = torch.tensor([1, 2])

        with patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer_quant", create=True, return_value=torch.tensor([[0]])
        ):
            result = self.impl.indexer_select_post_process(
                x, q_c, kv_cache, attn_metadata, cos, sin, actual_seq_lengths_query, actual_seq_lengths_key
            )
        self.assertIsNotNone(result)
        AscendSFAImpl.q_hadamard = None

    def test_execute_sparse_flash_attention_process(self):
        ql_nope = torch.randn(2, self.impl.local_num_heads, self.impl.kv_lora_rank)
        q_pe = torch.randn(2, self.impl.local_num_heads, self.impl.qk_rope_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )
        topk_indices = torch.tensor([[0]])
        attn_metadata = MagicMock()
        attn_metadata.block_table = torch.tensor([[0]])

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(2, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ) as mock_sfa:
            result = self.impl._execute_sparse_flash_attention_process(
                ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, torch.tensor([1, 2]), torch.tensor([1, 2])
            )
        mock_sfa.assert_called_once()
        self.assertIsNotNone(result)

    def test_init_o_proj_tp_full_params(self):
        sample = torch.randn(8, 32)
        self.impl.tp_size = 2
        self.impl.o_proj.weight = sample.clone()
        self.impl.o_proj.aclnn_input_scale = torch.randn(8)
        self.impl.o_proj.aclnn_input_scale_reciprocal = torch.randn(8)
        self.impl.o_proj.aclnn_input_offset = torch.randn(8)

        AscendSFAImpl.o_proj_full_pool = None
        self.impl._init_o_proj_tp_full_params()

        self.assertIsNotNone(AscendSFAImpl.o_proj_full_pool)
        self.assertEqual(AscendSFAImpl.o_proj_full_pool.shape, (sample.shape[0] * 2, sample.shape[1]))
        self.assertTrue(hasattr(self.impl, "o_proj_tp_weight"))
        self.assertTrue(hasattr(self.impl, "o_proj_full_aclnn_input_scale"))
        self.assertEqual(self.impl.o_proj_full_aclnn_input_scale.shape[0], 8 * 2)
        AscendSFAImpl.o_proj_full_pool = None

    def test_init_o_proj_tp_full_params_pool_already_set(self):
        sample = torch.randn(8, 32)
        AscendSFAImpl.o_proj_full_pool = torch.empty((16, 32))
        self.impl.tp_size = 2
        self.impl.o_proj.weight = sample.clone()
        self.impl.o_proj.aclnn_input_scale = torch.randn(8)
        self.impl.o_proj.aclnn_input_scale_reciprocal = torch.randn(8)
        self.impl.o_proj.aclnn_input_offset = torch.randn(8)

        before_id = id(AscendSFAImpl.o_proj_full_pool)
        self.impl._init_o_proj_tp_full_params()
        # Ensure pool wasn't reallocated.
        self.assertEqual(id(AscendSFAImpl.o_proj_full_pool), before_id)
        AscendSFAImpl.o_proj_full_pool = None

    def test_handle_o_proj_weight_switch_should_shard(self):
        # Set up o_proj attrs and full-mode params
        self.impl.tp_size = 2
        sample = torch.randn(4, 8)
        self.impl.o_proj.weight = sample.clone()
        AscendSFAImpl.o_proj_full_pool = torch.empty((8, 8))

        self.impl.o_proj_tp_weight = sample.clone()
        self.impl.o_proj_tp_aclnn_input_scale = torch.randn(4)
        self.impl.o_proj_tp_aclnn_input_scale_reciprocal = torch.randn(4)
        self.impl.o_proj_tp_aclnn_input_offset = torch.randn(4)
        self.impl.o_proj_full_aclnn_input_scale = torch.randn(8)
        self.impl.o_proj_full_aclnn_input_scale_reciprocal = torch.randn(8)
        self.impl.o_proj_full_aclnn_input_offset = torch.randn(8)
        self.impl.o_proj.aclnn_input_scale = torch.randn(4)
        self.impl.o_proj.aclnn_input_scale_reciprocal = torch.randn(4)
        self.impl.o_proj.aclnn_input_offset = torch.randn(4)

        self.impl.o_proj.quant_method = MagicMock()
        self.impl.o_proj.quant_method.quant_method.apply.return_value = torch.zeros(2, 4)

        attn_output = torch.randn(2, 4)
        output = torch.zeros(2, 4)
        handle = MagicMock()

        result, require_forward = self.impl._handle_o_proj_weight_switch_and_forward(
            attn_output=attn_output,
            output=output,
            o_proj_full_handle=handle,
            should_shard_weight=True,
        )
        self.assertFalse(require_forward)
        self.assertIs(result, output)
        handle.wait.assert_called_once()
        AscendSFAImpl.o_proj_full_pool = None

    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("torch.distributed.all_to_all_single")
    def test_handle_o_proj_weight_switch_decode_path(self, mock_all_to_all, mock_get_tp_group):
        self.impl.tp_size = 2
        # attn_output shape: [batch_size * seq, num_heads * v_head_dim]
        attn_output = torch.randn(4, self.impl.num_heads * self.impl.v_head_dim)
        output = torch.zeros(4, self.impl.num_heads * self.impl.v_head_dim)
        mock_get_tp_group.return_value = MagicMock()

        result, require_forward = self.impl._handle_o_proj_weight_switch_and_forward(
            attn_output=attn_output,
            output=output,
            o_proj_full_handle=None,
            should_shard_weight=False,
        )
        self.assertTrue(require_forward)
        mock_all_to_all.assert_called_once()
        # The result should be a tensor different from the input (after permute/reshape).
        self.assertIsNotNone(result)

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_basic(self, mock_torch_npu, mock_maybe_trans_nz, mock_dispose_layer):
        mock_maybe_trans_nz.side_effect = lambda x: x
        # Set up kv_b_proj
        self.impl.kv_b_proj = MagicMock()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 256
        kv_b_proj_weight = torch.randn(
            256 * (self.impl.qk_nope_head_dim + self.impl.v_head_dim), self.impl.kv_lora_rank
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight

        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = False
        self.impl.enable_mlapo = False
        self.impl.use_sparse_c8_indexer = False

        self.impl.process_weights_after_loading(torch.float16)

        self.assertTrue(hasattr(self.impl, "W_UV"))
        self.assertTrue(hasattr(self.impl, "W_UK_T"))
        mock_dispose_layer.assert_called_once_with(self.impl.kv_b_proj)

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_existing_w_uv(self, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose):
        mock_maybe_trans_nz.side_effect = lambda x: x
        self.impl.kv_b_proj = MagicMock()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = False
        self.impl.enable_mlapo = False
        self.impl.use_sparse_c8_indexer = False

        # Pre-allocate W_UV/W_UK_T to exercise the in-place copy_ branch.
        self.impl.W_UV = torch.empty(self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim)
        self.impl.W_UK_T = torch.empty(self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank)

        self.impl.process_weights_after_loading(torch.float16)
        self.assertTrue(hasattr(self.impl, "W_UV"))

    @patch("vllm_ascend.attention.sfa_v1.is_hidden_layer", return_value=True)
    @patch("vllm_ascend.attention.sfa_v1.post_process_after_loading_for_shard_weight_series")
    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_with_dsa_cp_layer_shard(
        self,
        mock_torch_npu,
        mock_maybe_trans_nz,
        _mock_dispose,
        mock_post_process,
        _mock_is_hidden_layer,
    ):
        mock_maybe_trans_nz.side_effect = lambda x: x
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = True
        self.impl.enable_mlapo = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.layer_sharding_kwargs = ["layer_shard_0"]

        self.impl.process_weights_after_loading(torch.float16)
        mock_post_process.assert_called_once_with("layer_shard_0")

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_with_dsa_cp_o_proj_init(
        self, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose
    ):
        mock_maybe_trans_nz.side_effect = lambda x: x
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = False
        self.impl.enable_mlapo = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.tp_size = 2
        self.impl.o_proj = MagicMock()
        self.impl.o_proj.weight = torch.randn(8, 16)
        self.impl.o_proj.aclnn_input_scale = torch.randn(8)
        self.impl.o_proj.aclnn_input_scale_reciprocal = torch.randn(8)
        self.impl.o_proj.aclnn_input_offset = torch.randn(8)

        AscendSFAImpl.o_proj_full_pool = None
        self.impl.process_weights_after_loading(torch.float16)
        self.assertIsNotNone(AscendSFAImpl.o_proj_full_pool)
        AscendSFAImpl.o_proj_full_pool = None

    @patch.object(AscendSFAImpl, "_process_weights_for_fused_mlapo")
    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_mlapo_w8a8_path(
        self, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose, mock_mlapo_proc
    ):
        # mlapo enabled with W8A8 quant -> calls _process_weights_for_fused_mlapo
        mock_maybe_trans_nz.side_effect = lambda x: x
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = False
        self.impl.enable_mlapo = True
        self.impl.use_sparse_c8_indexer = False
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.quant_method.quant_method = MagicMock(spec=AscendW8A8LinearMethod)

        self.impl.process_weights_after_loading(torch.float16)
        mock_mlapo_proc.assert_called_once_with(torch.float16)
        self.assertTrue(self.impl.enable_mlapo)

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_mlapo_disabled_by_quant(
        self, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose
    ):
        # mlapo is enabled but the fused_qkv_a_proj is not W8A8 quantized -> mlapo gets disabled
        mock_maybe_trans_nz.side_effect = lambda x: x
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = False
        self.impl.enable_mlapo = True
        self.impl.use_sparse_c8_indexer = False
        # Default mock for fused_qkv_a_proj.quant_method.quant_method (not W8A8)
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.quant_method.quant_method = MagicMock()  # not AscendW8A8LinearMethod

        self.impl.process_weights_after_loading(torch.float16)
        self.assertFalse(self.impl.enable_mlapo)

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_after_loading_mlapo_disabled_by_dsa_cp(
        self, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose
    ):
        # mlapo + dsa_cp -> mlapo disabled
        mock_maybe_trans_nz.side_effect = lambda x: x
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj = MagicMock()
        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = False
        self.impl.enable_mlapo = True
        self.impl.use_sparse_c8_indexer = False
        self.impl.tp_size = 2
        self.impl.o_proj = MagicMock()
        self.impl.o_proj.weight = torch.randn(4, 8)
        self.impl.o_proj.aclnn_input_scale = torch.randn(4)
        self.impl.o_proj.aclnn_input_scale_reciprocal = torch.randn(4)
        self.impl.o_proj.aclnn_input_offset = torch.randn(4)
        # Provide a W8A8-like fused_qkv_a_proj
        from vllm_ascend.quantization.methods import AscendW8A8LinearMethod

        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.quant_method.quant_method = MagicMock(spec=AscendW8A8LinearMethod)

        AscendSFAImpl.o_proj_full_pool = None
        self.impl.process_weights_after_loading(torch.float16)
        self.assertFalse(self.impl.enable_mlapo)
        AscendSFAImpl.o_proj_full_pool = None

    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    @patch("vllm_ascend.attention.sfa_v1.scipy")
    def test_process_weights_after_loading_with_sparse_c8(
        self, mock_scipy, mock_torch_npu, mock_maybe_trans_nz, _mock_dispose
    ):
        mock_maybe_trans_nz.side_effect = lambda x: x
        self.impl.kv_b_proj = MagicMock()
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        self.impl.kv_b_proj.quant_method = MagicMock(spec=UnquantizedLinearMethod)
        self.impl.local_num_heads = 4
        kv_b_proj_weight = torch.randn(
            self.impl.local_num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim),
            self.impl.kv_lora_rank,
        )
        self.impl.kv_b_proj.weight.data = kv_b_proj_weight
        mock_torch_npu.npu_format_cast.return_value = kv_b_proj_weight
        self.impl.enable_dsa_cp = False
        self.impl.enable_mlapo = False
        self.impl.use_sparse_c8_indexer = True

        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None
        mock_scipy.linalg.hadamard.return_value = torch.eye(128).numpy()

        # Patch torch.tensor on npu device construction to avoid actual NPU usage.
        with patch("torch.tensor", side_effect=lambda *a, **kw: torch.zeros(128, 128, dtype=torch.bfloat16)):
            self.impl.process_weights_after_loading(torch.float16)
        self.assertIsNotNone(AscendSFAImpl.q_hadamard)
        self.assertIsNotNone(AscendSFAImpl.k_hadamard)
        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None

    @patch("vllm_ascend.attention.sfa_v1.trans_rope_weight")
    @patch("vllm_ascend.attention.sfa_v1.transdata")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_for_fused_mlapo(self, mock_torch_npu, mock_transdata, mock_trans_rope_weight):
        mock_transdata.return_value = torch.randn(128, 128)
        mock_torch_npu.npu_format_cast.return_value = torch.randn(1, 128, 128)
        mock_trans_rope_weight.side_effect = lambda x, *a, **kw: x

        self.impl.enable_mlapo = True
        self.impl.fused_qkv_a_proj = MagicMock()
        q_lora_rank = 64
        kv_lora_rank_plus_rope = 32 + 32
        total_rank = q_lora_rank + kv_lora_rank_plus_rope
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, total_rank)
        self.impl.fused_qkv_a_proj.deq_scale = torch.randn(total_rank)
        self.impl.fused_qkv_a_proj.quant_bias = torch.randn(total_rank)
        self.impl.fused_qkv_a_proj.input_scale.data = torch.randn(1)
        self.impl.fused_qkv_a_proj.input_offset.data = torch.randn(1)
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.weight.data = torch.randn(128, 256 * 96)
        self.impl.q_proj.weight.device = torch.device("cpu")
        self.impl.q_proj.deq_scale.data = torch.randn(256 * 96)
        self.impl.q_proj.quant_bias.data = torch.randn(256 * 96)
        self.impl.q_proj.input_scale.data = torch.randn(1)
        self.impl.q_proj.input_offset.data = torch.randn(1)
        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.weight.data = torch.randn(128)
        self.impl.q_a_layernorm.bias.data = torch.randn(128)
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight.data = torch.randn(128)
        self.impl.kv_a_proj_with_mqa = None
        self.impl.q_lora_rank = q_lora_rank
        self.impl.kv_lora_rank = 32
        self.impl.qk_rope_head_dim = 32
        self.impl.qk_nope_head_dim = 64
        self.impl.num_heads = 256
        self.impl.vllm_config = MagicMock()
        self.impl.vllm_config.kv_transfer_config = None

        self.impl._process_weights_for_fused_mlapo(torch.float16)
        self.assertTrue(hasattr(self.impl, "wd_qkv"))
        self.assertTrue(hasattr(self.impl, "deq_scale_qkv"))
        self.assertTrue(hasattr(self.impl, "quant_bias_qkv"))
        self.assertTrue(hasattr(self.impl, "wu_q"))
        self.assertTrue(hasattr(self.impl, "ctkv_scale"))

    @patch("vllm_ascend.attention.sfa_v1.trans_rope_weight")
    @patch("vllm_ascend.attention.sfa_v1.transdata")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_process_weights_for_fused_mlapo_kv_consumer_drops_weights(
        self, mock_torch_npu, mock_transdata, mock_trans_rope_weight
    ):
        mock_transdata.return_value = torch.randn(128, 128)
        mock_torch_npu.npu_format_cast.return_value = torch.randn(1, 128, 128)
        mock_trans_rope_weight.side_effect = lambda x, *a, **kw: x

        self.impl.enable_mlapo = True
        self.impl.fused_qkv_a_proj = MagicMock()
        q_lora_rank = 64
        total_rank = q_lora_rank + 64
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, total_rank)
        self.impl.fused_qkv_a_proj.deq_scale = torch.randn(total_rank)
        self.impl.fused_qkv_a_proj.quant_bias = torch.randn(total_rank)
        self.impl.fused_qkv_a_proj.input_scale.data = torch.randn(1)
        self.impl.fused_qkv_a_proj.input_offset.data = torch.randn(1)
        self.impl.q_proj = MagicMock()
        self.impl.q_proj.weight.data = torch.randn(128, 256 * 96)
        self.impl.q_proj.weight.device = torch.device("cpu")
        self.impl.q_proj.deq_scale.data = torch.randn(256 * 96)
        self.impl.q_proj.quant_bias.data = torch.randn(256 * 96)
        self.impl.q_proj.input_scale.data = torch.randn(1)
        self.impl.q_proj.input_offset.data = torch.randn(1)
        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.weight.data = torch.randn(128)
        self.impl.q_a_layernorm.bias.data = torch.randn(128)
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight.data = torch.randn(128)
        self.impl.kv_a_proj_with_mqa = None
        self.impl.q_lora_rank = q_lora_rank
        self.impl.kv_lora_rank = 32
        self.impl.qk_rope_head_dim = 32
        self.impl.qk_nope_head_dim = 64
        self.impl.num_heads = 256
        self.impl.vllm_config = MagicMock()
        self.impl.vllm_config.kv_transfer_config = MagicMock()
        self.impl.vllm_config.kv_transfer_config.is_kv_consumer = True
        self.impl.vllm_config.scheduler_config.max_num_batched_tokens = 256

        with patch("torch.npu.empty_cache", create=True):
            self.impl._process_weights_for_fused_mlapo(torch.float16)
        self.assertIsNone(self.impl.fused_qkv_a_proj.weight)
        self.assertIsNone(self.impl.q_proj.weight)

    def test_sfa_preprocess_with_mlapo(self):
        num_input_tokens = 4
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
        )
        hidden_states = torch.randn(num_input_tokens, 128)
        cos = torch.randn(num_input_tokens, self.impl.qk_rope_head_dim)
        sin = torch.randn(num_input_tokens, self.impl.qk_rope_head_dim)
        slot_mapping = torch.tensor([0, 1, 2, 3])

        self.impl.W_UK_T = torch.randn(self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank)
        self.impl.q_lora_rank = 64
        self.impl.wd_qkv = torch.randn(1, 128, 128)
        self.impl.deq_scale_qkv = torch.randn(128)
        self.impl.gamma1 = torch.randn(128)
        self.impl.beta1 = torch.randn(128)
        self.impl.wu_q = torch.randn(1, 128, 128)
        self.impl.qb_deq_scl = torch.randn(128)
        self.impl.gamma2 = torch.randn(128)
        self.impl.quant_scale0 = torch.randn(1)
        self.impl.quant_offset0 = torch.randn(1)
        self.impl.quant_bias_qkv = torch.randn(128)
        self.impl.quant_scale1 = torch.randn(1)
        self.impl.quant_offset1 = torch.randn(1)
        self.impl.qb_qt_bias = torch.randn(128)
        self.impl.ctkv_scale = torch.randn(1)
        self.impl.q_nope_scale = torch.randn(1)

        with patch.object(torch.ops._C_ascend, "mla_preprocess", create=True) as mock_mla_pre:
            hs, ql_nope, q_pe, q_c = self.impl._sfa_preprocess_with_mlapo(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                cos=cos,
                sin=sin,
                slot_mapping=slot_mapping,
                num_input_tokens=num_input_tokens,
            )
        mock_mla_pre.assert_called_once()
        self.assertIs(hs, hidden_states)
        self.assertEqual(ql_nope.shape[0], num_input_tokens)
        self.assertEqual(q_pe.shape[0], num_input_tokens)
        self.assertEqual(q_c.shape, (num_input_tokens, self.impl.q_lora_rank))


class TestAscendSFAImplForward(TestBase):
    """Tests for AscendSFAImpl.forward (the main entrypoint)."""

    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_layer_shard", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    def setUp(
        self,
        mock_get_current_vllm_config,
        _mock_enable_dsa_cp,
        _mock_enable_dsa_cp_with_layer_shard,
        _mock_enable_dsa_cp_with_o_proj_tp,
        mock_tp,
    ):
        mock_tp.world_size = 1
        mock_tp.rank_in_group = MagicMock()
        mock_tp.device_group = MagicMock()
        vllm_config = MagicMock()
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 0
        vllm_config.speculative_config = speculative_config
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_config.model_type = "deepseek_v3"
        vllm_config.kv_transfer_config = None
        vllm_config.additional_config = {"refresh": True}
        parallel_config = MagicMock()
        parallel_config.prefill_context_parallel_size = 1
        parallel_config.decode_context_parallel_size = 1
        parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config = parallel_config
        mock_get_current_vllm_config.return_value = vllm_config
        init_ascend_config(vllm_config)

        self.impl = AscendSFAImpl(
            num_heads=4,
            head_size=128,
            scale=0.1,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **_make_impl_kwargs(),
        )

        AscendSFAImpl.o_proj_full_pool = None
        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None

    def test_forward_requires_output(self):
        with self.assertRaises(AssertionError):
            self.impl.forward(
                layer_name="layer_0",
                hidden_states=torch.randn(2, 8),
                kv_cache=(torch.zeros(2), torch.zeros(2), torch.zeros(2)),
                attn_metadata=MagicMock(),
                output=None,
            )

    def test_forward_profiling_run_returns_zeros(self):
        # When attn_metadata is None, forward fills output with zeros (profiling).
        self.impl.enable_dsa_cp_with_layer_shard = False
        output = torch.ones(4, 8)
        result = self.impl.forward(
            layer_name="layer_0",
            hidden_states=torch.randn(4, 8),
            kv_cache=(torch.zeros(2), torch.zeros(2), torch.zeros(2)),
            attn_metadata=None,
            output=output,
        )
        self.assertTrue(torch.all(result == 0))

    @patch("vllm_ascend.attention.sfa_v1._EXTRA_CTX")
    def test_forward_profiling_with_layer_shard(self, mock_extra_ctx):
        mock_extra_ctx.in_profile_run = False
        self.impl.enable_dsa_cp_with_layer_shard = True
        self.impl.layer_sharding_kwargs = []
        output = torch.ones(4, 8)
        result = self.impl.forward(
            layer_name="layer_0",
            hidden_states=torch.randn(4, 8),
            kv_cache=(torch.zeros(2), torch.zeros(2), torch.zeros(2)),
            attn_metadata=None,
            output=output,
        )
        self.assertTrue(torch.all(result == 0))

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_native_path(
        self,
        mock_torch_npu,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        # Configure the impl for the simplest native (non-mlapo, non-dsa-cp) path.
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = False
        self.impl.enable_dsa_cp_with_o_proj_tp = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False

        # Pre-set required projection-derived weights/buffers.
        self.impl.W_UK_T = torch.randn(self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank)
        self.impl.W_UV = torch.randn(self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim)

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )
        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.cum_query_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.seq_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.block_table = torch.tensor([[0]])

        # Mock fused_qkv_a_proj: returns (lora, ?) where lora has shape
        # [num_tokens, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
        qkv_lora = torch.randn(num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim)
        self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)

        # q_a_layernorm returns the same q_c
        self.impl.q_a_layernorm.side_effect = lambda x: x

        # wk_weights_proj returns kw of shape [num_tokens, head_dim*2]
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)

        # mock_torch_npu calls used in the function
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None)
        # npu_rotary_mul preserves input shape
        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_args, **_kwargs: torch.randn_like(x)
        mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
            num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
        )
        mock_torch_npu.npu_scatter_nd_update_.return_value = None

        # q_proj output for _q_proj_and_k_up_proj
        q_proj_output = torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim)
        self.impl.q_proj.return_value = (q_proj_output,)

        # wq_b for indexer_select_post_process
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        # weight prefetch method
        mock_prefetch_method = MagicMock()
        mock_get_weight_prefetch.return_value = mock_prefetch_method

        # o_proj output
        self.impl.o_proj.return_value = (torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),)
        self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

        output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

        with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
        ), patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ):
            result = self.impl.forward(
                layer_name="layer_0",
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, output)

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_kv_producer_path(
        self,
        mock_torch_npu,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover the kv-producer reshape_cache_event branches."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = False
        self.impl.enable_dsa_cp_with_o_proj_tp = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = True

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )
        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.cum_query_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.seq_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.block_table = torch.tensor([[0]])

        qkv_lora = torch.randn(
            num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        )
        self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
        self.impl.q_a_layernorm.side_effect = lambda x: x
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
        self.impl.q_proj.return_value = (torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),)
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None)
        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
        mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
            num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
        )

        # Mock npu Event for kv_producer
        mock_torch_npu.Event = MagicMock()

        mock_prefetch_method = MagicMock()
        mock_get_weight_prefetch.return_value = mock_prefetch_method

        self.impl.o_proj.return_value = (
            torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
        )
        self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

        output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

        with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch(
            "torch.npu.Event", create=True, return_value=MagicMock()
        ), patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
        ), patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ):
            self.impl.forward(
                layer_name="layer_0",
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        # Verify the kv-producer reshape_cache_event branch was taken
        self.assertIsNotNone(attn_metadata.reshape_cache_event)

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_mlapo_path(
        self,
        mock_torch_npu,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover the MLAPO short-circuit path in forward."""
        self.impl.enable_mlapo = True
        self.impl.enable_dsa_cp = False
        self.impl.enable_dsa_cp_with_o_proj_tp = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        # Set MLAPO-specific attributes used by _sfa_preprocess_with_mlapo
        self.impl.wd_qkv = torch.randn(1, 128, 128)
        self.impl.deq_scale_qkv = torch.randn(128)
        self.impl.gamma1 = torch.randn(128)
        self.impl.beta1 = torch.randn(128)
        self.impl.wu_q = torch.randn(1, 128, 128)
        self.impl.qb_deq_scl = torch.randn(128)
        self.impl.gamma2 = torch.randn(128)
        self.impl.quant_scale0 = torch.randn(1)
        self.impl.quant_offset0 = torch.randn(1)
        self.impl.quant_bias_qkv = torch.randn(128)
        self.impl.quant_scale1 = torch.randn(1)
        self.impl.quant_offset1 = torch.randn(1)
        self.impl.qb_qt_bias = torch.randn(128)
        self.impl.ctkv_scale = torch.randn(1)
        self.impl.q_nope_scale = torch.randn(1)

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, 128)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )
        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.cum_query_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.seq_lens = torch.tensor([1, 2, 3, 4])
        attn_metadata.block_table = torch.tensor([[0]])

        # wk_weights_proj/k_norm for indexer_select_pre_process (uses hidden_states)
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)

        mock_prefetch_method = MagicMock()
        mock_get_weight_prefetch.return_value = mock_prefetch_method

        self.impl.o_proj.return_value = (
            torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
        )
        self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

        output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

        with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
            torch.ops._C_ascend, "mla_preprocess", create=True
        ), patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
        ), patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ):
            result = self.impl.forward(
                layer_name="layer_0",
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, output)

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.DeviceOperator")
    @patch("vllm_ascend.attention.sfa_v1.all_gather_async")
    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_dsa_cp_path(
        self,
        mock_torch_npu,
        mock_get_tp_group,
        mock_all_gather_async,
        mock_device_operator,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover DSA-CP enabled branches in forward."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = False
        self.impl.enable_dsa_cp_with_o_proj_tp = False
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False
        self.impl.layer_sharding_kwargs = []

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        mock_get_tp_group.return_value = MagicMock()

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )

        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.block_table = torch.tensor([[0]])
        attn_metadata.dsa_cp_context = MagicMock()
        attn_metadata.dsa_cp_context.slot_mapping_cp = torch.tensor([0, 1, 2, 3])
        attn_metadata.dsa_cp_context.actual_seq_lengths_query = torch.tensor([1, 2, 3, 4])
        attn_metadata.dsa_cp_context.actual_seq_lengths_key = torch.tensor([1, 2, 3, 4])

        qkv_lora = torch.randn(
            num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        )
        self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
        self.impl.q_a_layernorm.side_effect = lambda x: x
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
        self.impl.q_proj.return_value = (torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),)
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        # exec_kv with DSA-CP returns tensors for k_pe and k_nope.
        fake_k_pe = torch.randn(num_tokens, 1, self.impl.qk_rope_head_dim)
        fake_k_nope = torch.randn(num_tokens, 1, self.impl.kv_lora_rank)
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None, fake_k_pe, fake_k_nope)
        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
        mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
            num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
        )

        # all_gather_async returns the input tensor and a handle
        all_gathered = torch.cat(
            [
                fake_k_pe.view(-1, self.impl.qk_rope_head_dim),
                fake_k_nope.view(-1, self.impl.kv_lora_rank),
                torch.randn(num_tokens, self.impl.head_dim),
            ],
            dim=1,
        )
        kv_handle = MagicMock()
        mock_all_gather_async.return_value = (all_gathered, kv_handle)

        mock_get_weight_prefetch.return_value = MagicMock()

        self.impl.o_proj.return_value = (
            torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
        )
        self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

        output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

        with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
        ), patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ):
            result = self.impl.forward(
                layer_name="layer_0",
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        self.assertIs(result, output)
        kv_handle.wait.assert_called_once()
        mock_device_operator.reshape_and_cache.assert_called_once()

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_dsa_cp_with_o_proj_tp_decode(
        self,
        mock_torch_npu,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover the DSA-CP-with-o_proj-tp path for decode (require_o_proj_forward=True)."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = False
        self.impl.enable_dsa_cp_with_o_proj_tp = True
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        # Patch _handle_o_proj_weight_switch_and_forward to a controllable mock
        with patch.object(
            self.impl,
            "_handle_o_proj_weight_switch_and_forward",
            return_value=(torch.randn(4, self.impl.local_num_heads * self.impl.v_head_dim), True),
        ) as mock_handler:
            num_tokens = 4
            hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
            kv_cache = (
                torch.randn(2, 4, 1, self.impl.kv_lora_rank),
                torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
                torch.randn(2, 4, 1, self.impl.head_dim),
            )
            attn_metadata = MagicMock()
            attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
            attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
            attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
            attn_metadata.num_input_tokens = num_tokens
            attn_metadata.num_actual_tokens = num_tokens
            attn_metadata.attn_state = AscendAttentionState.DecodeOnly  # decode -> all-to-all path
            attn_metadata.cum_query_lens = torch.tensor([1, 2, 3, 4])
            attn_metadata.seq_lens = torch.tensor([1, 2, 3, 4])
            attn_metadata.block_table = torch.tensor([[0]])

            qkv_lora = torch.randn(
                num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
            )
            self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
            self.impl.q_a_layernorm.side_effect = lambda x: x
            self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
            self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
            self.impl.q_proj.return_value = (
                torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),
            )
            self.impl.wq_b.return_value = (
                torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim),
                None,
            )

            mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None)
            mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
            mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
                num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
            )

            mock_get_weight_prefetch.return_value = MagicMock()

            self.impl.o_proj.return_value = (
                torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
            )
            self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

            output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

            with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
                torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
            ), patch.object(
                torch.ops._C_ascend,
                "npu_sparse_flash_attention",
                create=True,
                return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
            ):
                self.impl.forward(
                    layer_name="layer_0",
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )

            mock_handler.assert_called_once()

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_dsa_cp_with_o_proj_tp_returns_early(
        self,
        mock_torch_npu,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover the early-return when require_o_proj_forward is False."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = False
        self.impl.enable_dsa_cp_with_o_proj_tp = True
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        sentinel = torch.full((4, self.impl.local_num_heads * self.impl.v_head_dim), 7.0)
        with patch.object(
            self.impl,
            "_handle_o_proj_weight_switch_and_forward",
            return_value=(sentinel, False),
        ):
            num_tokens = 4
            hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
            kv_cache = (
                torch.randn(2, 4, 1, self.impl.kv_lora_rank),
                torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
                torch.randn(2, 4, 1, self.impl.head_dim),
            )
            attn_metadata = MagicMock()
            attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
            attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
            attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
            attn_metadata.num_input_tokens = num_tokens
            attn_metadata.num_actual_tokens = num_tokens
            # Use ChunkedPrefill state so that full_gather_o_proj_enabled is True.
            attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
            attn_metadata.cum_query_lens = torch.tensor([1, 2, 3, 4])
            attn_metadata.seq_lens = torch.tensor([1, 2, 3, 4])
            attn_metadata.block_table = torch.tensor([[0]])

            qkv_lora = torch.randn(
                num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
            )
            self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
            self.impl.q_a_layernorm.side_effect = lambda x: x
            self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
            self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
            self.impl.q_proj.return_value = (
                torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),
            )
            self.impl.wq_b.return_value = (
                torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim),
                None,
            )

            mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None)
            mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
            mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
                num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
            )

            mock_get_weight_prefetch.return_value = MagicMock()

            self.impl.o_proj.return_value = (
                torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
            )
            self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

            output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

            with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
                torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
            ), patch.object(
                torch.ops._C_ascend,
                "npu_sparse_flash_attention",
                create=True,
                return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
            ):
                result = self.impl.forward(
                    layer_name="layer_0",
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )

            # Early return: result should be the sentinel tensor returned by the handler.
            self.assertTrue(torch.allclose(result, sentinel))

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.DeviceOperator")
    @patch("vllm_ascend.attention.sfa_v1.all_gather_async")
    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.is_hidden_layer", return_value=True)
    @patch("vllm_ascend.attention.sfa_v1.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_dsa_cp_sparse_c8_with_layer_shard(
        self,
        mock_torch_npu,
        mock_reach,
        _mock_is_hidden_layer,
        mock_get_tp_group,
        mock_all_gather_async,
        mock_device_operator,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover the DSA-CP+sparse_c8 split-all-gather path AND layer_shard reach branch."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = True
        self.impl.enable_dsa_cp_with_o_proj_tp = False
        self.impl.use_sparse_c8_indexer = True
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = True
        self.impl.layer_sharding_kwargs = ["layer_shard_0"]
        self.impl.c8_k_cache_dtype = torch.int8
        self.impl.c8_k_scale_cache_dtype = torch.float16
        AscendSFAImpl.q_hadamard = torch.randn(self.impl.head_dim, self.impl.head_dim)
        AscendSFAImpl.k_hadamard = torch.randn(self.impl.head_dim, self.impl.head_dim)

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        mock_get_tp_group.return_value = MagicMock()

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.zeros(2, 4, 1, self.impl.head_dim, dtype=torch.int8),
            torch.randn(2, 4, 1, 1, dtype=torch.float16),
        )

        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata.block_table = torch.tensor([[0]])
        attn_metadata.dsa_cp_context = MagicMock()
        attn_metadata.dsa_cp_context.slot_mapping_cp = torch.tensor([0, 1, 2, 3])
        attn_metadata.dsa_cp_context.actual_seq_lengths_query = torch.tensor([1, 2, 3, 4])
        attn_metadata.dsa_cp_context.actual_seq_lengths_key = torch.tensor([1, 2, 3, 4])

        qkv_lora = torch.randn(
            num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        )
        self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
        self.impl.q_a_layernorm.side_effect = lambda x: x
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
        self.impl.q_proj.return_value = (torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),)
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        # exec_kv with DSA-CP
        fake_k_pe = torch.randn(num_tokens, 1, self.impl.qk_rope_head_dim)
        fake_k_nope = torch.randn(num_tokens, 1, self.impl.kv_lora_rank)
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None, fake_k_pe, fake_k_nope)
        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
        mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
            num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
        )
        mock_torch_npu.npu_dynamic_quant.return_value = (
            torch.zeros(num_tokens * self.impl.n_head, self.impl.head_dim, dtype=torch.int8),
            torch.randn(num_tokens * self.impl.n_head, dtype=torch.float32),
        )
        # simulate npu Event creation
        mock_torch_npu.Event = MagicMock()

        # The DSA-CP+sparse_c8 path calls all_gather_async 3 times (kv, k_li, k_li_scale)
        # and then once more for the o_proj_tp path? No - o_proj_tp is False here.
        kv_handle = MagicMock()
        # We need the all_gather output to satisfy split into [qk_rope_head_dim, kv_lora_rank]
        gathered_kv = torch.cat(
            [fake_k_pe.view(-1, self.impl.qk_rope_head_dim), fake_k_nope.view(-1, self.impl.kv_lora_rank)],
            dim=1,
        )
        gathered_k_li = torch.zeros(num_tokens, self.impl.head_dim, dtype=torch.int8)
        gathered_k_li_scale = torch.randn(num_tokens, 1, dtype=torch.float16)

        mock_all_gather_async.side_effect = [
            (gathered_kv, MagicMock()),
            (gathered_k_li, MagicMock()),
            (gathered_k_li_scale, kv_handle),
        ]

        mock_get_weight_prefetch.return_value = MagicMock()

        self.impl.o_proj.return_value = (
            torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
        )
        self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

        output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

        with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch(
            "torch.npu.Event", create=True, return_value=MagicMock()
        ), patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer_quant", create=True, return_value=torch.tensor([[0]])
        ), patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
        ):
            self.impl.forward(
                layer_name="layer_0",
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        # Verify the layer-shard branch was taken in the DSA-CP forward.
        mock_reach.assert_called()
        AscendSFAImpl.q_hadamard = None
        AscendSFAImpl.k_hadamard = None

    @patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector")
    @patch("vllm_ascend.attention.sfa_v1.get_weight_prefetch_method")
    @patch("vllm_ascend.attention.sfa_v1.DeviceOperator")
    @patch("vllm_ascend.attention.sfa_v1.all_gather_async")
    @patch("vllm_ascend.attention.sfa_v1.get_tp_group")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu")
    def test_forward_dsa_cp_with_full_gather_o_proj_enabled(
        self,
        mock_torch_npu,
        mock_get_tp_group,
        mock_all_gather_async,
        mock_device_operator,
        mock_get_weight_prefetch,
        _mock_wait,
        _mock_save,
    ):
        """Cover full_gather_o_proj_enabled path inside DSA-CP forward (line 1192)."""
        self.impl.enable_mlapo = False
        self.impl.enable_dsa_cp = True
        self.impl.enable_dsa_cp_with_layer_shard = False
        self.impl.enable_dsa_cp_with_o_proj_tp = True
        self.impl.use_sparse_c8_indexer = False
        self.impl.use_torch_npu_lightning_indexer = False
        self.impl.is_kv_producer = False
        self.impl.layer_sharding_kwargs = []
        # Required for full_gather path
        self.impl.o_proj_tp_weight = torch.randn(8, 16)
        AscendSFAImpl.o_proj_full_pool = torch.empty((16, 16))

        self.impl.W_UK_T = torch.randn(
            self.impl.local_num_heads, self.impl.qk_nope_head_dim, self.impl.kv_lora_rank
        )
        self.impl.W_UV = torch.randn(
            self.impl.local_num_heads, self.impl.kv_lora_rank, self.impl.v_head_dim
        )

        mock_get_tp_group.return_value = MagicMock()

        num_tokens = 4
        hidden_states = torch.randn(num_tokens, self.impl.local_num_heads * self.impl.qk_head_dim)
        kv_cache = (
            torch.randn(2, 4, 1, self.impl.kv_lora_rank),
            torch.randn(2, 4, 1, self.impl.qk_rope_head_dim),
            torch.randn(2, 4, 1, self.impl.head_dim),
        )

        attn_metadata = MagicMock()
        attn_metadata.cos = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.sin = torch.randn(num_tokens, self.impl.qk_rope_head_dim)
        attn_metadata.slot_mapping = torch.tensor([0, 1, 2, 3])
        attn_metadata.num_input_tokens = num_tokens
        attn_metadata.num_actual_tokens = num_tokens
        # Use ChunkedPrefill so full_gather_o_proj_enabled is True
        attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        attn_metadata.block_table = torch.tensor([[0]])
        attn_metadata.dsa_cp_context = MagicMock()
        attn_metadata.dsa_cp_context.slot_mapping_cp = torch.tensor([0, 1, 2, 3])
        attn_metadata.dsa_cp_context.actual_seq_lengths_query = torch.tensor([1, 2, 3, 4])
        attn_metadata.dsa_cp_context.actual_seq_lengths_key = torch.tensor([1, 2, 3, 4])

        qkv_lora = torch.randn(
            num_tokens, self.impl.q_lora_rank + self.impl.kv_lora_rank + self.impl.qk_rope_head_dim
        )
        self.impl.fused_qkv_a_proj.return_value = (qkv_lora,)
        self.impl.q_a_layernorm.side_effect = lambda x: x
        self.impl.wk_weights_proj.return_value = (torch.randn(num_tokens, self.impl.head_dim * 2), None)
        self.impl.k_norm.return_value = torch.randn(num_tokens, self.impl.head_dim)
        self.impl.q_proj.return_value = (torch.randn(num_tokens, self.impl.local_num_heads, self.impl.qk_head_dim),)
        self.impl.wq_b.return_value = (torch.randn(num_tokens, self.impl.n_head * self.impl.head_dim), None)

        fake_k_pe = torch.randn(num_tokens, 1, self.impl.qk_rope_head_dim)
        fake_k_nope = torch.randn(num_tokens, 1, self.impl.kv_lora_rank)
        mock_torch_npu.npu_kv_rmsnorm_rope_cache.return_value = (None, None, fake_k_pe, fake_k_nope)
        mock_torch_npu.npu_rotary_mul.side_effect = lambda x, *_a, **_kw: torch.randn_like(x)
        mock_torch_npu.npu_interleave_rope.return_value = torch.randn(
            num_tokens, self.impl.local_num_heads, 1, self.impl.qk_rope_head_dim
        )

        gathered_kv = torch.cat(
            [
                fake_k_pe.view(-1, self.impl.qk_rope_head_dim),
                fake_k_nope.view(-1, self.impl.kv_lora_rank),
                torch.randn(num_tokens, self.impl.head_dim),
            ],
            dim=1,
        )
        # DSA-CP non-sparse path: 1 all_gather call for kv, then a 2nd for the
        # o_proj weight gather (full_gather_o_proj_enabled branch).
        mock_all_gather_async.side_effect = [
            (gathered_kv, MagicMock()),
            (None, MagicMock()),  # the o_proj_tp_weight gather
        ]

        mock_get_weight_prefetch.return_value = MagicMock()

        # Patch _handle_o_proj_weight_switch_and_forward so we don't need its full setup.
        with patch.object(
            self.impl,
            "_handle_o_proj_weight_switch_and_forward",
            return_value=(torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim), True),
        ):
            self.impl.o_proj.return_value = (
                torch.randn(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim),
            )
            self.impl.o_proj.weight = torch.randn(self.impl.local_num_heads * self.impl.v_head_dim, 128)

            output = torch.zeros(num_tokens, self.impl.local_num_heads * self.impl.v_head_dim)

            with patch("vllm_ascend.attention.sfa_v1.HAS_TRITON", False), patch.object(
                torch.ops._C_ascend, "npu_lightning_indexer", create=True, return_value=torch.tensor([[0]])
            ), patch.object(
                torch.ops._C_ascend,
                "npu_sparse_flash_attention",
                create=True,
                return_value=torch.randn(num_tokens, self.impl.local_num_heads, self.impl.kv_lora_rank),
            ):
                self.impl.forward(
                    layer_name="layer_0",
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )

        # Verify all_gather_async was called twice (kv + o_proj weight)
        self.assertEqual(mock_all_gather_async.call_count, 2)
        AscendSFAImpl.o_proj_full_pool = None

    @patch("vllm_ascend.attention.sfa_v1.is_hidden_layer", return_value=True)
    @patch("vllm_ascend.attention.sfa_v1.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.attention.sfa_v1._EXTRA_CTX")
    def test_forward_profiling_with_layer_shard_with_hidden_layers(
        self, mock_extra_ctx, mock_reach, _mock_is_hidden_layer
    ):
        """Cover the profiling-path branch that calls reach_layer_for_shard_weight_series."""
        mock_extra_ctx.in_profile_run = False
        self.impl.enable_dsa_cp_with_layer_shard = True
        self.impl.layer_sharding_kwargs = ["layer_shard_0"]

        output = torch.ones(4, 8)
        self.impl.forward(
            layer_name="layer_0",
            hidden_states=torch.randn(4, 8),
            kv_cache=(torch.zeros(2), torch.zeros(2), torch.zeros(2)),
            attn_metadata=None,
            output=output,
        )
        mock_reach.assert_called_once_with("layer_shard_0")
