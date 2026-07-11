import sys
from unittest.mock import MagicMock, patch

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()
        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1
        self.mock_config.parallel_config = mock_parallel_config
        self.mock_config.model_config = MagicMock(spec=[])
        self.config_context = set_current_vllm_config(self.mock_config)
        self.config_context.__enter__()

        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()

        from vllm_ascend.attention.utils import enable_cp

        enable_cp.cache_clear()

    def tearDown(self):
        self.utils_patcher.stop()
        self.config_context.__exit__(None, None, None)

    def test_get_name(self):
        self.assertEqual(AscendSFABackend.get_name(), "ASCEND_SFA")

    def test_get_builder_cls(self):
        self.assertEqual(AscendSFABackend.get_builder_cls(), AscendSFAMetadataBuilder)

    def test_get_kv_cache_shape(self):
        result = AscendSFABackend.get_kv_cache_shape(2, 4, 8, 128)
        self.assertEqual(result, (2, 4, 8, 128))

    def test_get_impl_cls(self):
        result = AscendSFABackend.get_impl_cls()
        self.assertEqual(result, AscendSFAImpl)

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


class TestAscendSFADeviceOperator(TestBase):
    def _make_common_inputs(self):
        ql_nope = torch.randn(3, 4, 8)
        q_pe = torch.randn(3, 4, 2)
        topk_indices = torch.zeros(3, 1, dtype=torch.int32)
        attn_metadata = MagicMock()
        attn_metadata.block_table = torch.zeros(1, 4, dtype=torch.int32)
        actual_seq_lengths_query = torch.tensor([3], dtype=torch.int32)
        actual_seq_lengths_key = torch.tensor([3], dtype=torch.int32)
        impl = MagicMock()
        impl.scale = 0.125
        impl.qk_rope_head_dim = 2
        impl.sfa_qsfa_tile_size = 128
        return (
            impl,
            ql_nope,
            q_pe,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

    def test_execute_sparse_flash_attention_returns_lse(self):
        (
            impl,
            ql_nope,
            q_pe,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        ) = self._make_common_inputs()
        kv_cache = (
            torch.randn(4, 1, 1, 8),
            torch.randn(4, 1, 1, 2),
        )
        attn_output = torch.randn(3, 4, 8)
        softmax_max = torch.zeros(1, 3, 4)
        softmax_sum = torch.full((1, 3, 4), 2.0)

        with patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            create=True,
            return_value=(attn_output, softmax_max, softmax_sum),
        ) as mock_sfa:
            output, softmax_lse = DeviceOperator.execute_sparse_flash_attention_process(
                impl,
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                return_lse=True,
            )

        self.assertIs(output, attn_output)
        self.assertEqual(softmax_lse.shape, (3, 4, 1))
        expected_lse = torch.full((3, 4, 1), torch.log(torch.tensor(2.0)).item())
        self.assertTrue(torch.allclose(softmax_lse, expected_lse))
        self.assertTrue(mock_sfa.call_args.kwargs["return_softmax_lse"])

    def test_execute_sparse_flash_attention_c8_returns_lse(self):
        (
            impl,
            ql_nope,
            q_pe,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        ) = self._make_common_inputs()
        packed_kv_cache = (torch.empty(4, 1, 1, 12, dtype=torch.int8),)
        attn_output = torch.randn(3, 4, 8)
        softmax_max = torch.ones(1, 3, 4)
        softmax_sum = torch.full((1, 3, 4), 3.0)

        with patch(
            "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
            create=True,
            return_value=(attn_output, softmax_max, softmax_sum),
        ) as mock_qsfa:
            output, softmax_lse = DeviceOperator.execute_sparse_flash_attention_process(
                impl,
                ql_nope,
                q_pe,
                packed_kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                sparse_mode=0,
                return_lse=True,
            )

        self.assertIs(output, attn_output)
        expected_lse = torch.full((3, 4, 1), 1.0 + torch.log(torch.tensor(3.0)).item())
        self.assertTrue(torch.allclose(softmax_lse, expected_lse))
        call_kwargs = mock_qsfa.call_args.kwargs
        self.assertIs(call_kwargs["key"], packed_kv_cache[0])
        self.assertIs(call_kwargs["value"], packed_kv_cache[0])
        self.assertEqual(call_kwargs["query"].shape, (3, 4, 10))
        self.assertEqual(call_kwargs["sparse_mode"], 0)
        self.assertTrue(call_kwargs["return_softmax_lse"])


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

        self.mock_cfg.additional_config = {"refresh": True}
        init_ascend_config(self.mock_cfg)

        self.patcher = patch("vllm.config.get_current_vllm_config", return_value=self.mock_cfg)
        self.patcher.start()

        mock_ascend_config = MagicMock()
        mock_ascend_config.c8_enable_reshape_optim = False
        mock_ascend_config.enable_mlapo = True
        mock_ascend_config.enable_shared_expert_dp = False
        mock_ascend_config.layer_sharding = None
        self.ascend_config_patcher = patch(
            "vllm_ascend.attention.sfa_v1.get_ascend_config",
            return_value=mock_ascend_config,
        )
        self.ascend_config_patcher.start()

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
        self.ascend_config_patcher.stop()
        self.parent_init_patcher.stop()

    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_default(self):
        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = 128
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        assert builder.device == device
        assert builder.vllm_config == vllm_config

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
        kv_cache_spec.block_size = 128
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
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
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("vllm.distributed.parallel_state.get_tp_group")
    @patch_distributed_groups(dcp_size=2, pcp_size=2, needs_mocks=False)
    def test_ascend_sfa_metadata_builder_build_for_graph_capture(
        self, mock_get_tp_group, mock_enable_dsa_cp, mock_get_cos_and_sin_mla, mock_get_current_vllm_config
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg

        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = 128
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
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

    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.get_cos_and_sin_mla")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp", return_value=False)
    @patch("torch.ops._C_ascend.store_kv_block_pre", create=True)
    def test_ascend_sfa_metadata_builder_build_with_c8_reshape_optim(
        self,
        mock_store_kv_block_pre,
        mock_enable_dsa_cp,
        mock_get_cos_and_sin_mla,
        mock_get_current_vllm_config,
    ):
        cfg = MagicMock()
        cfg.model_config = MagicMock()
        cfg.model_config.hf_text_config = MagicMock()

        mock_get_current_vllm_config.return_value = cfg
        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = 128
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.scheduler_config.max_num_seqs = 16
        vllm_config.model_config.max_model_len = 1024
        vllm_config.model_config.get_head_size.return_value = 64
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_text_config.qk_rope_head_dim = 64
        speculative_config = MagicMock()
        speculative_config.num_speculative_tokens = 4
        vllm_config.speculative_config = speculative_config
        device = torch.device("cpu")

        builder = AscendSFAMetadataBuilder(
            kv_cache_spec=kv_cache_spec, layer_names=layer_names, vllm_config=vllm_config, device=device
        )

        slot_mapping_cpu = torch.randint(0, 10000, (100,))

        common_attn_metadata = MagicMock()
        common_attn_metadata.num_reqs = 10
        common_attn_metadata.num_actual_tokens = 100
        common_attn_metadata.query_start_loc = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.query_start_loc_cpu = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        common_attn_metadata.slot_mapping = torch.randn(100, 4, 1024)
        common_attn_metadata.slot_mapping_cpu = slot_mapping_cpu
        common_attn_metadata.seq_lens_cpu = torch.tensor([2] * 10)
        common_attn_metadata.positions = torch.randn(100)
        common_attn_metadata.attn_mask = None
        common_attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill
        common_attn_metadata.block_table_tensor = torch.randn(100, 4)
        common_attn_metadata.cos = None
        common_attn_metadata.sin = None
        common_attn_metadata.num_input_tokens = 100

        mock_get_cos_and_sin_mla.return_value = (torch.randn(100), torch.randn(100))

        mock_group_len = torch.tensor([1, 2, 3])
        mock_group_key_idx = torch.tensor([0, 1, 2])
        mock_group_key_cache_idx = torch.tensor([4, 5, 6])
        mock_store_kv_block_pre.return_value = (mock_group_len, mock_group_key_idx, mock_group_key_cache_idx)

        with patch("vllm_ascend.attention.sfa_v1.get_ascend_config") as mock_get_ascend_config:
            mock_ascend_config = MagicMock()
            mock_ascend_config.c8_enable_reshape_optim = True
            mock_get_ascend_config.return_value = mock_ascend_config

            metadata = builder.build(
                common_prefix_len=10,
                common_attn_metadata=common_attn_metadata,
            )

        assert isinstance(metadata, AscendSFAMetadata)
        assert metadata.num_actual_tokens == common_attn_metadata.num_actual_tokens
        assert metadata.slot_mapping.shape == (100, 4, 1024)

        mock_store_kv_block_pre.assert_called_once()
        actual_args, _ = mock_store_kv_block_pre.call_args
        assert torch.equal(actual_args[0], common_attn_metadata.slot_mapping)
        assert actual_args[1] == slot_mapping_cpu.tolist()
        assert actual_args[2] == 128

        assert metadata.block_size == 128
        assert metadata.group_len is mock_group_len
        assert metadata.group_key_idx is mock_group_key_idx
        assert metadata.group_key_cache_idx is mock_group_key_cache_idx
