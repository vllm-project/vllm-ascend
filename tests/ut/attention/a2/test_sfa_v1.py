import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from tests.ut.attention.utils import patch_distributed_groups
from tests.ut.base import TestBase
from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.attention.attention_v1 import AscendAttentionState

if "torch_npu._inductor" not in sys.modules:
    sys.modules["torch_npu._inductor"] = MagicMock()

from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl, AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.utils import enable_dsa_cp


class TestAscendSFABackend(TestBase):
    def setUp(self):
        self.mock_config = MagicMock()
        mock_parallel_config = MagicMock()
        mock_parallel_config.prefill_context_parallel_size = 1
        mock_parallel_config.decode_context_parallel_size = 1
        self.mock_config.parallel_config = mock_parallel_config

        self.utils_patcher = patch("vllm_ascend.attention.utils.get_current_vllm_config", return_value=self.mock_config)
        self.utils_patcher.start()

        from vllm_ascend.attention.utils import enable_cp

        enable_cp.cache_clear()

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


class TestAscendSFAPrologV3(TestBase):
    @staticmethod
    def _make_prolog_impl(has_indexer: bool) -> AscendSFAImpl:
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.has_indexer = has_indexer
        impl.local_num_heads = 2
        impl.num_kv_heads = 1
        impl.kv_lora_rank = 128
        impl.qk_rope_head_dim = 16
        impl.q_lora_rank = 8
        impl.q_a_layernorm = SimpleNamespace(
            weight=SimpleNamespace(data=torch.ones(8)),
            variance_epsilon=1e-5,
        )
        impl.kv_a_layernorm = SimpleNamespace(
            weight=SimpleNamespace(data=torch.ones(128)),
            variance_epsilon=1e-5,
        )
        impl.wq_b = MagicMock() if has_indexer else None
        impl.weight_dq = torch.empty(1)
        impl.weight_uq_qr = torch.empty(1)
        impl.W_UK_T = torch.empty(1)
        impl.weight_dkv_kr = torch.empty(1)
        impl.dequant_scale_w_dq = torch.empty(1)
        impl.dequant_scale_w_uq_qr = torch.empty(1)
        impl.dequant_scale_w_dkv_kr = torch.empty(1)
        return impl

    def test_prolog_v3_uses_unquantized_kv_cache(self):
        impl = self._make_prolog_impl(has_indexer=True)

        hidden_states = torch.randn(2, 8)
        k_cache = torch.empty(4, 16, 1, 128, dtype=torch.bfloat16)
        kr_cache = torch.empty(4, 16, 1, 16, dtype=torch.bfloat16)
        query_norm = torch.randint(-128, 127, (2, 8), dtype=torch.int8)
        query_norm_scale = torch.ones(2, 1, dtype=torch.float32)

        with (
            patch.object(
                impl,
                "_format_prolog_v3_inputs",
                return_value=(
                    torch.empty(2, 8, dtype=torch.int8),
                    torch.ones(2, 1),
                    torch.randn(2, 16),
                    torch.randn(2, 16),
                ),
            ),
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_mla_prolog_v3",
                create=True,
                return_value=(
                    torch.randn(2, 2, 128),
                    torch.randn(2, 2, 16),
                    None,
                    query_norm,
                    query_norm_scale,
                ),
            ) as mock_prolog,
            patch.object(impl, "_is_w8a8_dynamic_linear", return_value=True),
        ):
            result = impl._sfa_preprocess_with_prolog_v3(
                hidden_states=hidden_states,
                kv_cache=(k_cache, kr_cache),
                cos=torch.randn(2, 1, 1, 16),
                sin=torch.randn(2, 1, 1, 16),
                slot_mapping=torch.arange(2),
                cache_mode="PA_BSND",
            )

        call_kwargs = mock_prolog.call_args.kwargs
        self.assertIs(call_kwargs["kv_cache"], k_cache)
        self.assertIs(call_kwargs["kr_cache"], kr_cache)
        self.assertEqual(call_kwargs["kv_cache_quant_mode"], 0)
        self.assertEqual(call_kwargs["weight_quant_mode"], 2)
        self.assertTrue(call_kwargs["query_norm_flag"])
        self.assertEqual(call_kwargs["cache_mode"], "PA_BSND")
        self.assertEqual(call_kwargs["cache_index"].tolist(), [0, 1])
        self.assertIsInstance(result[3], tuple)
        self.assertTrue(torch.equal(result[3][0], query_norm))
        self.assertEqual(result[3][1].shape, (2,))

    def test_prolog_v3_skips_query_norm_without_indexer(self):
        impl = self._make_prolog_impl(has_indexer=False)
        hidden_states = torch.randn(2, 8)
        k_cache = torch.empty(4, 16, 1, 128, dtype=torch.bfloat16)
        kr_cache = torch.empty(4, 16, 1, 16, dtype=torch.bfloat16)

        with (
            patch.object(
                impl,
                "_format_prolog_v3_inputs",
                return_value=(
                    torch.empty(2, 8, dtype=torch.int8),
                    torch.ones(2, 1),
                    torch.randn(2, 16),
                    torch.randn(2, 16),
                ),
            ),
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_mla_prolog_v3",
                create=True,
                return_value=(
                    torch.randn(2, 2, 128),
                    torch.randn(2, 2, 16),
                    None,
                    None,
                    None,
                ),
            ) as mock_prolog,
        ):
            result = impl._sfa_preprocess_with_prolog_v3(
                hidden_states=hidden_states,
                kv_cache=(k_cache, kr_cache),
                cos=torch.randn(2, 1, 1, 16),
                sin=torch.randn(2, 1, 1, 16),
                slot_mapping=torch.arange(2),
                cache_mode="PA_BSND",
            )

        self.assertFalse(mock_prolog.call_args.kwargs["query_norm_flag"])
        self.assertIsNone(result[3])


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
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
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
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
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
        layer_names = ["layer1", "layer2"]
        vllm_config = MagicMock()
        vllm_config.cache_config.block_size = 16
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
