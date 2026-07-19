import sys
from types import SimpleNamespace
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

from vllm_ascend.attention.sfa_v1 import (
    AscendSFABackend,
    AscendSFAImpl,
    AscendSFAMetadata,
    AscendSFAMetadataBuilder,
    custom_kv_rmsnorm_rope,
)
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
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


class TestAscendSFACacheComposition(TestBase):
    def test_compose_independent_sfa_and_li_c8_layouts(self):
        for enable_sfa_c8, enable_li_c8 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                enable_sfa_c8=enable_sfa_c8,
                enable_li_c8=enable_li_c8,
            ):
                impl = AscendSFAImpl.__new__(AscendSFAImpl)
                impl.layer_name = "model.layers.0.self_attn.attn"
                impl.has_indexer = True
                impl.enable_sparse_sfa_c8 = enable_sfa_c8
                impl.enable_sparse_li_c8 = enable_li_c8

                main_cache = tuple(torch.empty(1) for _ in range(1 if enable_sfa_c8 else 2))
                indexer_cache = tuple(torch.empty(1) for _ in range(2 if enable_li_c8 else 1))
                impl.indexer = SimpleNamespace(
                    k_cache=SimpleNamespace(kv_cache=indexer_cache)
                )

                composed = impl._compose_sfa_kv_cache(main_cache)

                expected = (*main_cache, *indexer_cache)
                self.assertIsNotNone(composed)
                assert composed is not None
                self.assertEqual(len(composed), len(expected))
                for actual_tensor, expected_tensor in zip(composed, expected):
                    self.assertIs(actual_tensor, expected_tensor)


class TestAscendSFAKVQuantSparseAttention(TestBase):
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_dynamic_block_quant")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_interleave_rope")
    @patch("vllm_ascend.attention.sfa_v1.torch_npu.npu_rms_norm")
    def test_pack_prefill_kv_cache(self, mock_rms_norm, mock_rope, mock_block_quant):
        k_nope = torch.randn(2, 1, 1, 256, dtype=torch.bfloat16)
        k_pe = torch.randn(2, 1, 1, 16, dtype=torch.bfloat16)
        quantized = torch.randint(-128, 127, (2, 1, 256), dtype=torch.int8)
        scales = torch.arange(1, 5, dtype=torch.float32).view(2, 1, 2)
        mock_rms_norm.return_value = k_nope, None
        mock_rope.return_value = k_pe
        mock_block_quant.return_value = quantized, scales

        actual_k_pe, actual_k_nope, actual_scales = custom_kv_rmsnorm_rope(
            torch.randn(2, 1, 1, 272, dtype=torch.bfloat16),
            torch.ones(256, dtype=torch.bfloat16),
            torch.randn(2, 1, 1, 16),
            torch.randn(2, 1, 1, 16),
            256,
            16,
            dst_type=1,
            tile_size=128,
        )
        packed_kv = torch.cat([actual_k_nope, actual_k_pe, actual_scales], dim=-1)

        self.assertEqual(mock_block_quant.call_args.kwargs["dst_type"], 1)
        self.assertEqual(mock_block_quant.call_args.kwargs["row_block_size"], 1)
        self.assertEqual(mock_block_quant.call_args.kwargs["col_block_size"], 128)
        self.assertEqual(packed_kv.shape, (2, 1, 1, 296))
        self.assertTrue(torch.equal(packed_kv[..., :256], quantized.view_as(k_nope)))
        self.assertTrue(torch.equal(packed_kv[..., 256:288], k_pe.contiguous().view(torch.int8)))
        self.assertTrue(torch.equal(packed_kv[..., 288:], scales.view(2, 1, 1, 2).view(torch.int8)))

    def test_execute_kv_quant_sparse_flash_attention(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_sparse_sfa_c8 = True
        impl.scale = 0.125
        impl.sfa_qsfa_tile_size = 128
        impl.qk_rope_head_dim = 16
        ql_nope = torch.randn(3, 2, 32)
        q_pe = torch.randn(3, 2, 16)
        kv_cache = (torch.empty(4, 16, 1, 80, dtype=torch.int8),)
        topk_indices = torch.zeros(3, 1, dtype=torch.int32)
        attn_metadata = SimpleNamespace(block_table=torch.zeros(1, 4, dtype=torch.int32))
        actual_seq_lengths = torch.tensor([3], dtype=torch.int32)
        expected = torch.randn(3, 2, 32)

        with patch(
            "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
            create=True,
            return_value=expected,
        ) as mock_qsfa:
            result = impl._execute_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths,
                actual_seq_lengths,
            )

        self.assertIs(result, expected)
        call_kwargs = mock_qsfa.call_args.kwargs
        self.assertIs(call_kwargs["key"], kv_cache[0])
        self.assertEqual(call_kwargs["query"].shape, (3, 2, 48))
        self.assertEqual(call_kwargs["key_quant_mode"], 2)
        self.assertEqual(call_kwargs["tile_size"], 128)

    def test_prolog_v3_enables_packed_int8_kv_cache(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.enable_sparse_sfa_c8 = True
        impl.has_indexer = True
        impl.sfa_qsfa_tile_size = 128
        impl.sfa_qsfa_k_nope_clip_alpha = torch.ones(1)
        impl.sfa_qsfa_kr_cache_dummy = torch.empty(0, dtype=torch.bfloat16)
        impl.local_num_heads = 2
        impl.kv_lora_rank = 128
        impl.qk_rope_head_dim = 16
        impl.q_lora_rank = 8
        impl.q_a_layernorm = SimpleNamespace(weight=SimpleNamespace(data=torch.ones(8)), variance_epsilon=1e-5)
        impl.kv_a_layernorm = SimpleNamespace(weight=SimpleNamespace(data=torch.ones(128)), variance_epsilon=1e-5)
        impl.weight_dq = torch.empty(1)
        impl.weight_uq_qr = torch.empty(1)
        impl.W_UK_T = torch.empty(1)
        impl.weight_dkv_kr = torch.empty(1)
        impl.dequant_scale_w_dq = torch.empty(1)
        impl.dequant_scale_w_uq_qr = torch.empty(1)
        impl.dequant_scale_w_dkv_kr = torch.empty(1)
        k_cache = torch.empty(4, 16, 1, get_sfa_qsfa_packed_head_dim(128, 16), dtype=torch.int8)
        dsa_k_cache = torch.empty(4, 16, 1, 128, dtype=torch.bfloat16)

        with (
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_dynamic_quant",
                return_value=(torch.empty(2, 8, dtype=torch.int8), torch.ones(2, 1)),
            ),
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_mla_prolog_v3",
                create=True,
                return_value=(torch.randn(2, 2, 128), torch.randn(2, 2, 16), None, torch.randn(2, 8), None),
            ) as mock_prolog,
        ):
            impl._sfa_preprocess_with_prolog_v3(
                hidden_states=torch.randn(2, 8),
                kv_cache=(k_cache, dsa_k_cache),
                cos=torch.randn(2, 1, 1, 16),
                sin=torch.randn(2, 1, 1, 16),
                slot_mapping=torch.arange(2),
                cache_mode="PA_BSND",
            )

        call_kwargs = mock_prolog.call_args.kwargs
        self.assertIs(call_kwargs["kv_cache"], k_cache)
        self.assertIs(call_kwargs["kr_cache"], impl.sfa_qsfa_kr_cache_dummy)
        self.assertEqual(call_kwargs["kv_cache_quant_mode"], 3)
        self.assertEqual(call_kwargs["ckvkr_repo_mode"], 1)
        self.assertEqual(call_kwargs["quant_scale_repo_mode"], 1)


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
    @patch("torch.ops._C_ascend.store_kv_block_metadata", create=True)
    def test_ascend_sfa_metadata_builder_build_with_c8_reshape_optim(
        self,
        store_kv_block_metadata,
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

        store_kv_block_metadata.assert_called_once()
        actual_args, _ = store_kv_block_metadata.call_args
        assert torch.equal(actual_args[0], common_attn_metadata.slot_mapping)
        assert actual_args[4] == 128

        assert metadata.block_size == 128
        assert metadata.group_len is actual_args[1]
        assert metadata.group_key_idx is actual_args[2]
        assert metadata.group_key_cache_idx is actual_args[3]
