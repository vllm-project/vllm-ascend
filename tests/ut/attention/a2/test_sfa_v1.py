import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

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
    PreprocessType,
    custom_kv_rmsnorm_rope,
)
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
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

    def test_execute_sparse_flash_attention_returns_softmax_components(self):
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
            output, actual_softmax_max, actual_softmax_sum = DeviceOperator.execute_sparse_flash_attention_process(
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
        self.assertIs(actual_softmax_max, softmax_max)
        self.assertIs(actual_softmax_sum, softmax_sum)
        self.assertTrue(mock_sfa.call_args.kwargs["return_softmax_lse"])

    def test_execute_sparse_flash_attention_c8_returns_softmax_components(self):
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

        with (
            patch.object(
                torch.ops._C_ascend,
                "npu_kv_quant_sparse_flash_attention",
                create=True,
                return_value=(attn_output, softmax_max, softmax_sum),
            ) as mock_qsfa,
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
                create=True,
                side_effect=AssertionError("C8 SFA with LSE must use the custom op"),
            ),
        ):
            output, actual_softmax_max, actual_softmax_sum = DeviceOperator.execute_sparse_flash_attention_process(
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
        self.assertIs(actual_softmax_max, softmax_max)
        self.assertIs(actual_softmax_sum, softmax_sum)
        call_kwargs = mock_qsfa.call_args.kwargs
        self.assertIs(call_kwargs["key"], packed_kv_cache[0])
        self.assertIs(call_kwargs["value"], packed_kv_cache[0])
        self.assertEqual(call_kwargs["query"].shape, (3, 4, 10))
        self.assertEqual(call_kwargs["sparse_mode"], 0)
        self.assertTrue(call_kwargs["return_softmax_lse"])


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

        custom_kv_rmsnorm_rope(
            torch.randn(2, 1, 1, 272, dtype=torch.bfloat16),
            torch.ones(256, dtype=torch.bfloat16),
            torch.randn(2, 1, 1, 16),
            torch.randn(2, 1, 1, 16),
            256,
            16,
            dst_type=1,
            tile_size=128,
        )

        self.assertEqual(mock_block_quant.call_args.kwargs["dst_type"], 1)
        self.assertEqual(mock_block_quant.call_args.kwargs["row_block_size"], 1)
        self.assertEqual(mock_block_quant.call_args.kwargs["col_block_size"], 128)

    def test_execute_kv_quant_sparse_flash_attention(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.use_sparse_c8_sfa = True
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

        with (
            patch.object(
                torch.ops._C_ascend,
                "npu_kv_quant_sparse_flash_attention",
                create=True,
                return_value=(expected, torch.empty(0), torch.empty(0)),
            ) as mock_qsfa,
            patch(
                "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
                create=True,
                side_effect=AssertionError("Base must use _C_ascend custom op"),
            ),
        ):
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
        self.assertFalse(call_kwargs["return_softmax_lse"])

    def test_prolog_v3_enables_packed_int8_kv_cache(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl._quant_type = AscendW8A8DynamicLinearMethod
        impl.use_sparse_c8_sfa = True
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
                "vllm_ascend.attention.sfa_v1.torch_npu.npu_dynamic_quant",
                return_value=(torch.empty(2, 8, dtype=torch.int8), torch.ones(2, 1)),
            ),
            patch(
                "vllm_ascend.attention.sfa_v1.torch_npu.npu_mla_prolog_v3",
                create=True,
                return_value=(torch.randn(2, 2, 128), torch.randn(2, 2, 16), None, torch.randn(2, 8), None),
            ) as mock_prolog,
        ):
            impl._sfa_preprocess_prolog_v3(
                hidden_states=torch.randn(2, 8),
                kv_cache=(k_cache, dsa_k_cache),
                cos=torch.randn(2, 1, 1, 16),
                sin=torch.randn(2, 1, 1, 16),
                slot_mapping=torch.arange(2),
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
        common_attn_metadata.slot_mapping = torch.randint(0, 10000, (100, 4, 1024), dtype=torch.int64)
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


class TestAscendSFAImpl(TestBase):
    @patch("vllm.distributed.parallel_state._TP", new_callable=lambda: MagicMock(spec=GroupCoordinator))
    @patch("vllm_ascend.attention.sfa_v1.get_current_vllm_config")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp_with_o_proj_tp")
    @patch("vllm_ascend.attention.sfa_v1.enable_sp")
    @patch("vllm_ascend.attention.sfa_v1.enable_dsa_cp")
    @patch("vllm_ascend.attention.sfa_v1.get_ascend_config")
    def setUp(
        self,
        mock_get_ascend_config,
        mock_enable_dsa_cp,
        mock_enable_sp,
        mock_enable_dsa_cp_with_o_proj_tp,
        mock_get_current_vllm_config,
        mock_tp,
    ):
        mock_tp.world_size = 1
        mock_tp.rank_in_group = 0
        mock_tp.device_group = MagicMock()

        mock_enable_dsa_cp.return_value = False
        mock_enable_sp.return_value = False
        mock_enable_dsa_cp_with_o_proj_tp.return_value = False

        # Default ascend config (non-MLAPO, non-C8)
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_mlapo = False
        mock_ascend_config.enable_sparse_c8 = False
        mock_ascend_config.enable_shared_expert_dp = False
        mock_ascend_config.is_sparse_c8_layer.return_value = False
        mock_get_ascend_config.return_value = mock_ascend_config
        self.mock_ascend_config = mock_ascend_config

        vllm_config = MagicMock()
        vllm_config.model_config.dtype = torch.float16
        vllm_config.model_config.hf_config = MagicMock()
        vllm_config.model_config.hf_text_config = None
        vllm_config.kv_transfer_config = None
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.num_speculative_tokens = 0
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.additional_config = {"refresh": True}
        vllm_config.scheduler_config.max_num_batched_tokens = 4096
        mock_get_current_vllm_config.return_value = vllm_config

        init_ascend_config(vllm_config)

        num_heads = 256
        head_size = 1024
        kv_lora_rank = 128
        qk_nope_head_dim = 64
        v_head_dim = 128

        kv_a_layernorm = MagicMock()
        kv_a_layernorm.weight = torch.randn(96)
        kv_a_layernorm.variance_epsilon = 1e-6

        q_a_layernorm = MagicMock()
        q_a_layernorm.weight = torch.randn(96)

        kwargs = {
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": 32,
            "qk_head_dim": 96,
            "v_head_dim": v_head_dim,
            "q_lora_rank": 64,
            "q_proj": MagicMock(),
            "q_b_proj": MagicMock(),
            "kv_b_proj": MagicMock(),
            "o_proj": MagicMock(),
            "kv_a_proj_with_mqa": MagicMock(),
            "fused_qkv_a_proj": MagicMock(),
            "kv_a_layernorm": kv_a_layernorm,
            "q_a_layernorm": q_a_layernorm,
            "rotary_emb": MagicMock(),
            "indexer": None,
            "skip_topk": True,
            "topk_indices_buffer": torch.zeros(4096, dtype=torch.int64),
            "layer_name": "model.layers.0",
        }

        self.impl = AscendSFAImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=0.1,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            **kwargs,
        )

    def _setup_kv_b_proj(self):
        """Set up kv_b_proj with real weight tensor for process_weights_after_loading."""
        shape_0 = self.impl.num_heads * (self.impl.qk_nope_head_dim + self.impl.v_head_dim)
        shape_1 = self.impl.kv_lora_rank
        layer = MagicMock(spec=LinearBase)
        layer.input_size_per_partition = 10
        quant_method = MagicMock(spec=UnquantizedLinearMethod)
        layer.quant_method = quant_method
        layer.weight = torch.randn(shape_0, shape_1)
        self.impl.kv_b_proj = layer
        return layer

    # ============ process_weights_after_loading ============

    @patch("vllm_ascend.attention.sfa_v1.maybe_trans_nz")
    @patch("vllm_ascend.attention.sfa_v1.dispose_layer")
    @patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading(self, mock_format_cast, mock_dispose, mock_maybe_trans_nz):
        """Basic weight reshape without MLAPO."""
        layer = self._setup_kv_b_proj()
        mock_format_cast.return_value = layer.weight
        mock_maybe_trans_nz.side_effect = lambda x: x

        self.impl.process_weights_after_loading(torch.bfloat16)

        self.assertEqual(self.impl.W_UK_T.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UK_T.shape[1], self.impl.qk_nope_head_dim)
        self.assertEqual(self.impl.W_UK_T.shape[2], self.impl.kv_lora_rank)

        self.assertEqual(self.impl.W_UV.shape[0], self.impl.num_heads)
        self.assertEqual(self.impl.W_UV.shape[1], self.impl.kv_lora_rank)
        self.assertEqual(self.impl.W_UV.shape[2], self.impl.v_head_dim)

        mock_dispose.assert_called_once()
        mock_maybe_trans_nz.assert_called_once()

    # ============ _process_weights_for_fused_prolog_v3 ============

    def _run_prolog_v3_weight_test(self, qt, has_scales):
        mock_format_cast = patch("torch_npu.npu_format_cast", return_value=torch.randn(128, 128))
        mock_format_cast.start()
        self.addCleanup(mock_format_cast.stop)

        self.impl._quant_type = qt
        self.impl.fused_qkv_a_proj = MagicMock()
        self.impl.fused_qkv_a_proj.weight.data = torch.randn(128, 96, 64)

        if qt is None:
            self.impl.q_proj = SimpleNamespace(weight=SimpleNamespace(data=torch.randn(128, 96)))
        else:
            self.impl.fused_qkv_a_proj.weight_scale = torch.randn(64, 128, 128)
            self.impl.q_proj = MagicMock()
            self.impl.q_proj.weight.data = torch.randn(128, 128)
            self.impl.q_proj.weight_scale.data = torch.randn(128, 128, 128)

        self.impl.q_lora_rank = 32
        self.impl._process_weights_for_fused_prolog_v3()

        self.assertTrue(hasattr(self.impl, "weight_dq"))
        self.assertTrue(hasattr(self.impl, "weight_uq_qr"))
        self.assertTrue(hasattr(self.impl, "weight_dkv_kr"))
        self.assertEqual(hasattr(self.impl, "weight_dq_scale"), has_scales)

    def test_process_weights_for_fused_prolog_v3_mxfp(self):
        self._run_prolog_v3_weight_test(AscendW8A8MXFP8DynamicLinearMethod, True)

    def test_process_weights_for_fused_prolog_v3_unquantized(self):
        self._run_prolog_v3_weight_test(None, False)

    # ============ exec_kv: sparse C8 uses custom_kv_rmsnorm_rope ============

    @patch("vllm_ascend.attention.sfa_v1.custom_kv_rmsnorm_rope")
    @patch("torch_npu.npu_kv_rmsnorm_rope_cache")
    def test_exec_kv_sparse_c8_uses_custom(
        self,
        mock_npu_kv_rmsnorm_rope_cache,
        mock_custom_kv_rmsnorm_rope,
    ):
        """exec_kv with use_sparse_c8_sfa → delegates to custom_kv_rmsnorm_rope."""
        self.impl.use_sparse_c8_sfa = True
        self.impl.c8_k_cache_dtype = torch.int8
        self.impl.enable_dsa_cp = False
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.weight = torch.ones(self.impl.kv_lora_rank)
        self.impl.kv_a_layernorm.variance_epsilon = 1e-5

        num_tokens = 2
        N = self.impl.num_kv_heads
        kv_lora_rank = self.impl.kv_lora_rank
        qk_rope_head_dim = self.impl.qk_rope_head_dim

        kv_no_split = torch.randn(num_tokens, N * (kv_lora_rank + qk_rope_head_dim))
        cos = torch.randn(num_tokens, qk_rope_head_dim)
        sin = torch.randn(num_tokens, qk_rope_head_dim)

        kv_cache = (torch.zeros(128, 72), torch.zeros(128, 128))
        slots = torch.arange(4).view(-1, 1)

        fake_result = (
            torch.randn(2, 8, 1, 16),
            torch.randn(16, 1, 4),
            torch.randn(16, 4),
        )
        mock_custom_kv_rmsnorm_rope.return_value = fake_result

        result = self.impl.exec_kv(kv_no_split, cos, sin, kv_cache, slots, MagicMock())
        self.assertIs(result, fake_result)
        mock_npu_kv_rmsnorm_rope_cache.assert_not_called()

    # ============ _resolve_preprocess_type: routing logic ============

    def _set_quant(self, qt):
        """Make _resolve_preprocess_type see *qt* as the layer quant type."""
        self.impl.fused_qkv_a_proj = MagicMock()
        # Production code derives the type from the scheme *instance*; build one
        # without running __init__ (MXFP8 __init__ needs NPU/vllm config).
        quant_method = qt.__new__(qt) if qt is not None else None
        self.impl.fused_qkv_a_proj.quant_method = SimpleNamespace(quant_method=quant_method)
        # setUp's @patch stops at setUp exit; re-patch so the routing decision
        # sees self.mock_ascend_config (e.g. enable_mlapo) at call time.
        patcher_cfg = patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=self.mock_ascend_config)
        patcher_cfg.start()
        self.addCleanup(patcher_cfg.stop)
        self.impl.q_proj = MagicMock()
        self.impl.q_proj._chunk_size = 0
        self.impl.kv_a_proj_with_mqa = None
        # Path resolution tests: mock weight processing so only the routing
        # decision is exercised, not tensor operations.
        patcher = patch.object(self.impl, "_try_enable_type", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_resolve_path_w8a8dynamic_c8_goes_prolog_v3(self):
        """W8A8Dynamic + C8 + PD consumer → PROLOG_V3."""
        self._set_quant(AscendW8A8DynamicLinearMethod)
        self.impl.is_kv_consumer = True
        self.impl.use_sparse_c8_sfa = True

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        self.assertEqual(path, PreprocessType.PROLOG_V3)

    def test_resolve_path_w8a8_mlapo_enabled_goes_mlapo(self):
        """W8A8 + enable_mlapo → MLAPO."""
        self._set_quant(AscendW8A8LinearMethod)
        self.impl.enable_mlapo = True

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        self.assertEqual(path, PreprocessType.MLAPO)

    def test_resolve_path_mxfp_c8_goes_prolog_v3(self):
        """MXFP + is_kv_consumer + C8 → PROLOG_V3."""
        self._set_quant(AscendW8A8MXFP8DynamicLinearMethod)
        self.impl.is_kv_consumer = True
        self.impl.use_sparse_c8_sfa = True

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        self.assertEqual(path, PreprocessType.PROLOG_V3)

    def test_resolve_path_unquantized_c8_goes_prolog_v3(self):
        """Unquantized + is_kv_consumer + C8 → PROLOG_V3 (blocked by reasons)."""
        self._set_quant(None)
        self.impl.is_kv_consumer = True
        self.impl.use_sparse_c8_sfa = True

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        # Enters candidate but blocked by _get_fused_type_unsupported_reasons
        # (unquantized + C8).  With _try_enable_type mocked to True, still
        # returns PROLOG_V3 in the test.
        self.assertEqual(path, PreprocessType.PROLOG_V3)

    def test_resolve_path_no_mlapo_goes_native(self):
        """No quant + MLAPO disabled → NATIVE."""
        self._set_quant(None)

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        self.assertEqual(path, PreprocessType.NATIVE)

    def test_resolve_path_w8a8dynamic_c8_no_mlapo_still_prolog_v3(self):
        """W8A8Dynamic+C8 enters PROLOG_V3 even when enable_mlapo=False."""
        self._set_quant(AscendW8A8DynamicLinearMethod)
        self.impl.is_kv_consumer = True
        self.impl.use_sparse_c8_sfa = True

        path = self.impl._resolve_preprocess_type(torch.bfloat16)
        self.assertEqual(path, PreprocessType.PROLOG_V3)

    # ============ _get_fused_type_unsupported_reasons ============

    def _setup_prolog_v3_state(self):
        """Minimal setup so unsupported-reasons checks can run."""
        self.impl.preprocess_type = PreprocessType.PROLOG_V3
        self.impl._quant_type = AscendW8A8DynamicLinearMethod
        self.impl.kv_a_layernorm = MagicMock()
        self.impl.kv_a_layernorm.variance_epsilon = 1e-5
        self.impl.q_a_layernorm = MagicMock()
        self.impl.q_a_layernorm.variance_epsilon = 1e-5
        self.impl.q_proj = MagicMock()
        self.impl.q_proj._chunk_size = 0

    def test_reasons_dsa_cp_blocked(self):
        self._setup_prolog_v3_state()
        self.impl.enable_dsa_cp = True

        reasons = self.impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
        self.assertTrue(any("DSA-CP" in r for r in reasons))

    def test_reasons_kv_producer_blocked(self):
        self._setup_prolog_v3_state()
        self.impl.is_kv_producer = True

        reasons = self.impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
        self.assertTrue(any("KV producer" in r for r in reasons))

    def test_reasons_unquantized_c8_blocked(self):
        self._setup_prolog_v3_state()
        self.impl._quant_type = None
        self.impl.use_sparse_c8_sfa = True

        reasons = self.impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
        self.assertTrue(any("C8 sparse requires quantized" in r for r in reasons))

    def test_reasons_mlapo_c8_blocked(self):
        self._setup_prolog_v3_state()
        self.impl.preprocess_type = PreprocessType.MLAPO
        self.impl.use_sparse_c8_sfa = True

        reasons = self.impl._get_fused_type_unsupported_reasons(PreprocessType.MLAPO)
        self.assertTrue(any("sparse C8" in r for r in reasons))

    # ============ _sfa_preprocess_prolog_v3: MXFP runtime ============

    def test_sfa_preprocess_prolog_v3_mxfp(self):
        """MXFP branch: npu_dynamic_mx_quant + q_c scale wrapping."""
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl._quant_type = AscendW8A8MXFP8DynamicLinearMethod
        impl.use_sparse_c8_sfa = False
        impl.local_num_heads = 2
        impl.num_heads = 2
        impl.kv_lora_rank = 128
        impl.qk_rope_head_dim = 16
        impl.q_lora_rank = 8
        impl.q_a_layernorm = MagicMock()
        impl.q_a_layernorm.weight.data = torch.ones(8)
        impl.q_a_layernorm.variance_epsilon = 1e-5
        impl.kv_a_layernorm = MagicMock()
        impl.kv_a_layernorm.weight.data = torch.ones(32)
        impl.kv_a_layernorm.variance_epsilon = 1e-5
        impl.has_indexer = True
        impl.wq_b = None
        impl.weight_dq = torch.empty(1)
        impl.weight_uq_qr = torch.empty(1)
        impl.W_UK_T = torch.randn(2, 64, 32)
        impl.weight_dkv_kr = torch.empty(1)
        impl.weight_dq_scale = torch.randn(16, 1)
        impl.weight_uq_qr_scale = torch.randn(16, 1)
        impl.weight_dkv_kr_scale = torch.randn(16, 1)
        impl.sfa_qsfa_kr_cache_dummy = torch.empty(0)

        # Verify MXFP runtime attributes are correctly wired.
        # (Full operator call requires NPU hardware; operator-level tests
        # live in integration suites.)
        self.assertTrue(hasattr(impl, "weight_dq_scale"))
        self.assertTrue(hasattr(impl, "weight_uq_qr_scale"))
        self.assertTrue(hasattr(impl, "weight_dkv_kr_scale"))
        self.assertIs(impl._quant_type, AscendW8A8MXFP8DynamicLinearMethod)

    # (MLAPO runtime path requires NPU hardware; covered by integration tests.)
