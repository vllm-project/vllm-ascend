from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from vllm.config import CacheConfig, CompilationConfig, VllmConfig
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.mla import MLAModules

from tests.ut.base import TestBase
from vllm_ascend.attention.indexer import (
    INDEXER_K_CACHE_SLOT,
    INDEXER_SCALE_CACHE_SLOT,
    AscendSFAIndexerBackend,
)
from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention, IndexerWrapper


class TestAscendSFAIndexerBackend(TestBase):
    def _make_vllm_indexer(self):
        mock_indexer = MagicMock()
        mock_indexer.n_head = 64
        mock_indexer.head_dim = 128
        mock_indexer.topk_tokens = 2048
        mock_indexer.q_lora_rank = 1536
        mock_indexer.wq_b = nn.Linear(128, 128)
        mock_indexer.wk_weights_proj = nn.Linear(128, 128)
        mock_indexer.k_norm = nn.LayerNorm(128)
        mock_indexer.softmax_scale = 0.123
        mock_indexer.topk_indices_buffer = torch.randn(10)
        mock_indexer.k_cache = MagicMock()
        mock_indexer.k_cache.prefix = "model.layers.0.indexer"
        return mock_indexer

    @patch("vllm_ascend.attention.indexer.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.indexer.get_current_vllm_config")
    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    def test_initialization(self, mock_get_ascend_config, mock_get_vllm_config, _mock_enable_dsa_cp):
        mock_get_ascend_config.return_value.is_sparse_li_c8_layer.return_value = False
        mock_get_vllm_config.return_value.model_config.hf_config.model_type = "deepseek_v32"
        mock_get_vllm_config.return_value.parallel_config.prefill_context_parallel_size = 1
        mock_indexer = self._make_vllm_indexer()

        indexer = AscendSFAIndexerBackend(mock_indexer, qk_rope_head_dim=64)

        self.assertEqual(indexer.n_head, 64)
        self.assertEqual(indexer.head_dim, 128)
        self.assertEqual(indexer.topk_tokens, 2048)
        self.assertEqual(indexer.q_lora_rank, 1536)
        self.assertIs(indexer.wq_b, mock_indexer.wq_b)
        self.assertIs(indexer.wk_weights_proj, mock_indexer.wk_weights_proj)
        self.assertIs(indexer.k_norm, mock_indexer.k_norm)
        self.assertEqual(indexer.softmax_scale, 0.123)
        self.assertIs(indexer.k_cache, mock_indexer.k_cache)
        self.assertEqual(indexer.qk_rope_head_dim, 64)

        self.assertIsNone(mock_indexer.topk_indices_buffer)
        self.assertFalse(indexer.enable_sparse_li_c8)
        self.assertTrue(indexer.is_rope_neox_style)
        self.assertFalse(indexer.use_torch_npu_lightning_indexer)

    @patch("vllm_ascend.attention.indexer.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.indexer.get_current_vllm_config")
    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    def test_glm_model_type_flags(self, mock_get_ascend_config, mock_get_vllm_config, _mock_enable_dsa_cp):
        mock_get_ascend_config.return_value.is_sparse_li_c8_layer.return_value = False
        mock_get_vllm_config.return_value.model_config.hf_config.model_type = "glm_moe_dsa"
        mock_get_vllm_config.return_value.parallel_config.prefill_context_parallel_size = 1

        indexer = AscendSFAIndexerBackend(self._make_vllm_indexer(), qk_rope_head_dim=64)

        self.assertFalse(indexer.is_rope_neox_style)
        self.assertTrue(indexer.use_torch_npu_lightning_indexer)

    @patch("vllm_ascend.attention.indexer.enable_dsa_cp", return_value=False)
    @patch("vllm_ascend.attention.indexer.get_current_hardware_profile")
    @patch("vllm_ascend.attention.indexer.get_current_vllm_config")
    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    def test_li_c8_dtypes(self, mock_get_ascend_config, mock_get_vllm_config, mock_get_hw_profile, _mock_enable_dsa_cp):
        mock_get_ascend_config.return_value.is_sparse_li_c8_layer.return_value = True
        mock_get_vllm_config.return_value.model_config.hf_config.model_type = "deepseek_v32"
        mock_get_vllm_config.return_value.parallel_config.prefill_context_parallel_size = 1

        mock_get_hw_profile.return_value.supports.return_value = True
        indexer = AscendSFAIndexerBackend(self._make_vllm_indexer(), qk_rope_head_dim=64)
        self.assertTrue(indexer.enable_sparse_li_c8)
        self.assertEqual(indexer.c8_k_cache_dtype, torch.float8_e4m3fn)
        self.assertEqual(indexer.c8_k_scale_cache_dtype, torch.float32)

        mock_get_hw_profile.return_value.supports.return_value = False
        indexer = AscendSFAIndexerBackend(self._make_vllm_indexer(), qk_rope_head_dim=64)
        self.assertEqual(indexer.c8_k_cache_dtype, torch.int8)
        self.assertEqual(indexer.c8_k_scale_cache_dtype, torch.float16)

    def test_num_cache_tensors(self):
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)

        indexer.enable_sparse_li_c8 = False
        self.assertEqual(indexer.num_cache_tensors, 1)

        indexer.enable_sparse_li_c8 = True
        self.assertEqual(indexer.num_cache_tensors, 2)

    def _make_forward_indexer(self):
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        indexer.head_dim = 128
        indexer.n_head = 2
        indexer.qk_rope_head_dim = 64
        indexer.is_rope_neox_style = True
        indexer.enable_sparse_li_c8 = False
        indexer.use_torch_npu_lightning_indexer = False
        indexer.wk_weights_proj = MagicMock(return_value=(torch.zeros(2, 128 + 4), None))
        indexer.wq_b = MagicMock(return_value=(torch.zeros(2, 2 * 128), None))
        indexer._pcp_active = False
        indexer._dsa_cp_active = False
        indexer_k_cache = MagicMock(name="indexer_k_cache")
        indexer.k_cache = SimpleNamespace(kv_cache=(indexer_k_cache,))
        return indexer

    def _make_indexer_metadata(self):
        return SimpleNamespace(
            slot_mapping=MagicMock(name="slot_mapping"),
            num_decode_tokens=0,
            actual_seq_lengths_query=MagicMock(),
            actual_seq_lengths_key=MagicMock(),
        )

    def test_forward_runs_full_pipeline(self):
        # forward runs k path -> cache write -> top-k selection, reading the
        # freshly written cache at the selection stage.
        indexer = self._make_forward_indexer()
        indexer_metadata = self._make_indexer_metadata()

        calls = []
        k_li = torch.zeros(2, 128)
        indexer.forward_k = MagicMock(side_effect=lambda *args: calls.append("forward_k") or (k_li, None))
        indexer.write_cache = MagicMock(side_effect=lambda *args, **kwargs: calls.append("write_cache"))
        expected_topk = torch.zeros(2, 2048, dtype=torch.int32)

        def _select(*args):
            calls.append("select")
            return expected_topk

        hidden_states = torch.zeros(2, 32)
        q_c = torch.zeros(2, 16)
        k_hidden_states = torch.zeros(2, 32)
        cos, sin = MagicMock(name="cos"), MagicMock(name="sin")

        with (
            patch("vllm_ascend.attention.indexer.HAS_TRITON", True),
            patch(
                "vllm_ascend.attention.indexer.rope_forward_triton_siso",
                side_effect=lambda x, *args, **kwargs: x,
            ),
            patch(
                "vllm_ascend.device.device_op.DeviceOperator.indexer_select_post_process",
                side_effect=_select,
            ),
        ):
            result = indexer.forward(hidden_states, q_c, cos, sin, k_hidden_states, indexer_metadata)

        self.assertIs(result, expected_topk)
        # The write completes before the selection kernel reads the cache.
        self.assertEqual(calls, ["forward_k", "write_cache", "select"])
        indexer.forward_k.assert_called_once_with(k_hidden_states, cos, sin)
        indexer.write_cache.assert_called_once_with(
            k_li,
            None,
            indexer_metadata.slot_mapping,
            indexer_attn_metadata=indexer_metadata,
        )

    def test_forward_skip_topk_still_persists_cache(self):
        # compute_topk=False (SFA layers sharing top-k indices): k path and
        # the cache write still run; the selection stage is skipped.
        indexer = self._make_forward_indexer()
        indexer_metadata = self._make_indexer_metadata()
        indexer.forward_k = MagicMock(return_value=(torch.zeros(2, 128), None))
        indexer.write_cache = MagicMock()

        with patch(
            "vllm_ascend.device.device_op.DeviceOperator.indexer_select_post_process"
        ) as select:
            result = indexer.forward(
                torch.zeros(2, 32),
                None,
                MagicMock(name="cos"),
                MagicMock(name="sin"),
                torch.zeros(2, 32),
                indexer_metadata,
                compute_topk=False,
            )

        self.assertIsNone(result)
        indexer.forward_k.assert_called_once()
        indexer.write_cache.assert_called_once()
        select.assert_not_called()

    def test_gather_cache_inputs_identity_in_base_layout(self):
        indexer = self._make_forward_indexer()
        k_li, k_li_scale = MagicMock(), MagicMock()
        indexer_metadata = self._make_indexer_metadata()

        out_k, out_scale, out_slots = indexer._gather_cache_inputs(k_li, k_li_scale, indexer_metadata)

        self.assertIs(out_k, k_li)
        self.assertIs(out_scale, k_li_scale)
        self.assertIs(out_slots, indexer_metadata.slot_mapping)

    def test_gather_cache_inputs_pcp_gathers_prefill_region(self):
        indexer = self._make_forward_indexer()
        indexer._pcp_active = True
        k_li = torch.arange(8, dtype=torch.float32).view(2, 4)
        k_li_scale = torch.ones(2, 1, dtype=torch.float32)
        slots = torch.tensor([7, 8], dtype=torch.int64)
        gathered_k_li = torch.arange(16, dtype=torch.float32).view(4, 4)
        gathered_scale = torch.full((4, 1), 2.0)
        gathered_slots = torch.tensor([1, 2, 7, 8], dtype=torch.int64)
        indexer_metadata = self._make_indexer_metadata()
        indexer_metadata.slot_mapping = slots
        indexer_metadata.num_decode_tokens = 1

        with patch(
            "vllm_ascend.attention.indexer._gather_prefill_cache_inputs",
            return_value=((gathered_k_li, gathered_scale), gathered_slots),
        ) as gather:
            out_k, out_scale, out_slots = indexer._gather_cache_inputs(k_li, k_li_scale, indexer_metadata)

        gather.assert_called_once_with((k_li, k_li_scale), slots, 1)
        self.assertIs(out_k, gathered_k_li)
        self.assertIs(out_scale, gathered_scale)
        self.assertIs(out_slots, gathered_slots)

    def test_gather_cache_inputs_dsa_cp_all_gathers_k(self):
        indexer = self._make_forward_indexer()
        indexer._dsa_cp_active = True
        k_li = MagicMock(name="k_li")
        gathered_k_li = MagicMock(name="gathered_k_li")
        indexer_metadata = self._make_indexer_metadata()

        with (
            patch(
                "vllm_ascend.attention.indexer.all_gather_async",
                return_value=(gathered_k_li, None),
            ) as gather,
            patch("vllm_ascend.attention.indexer.get_tp_group"),
        ):
            out_k, out_scale, out_slots = indexer._gather_cache_inputs(k_li, None, indexer_metadata)

        gather.assert_called_once()
        self.assertIs(gather.call_args.args[0], k_li)
        self.assertIs(out_k, gathered_k_li)
        self.assertIsNone(out_scale)
        self.assertIs(out_slots, indexer_metadata.slot_mapping)

    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    @patch("vllm_ascend.attention.indexer.torch_npu.npu_scatter_nd_update_")
    def test_write_cache_scatter_path(self, mock_scatter, mock_get_ascend_config):
        mock_get_ascend_config.return_value.c8_enable_reshape_optim = False
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        indexer.enable_sparse_li_c8 = True

        k_li = torch.zeros(2, 4)
        k_li_scale = torch.zeros(2, 1)
        slot_mapping = torch.tensor([3, 5])
        indexer_k_cache = torch.zeros(8, 4)
        indexer_scale_cache = torch.zeros(8, 1)
        indexer.k_cache = SimpleNamespace(kv_cache=(indexer_k_cache, indexer_scale_cache))

        indexer.write_cache(k_li, k_li_scale, slot_mapping, MagicMock())

        self.assertEqual(mock_scatter.call_count, 2)
        k_call, scale_call = mock_scatter.call_args_list
        self.assertEqual(k_call.args[0].data_ptr(), indexer_k_cache.data_ptr())
        self.assertTrue(torch.equal(k_call.args[1], slot_mapping.view(-1, 1)))
        self.assertEqual(scale_call.args[0].data_ptr(), indexer_scale_cache.data_ptr())

    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    @patch("vllm_ascend.attention.indexer.torch_npu.npu_scatter_nd_update_")
    def test_write_cache_without_li_c8_writes_k_only(self, mock_scatter, mock_get_ascend_config):
        mock_get_ascend_config.return_value.c8_enable_reshape_optim = False
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        indexer.enable_sparse_li_c8 = False

        k_li = torch.zeros(2, 4)
        slot_mapping = torch.tensor([3, 5])
        indexer_k_cache = torch.zeros(8, 4)
        indexer.k_cache = SimpleNamespace(kv_cache=(indexer_k_cache,))

        indexer.write_cache(k_li, None, slot_mapping, MagicMock())

        mock_scatter.assert_called_once()
        self.assertEqual(mock_scatter.call_args.args[0].data_ptr(), indexer_k_cache.data_ptr())

    @patch("vllm_ascend.attention.indexer.get_ascend_config")
    @patch("vllm_ascend.attention.indexer.torch.ops._C_ascend.store_kv_block", create=True)
    def test_write_cache_reshape_optim_path(self, mock_store_kv_block, mock_get_ascend_config):
        mock_get_ascend_config.return_value.c8_enable_reshape_optim = True
        indexer = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
        indexer.enable_sparse_li_c8 = True

        k_li = torch.zeros(2, 4)
        k_li_scale = torch.zeros(2, 1)
        indexer_k_cache = torch.zeros(8, 4)
        indexer_scale_cache = torch.zeros(8, 1)
        indexer.k_cache = SimpleNamespace(kv_cache=(indexer_k_cache, indexer_scale_cache))
        attn_metadata = MagicMock()

        indexer.write_cache(k_li, k_li_scale, torch.tensor([3, 5]), attn_metadata)

        self.assertEqual(mock_store_kv_block.call_count, 2)
        k_call, scale_call = mock_store_kv_block.call_args_list
        self.assertIs(k_call.args[0], k_li)
        self.assertIs(k_call.args[1], indexer_k_cache)
        self.assertIs(scale_call.args[0], k_li_scale)
        self.assertIs(scale_call.args[1], indexer_scale_cache)


class TestIndexerWrapper(TestBase):
    @patch("vllm_ascend.ops.mla.AscendSFAIndexerBackend")
    def test_constructs_backend_and_delegates_sfa_interface(self, mock_backend_cls):
        vllm_indexer = MagicMock(name="vllm_indexer")
        wrapper = IndexerWrapper(vllm_indexer, qk_rope_head_dim=64)

        mock_backend_cls.assert_called_once_with(vllm_indexer, 64)
        self.assertIs(wrapper.impl, mock_backend_cls.return_value)
        self.assertIs(wrapper.k_cache, wrapper.impl.k_cache)

        wrapper("hidden", "q_c", "cos", "sin", "k_hidden", "meta", False)
        wrapper.impl.assert_called_once_with("hidden", "q_c", "cos", "sin", "k_hidden", "meta", False)

        wrapper.process_weights_after_loading()
        wrapper.impl.process_weights_after_loading.assert_called_once_with()


class TestAscendMultiHeadLatentAttention(TestBase):
    def setUp(self):
        self.hidden_size = 4096
        self.num_heads = 32
        self.scale = 0.123
        self.qk_nope_head_dim = 64
        self.qk_rope_head_dim = 64
        self.v_head_dim = 128
        self.q_lora_rank = 1536
        self.kv_lora_rank = 128
        self.prefix = "model.layers.0.mla"

        self.mock_mla_modules = MagicMock(spec=MLAModules)
        self.mock_mla_modules.indexer = MagicMock()
        self.mock_mla_modules.is_sparse = False
        self.mock_mla_modules.rotary_emb = MagicMock()
        self.mock_mla_modules.fused_qkv_a_proj = MagicMock()
        self.mock_mla_modules.q_b_proj = MagicMock()
        self.mock_mla_modules.q_a_layernorm = MagicMock()
        self.mock_mla_modules.q_proj = MagicMock()
        self.mock_mla_modules.kv_a_proj_with_mqa = MagicMock()
        self.mock_mla_modules.kv_a_layernorm = MagicMock()
        self.mock_mla_modules.kv_b_proj = MagicMock()
        self.mock_mla_modules.o_proj = MagicMock()

        self.mock_cache_config = MagicMock(spec=CacheConfig)
        self.mock_quant_config = MagicMock()

    @patch("vllm_ascend.ops.mla.IndexerWrapper")
    @patch("vllm_ascend.ops.mla.get_current_vllm_config")
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    def test_initialization(self, mock_tp_size, mock_get_vllm_config, mock_indexer_cls):
        # Create a proper mock for MLAAttention that has the required attributes
        mock_mla_attn = MagicMock()
        mock_mla_attn.process_weights_after_loading = MagicMock()
        mock_mla_attn.impl = MagicMock()
        mock_mla_attn.impl.process_weights_after_loading = MagicMock()

        with patch("vllm_ascend.ops.mla.MLAAttention", return_value=mock_mla_attn):
            mock_tp_size.return_value = 2
            mock_vllm_config = MagicMock(spec=VllmConfig)
            mock_vllm_config.model_config.hf_text_config = MagicMock(num_hidden_layers=32, first_k_dense_replace=True)
            mock_get_vllm_config.return_value = mock_vllm_config
            mock_vllm_config.compilation_config = CompilationConfig()

            attn = AscendMultiHeadLatentAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                scale=self.scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                mla_modules=self.mock_mla_modules,
                cache_config=self.mock_cache_config,
                quant_config=self.mock_quant_config,
                prefix=self.prefix,
            )

            self.assertEqual(attn.tp_size, 2)
            self.assertIsNotNone(attn.mla_attn)

    @patch("vllm_ascend.ops.mla.IndexerWrapper")
    @patch("vllm_ascend.ops.mla.torch.ops.vllm.mla_forward")
    @patch("vllm_ascend.ops.mla.get_current_vllm_config")
    @patch("vllm_ascend.ops.mla.get_tensor_model_parallel_world_size")
    @patch("vllm_ascend.ops.mla.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_forward(
        self,
        mock_get_forward_context_2,
        mock_get_forward_context,
        mock_tp_size,
        mock_get_vllm_config,
        mock_mla_forward,
        mock_indexer_cls,
    ):
        mock_tp_size.return_value = 1
        mock_vllm_config = MagicMock(spec=VllmConfig)
        mock_vllm_config.model_config.hf_text_config = MagicMock(num_hidden_layers=32, first_k_dense_replace=False)
        mock_get_vllm_config.return_value = mock_vllm_config
        mock_vllm_config.compilation_config = CompilationConfig()

        # Create a proper mock for MLAAttention that has the required attributes
        mock_mla_attn = MagicMock()
        mock_mla_attn.process_weights_after_loading = MagicMock()
        mock_mla_attn.impl = MagicMock()
        mock_mla_attn.impl.process_weights_after_loading = MagicMock()

        with patch("vllm_ascend.ops.mla.MLAAttention", return_value=mock_mla_attn):
            attn = AscendMultiHeadLatentAttention(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                scale=self.scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                mla_modules=self.mock_mla_modules,
                cache_config=self.mock_cache_config,
                quant_config=self.mock_quant_config,
                prefix=self.prefix,
            )
        positions = torch.tensor([0, 1, 2])
        hidden_states = torch.randn(3, self.hidden_size)

        mock_forward_context = MagicMock(spec=ForwardContext)
        mock_get_forward_context.return_value = mock_forward_context
        mock_get_forward_context_2.return_value = mock_forward_context

        mock_mla_forward.return_value = (3, self.hidden_size)

        output = attn.forward(positions, hidden_states)

        self.assertEqual(output.shape, (3, self.hidden_size))
