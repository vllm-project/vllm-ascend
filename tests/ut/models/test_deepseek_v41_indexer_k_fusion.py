# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer

INDEXER_MODULE = "vllm_ascend.models.deepseek_v41.indexer"
FUSION_MODULE = "vllm_ascend.ops.triton.indexer_k_fusion"


class TestIndexerKFusionDispatch(TestCase):
    def setUp(self):
        self.latent = torch.randn(3, 128, dtype=torch.bfloat16)
        self.slots = torch.tensor([[0, 0], [1, 1], [-1, -1]], dtype=torch.int32)
        self.cos = torch.ones(3, 1, 1, 64)
        self.sin = torch.zeros_like(self.cos)
        self.projected = torch.randn_like(self.latent)
        self.key_cache = torch.zeros(2, 4, 1, 128, dtype=torch.int8)
        self.scale_cache = torch.zeros(2, 4, 1, 1, dtype=torch.float16)
        self.indexer = SimpleNamespace(
            owns_k=True,
            width=128,
            rope_width=64,
            wk=MagicMock(return_value=self.projected),
            k_norm=MagicMock(
                return_value=self.projected,
                weight=torch.ones(128, dtype=torch.bfloat16),
                variance_epsilon=1e-6,
                bias_loaded=False,
            ),
            packed_cache_ops=None,
            k_cache=SimpleNamespace(kv_cache=[(self.key_cache, self.scale_cache)]),
            k_cache_folded=None,
        )

    def _update(self):
        DeepseekV41Indexer.update_keys(self.indexer, self.latent, self.slots, self.cos, self.sin)

    def test_fused_path_receives_pre_norm_projection_and_original_buffers(self):
        latent_before, slots_before = self.latent.clone(), self.slots.clone()
        with (
            patch(f"{FUSION_MODULE}.indexer_k_norm_rope_int8_store") as fused,
            patch(f"{INDEXER_MODULE}.torch_npu.npu_dynamic_quant") as quant,
            patch(f"{INDEXER_MODULE}.scatter_cache_sk") as scatter,
            patch(f"{INDEXER_MODULE}.DeviceOperator.apply_partial_rotary_inplace") as rope,
        ):
            self._update()
        self.indexer.wk.assert_called_once_with(self.latent)
        self.indexer.k_norm.assert_not_called()
        fused.assert_called_once_with(
            self.projected,
            self.indexer.k_norm.weight,
            self.indexer.k_norm.variance_epsilon,
            self.cos,
            self.sin,
            self.slots,
            self.key_cache,
            self.scale_cache,
        )
        quant.assert_not_called()
        scatter.assert_not_called()
        rope.assert_not_called()
        torch.testing.assert_close(self.latent, latent_before, rtol=0, atol=0)
        torch.testing.assert_close(self.slots, slots_before, rtol=0, atol=0)

    def test_non_owner_and_empty_batch_return_before_accessing_cache(self):
        DeepseekV41Indexer.update_keys(SimpleNamespace(owns_k=False), self.latent, None, None, None)
        DeepseekV41Indexer.update_keys(SimpleNamespace(owns_k=True), self.latent[:0], None, None, None)

    def test_mismatched_source_rope_width_rejected_before_projection(self):
        self.cos = self.cos[..., :32]
        with self.assertRaisesRegex(ValueError, "rope_width"):
            self._update()
        self.indexer.wk.assert_not_called()

    def test_loaded_norm_bias_keeps_norm_rope_quant_and_scatter_path(self):
        self.indexer.k_norm.bias_loaded = True
        quantized = torch.ones_like(self.projected, dtype=torch.int8)
        scale = torch.ones(3, 1)
        with (
            patch(f"{FUSION_MODULE}.indexer_k_norm_rope_int8_store") as fused,
            patch(f"{INDEXER_MODULE}.DeviceOperator.apply_partial_rotary_inplace") as rope,
            patch(f"{INDEXER_MODULE}.torch_npu.npu_dynamic_quant", return_value=(quantized, scale)) as quant,
            patch(f"{INDEXER_MODULE}.scatter_cache_sk") as scatter,
        ):
            self._update()
        fused.assert_not_called()
        self.indexer.k_norm.assert_called_once_with(self.projected)
        self.assertEqual(rope.call_args.kwargs, dict(start=64, end=128))
        self.assertEqual(quant.call_args.kwargs, dict(dst_type=torch.int8))
        self.assertEqual(scatter.call_count, 2)
        self.assertIs(scatter.call_args_list[0].args[0], self.key_cache)
        self.assertIs(scatter.call_args_list[0].args[1], self.slots)
        self.assertIs(scatter.call_args_list[0].args[2], quantized)
        self.assertIs(scatter.call_args_list[1].args[0], self.scale_cache)
        torch.testing.assert_close(scatter.call_args_list[1].args[2], scale.unsqueeze(-1).half())

    def test_packed_cache_and_optional_folded_publication_keep_original_path(self):
        for folded in (False, True):
            with self.subTest(folded=folded):
                packed = self.indexer.packed_cache_ops = MagicMock()
                self.indexer.k_cache_folded = SimpleNamespace(kv_cache=[object()]) if folded else None
                with (
                    patch(f"{FUSION_MODULE}.indexer_k_norm_rope_int8_store") as fused,
                    patch(f"{INDEXER_MODULE}.DeviceOperator.apply_partial_rotary_inplace"),
                    patch(f"{INDEXER_MODULE}.torch_npu.npu_dynamic_quant") as quant,
                    patch(f"{INDEXER_MODULE}.fold_indexer_cache_rows") as fold,
                ):
                    self._update()
                fused.assert_not_called()
                quant.assert_not_called()
                packed.write_index_cache.assert_called_once()
                self.assertEqual(fold.call_count, int(folded))
                if folded:
                    self.assertIs(fold.call_args.args[1], self.indexer.k_cache_folded.kv_cache[0])
                    self.assertIs(fold.call_args.args[2], self.slots)
