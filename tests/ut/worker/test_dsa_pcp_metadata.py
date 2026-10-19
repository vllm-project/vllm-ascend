from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from vllm_ascend.worker.v2 import dsa_pcp_metadata as metadata_module


class FakeBuilder:
    def __init__(self, ratio=0):
        self.compressor_ratio = ratio
        self.layer_names = ["swa"]
        self.speculative_config = SimpleNamespace(method="dspark", num_speculative_tokens=2)
        self.model_config = SimpleNamespace(hf_config=SimpleNamespace(sliding_window=4), get_head_size=lambda: 8)
        self.vllm_config = SimpleNamespace(speculative_config=self.speculative_config, model_config=self.model_config)
        self.start_pos_prefill = torch.zeros(4, dtype=torch.int32)
        self.slot_mapping = torch.zeros(8, dtype=torch.int32)
        self._global_metadata_builder = SimpleNamespace(
            start_pos_prefill=torch.zeros(4), _device_metadata_enabled=False
        )
        self._hidden_restore_idx_buffer = torch.zeros(8, dtype=torch.int64)
        self.sas_metadata_buffer = torch.zeros(4, dtype=torch.int32)
        self.dspark_swa_indices_buffer = torch.zeros((8, 1, 8), dtype=torch.int32)
        self.cu_seqlens_ori_kv = torch.empty(0)
        self.cu_seqlens_cmp_kv = torch.empty(0)
        self._zero_i32 = torch.zeros(1, dtype=torch.int32)
        self.storage_block_size = 4
        self.cache_group_key = "swa"
        self.hadamard = None
        self._device_metadata_enabled = False
        self.calls = []

    def _build_sas_metadata(self, **kwargs):
        self.calls.append(kwargs)
        # Model the original builder's ratio-only cache and persistent write.
        value = kwargs["metadata_cache"].setdefault(kwargs["layer_name"], 2 if kwargs["dspark_swa"] else 1)
        self.sas_metadata_buffer.fill_(value)
        return self.sas_metadata_buffer


class TestDSparkPCPMetadata(TestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.builder = FakeBuilder()
        self.plan = metadata_module.PCPMetadataPlan([self.builder], {id(self.builder): metadata_module.MetadataUse.SWA})
        self.plan._limits = lambda scope: (2, 6)
        self.plan._rope = lambda *args: (None, None)
        self.common = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=2,
            num_input_tokens=2,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            seq_lens=torch.tensor([6], dtype=torch.int32),
            block_table_tensor=torch.tensor([[3, 4]], dtype=torch.int32),
            causal=False,
            attn_state=None,
        )
        self.scope = metadata_module.BatchScope(self.common, 1, 1, 0, 2, 0, sas={}, op_cache={})
        self.indices = torch.full((2, 1, 8), 7, dtype=torch.int32)
        self.index_op = stack.enter_context(
            patch.object(metadata_module, "build_dspark_swa_indices", return_value=(self.indices, None))
        )
        self.turboquant = stack.enter_context(patch.object(metadata_module, "is_turboquant", return_value=False))
        stack.enter_context(
            patch.object(
                metadata_module,
                "DeviceOperator",
                SimpleNamespace(
                    get_dsa_decode_cu_seqlens_ori_kv=lambda *args: None,
                    get_dsa_decode_cu_seqlens_cmp_kv=lambda *args: None,
                ),
            )
        )

    def build(self, *, query=True):
        return self.plan._build_one(
            self.builder,
            metadata_module.MetadataUse.SWA,
            self.scope,
            self.common,
            "local",
            {"full_graph_mode": False},
            build_query=query,
            update_cache=False,
        ).req_metadata

    def test_special_indices_window_and_mask(self):
        req = self.build()
        self.assertIs(req.dspark_swa_indices, self.indices)
        self.assertEqual((req.ori_win_left, req.ori_win_right), (5, 0))
        self.assertTrue(self.builder.calls[0]["dspark_swa"])
        self.assertIs(self.index_op.call_args.kwargs["buffer"], self.builder.dspark_swa_indices_buffer)
        torch.testing.assert_close(self.index_op.call_args.args[0], self.common.block_table_tensor)

    def test_causal_prefill_and_cache_only_skip_special_indices(self):
        for causal, prefills, query in ((True, 0, True), (False, 1, True), (False, 0, False)):
            with self.subTest(causal=causal, prefills=prefills, query=query):
                self.common.causal = causal
                self.scope.num_prefills = prefills
                self.scope.sas.clear()
                self.scope.op_cache.clear()
                req = self.build(query=query)
                self.assertIsNone(req.dspark_swa_indices)
                self.assertEqual(req.ori_win_left, 3)
        self.index_op.assert_not_called()

    def test_sas_variants_do_not_overwrite_each_other(self):
        self.common.causal = True
        ordinary = self.build().sas_metadata
        self.common.causal = False
        special = self.build().sas_metadata
        self.assertNotEqual(ordinary.data_ptr(), special.data_ptr())
        torch.testing.assert_close(ordinary, torch.ones_like(ordinary))
        torch.testing.assert_close(special, torch.full_like(special, 2))
        self.assertIs(self.build().sas_metadata, special)
        self.assertEqual(len(self.builder.calls), 2)

    def test_indices_remain_group_specific_when_sas_is_shared(self):
        first = self.build().sas_metadata
        self.common.block_table_tensor = torch.tensor([[9, 10]], dtype=torch.int32)
        second = self.build().sas_metadata
        self.assertIs(first, second)
        self.assertEqual(self.index_op.call_count, 2)
        torch.testing.assert_close(self.index_op.call_args.args[0], self.common.block_table_tensor)
        self.assertEqual(len(self.builder.calls), 1)

    def test_turboquant_compressed_group_skips_special_indices(self):
        self.builder.compressor_ratio = 4
        self.turboquant.return_value = True
        self.plan._sas = Mock(return_value=(torch.zeros(4), None))
        req = self.build()
        self.index_op.assert_not_called()
        self.assertIsNone(req.dspark_swa_indices)
        self.assertFalse(self.plan._sas.call_args.kwargs["dspark_swa"])

    def test_new_scope_refreshes_same_stable_output(self):
        first = self.build().sas_metadata
        self.scope.sas = {}
        self.scope.op_cache = {}
        second = self.build().sas_metadata
        self.assertIs(first, second)
        self.assertEqual(len(self.builder.calls), 2)

    def test_draft_needs_no_pcp_buffers_and_respects_device_metadata(self):
        del self.builder._global_metadata_builder
        del self.builder._hidden_restore_idx_buffer
        plan = metadata_module.PCPMetadataPlan(
            [self.builder], {id(self.builder): metadata_module.MetadataUse.SWA}, draft=True
        )
        self.assertTrue(plan.enable())
        self.assertNotIn("global", plan.start_buffers)
        self.builder._device_metadata_enabled = True
        self.assertFalse(plan.enable())

    def test_draft_shares_scope_only_within_one_execution(self):
        plan = metadata_module.PCPMetadataPlan(
            [self.builder], {id(self.builder): metadata_module.MetadataUse.SWA}, draft=True
        )
        plan._scope = Mock(side_effect=[self.scope, object()])
        plan._build_one = Mock()
        shared: dict[tuple[str, int], Any] = {}
        plan.build_draft(self.builder, self.common, 1, shared, False)
        other_common = SimpleNamespace(causal=True)
        plan.build_draft(self.builder, other_common, 1, shared, False)
        self.assertEqual(plan._scope.call_count, 1)
        self.assertIs(plan._build_one.call_args.args[3], other_common)
        self.assertTrue(plan._build_one.call_args.kwargs["update_cache"])
        self.assertTrue(plan._build_one.call_args.kwargs["build_query"])
        plan.build_draft(self.builder, self.common, 1, {}, False)
        self.assertEqual(plan._scope.call_count, 2)

    def test_state_cache_skips_query_and_compressor_preparation(self):
        self.plan._rope = Mock()
        self.plan._sas = Mock()
        self.plan._qli = Mock()
        self.plan._start_pos = Mock()
        result = self.plan._build_one(
            self.builder,
            metadata_module.MetadataUse.STATE,
            self.scope,
            self.common,
            "local",
            {"full_graph_mode": False},
            build_query=True,
            update_cache=True,
        )
        self.assertIsNotNone(result.req_metadata.block_table)
        for method in (self.plan._rope, self.plan._sas, self.plan._qli, self.plan._start_pos):
            method.assert_not_called()
