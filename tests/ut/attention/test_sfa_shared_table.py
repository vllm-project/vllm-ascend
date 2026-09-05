# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import (
    SFA_FULL_VISIBLE_TEMPLATE_BLOCK_SIZE,
    SFA_INDEXER_SPARSE_COUNT,
    AscendSFAImpl,
    PreprocessType,
)
from vllm_ascend.device.device_op import BaseDeviceAdaptor


class TestFullVisibleHostDispatch(unittest.TestCase):
    """Real CPU tensors and forward body; projections and NPU kernels are mocked.

    Cache helpers, eligibility and the final Python adapter are real. The CPU
    scatter checks the call's destination, indices and values, not native
    numerical or lifetime behavior. Weight-loading setup is excluded.
    """

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.config = SimpleNamespace(enable_sfa_full_visible_index_bypass=True)
        self.stack.enter_context(patch("vllm_ascend.attention.sfa_v1.get_ascend_config", return_value=self.config))
        # Do not clear or leak the per-device table owned by other tests.
        self.stack.enter_context(patch.object(AscendSFAImpl, "_full_visible_index_tables", {}))

    def make_impl(self):
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.allow_short_prefill_indexer_scoring_skip = True
        impl.skip_topk = False
        impl.has_indexer = True
        impl.enable_sparse_sfa_c8 = False
        impl.enable_sparse_li_c8 = False
        impl._full_visible_index_table = impl._get_or_create_full_visible_index_table(torch.device("cpu"))
        return impl

    @staticmethod
    def metadata(state, total, query=64):
        return SimpleNamespace(
            attn_state=state,
            block_size=128,
            seq_lens_cpu=torch.tensor([total]),
            num_actual_tokens=query,
            num_input_tokens=query,
            cos=torch.ones(query, 2),
            sin=torch.zeros(query, 2),
            slot_mapping=torch.arange(total - query, total),
            block_table=torch.arange(17, dtype=torch.int32).unsqueeze(0),
        )

    @staticmethod
    def rr_order(visible):
        # Independent scalar oracle: visit each offset across all visible blocks.
        return [block + offset for offset in range(128) for block in range(0, visible, 128) if block + offset < visible]

    @staticmethod
    def storage_interval(tensor):
        storage = tensor.untyped_storage()
        return storage.data_ptr(), storage.data_ptr() + storage.nbytes()

    def assert_disjoint_storage(self, left, right):
        a, b = self.storage_interval(left), self.storage_interval(right)
        self.assertTrue(a[1] <= b[0] or b[1] <= a[0], (a, b))

    def test_two_layers_short_longer_short_preserve_entire_template(self):
        first, second = self.make_impl(), self.make_impl()
        table = first._full_visible_index_table
        self.assertIs(table, second._full_visible_index_table)
        before = torch.empty_like(table)
        before.copy_(table)
        self.assert_disjoint_storage(table, before)
        # Match the model's one allocation passed to multiple decoder layers.
        cache = torch.full((257, 2048), -17, dtype=torch.int32)
        first.topk_indices_buffer = second.topk_indices_buffer = cache
        for impl, query, total in ((first, 64, 128), (second, 192, 192), (first, 64, 128)):
            with self.subTest(query=query, total=total):
                self.forward_case(AscendAttentionState.PrefillCacheHit, total, query=query, impl=impl)
                torch.testing.assert_close(table, before, rtol=0, atol=0)
                expected = before[total - query + 1 : total + 1]
                torch.testing.assert_close(cache[:query], expected, rtol=0, atol=0)

    def test_complete_cache_update_preserves_strided_destination_guards(self):
        impl = self.make_impl()
        table = impl._full_visible_index_table
        before = torch.empty_like(table)
        before.copy_(table)
        rows = impl._get_full_visible_topk_indices(
            self.metadata(AscendAttentionState.PrefillCacheHit, 128), 64, torch.device("cpu")
        )
        for source in (rows, rows.squeeze(1)):
            backing = torch.full((66, 2050), -17, dtype=torch.int32)
            impl.topk_indices_buffer = backing[1:65, 1:2049]
            self.assertFalse(impl.topk_indices_buffer.is_contiguous())
            self.assert_disjoint_storage(table, backing)
            expected = backing.clone()
            expected[1:65, 1:2049] = before[65:129]
            impl._update_indexcache_topk_indices(source)
            torch.testing.assert_close(backing, expected, rtol=0, atol=0)
            torch.testing.assert_close(table, before, rtol=0, atol=0)
        impl.topk_indices_buffer = None
        impl._update_indexcache_topk_indices(rows)
        torch.testing.assert_close(table, before, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            impl._get_indexcache_topk_indices(64)

    def test_snapshot_oracle_detects_injected_writes(self):
        # Harness sensitivity controls, not observed production corruption.
        impl = self.make_impl()
        table = impl._full_visible_index_table
        before = torch.empty_like(table)
        before.copy_(table)
        self.assert_disjoint_storage(table, before)
        rows = table[65:129].unsqueeze(1)
        self.assertNotEqual(rows.data_ptr(), table.data_ptr())
        self.assertEqual(self.storage_interval(rows), self.storage_interval(table))
        rows[0, 0, 0] = -99
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(table, before, rtol=0, atol=0)
        table.copy_(before)
        # A view-only oracle misses writes elsewhere in the persistent template.
        table[-1, -1] = -99
        torch.testing.assert_close(rows[:, 0], before[65:129], rtol=0, atol=0)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(table, before, rtol=0, atol=0)
        table.copy_(before)
        # Demonstrate why the separately allocated destination is necessary.
        impl.topk_indices_buffer = table[1:65]
        with self.assertRaises(AssertionError):
            self.assert_disjoint_storage(rows, impl.topk_indices_buffer)
        impl._update_indexcache_topk_indices(rows)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(table, before, rtol=0, atol=0)
        table.copy_(before)

    def test_complete_rr_rows_and_storage(self):
        self.assertEqual(SFA_FULL_VISIBLE_TEMPLATE_BLOCK_SIZE, 128)
        self.assertEqual(SFA_INDEXER_SPARSE_COUNT, 2048)
        impl = self.make_impl()
        table = impl._full_visible_index_table
        self.assertEqual(table.shape, (2049, 2048))
        self.assertEqual(table.dtype, torch.int32)
        self.assertEqual(table.device, torch.device("cpu"))
        self.assertTrue(table.is_contiguous())
        self.assertTrue(torch.all(table[0] == -1))
        self.assertIs(table, impl._get_or_create_full_visible_index_table(torch.device("cpu")))
        for total in (1, 64, 128, 129, 192, 2048):
            with self.subTest(total=total):
                metadata = self.metadata(AscendAttentionState.PrefillNoCache, total, total)
                rows = impl._get_full_visible_topk_indices(metadata, total, torch.device("cpu"))
                self.assertEqual(rows.shape, (total, 1, 2048))
                self.assertEqual(rows.dtype, torch.int32)
                self.assertTrue(rows.is_contiguous())
                self.assertEqual(rows.untyped_storage().data_ptr(), table.untyped_storage().data_ptr())
                self.assertEqual(rows.storage_offset(), 2048)
                for visible, row in enumerate(rows[:, 0].tolist(), start=1):
                    self.assertEqual(row[:visible], self.rr_order(visible))
                    self.assertEqual(len(set(row[:visible])), visible)
                    self.assertEqual(row[visible:], [-1] * (2048 - visible))

    def test_short_cache_hit_total_visible_boundary(self):
        impl = self.make_impl()
        for total in (2048, 2049):
            with self.subTest(total=total):
                metadata = self.metadata(AscendAttentionState.PrefillCacheHit, total)
                rows = impl._get_full_visible_topk_indices(metadata, 64, torch.device("cpu"))
                if total == 2049:
                    self.assertIsNone(rows)
                    continue
                self.assertEqual(rows.shape, (64, 1, 2048))
                self.assertEqual(rows.storage_offset(), 1985 * 2048)
                self.assertEqual(
                    rows.untyped_storage().data_ptr(), impl._full_visible_index_table.untyped_storage().data_ptr()
                )
                for visible, row in enumerate(rows[:, 0].tolist(), start=1985):
                    self.assertEqual(row[:visible], self.rr_order(visible))
                    self.assertEqual(row[visible:], [-1] * (2048 - visible))

    def test_chunked_and_speculative_prefill_remain_ineligible(self):
        impl = self.make_impl()
        for state in (AscendAttentionState.ChunkedPrefill, AscendAttentionState.SpecDecoding):
            with self.subTest(state=state):
                metadata = self.metadata(state, 64)
                self.assertIsNone(impl._get_full_visible_topk_indices(metadata, 64, torch.device("cpu")))

    def forward_case(
        self, state, total, *, enabled=True, skip_topk=False, has_indexer=True, owner=True, query=64, impl=None
    ):
        """Run actual forward and cache methods, with explicit device boundaries."""
        impl = self.make_impl() if impl is None else impl
        self.config.enable_sfa_full_visible_index_bypass = enabled
        impl.skip_topk = skip_topk
        impl.has_indexer = has_indexer
        impl.use_index_cache = owner
        impl.layer_name = "model.layers.2.self_attn.attn"
        impl.preprocess_type = PreprocessType.NATIVE
        impl.q_lora_rank = 2
        impl.kv_lora_rank = 2
        impl.qk_rope_head_dim = 2
        impl.scale = 0.5
        impl.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=1))
        if not hasattr(impl, "topk_indices_buffer"):
            impl.topk_indices_buffer = torch.full((257, 2048), -17, dtype=torch.int32)
        table = impl._full_visible_index_table
        template_before = torch.empty_like(table)
        template_before.copy_(table)
        self.assert_disjoint_storage(table, template_before)
        self.assert_disjoint_storage(table, impl.topk_indices_buffer)
        if skip_topk:
            impl.topk_indices_buffer[:query, 0] = 7
        cached_before = impl.topk_indices_buffer.clone()
        metadata = self.metadata(state, total, query)
        hidden = torch.zeros(query, 4)
        output = torch.empty_like(hidden)
        main_cache = (torch.zeros(17, 128, 1, 2), torch.zeros(17, 128, 1, 2))
        indexer_cache = torch.full((17, 128, 1, 2), -37.0)
        impl.indexer = SimpleNamespace(k_cache=SimpleNamespace(kv_cache=(indexer_cache,)))
        keys = torch.arange(query * 2, dtype=torch.float32).reshape(query, 1, 2)
        parallel = SimpleNamespace(
            actual_seq_lengths_query=torch.tensor([query]),
            actual_seq_lengths_key=torch.tensor([total]),
            kv_slot_mapping=metadata.slot_mapping,
            gather_full_o_proj=False,
            topk_num_tokens=query,
        )
        impl._get_parallel_forward_context = MagicMock(return_value=parallel)
        impl._prepare_native_hidden_states = MagicMock(return_value=hidden)
        impl.fused_qkv_a_proj = MagicMock(return_value=(torch.zeros(query, 6),))
        impl.q_a_layernorm = MagicMock(side_effect=lambda q: q)
        impl.indexer_select_pre_process = MagicMock(return_value=(keys, None))
        impl.exec_kv = MagicMock(return_value=(main_cache[1], main_cache[0]))
        impl._prepare_kv_for_parallel = MagicMock(return_value=(keys, None, None, None))
        impl._q_proj_and_k_up_proj = MagicMock(return_value=(hidden, hidden))
        impl.rope_single = MagicMock(side_effect=lambda q, cos, sin: q)
        impl._record_query_gather_context = MagicMock()
        impl._store_parallel_kv = MagicMock(return_value=(main_cache[1], main_cache[0], keys))
        scored = torch.full((query, 1, 2048), -1, dtype=torch.int32)
        scored[:, :, 0] = 11  # Deliberately different from both table and cache.
        impl.indexer_select_post_process = MagicMock(return_value=scored)
        impl._execute_sparse_flash_attention_process = MagicMock(
            wraps=AscendSFAImpl._execute_sparse_flash_attention_process.__get__(impl)
        )
        impl._v_up_proj = MagicMock(side_effect=lambda x: x)
        impl._finalize_o_proj = MagicMock(return_value=output)
        events = MagicMock()
        for name in (
            "indexer_select_pre_process", "exec_kv", "_store_parallel_kv", "indexer_select_post_process",
            "_execute_sparse_flash_attention_process",
        ):
            events.attach_mock(getattr(impl, name), name)
        for name in ("_write_indexer_cache", "_update_indexcache_topk_indices", "_get_indexcache_topk_indices",
                     "_get_full_visible_topk_indices"):
            spy = MagicMock(wraps=getattr(AscendSFAImpl, name).__get__(impl))
            setattr(impl, name, spy)
            events.attach_mock(spy, name)

        def scatter_cpu(cache, slots, values):
            cache[slots.flatten()] = values
            return cache

        def native_consumer(**kwargs):
            received = kwargs["sparse_indices"]
            # Observe the cache at the consumption boundary, before returning.
            if owner and not skip_topk:
                torch.testing.assert_close(impl.topk_indices_buffer[:query], received[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(table, template_before, rtol=0, atol=0)
            self.assertEqual(kwargs["layout_query"], "TND")
            self.assertEqual(kwargs["layout_kv"], "PA_BSND")
            self.assertEqual(kwargs["sparse_mode"], 3)
            return hidden, torch.empty(0), torch.empty(0)

        with ExitStack() as stack:
            stack.enter_context(patch("vllm_ascend.attention.sfa_v1.DeviceOperator", BaseDeviceAdaptor))
            native = stack.enter_context(patch(
                "torch.ops._C_ascend.npu_sparse_flash_attention", create=True, side_effect=native_consumer
            ))
            for name in ("wait_for_kv_layer_from_connector", "notify_kv_cache_written",
                         "record_attention_compute_start", "maybe_save_kv_layer_to_connector"):
                spy = stack.enter_context(patch("vllm_ascend.attention.sfa_v1." + name))
                events.attach_mock(spy, name)
            scatter = stack.enter_context(patch(
                "vllm_ascend.attention.sfa_v1.torch_npu.npu_scatter_nd_update_", side_effect=scatter_cpu
            ))
            result = impl.forward(impl.layer_name, hidden, main_cache, metadata, output)

        native.assert_called_once()
        torch.testing.assert_close(table, template_before, rtol=0, atol=0)
        self.assertIs(result, output)
        impl.exec_kv.assert_called_once()
        impl._store_parallel_kv.assert_called_once()
        if has_indexer:
            impl.indexer_select_pre_process.assert_called_once()
            impl._write_indexer_cache.assert_called_once()
            scatter.assert_called_once()
            destination, slots, values = scatter.call_args.args
            self.assertEqual(
                destination.untyped_storage().data_ptr(), indexer_cache.untyped_storage().data_ptr()
            )
            self.assertTrue(torch.equal(slots.flatten(), metadata.slot_mapping))
            self.assertTrue(torch.equal(values, keys.view(query, 2)))
            expected_cache = torch.full_like(indexer_cache.view(-1, 2), -37)
            expected_cache[metadata.slot_mapping] = keys.view(query, 2)
            self.assertTrue(torch.equal(indexer_cache.view(-1, 2), expected_cache))
        else:
            impl.indexer_select_pre_process.assert_not_called()
            impl._write_indexer_cache.assert_not_called()
            scatter.assert_not_called()
        consumer = impl._execute_sparse_flash_attention_process
        consumer.assert_called_once()
        received = consumer.call_args.args[3]
        self.assertEqual(received.shape, (query, 1, 2048))
        self.assertEqual(received.dtype, torch.int32)
        self.assertTrue(received.is_contiguous())
        eligible = enabled and not skip_topk and total <= 2048 and state in (
            AscendAttentionState.PrefillNoCache, AscendAttentionState.PrefillCacheHit
        )
        if skip_topk:
            impl._get_full_visible_topk_indices.assert_not_called()
            impl.indexer_select_post_process.assert_not_called()
            impl._get_indexcache_topk_indices.assert_called_once_with(query)
            impl._update_indexcache_topk_indices.assert_not_called()
            self.assertTrue(torch.equal(received[:, 0], cached_before[:query]))
            self.assertEqual(
                received.untyped_storage().data_ptr(), impl.topk_indices_buffer.untyped_storage().data_ptr()
            )
        else:
            impl._get_full_visible_topk_indices.assert_called_once_with(metadata, query, hidden.device)
            impl._get_indexcache_topk_indices.assert_not_called()
            if eligible:
                impl.indexer_select_post_process.assert_not_called()
                table = impl._full_visible_index_table
                self.assertEqual(received.untyped_storage().data_ptr(), table.untyped_storage().data_ptr())
                self.assertEqual(received.storage_offset(), (total - query + 1) * 2048)
                for visible, row in enumerate(received[:, 0].tolist(), start=total - query + 1):
                    self.assertEqual(row[:visible], self.rr_order(visible))
                    self.assertEqual(row[visible:], [-1] * (2048 - visible))
            else:
                impl.indexer_select_post_process.assert_called_once()
                self.assertIs(received, scored)
            if owner:
                impl._update_indexcache_topk_indices.assert_called_once()
                self.assertIs(impl._update_indexcache_topk_indices.call_args.args[0], received)
                self.assertTrue(torch.equal(impl.topk_indices_buffer[:query], received[:, 0]))
                self.assertTrue(torch.equal(impl.topk_indices_buffer[query:], cached_before[query:]))
            else:
                impl._update_indexcache_topk_indices.assert_not_called()
                self.assertTrue(torch.equal(impl.topk_indices_buffer, cached_before))
        self.assertIs(native.call_args.kwargs["sparse_indices"], received)
        names = [event[0] for event in events.mock_calls]
        for name in ("exec_kv", "_store_parallel_kv", "notify_kv_cache_written", "record_attention_compute_start"):
            self.assertLess(names.index(name), names.index("_execute_sparse_flash_attention_process"))
        if has_indexer:
            route = "_get_indexcache_topk_indices" if skip_topk else "_get_full_visible_topk_indices"
            self.assertLess(names.index("indexer_select_pre_process"), names.index("_write_indexer_cache"))
            self.assertLess(names.index("_write_indexer_cache"), names.index(route))
        if owner and not skip_topk:
            self.assertLess(
                names.index("_update_indexcache_topk_indices"), names.index("_execute_sparse_flash_attention_process")
            )

    def test_forward_eligible_prefills_and_owning_cache(self):
        for state, total in ((AscendAttentionState.PrefillNoCache, 64),
                             (AscendAttentionState.PrefillCacheHit, 192),
                             (AscendAttentionState.PrefillCacheHit, 2048)):
            for owner in (False, True):
                with self.subTest(state=state, total=total, owner=owner):
                    self.forward_case(state, total, owner=owner)

    def test_forward_fallback_calls_scorer_exactly_once(self):
        for state, total, enabled in (
            (AscendAttentionState.PrefillNoCache, 64, False),
            (AscendAttentionState.PrefillCacheHit, 2048, False),
            (AscendAttentionState.PrefillCacheHit, 2049, True),
            (AscendAttentionState.ChunkedPrefill, 64, True),
            (AscendAttentionState.SpecDecoding, 64, True),
            (AscendAttentionState.DecodeOnly, 64, True),
        ):
            with self.subTest(state=state, total=total, enabled=enabled):
                self.forward_case(state, total, enabled=enabled)

    def test_forward_skip_topk_keeps_cached_route(self):
        for has_indexer in (False, True):
            with self.subTest(has_indexer=has_indexer):
                self.forward_case(AscendAttentionState.PrefillCacheHit, 192, skip_topk=True, has_indexer=has_indexer)

    def test_forward_absent_metadata_only_zeros_output(self):
        impl = self.make_impl()
        impl._compose_sfa_kv_cache = MagicMock(side_effect=AssertionError("profiling must not access cache"))
        output = torch.ones(64, 4)
        self.assertIs(impl.forward("layer", torch.empty_like(output), (), None, output), output)
        self.assertTrue(torch.all(output == 0))
        impl._compose_sfa_kv_cache.assert_not_called()
