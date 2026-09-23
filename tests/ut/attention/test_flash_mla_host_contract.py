# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests, not NPU/operator correctness evidence.

Run this file directly to avoid the NPU-wide pytest conftest. Real tensor
buffer operations run on CPU; the external package and NPU stream are mocked.
"""

import ast
import sys
import types
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names=None, **namespace):
    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    ast.fix_missing_locations(tree)
    module = types.ModuleType("_flash_host_test")
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(ROOT / path), "exec"), module.__dict__)
    return module


class FlashMLAHostContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.generations = 0
        package = types.ModuleType("cann_ops_transformer.ops")

        def metadata(lengths, heads, kv_heads, **kwargs):
            self.calls.append(("metadata", lengths.clone(), heads, kv_heads, kwargs))
            if lengths.device.type != "meta":
                self.generations += 1
            return torch.full((lengths.numel() * 8,), self.generations, dtype=torch.int32, device=lengths.device)

        def main(query, cache, **kwargs):
            self.calls.append(("main", query, cache, kwargs))
            output = torch.zeros(query.shape[1], query.shape[0], 512, dtype=query.dtype)
            lse = torch.zeros(query.shape[1], query.shape[0]) if kwargs["return_softmax_lse"] else torch.empty(0)
            return output, lse

        package.flash_mla_with_kvcache_metadata = metadata
        package.flash_mla_with_kvcache = main
        self.modules = patch.dict(sys.modules, {"cann_ops_transformer.ops": package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        cp = load_functions(
            "vllm_ascend/attention/context_parallel/common_cp.py",
            {"get_flash_dcp_local_seq_lens"},
            torch=torch,
        )
        self.api = load_functions(
            "vllm_ascend/attention/flash_mla.py",
            torch=torch,
            dataclass=dataclass,
            cdiv=lambda a, b: (a + b - 1) // b,
            MLA_FLASH_SUPPORTED_Q_HEADS={8, 12, 64, 96},
            DeviceMetadataStage=SimpleNamespace(ATTENTION=2),
            DeviceMetadataTask=lambda stage, run, group: SimpleNamespace(stage=stage, run=run, group_id=group),
            get_flash_dcp_local_seq_lens=cp.get_flash_dcp_local_seq_lens,
        )

    def builder(self, heads=8, deferred=False, dcp=1, rank=0):
        builder = SimpleNamespace(
            device="cpu",
            kv_cache_spec=SimpleNamespace(dtype=torch.bfloat16, block_size=128),
            kernel_block_size=128,
            decode_threshold=16,
            dcp_size=dcp,
            dcp_rank=rank,
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=16)),
            _device_metadata_enabled=deferred,
        )
        self.api._init_flash_attention_metadata(builder, SimpleNamespace(num_heads=heads, num_kv_heads=1))
        return builder

    @staticmethod
    def common(batch=1, query=2, padding=2, causal=True, kv=128000):
        tokens = batch * query
        return SimpleNamespace(
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens + padding,
            max_query_len=query,
            causal=causal,
            block_table_tensor=torch.arange(batch * 1000, dtype=torch.int32).view(batch, 1000),
            query_start_loc=torch.arange(0, tokens + 1, query, dtype=torch.int32),
            seq_lens=torch.full((batch,), kv, dtype=torch.int32),
            slot_mapping=torch.arange(tokens, dtype=torch.int64),
            positions=torch.arange(tokens),
        )

    @staticmethod
    def cache():
        # Both page and token axes strided, non-zero offset, one KV head.
        backing = torch.full((4 * 128 * 2 * 576 + 37,), -9, dtype=torch.bfloat16)
        cache = torch.as_strided(backing, (2, 128, 1, 576), (2 * 128 * 2 * 576, 2 * 576, 576, 1), 37)
        return backing, cache

    def test_head_batch_length_matrix(self):
        for heads in (8, 12, 64, 96):
            for batch, query, kv in (
                (1, 1, 100000),
                (29, 2, 128000),
                (30, 4, 100000),
                (31, 8, 128000),
                (32, 16, 128000),
            ):
                with self.subTest(heads=heads, batch=batch, query=query, kv=kv):
                    b = self.builder(heads)
                    flash = self.api._build_flash_attention_metadata(b, self.common(batch, query, kv=kv))
                    self.assertEqual(flash.query.shape, (batch * query + 2, heads, 576))
                    self.assertEqual(flash.cu[-1], batch * query + 2)
                    self.assertEqual(flash.used_q[-1], 0)
                    self.assertTrue(torch.equal(flash.cache_lens[:-1], torch.full((batch,), kv)))
                    self.assertTrue((flash.slots[-2:] == -1).all())
                    call = self.calls[-1]
                    self.assertEqual(call[2:4], (heads, 1))
                    self.assertEqual(call[-1]["max_seqlen_q"], -1)
                    self.assertEqual(call[-1]["max_seqlen_kv"], -1)
                    self.assertEqual(call[-1]["layout_q"], "TND")

    def test_stable_buffers_refresh_each_batch(self):
        builder, common = self.builder(), self.common()
        first = self.api._build_flash_attention_metadata(builder, common)
        old_schedule = first.schedule.clone()
        pointers = [x.data_ptr() for x in (first.schedule, first.cu, first.block_table, first.query)]
        common.seq_lens.fill_(127)
        common.block_table_tensor.add_(7)
        second = self.api._build_flash_attention_metadata(builder, common)
        self.assertIs(first, second)
        self.assertEqual(
            pointers, [x.data_ptr() for x in (second.schedule, second.cu, second.block_table, second.query)]
        )
        self.assertFalse(torch.equal(old_schedule, second.schedule))
        self.assertEqual(second.cache_lens[0], 127)
        self.assertEqual(second.block_table[0, 0], 7)

    def test_deferred_metadata_reads_device_lengths_at_execution(self):
        builder, common = self.builder(deferred=True), self.common()
        flash = self.api._build_flash_attention_metadata(builder, common)
        common.seq_lens.fill_(129)
        self.assertEqual(flash.cache_lens[0], 0)
        builder._device_metadata_tasks[0].run()
        self.assertEqual(flash.cache_lens[0], 129)
        self.assertEqual(builder._device_metadata_tasks[0].group_id, id(flash.schedule))

    def test_main_keeps_cache_alias_and_contract(self):
        for causal in (True, False):
            flash = self.api._build_flash_attention_metadata(self.builder(), self.common(causal=causal))
            backing, cache = self.cache()
            before = backing.clone()
            output, lse = self.api.run_flash_mla(flash.query, cache, flash, 0.125)
            _, q, consumed, kwargs = self.calls[-1]
            self.assertIs(consumed, cache)
            self.assertIs(q, flash.query)
            self.assertEqual(cache.storage_offset(), 37)
            self.assertTrue(torch.equal(before, backing))
            self.assertEqual((output.shape, lse.shape), ((8, 4, 512), (0,)))
            self.assertEqual((kwargs["layout_q"], kwargs["layout_kv"], kwargs["layout_out"]), ("TND", "PA_BBND", "NTD"))
            self.assertEqual((kwargs["max_seqlen_q"], kwargs["max_seqlen_kv"]), (-1, -1))
            self.assertIs(kwargs["metadata"], flash.schedule)
            self.assertIs(kwargs["cu_seqlens_q"], flash.cu)
            self.assertIs(kwargs["attn_mask"], flash.attn_mask if causal else None)
            self.assertEqual(kwargs["mask_mode"], 3 if causal else 0)
            self.assertEqual(flash.attn_mask[0, 0], 0)
            self.assertEqual(flash.attn_mask[0, 1], 1)
            self.assertEqual(flash.attn_mask[1, 0], 0)

    def test_inactive_request_and_padding_do_not_write(self):
        common = self.common(batch=2)
        common.seq_lens[1] = 0
        flash = self.api._build_flash_attention_metadata(self.builder(), common)
        self.assertEqual(flash.used_q.tolist(), [2, 0, 0])
        self.assertEqual(flash.slots.tolist(), [0, 1, -1, -1, -1, -1])
        self.assertEqual(flash.token_live.tolist(), [True, True, False, False, False, False])

    def test_dcp_history_current_boundary_preparation(self):
        # Preparatory helpers only: configuration still gates runtime to DCP=1.
        common = self.common(query=3, kv=321)
        flash = self.api._build_flash_attention_metadata(self.builder(dcp=8, rank=0), common)
        self.assertTrue(flash.split_kv)
        self.assertEqual(flash.cache_lens[0], 48)
        self.assertEqual(flash.current_cache.shape[1:], (128, 1, 576))
        self.assertEqual(flash.current_slots.tolist(), [0, 1, 2, -1, -1])
        _, cache = self.cache()
        self.api.run_flash_mla(flash.query, cache, flash, 1)
        history = self.calls[-1][-1]
        self.assertEqual(history["mask_mode"], 0)
        self.assertTrue(history["return_softmax_lse"])
        self.api.run_flash_mla(flash.query[:, :8].contiguous(), flash.current_cache, flash, 1, current=True)
        current = self.calls[-1][-1]
        self.assertEqual(current["mask_mode"], 3)
        self.assertIs(current["cache_seqlens"], flash.used_q)
        self.assertIs(current["metadata"], flash.current_schedule)

    def test_reject_incompatible_cache_or_heads(self):
        for shape in ((2, 1, 128, 576), (2, 16, 1, 576)):
            with self.assertRaises(ValueError):
                self.api.validate_flash_cache(torch.empty(shape))
        with self.assertRaises(ValueError):
            self.builder(heads=24)

    def test_eager_prefill_buffers_not_retained(self):
        b = self.builder()
        self.api._build_flash_attention_metadata(b, self.common(query=17))
        self.assertFalse(b._flash_buffers)
        self.api._build_flash_attention_metadata(b, self.common(query=17, causal=False))
        self.assertEqual(len(b._flash_buffers), 1)

    def test_executor_ownership_nested_target_draft_and_exception(self):
        # Execute #16468's context body with fake stream executors; no NPU.
        var = ContextVar("host_executor", default=None)
        lifecycle = load_functions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"device_metadata_context"},
            contextmanager=contextmanager,
            DeviceMetadataExecutor=object,
            _device_metadata_executor=var,
        )
        released = []

        def executor(name):
            owner = SimpleNamespace(submission_in_flight=True)
            owner.release = lambda: released.append(name)
            return owner

        target, draft = executor("target"), executor("draft")
        with self.assertRaisesRegex(RuntimeError, "consumer"), lifecycle.device_metadata_context(target):
            with lifecycle.device_metadata_context(target):
                self.assertIs(var.get(), target)
            self.assertFalse(released)
            with lifecycle.device_metadata_context(draft):
                self.assertIs(var.get(), draft)
            self.assertIs(var.get(), target)
            self.assertEqual(released, ["draft"])
            raise RuntimeError("consumer")
        self.assertIsNone(var.get())
        self.assertEqual(released, ["draft", "target"])

    def test_graph_and_dspark_wiring_contract(self):
        runner = (ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text(encoding="utf-8")
        capture = (ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py").read_text(encoding="utf-8")
        draft = (ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py").read_text(encoding="utf-8")
        self.assertIn("device_metadata_context(self.device_metadata_executor)", runner)
        self.assertIn("device_metadata_context(self.model_runner.device_metadata_executor)", capture)
        self.assertIn("dflash_cudagraph.build_attn_metadata = original", draft)
        self.assertIn('if getattr(metadata, "flash", None) is not None:', draft)
        propose = next(n for n in ast.walk(ast.parse(draft)) if isinstance(n, ast.FunctionDef) and n.name == "propose")
        self.assertTrue(
            any(
                isinstance(n, ast.With)
                and any(
                    isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == "device_metadata_context"
                    for item in n.items
                )
                for n in ast.walk(propose)
            )
        )
        attn_utils = (ROOT / "vllm_ascend/worker/v2/attn_utils.py").read_text(encoding="utf-8")
        self.assertIn("assert not torch.npu.is_current_stream_capturing()", attn_utils)
        self.assertLess(
            attn_utils.index("executor.submit(device_metadata_tasks)"),
            attn_utils.index("executor.wait(task.stage, task.group_id)"),
        )

    def test_metadata_does_not_retain_cpu_lengths(self):
        # A large KV length is an int32 metadata input, not a host list bound.
        b, common = self.builder(), self.common()
        common.seq_lens_cpu = torch.tensor([17])
        flash = self.api._build_flash_attention_metadata(b, common)
        self.assertEqual(flash.cache_lens[0], 128000)
        self.assertEqual(self.calls[-1][1][0], 128000)

    def test_accept_reject_refresh_uses_device_lengths_and_masks_padding(self):
        b, common = self.builder(deferred=True), self.common(batch=3, query=5, padding=3, causal=False)
        common.seq_lens_cpu = torch.tensor([900, 900, 900])
        first = self.api._build_flash_attention_metadata(b, common)
        tensors = (
            first.schedule,
            first.query,
            first.cu,
            first.used_q,
            first.cache_lens,
            first.block_table,
            first.slots,
            first.positions,
            first.token_live,
        )
        pointers = [t.data_ptr() for t in tensors]
        for lengths, boundaries in (
            ([25, 19, 0], [0, 5, 10, 10]),
            ([21, 0, 0], [0, 5, 5, 5]),
            ([26, 20, 9], [0, 5, 10, 15]),
        ):
            flash = self.api._build_flash_attention_metadata(b, common)
            # Simulate the upstream device acceptance/rejection correction
            # after Python built metadata, before its producer task executes.
            common.seq_lens.copy_(torch.tensor(lengths))
            common.query_start_loc.copy_(torch.tensor(boundaries))
            common.block_table_tensor.add_(1)
            b._device_metadata_tasks[0].run()
            self.assertIs(flash, first)
            self.assertEqual([t.data_ptr() for t in tensors], pointers)
            self.assertEqual(flash.cache_lens.tolist(), lengths + [0])
            self.assertEqual(flash.cu.tolist(), boundaries + [18])
            live = boundaries[-1]
            self.assertEqual(flash.token_live.tolist(), [True] * live + [False] * (18 - live))
            self.assertEqual(flash.slots[live:].tolist(), [-1] * (18 - live))
            self.assertEqual(self.calls[-1][1].tolist(), lengths + [0])
            self.assertEqual(self.calls[-1][-1]["mask_mode"], 0)

    def test_target_and_draft_own_distinct_schedules_and_cache_views(self):
        target = self.api._build_flash_attention_metadata(self.builder(), self.common())
        draft = self.api._build_flash_attention_metadata(self.builder(), self.common(causal=False))
        _, target_cache = self.cache()
        _, draft_cache = self.cache()
        for name in ("schedule", "query", "cache_lens", "block_table", "slots"):
            self.assertNotEqual(getattr(target, name).data_ptr(), getattr(draft, name).data_ptr())
        for flash, cache in ((target, target_cache), (draft, draft_cache)):
            self.api.run_flash_mla(flash.query, cache, flash, 1)
            self.assertIs(self.calls[-1][2], cache)
            self.assertIs(self.calls[-1][-1]["metadata"], flash.schedule)

    def test_missing_operator_fails_without_fia_fallback(self):
        with (
            patch.dict(sys.modules, {"cann_ops_transformer.ops": types.ModuleType("cann_ops_transformer.ops")}),
            self.assertRaises(ImportError),
        ):
            self.api._build_flash_attention_metadata(self.builder(), self.common())

    def test_external_package_is_used_even_if_in_tree_bindings_exist(self):
        in_tree = SimpleNamespace(
            flash_mla_with_kvcache_metadata=Mock(side_effect=AssertionError("in-tree metadata called")),
            flash_mla_with_kvcache=Mock(side_effect=AssertionError("in-tree FlashMLA called")),
        )
        with patch.object(torch.ops, "_C_ascend", in_tree):
            flash = self.api._build_flash_attention_metadata(self.builder(), self.common())
            _, cache = self.cache()
            self.api.run_flash_mla(flash.query, cache, flash, 1)
            self.assertEqual([call[0] for call in self.calls], ["metadata", "metadata", "main"])
            with patch.dict(sys.modules, {"cann_ops_transformer.ops": types.ModuleType("cann_ops_transformer.ops")}):
                with self.assertRaises(ImportError):
                    self.api._build_flash_attention_metadata(self.builder(), self.common())
                with self.assertRaises(ImportError):
                    self.api.run_flash_mla(flash.query, cache, flash, 1)
        in_tree.flash_mla_with_kvcache_metadata.assert_not_called()
        in_tree.flash_mla_with_kvcache.assert_not_called()

    def test_reject_overlapping_page_or_channel_views(self):
        raw = torch.empty(2 * 128 * 576)
        for strides in ((576, 576, 576, 1), (128 * 576, 1, 576, 1), (128 * 576, 576, 576, 0)):
            with self.subTest(strides=strides), self.assertRaises(ValueError):
                self.api.validate_flash_cache(torch.as_strided(raw, (2, 128, 1, 576), strides))

    def test_dspark_context_writer_preserves_strides_and_nope(self):
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        impl = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
        method = next(
            node for node in impl.body if isinstance(node, ast.FunctionDef) and node.name == "exec_kv_prefill"
        )
        method.body = [method.body[0]]  # The opt-in branch; no unrelated NPU imports.
        method.args.args = [ast.arg(arg=arg.arg) for arg in method.args.args]
        method.args.kwonlyargs = [ast.arg(arg=arg.arg) for arg in method.args.kwonlyargs]
        method.returns = None
        backing, cache = self.cache()
        expected = backing.clone()
        expected_cache = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
        data = torch.arange(3 * 576, dtype=torch.float32).to(torch.bfloat16).view(3, 576)
        slots = torch.tensor([0, 128, -1])
        expected_cache[0, 0, 0] = data[0]
        expected_cache[1, 0, 0] = data[1]

        def scatter(**kwargs):
            self.assertEqual(kwargs["key_cache"].stride(), cache.stride())
            for index, slot in enumerate(kwargs["slot_mapping"].tolist()):
                if slot >= 0:
                    kwargs["key_cache"][slot // 128, slot % 128] = kwargs["key"][index]
                    kwargs["value_cache"][slot // 128, slot % 128] = kwargs["value"][index]

        scope = {
            "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            "validate_flash_cache": self.api.validate_flash_cache,
            "torch_npu": SimpleNamespace(npu_scatter_pa_kv_cache=scatter),
        }
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "writer", "exec"), scope)
        scope["exec_kv_prefill"](
            SimpleNamespace(use_mla_rope=False, kv_a_layernorm=lambda x: x), data, None, None, cache, slots
        )
        self.assertTrue(torch.equal(backing, expected))


if __name__ == "__main__":
    unittest.main()
