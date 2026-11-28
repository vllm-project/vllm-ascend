# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only dispatch tests; generated NPU kernels are mocked."""

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("tilelang.ascend.language")
from vllm_ascend.ops.tilelang_recoverssm import commit_kernel as ck
from vllm_ascend.ops.tilelang_recoverssm import compact as compact_module
from vllm_ascend.ops.tilelang_recoverssm import full_backend as fb
from vllm_ascend.ops.tilelang_recoverssm import state_commit as state_module


def config(align=True, conv=False, dtype="bfloat16", window=4):
    return (
        8,
        12,
        19,
        24,
        window,
        5120,
        3,
        16,
        (8, (1,), "int32"),
        dtype,
        "bfloat16",
        (16384, 128, 1),
        (window * 257, 257, 1),
        align,
        conv,
    )


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.compiles = []
        ck.compiled_commit.cache_clear()

        def state_factory(*cfg, **kw):
            row = kw.get("commit_lens_row", 0)
            return SimpleNamespace(kind={0: "single", 1: "plain", 2: "boundary"}[row], cfg=cfg, row=row)

        def compile_op(op, **kw):
            self.compiles.append(op)
            return lambda *args: self.events.append((op.kind, args))

        self.stack = __import__("contextlib").ExitStack()
        self.stack.enter_context(mock.patch.object(ck.tilelang, "compile", side_effect=compile_op))
        self.stack.enter_context(mock.patch.object(state_module, "state_commit_kernel", side_effect=state_factory))
        self.stack.enter_context(
            mock.patch.object(
                compact_module, "compact_kernel", side_effect=lambda *cfg, **kw: SimpleNamespace(kind="compact")
            )
        )
        self.stack.enter_context(
            mock.patch.object(ck, "commit_kernel", side_effect=lambda *cfg: SimpleNamespace(kind="generic"))
        )
        self.addCleanup(self.stack.close)
        self.addCleanup(ck.compiled_commit.cache_clear)

    def test_aligned_split_uses_disjoint_rows_and_aligned_destinations(self):
        fn = ck.compiled_commit(*config(), split_boundary=True)
        fn("ST", "CT", "Indices", "Plan")
        self.assertEqual([op.row for op in self.compiles], [1, 2])
        self.assertEqual([op.cfg[13] for op in self.compiles], [False, True])
        self.assertEqual([name for name, _ in self.events], ["plain", "boundary"])
        self.assertTrue(all(args == ("ST", "CT", "Indices", "Plan") for _, args in self.events))

    def test_default_keeps_single_launch(self):
        ck.compiled_commit(*config())("x")
        self.assertEqual([name for name, _ in self.events], ["single"])

    def test_unaligned_does_not_add_empty_boundary_launch(self):
        ck.compiled_commit(*config(align=False), split_boundary=True)("x")
        self.assertEqual([name for name, _ in self.events], ["single"])

    def test_pure_tilelang_compacts_once_before_both_state_paths(self):
        ck.compiled_commit(*config(conv=True), split_boundary=True)("x")
        self.assertEqual([name for name, _ in self.events], ["compact", "plain", "boundary"])

    def test_fp32_falls_back_without_duplicate_commit(self):
        ck.compiled_commit(*config(dtype="float32"), split_boundary=True)("x")
        self.assertEqual([name for name, _ in self.events], ["generic"])

    def test_long_window_falls_back_without_duplicate_commit(self):
        ck.compiled_commit(*config(window=6), split_boundary=True)("x")
        self.assertEqual([name for name, _ in self.events], ["generic"])

    def test_changed_runtime_inputs_do_not_change_factory_key(self):
        first = ck.compiled_commit(*config(), split_boundary=True)
        for plan_values in ([0, 0], [1, 3], [4, 2]):
            first(plan_values)
        self.assertIs(first, ck.compiled_commit(*config(), split_boundary=True))
        self.assertEqual(len(self.compiles), 2)

    def test_row_group_control_reaches_both_filtered_factories(self):
        # Inspect actual factory kwargs independently from the runtime inputs.
        with mock.patch.object(
            state_module,
            "state_commit_kernel",
            side_effect=lambda *cfg, **kw: SimpleNamespace(kind="state", group=kw["row_group"]),
        ):
            ck.compiled_commit(*config(), split_boundary=True, row_group=2)("x")
        self.assertEqual([op.group for op in self.compiles], [2, 2])


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.ctx = SimpleNamespace(conv_shared_dim_stride=1)
        self.indices = SimpleNamespace(numel=lambda: 8)
        self.stack = __import__("contextlib").ExitStack()
        self.original_compact = fb._compact_with_triton

        def prepare(*args):
            return ("plan_cfg",), ("plan_input",), config(conv=args[-1]), ("state_input",)

        self.prep = self.stack.enter_context(mock.patch.object(fb, "prepare_step", side_effect=prepare))
        self.stack.enter_context(
            mock.patch.object(ck, "compiled_plan", side_effect=lambda *cfg: lambda *args: self.events.append("plan"))
        )
        self.commit = self.stack.enter_context(
            mock.patch.object(
                ck, "compiled_commit", side_effect=lambda *cfg, **kw: lambda *args: self.events.append("state_group")
            )
        )
        self.compact = self.stack.enter_context(
            mock.patch.object(
                fb, "_compact_with_triton", side_effect=lambda *args: self.events.append("triton_compact")
            )
        )
        self.addCleanup(self.stack.close)

    def test_hybrid_does_not_compact_twice(self):
        fb.commit_step(self.ctx, "accepted", self.indices, "offsets", compact_backend="triton", split_boundary=True)
        self.assertEqual(self.events, ["plan", "triton_compact", "state_group"])
        self.assertFalse(self.prep.call_args.args[-1])
        self.assertFalse(self.commit.call_args.args[14])
        self.assertTrue(self.commit.call_args.kwargs["split_boundary"])

    def test_pure_tilelang_leaves_compact_inside_compiled_group(self):
        fb.commit_step(self.ctx, "accepted", self.indices, "offsets", compact_backend="tilelang")
        self.assertEqual(self.events, ["plan", "state_group"])
        self.assertTrue(self.commit.call_args.args[14])
        self.compact.assert_not_called()

    def test_state_only_never_compacts(self):
        fb.commit_step(self.ctx, "accepted", self.indices, "offsets", commit_conv=False, compact_backend="triton")
        self.assertEqual(self.events, ["plan", "state_group"])
        self.assertFalse(self.commit.call_args.args[14])

    def test_known_conv_stride_is_forwarded_without_reading_device_data(self):
        fb.commit_step(self.ctx, "accepted", self.indices, "offsets")
        self.assertEqual(self.commit.call_args.kwargs["conv_dim_stride"], 1)

    def test_empty_batch_is_noop(self):
        fb.commit_step(self.ctx, "accepted", SimpleNamespace(numel=lambda: 0), "offsets")
        self.prep.assert_not_called()
        self.assertEqual(self.events, [])

    def test_invalid_backend_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            fb.commit_step(self.ctx, "accepted", self.indices, "offsets", compact_backend="unknown")
        self.assertEqual(self.events, [])

    def test_shared_compact_receives_original_pool_tables_and_plan_views(self):
        op = mock.Mock()
        module_name = "vllm_ascend.ops.triton.mamba.compact_conv_state"
        ctx = SimpleNamespace(
            layer_caches=[SimpleNamespace(conv="original_conv_view")],
            _conv_cache_address_stride_table=["ptrs", "page_stride", "channel_stride", "time_stride"],
            commit_plan=SimpleNamespace(
                commit_lens="accepted",
                final_state_indices="final",
                boundary_state_indices="boundary",
                boundary_recovery_lens="cut",
            ),
            conv_history_len=3,
            conv_shared_dim_stride=1,
        )
        with mock.patch.dict(sys.modules, {module_name: SimpleNamespace(compact_conv_state=op)}):
            self.original_compact(ctx, "indices", True)
        op.assert_called_once_with(
            "original_conv_view",
            "ptrs",
            "page_stride",
            "channel_stride",
            "time_stride",
            "indices",
            "accepted",
            "final",
            "boundary",
            "cut",
            conv_history_len=3,
            align_mode=True,
            dim_stride=1,
            null_block_id=0,
        )


class GeometryTests(unittest.TestCase):
    def test_production_bf16_row_is_one_dma_task(self):
        self.assertEqual(compact_module.compact_geometry(5120, 3, "bfloat16", True), (5120, 122880))

    def test_larger_or_fp32_rows_split_within_budget(self):
        for channels, dtype in ((8192, "bfloat16"), (5120, "float32")):
            tile, ub = compact_module.compact_geometry(channels, 3, dtype, True)
            self.assertLess(tile, channels)
            self.assertLessEqual(ub, 128 * 1024)
            self.assertEqual(tile % 256, 0)

    def test_no_boundary_uses_less_ub(self):
        self.assertEqual(compact_module.compact_geometry(5120, 3, "bfloat16", False), (5120, 61440))

    def test_state_group_keeps_all_physical_workers_busy(self):
        self.assertEqual(state_module.state_task_geometry(32, 12, 24, 128, 64), (4, 6, 64))
        self.assertEqual(state_module.state_task_geometry(1, 1, 3, 128, 64), (1, 3, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
