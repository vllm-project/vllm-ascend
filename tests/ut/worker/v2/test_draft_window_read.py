# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_read import (
    DirectPrefixSource,
    materialize_direct_prefix,
)


class TestCanonicalRead(unittest.TestCase):
    def setUp(self):
        self.table = torch.arange(24, dtype=torch.int32).reshape(3, 8)
        self.owner = SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=self.table)],
            num_blocks=SimpleNamespace(np=np.array([[8, 8, 8]])),
            kernel_block_sizes=[4],
        )
        self.mapping = np.array([2, 0])

    def read(self, ends=(21, 11), starts=(12, 4), upper=(24, 15)):
        exact = torch.tensor(ends, dtype=torch.int32)
        return materialize_direct_prefix(
            DirectPrefixSource.from_block_tables(self.owner, 0, self.mapping),
            exact_cpu=exact,
            exact_device=exact,
            upper_cpu=torch.tensor(upper),
            starts_cpu=torch.tensor(starts),
            padded_rows=3,
        )

    def test_canonical_projection_preserves_upper_and_padding(self):
        view = self.read()
        self.assertEqual(view.block_table.tolist(), [[19, 20, 21], [1, 2, 0], [0, 0, 0]])
        self.assertEqual(view.seq_lens_cpu.tolist(), [9, 7, 0])
        self.assertEqual(view.seq_lens_cpu_upper_bound.tolist(), [12, 11, 0])
        self.assertNotEqual(view.block_table.untyped_storage().data_ptr(), self.table.untyped_storage().data_ptr())
        self.assertEqual(self.table.tolist(), torch.arange(24).reshape(3, 8).tolist())

    def test_owned_start_transfer_retains_actual_source_and_destination(self):
        retained: list[torch.Tensor] = []
        copied: list[tuple[torch.Tensor, torch.Tensor]] = []
        original_copy = torch.Tensor.copy_

        def copy(destination, source, *args, **kwargs):
            if kwargs.get("non_blocking"):
                self.assertTrue(any(value is source for value in retained))
                self.assertTrue(any(value is destination for value in retained))
                copied.append((source, destination))
            return original_copy(destination, source, *args, **kwargs)

        exact = torch.tensor([21, 11], dtype=torch.int32)
        with patch.object(torch.Tensor, "copy_", copy):
            view = materialize_direct_prefix(
                DirectPrefixSource.from_block_tables(self.owner, 0, self.mapping),
                exact_cpu=exact,
                exact_device=exact,
                upper_cpu=exact + 3,
                starts_cpu=torch.tensor([12, 4], dtype=torch.int32),
                padded_rows=3,
                retain=lambda *values: retained.extend(values),
            )
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0][0].dtype, torch.int64)
        self.assertEqual(view.seq_lens.tolist(), [9, 7, 0])
        self.assertTrue(any(value is view.block_table for value in retained))
        self.assertTrue(any(value is view.seq_lens for value in retained))

    def test_retention_failure_prevents_owned_start_copy(self):
        calls = []

        def retain(*values):
            calls.append(values)
            if len(calls) == 2:
                raise RuntimeError("retain destination failed")

        exact = torch.tensor([21, 11], dtype=torch.int32)
        with patch.object(torch.Tensor, "copy_") as copy:
            with self.assertRaisesRegex(RuntimeError, "retain destination"):
                materialize_direct_prefix(
                    DirectPrefixSource.from_block_tables(self.owner, 0, self.mapping),
                    exact_cpu=exact,
                    exact_device=exact,
                    upper_cpu=exact + 3,
                    starts_cpu=torch.tensor([12, 4], dtype=torch.int32),
                    padded_rows=3,
                    retain=retain,
                )
            copy.assert_not_called()

    def test_reorder_in_place_and_same_row_replacement(self):
        first = self.read()
        self.mapping[:] = [0, 2]
        self.table[0] = torch.arange(100, 108)
        second = self.read()
        self.assertEqual(second.block_table[0].tolist(), [103, 104, 105])
        self.assertEqual(first.block_table[0].tolist(), [19, 20, 21])

    def test_valid_count_cannot_be_replaced_by_capacity(self):
        self.owner.num_blocks.np[0, 2] = 5
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.read()

    def test_upper_is_not_used_as_exact(self):
        self.owner.num_blocks.np[0, 2] = 6
        view = self.read(upper=(31, 15))
        self.assertEqual(view.seq_lens.tolist(), [9, 7, 0])
        self.assertEqual(view.seq_lens_cpu_upper_bound.tolist(), [19, 11, 0])

    def test_current_prepared_rows_preserve_reorder_and_strided_sources(self):
        for strided in (False, True):
            with self.subTest(strided=strided):
                table = self.table if not strided else torch.arange(48, dtype=torch.int32).reshape(3, 16)[:, ::2]
                before = table.clone()
                source = DirectPrefixSource(table, (2, 0), (8, 8), 4)
                exact = torch.tensor([21, 11], dtype=torch.int32)
                view = materialize_direct_prefix(
                    source,
                    exact_cpu=exact,
                    exact_device=exact,
                    upper_cpu=torch.tensor([24, 15]),
                    starts_cpu=torch.tensor([12, 4]),
                    padded_rows=3,
                    row_indices=torch.tensor([2, 0]),
                )
                expected = torch.stack(
                    (
                        before[2, 3:6],
                        torch.cat((before[0, 1:3], torch.zeros(1, dtype=table.dtype))),
                        torch.zeros(3, dtype=table.dtype),
                    )
                )
                torch.testing.assert_close(view.block_table, expected, rtol=0, atol=0)
                torch.testing.assert_close(table, before, rtol=0, atol=0)
                self.assertEqual(view.seq_lens.tolist(), [9, 7, 0])

    def test_prepared_row_mapping_requires_integer_current_request_extent(self):
        source = DirectPrefixSource.from_block_tables(self.owner, 0, self.mapping)
        exact = torch.tensor([21, 11], dtype=torch.int32)
        for rows in (torch.tensor([2]), torch.tensor([2.0, 0.0])):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "Prepared Draft row mapping"):
                materialize_direct_prefix(
                    source,
                    exact_cpu=exact,
                    exact_device=exact,
                    upper_cpu=exact + 3,
                    starts_cpu=torch.tensor([12, 4]),
                    padded_rows=3,
                    row_indices=rows,
                )

    def test_typed_application_preserves_query_and_write_identity(self):
        slots, positions, query = object(), object(), object()
        common = SimpleNamespace(slot_mapping=slots, positions=positions, query_start_loc=query)
        self.read().apply(common)
        self.assertIs(common.slot_mapping, slots)
        self.assertIs(common.positions, positions)
        self.assertIs(common.query_start_loc, query)

    def test_reject_unrepresentable_start_and_inactive_mapping(self):
        with self.assertRaisesRegex(ValueError, "intra-block"):
            self.read(starts=(13, 4))
        self.mapping[0] = -1
        with self.assertRaisesRegex(ValueError, "inactive"):
            self.read()

    def test_continuing_full_row_and_window_row_share_batch(self):
        view = self.read(starts=(0, 4))
        self.assertEqual(view.seq_lens.tolist(), [21, 7, 0])
        self.assertEqual(view.block_table[0, :6].tolist(), [16, 17, 18, 19, 20, 21])

    def sequence_length_selector(self):
        # Run the complete shared selector without initializing a native backend.
        path = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/utils.py"
        node = next(
            node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) == "_select_seq_lens"
        )
        scope = {"CrossAttentionSpec": type("CrossAttentionSpec", (), {}), "_is_glm_model": lambda config: False}
        tree = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
        return scope["_select_seq_lens"]

    def select_parallel_lengths(self, common):
        return self.sequence_length_selector()(
            common,
            kv_cache_spec=None,
            speculative_config=SimpleNamespace(parallel_drafting=True, use_dspark=lambda: False),
            vllm_config=SimpleNamespace(model_config=object()),
        )

    def test_parallel_builder_reuses_the_current_window_exact_snapshot(self):
        view = self.read()
        common = SimpleNamespace(num_reqs=2)
        view.apply(common)
        selected = self.select_parallel_lengths(common)
        # Generic parallel drafting otherwise reselects device lengths and
        # synchronizes again at the GQA builder's tolist().
        self.assertEqual(selected.data_ptr(), view.seq_lens_cpu.data_ptr())
        self.assertNotEqual(selected.data_ptr(), view.seq_lens.data_ptr())
        self.assertEqual(selected.tolist(), [9, 7])

    def test_parallel_off_does_not_trust_an_uncertified_cpu_mirror(self):
        common = SimpleNamespace(
            num_reqs=2,
            seq_lens=torch.tensor([9, 7], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([99, 99], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([99, 99], dtype=torch.int32),
        )
        self.assertIs(self.select_parallel_lengths(common), common.seq_lens)
        self.assertFalse(hasattr(common, "_draft_window_read"))

    def test_glm_dspark_off_retains_official_cpu_choice_without_copy(self):
        selector = self.sequence_length_selector()
        selector.__globals__["_is_glm_model"] = lambda config: True
        for use_v2 in (False, True):
            with self.subTest(use_v2=use_v2):
                common = SimpleNamespace(
                    num_reqs=1,
                    seq_lens=torch.tensor([1151], dtype=torch.int32),
                    _seq_lens_cpu=torch.tensor([1152], dtype=torch.int32),
                    seq_lens_cpu=None,
                )
                with patch.object(torch.Tensor, "to", side_effect=AssertionError("OFF must not add an exact copy")):
                    selected = selector(
                        common,
                        None,
                        SimpleNamespace(parallel_drafting=True, use_dspark=lambda: True),
                        SimpleNamespace(model_config=object(), use_v2_model_runner=use_v2),
                    )
                self.assertEqual(selected.data_ptr(), common._seq_lens_cpu.data_ptr())
                self.assertEqual(selected.tolist(), [1152])
                self.assertFalse(hasattr(common, "_draft_window_read"))

    def test_glm_dspark_on_full_and_window_reads_keep_certified_exact(self):
        selector = self.sequence_length_selector()
        selector.__globals__["_is_glm_model"] = lambda config: True
        for starts in ((0, 4), (12, 4)):
            with self.subTest(starts=starts):
                view = self.read(starts=starts)
                common = SimpleNamespace(num_reqs=2)
                view.apply(common)
                selected = selector(
                    common,
                    None,
                    SimpleNamespace(parallel_drafting=True, use_dspark=lambda: True),
                    SimpleNamespace(model_config=object(), use_v2_model_runner=True),
                )
                self.assertEqual(selected.data_ptr(), view.seq_lens_cpu.data_ptr())
                self.assertEqual(selected.tolist(), [21 - starts[0], 7])
                self.assertNotEqual(selected.tolist(), view.seq_lens_cpu_upper_bound[:2].tolist())

    def test_replacing_a_window_read_operand_invalidates_its_cpu_certificate(self):
        for field in ("seq_lens", "_seq_lens_cpu", "seq_lens_cpu", "block_table_tensor"):
            with self.subTest(field=field):
                common = SimpleNamespace(num_reqs=2)
                self.read().apply(common)
                setattr(common, field, getattr(common, field).clone())
                with self.assertRaisesRegex(ValueError, "replaced"):
                    self.select_parallel_lengths(common)


if __name__ == "__main__":
    unittest.main()
