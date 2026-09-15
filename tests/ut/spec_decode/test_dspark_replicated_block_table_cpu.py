# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for MRv1 replicated GQA tables, without loading NPU modules."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def load_proposer():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/dspark_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDSparkProposer")
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_build_replicated_block_table", "set_per_group_attn_metadata"}
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["AscendDSparkProposer"]


class TestReplicatedBlockTable(unittest.TestCase):
    def test_batch_transitions_preserve_storage_and_target(self):
        for replication in (2, 8):
            for subblocks in (1, 3):
                with self.subTest(replication=replication, subblocks=subblocks):
                    proposer = load_proposer()()
                    proposer.device = torch.device("cpu")
                    proposer.max_batch_size = 4
                    proposer.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=24 * replication))
                    proposer._per_group_replication_sizes = {0: replication}
                    proposer._per_group_manager_block_sizes = {0: 4 * subblocks}
                    proposer._per_group_kernel_block_sizes = {0: 4}
                    proposer._replicated_block_table_storage = {}
                    proposer._replicated_block_table_arange = {}
                    proposer._per_group_block_tables = {}
                    proposer._per_group_slot_mappings = {}
                    proposer._per_group_block_table_buffers = {}
                    target = torch.arange(12, 36, dtype=torch.int32).reshape(4, 6)
                    original = target.clone()
                    slots = torch.full((16,), 777, dtype=torch.int32)
                    proposer.set_per_group_attn_metadata(0, target, slots)
                    buffer = proposer._per_group_block_table_buffers[0]
                    pointer = buffer.data_ptr()
                    self.assertEqual(buffer.shape, (5, 6 * replication))
                    self.assertEqual(torch.count_nonzero(buffer).item(), 0)

                    for batch in (1, 4, 2, 0, 3):
                        lengths = torch.ones(batch, dtype=torch.int32)
                        if batch > 1:
                            lengths[-1] = 0
                        actual = proposer._build_replicated_block_table(0, target[:batch], lengths)
                        expected = torch.zeros_like(actual)
                        for row in range(batch):
                            if lengths[row] == 0:
                                continue
                            expanded = []
                            for start in range(0, 6, subblocks):
                                for lane in range(replication):
                                    for block in target[row, start : start + subblocks].tolist():
                                        physical, offset = divmod(block, subblocks)
                                        expanded.append((physical * replication + lane) * subblocks + offset)
                            expected[row] = torch.tensor(expanded, dtype=torch.int32)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(proposer._replicated_block_table_storage[0].data_ptr(), pointer)
                        torch.testing.assert_close(buffer[:batch], expected)
                        self.assertEqual(torch.count_nonzero(buffer[batch:]).item(), 0)

                    # A runner metadata refresh must reuse the allocation too.
                    proposer.set_per_group_attn_metadata(0, target, slots)
                    self.assertEqual(proposer._per_group_block_table_buffers[0].data_ptr(), pointer)
                    torch.testing.assert_close(target, original)
                    self.assertTrue(torch.all(slots == 777))


if __name__ == "__main__":
    unittest.main()
