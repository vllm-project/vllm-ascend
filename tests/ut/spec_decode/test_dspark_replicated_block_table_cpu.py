# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for MRv1 replicated GQA tables, without loading NPU modules."""

import ast
import copy
import unittest
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch


def load_proposer(**overrides):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/utils.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DCPReplicatedDraftMixin")
    namespace = {"torch": torch, "copy": copy, "replace": replace, "VllmConfig": object, **overrides}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["DCPReplicatedDraftMixin"]


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
                        refreshed = proposer._get_draft_block_table(0, batch, lengths)
                        actual = refreshed[:batch]
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


class TestReplicatedDraftHooks(unittest.TestCase):
    def test_loading_restores_target_context_on_success_and_failure(self):
        @dataclass
        class Config:
            model_config: object
            parallel_config: object
            cache_config: object
            kv_transfer_config: object
            speculative_config: object

        for replicated in (False, True):
            for fail in (False, True):
                with self.subTest(replicated=replicated, fail=fail):
                    draft_model = SimpleNamespace(
                        hf_config=SimpleNamespace(
                            model_type="qwen3" if replicated else "mla",
                            architectures=["DSparkDraftModel"],
                        )
                    )
                    spec = SimpleNamespace(
                        draft_model_config=draft_model,
                        draft_parallel_config=SimpleNamespace(rank=0, decode_context_parallel_size=2),
                    )
                    target = Config(
                        SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3")),
                        SimpleNamespace(rank=3, decode_context_parallel_size=2),
                        SimpleNamespace(block_size=384),
                        object(),
                        spec,
                    )
                    current = [target]

                    @contextmanager
                    def set_config(config, current=current):
                        previous = current[0]
                        current[0] = config
                        try:
                            yield
                        finally:
                            current[0] = previous

                    @lru_cache(maxsize=1)
                    def enable_dcp(current=current):
                        return current[0].parallel_config.decode_context_parallel_size > 1

                    class Base:
                        def _create_draft_vllm_config(self):
                            return self.vllm_config

                        def _get_model(self, fail=fail):
                            config = self._create_draft_vllm_config()
                            self.loaded_config = config
                            with set_config(config):
                                self.loaded_dcp = enable_dcp()
                                if fail:
                                    raise RuntimeError("load failed")
                            return "model"

                    mixin = load_proposer(enable_dcp=enable_dcp, set_current_vllm_config=set_config)

                    class Proposer(mixin, Base):
                        pass

                    proposer = Proposer()
                    proposer.vllm_config = target
                    proposer.speculative_config = spec
                    proposer.runner = object()
                    proposer.dcp_size, proposer.dcp_rank = 2, 1
                    proposer._init_dcp_replicated_draft()
                    self.assertTrue(enable_dcp())
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "load failed"):
                            proposer._get_model()
                    else:
                        self.assertEqual(proposer._get_model(), "model")
                    self.assertIs(current[0], target)
                    self.assertTrue(enable_dcp())
                    self.assertEqual(proposer.loaded_dcp, not replicated)
                    if replicated:
                        self.assertEqual((proposer.dcp_size, proposer.dcp_rank), (1, 0))
                        self.assertIs(proposer.loaded_config.model_config, draft_model)
                        self.assertEqual(proposer.loaded_config.parallel_config.rank, 3)
                        self.assertIsNone(proposer.loaded_config.kv_transfer_config)
                        proposer.loaded_config.cache_config.block_size = 128
                        self.assertEqual(target.cache_config.block_size, 384)
                        self.assertEqual(spec.draft_parallel_config.decode_context_parallel_size, 2)
                    else:
                        self.assertEqual((proposer.dcp_size, proposer.dcp_rank), (2, 1))
                        self.assertIs(proposer.loaded_config, target)

    def test_nonreplicated_table_is_forwarded(self):
        proposer = load_proposer()()
        table = torch.tensor([[3, 7]], dtype=torch.int32)
        proposer._per_group_replication_sizes = {}
        proposer._per_group_block_tables = {0: table}
        self.assertIs(proposer._get_draft_block_table(0, 1, torch.tensor([8])), table)

    def test_context_slots_follow_expanded_pages(self):
        proposer = load_proposer()()
        proposer.device = torch.device("cpu")
        proposer._per_group_kernel_block_sizes = {0: 4}
        proposer._per_group_context_slot_mapping_buffers = {0: torch.empty(8, dtype=torch.int32)}
        table = torch.tensor([[6, 7, 14, 15], [10, 11, 18, 19]], dtype=torch.int32)
        slots = proposer._build_replicated_context_slot_mapping(
            0, table, torch.tensor([0, 4, 9, 3, 8]), torch.tensor([0, 3, 5]), 2, 5
        )
        torch.testing.assert_close(slots, torch.tensor([24, 28, 57, 43, 72, -1, -1, -1], dtype=torch.int32))
        proposer._build_replicated_context_slot_mapping(0, table, torch.empty(0), torch.tensor([0]), 0, 0)
        self.assertTrue(torch.all(slots == -1))


if __name__ == "__main__":
    unittest.main()
