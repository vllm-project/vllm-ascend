# SPDX-License-Identifier: Apache-2.0
"""CPU correctness of the shared-group wrapper around the existing LRU row."""

import importlib

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.fixture(scope="module", autouse=True)
def load_custom_ops():
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")


class Row:
    def __init__(self):
        self.slots = torch.full((4,), -1, dtype=torch.int32)
        self.lru = torch.arange(4, dtype=torch.int32)
        self.resident_pages = torch.full((2, 4), -1, dtype=torch.int32)
        self.identity = torch.full((9,), -1, dtype=torch.int64)
        self.mark = torch.zeros(16, dtype=torch.int32)
        self.pos = torch.full((16,), -1, dtype=torch.int32)
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.control = torch.zeros(2, dtype=torch.int64)
        self.meta = [7, 0, 0, 0, 10, 0, 0, 4, 8, 10, 0, 8]

    def plan(self, tokens, pages=None, finish=True):
        if pages is None:
            pages = [[1, 2, 3], [4, 5, 6]]
        result = torch.ops._C_ascend.selective_group_lru_plan_cpu(
            torch.tensor(tokens, dtype=torch.int32),
            torch.tensor(pages, dtype=torch.int32),
            self.slots, self.lru, self.resident_pages, self.identity, self.mark, self.pos, self.epoch,
            self.control, self.meta,
        )
        if finish:
            self.finish(result[4].item(), True)
        return result

    def finish(self, ticket, success):
        torch.ops._C_ascend.selective_group_lru_finish_cpu(self.identity, self.control, ticket, success)


def test_duplicates_tail_and_distinct_layer_pages():
    row = Row()
    physical, tokens, slots, _, _ = row.plan([0, 1, 1, 8])
    assert physical.tolist() == [0, 1, 1, 4]
    assert tokens.tolist() == [0, 1] and slots.tolist() == [0, 1]
    physical, tokens, slots, _, _ = row.plan([1, 0, 8, -1])
    assert physical.tolist() == [1, 0, 4, -1]
    assert tokens.numel() == slots.numel() == 0


def test_missing_history_fails_but_tail_needs_no_host_page():
    row = Row()
    physical, tokens, _, _, _ = row.plan([8, -1], pages=[[], []])
    assert physical.tolist() == [4, -1] and tokens.numel() == 0
    before = row.slots.clone()
    with pytest.raises(RuntimeError, match="missing a Host page"):
        row.plan([4, -1], pages=[[1], [4]])
    assert row.control[0].item() == 0
    with pytest.raises(RuntimeError, match="invalid Host page"):
        row.plan([4, -1], pages=[[1, 2], [4, -1]])
    assert row.control[0].item() == 0
    assert torch.equal(row.slots, before)
    assert row.plan([4, -1])[1].tolist() == [4]


def test_generation_data_and_buffer_versions_and_retire():
    row = Row()
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    assert row.plan([0, 1, -1, -1])[1].numel() == 0
    for index in (1, 2, 3):
        row.meta[index] += 1
        assert row.plan([0, 1, -1, -1])[1].numel() == 2
    retire = torch.ops._C_ascend.selective_group_lru_retire_cpu
    assert not retire(row.identity, row.control, 7, 0)
    assert row.plan([0, 1, -1, -1])[1].numel() == 0
    assert retire(row.identity, row.control, 7, 1)
    assert row.plan([0, 1, -1, -1])[1].numel() == 2


def test_physical_row_change_invalidates_resident_slots():
    row = Row()
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    assert row.plan([0, 1, -1, -1])[1].numel() == 0
    row.meta[10] = 1
    _, tokens, _, table, _ = row.plan([0, 1, -1, -1])
    assert tokens.tolist() == [0, 1]
    assert table.tolist() == [2, 1]


def test_geometry_change_invalidates_resident_slots():
    row = Row()
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    assert row.plan([0, 1, -1, -1])[1].numel() == 0
    row.meta[7] = 2  # Changing page/block mapping invalidates resident KV.
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    row.meta[11] = 12  # Changing per-row layout also invalidates it.
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    row.slots = torch.full((6,), -1, dtype=torch.int32)
    row.lru = torch.arange(6, dtype=torch.int32)
    row.resident_pages = torch.full((2, 6), -1, dtype=torch.int32)
    assert row.plan([0, 1, -1, -1])[1].numel() == 2


def test_layer_page_remap_invalidates_only_affected_slots():
    row = Row()
    hot = torch.zeros((2, 4, 8), dtype=torch.uint8)

    def simulate_copy_and_check(plan, tables):
        physical, misses, slots, _, _ = plan
        for token, slot in zip(misses.tolist(), slots.tolist()):
            for layer in range(2):
                page = tables[layer][token // 4]
                hot[layer, slot] = torch.tensor(
                    [(layer * 41 + page * 13 + token % 4 + i) % 256 for i in range(8)], dtype=torch.uint8
                )
        for token, slot in zip([0, 1, 4, 5], physical.tolist()):
            for layer in range(2):
                page = tables[layer][token // 4]
                expected = torch.tensor(
                    [(layer * 41 + page * 13 + token % 4 + i) % 256 for i in range(8)], dtype=torch.uint8
                )
                assert torch.equal(hot[layer, slot], expected)

    original = [[1, 2, 3], [4, 5, 6]]
    cold = row.plan([0, 1, 4, 5], original)
    assert cold[1].numel() == 4
    simulate_copy_and_check(cold, original)
    assert row.plan([0, 1, 4, 5], original)[1].numel() == 0
    # Only the second layer remaps block 1. Both shared layers reload the two
    # affected logical tokens; cached tokens in block 0 remain hits.
    changed = [[1, 2, 3], [4, 7, 6]]
    remapped = row.plan([0, 1, 4, 5], changed)
    _, misses, _, _, _ = remapped
    assert misses.tolist() == [4, 5]
    simulate_copy_and_check(remapped, changed)
    assert row.plan([0, 1, 4, 5], changed)[1].numel() == 0
    # Appending a previously unused page does not invalidate old entries.
    appended = [[1, 2, 3, 0], [4, 7, 6, 0]]
    assert row.plan([0, 1, 4, 5], appended)[1].numel() == 0
    # In-place content rewrite has identical page numbers; caller must bump
    # the request data version to invalidate even an unchanged mapping.
    row.meta[2] += 1
    assert row.plan([0, 1, 4, 5], appended)[1].numel() == 4


def test_stable_prefix_recomputation_invalidates_suffix():
    row = Row()
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    row.meta[9] = 1
    _, tokens, _, _, _ = row.plan([0, 1, -1, -1])
    assert tokens.tolist() == [1]


def test_failure_invalidates_and_blocks_reentry_and_retire():
    row = Row()
    _, tokens, _, _, ticket = row.plan([0, 1, -1, -1], finish=False)
    assert tokens.numel() == 2
    with pytest.raises(RuntimeError, match="in flight"):
        row.plan([0, 1, -1, -1])
    with pytest.raises(RuntimeError, match="in flight"):
        torch.ops._C_ascend.selective_group_lru_retire_cpu(row.identity, row.control, 7, 0)
    row.finish(ticket.item(), False)
    assert row.plan([0, 1, -1, -1])[1].numel() == 2
    with pytest.raises(RuntimeError, match="ticket"):
        row.finish(ticket.item(), True)


def test_current_token_requires_tail_and_table_uses_physical_tail_row():
    row = Row()
    row.meta[5] = -1
    with pytest.raises(RuntimeError, match="current token requires"):
        row.plan([9, -1])
    row.meta[5] = 2
    physical, _, _, table, _ = row.plan([9, -1], pages=[[], []])
    assert physical.tolist() == [5, -1] and table.tolist() == [0, 5]
    row.meta[4] = row.meta[9] = 9
    physical, _, _, table, _ = row.plan([8, -1], pages=[[], []])
    assert physical.tolist() == [4, -1] and table.tolist() == [0, 5]
    # Four-token blocks: token 4 begins a new block; token 3 is the preceding
    # block's last offset. The stable prefix must not exceed visible length.
    row.meta[4] = row.meta[9] = 5
    physical, _, _, table, _ = row.plan([4, -1], pages=[[], []])
    assert physical.tolist() == [4, -1] and table.tolist() == [0, 5]
    row.meta[4] = row.meta[9] = 4
    physical, _, _, table, _ = row.plan([3, -1], pages=[[], []])
    assert physical.tolist() == [7, -1] and table.tolist() == [0, 5]


def test_actual_128_token_tail_boundary_without_host_pages():
    row = Row()
    row.slots = torch.full((128,), -1, dtype=torch.int32)
    row.lru = torch.arange(128, dtype=torch.int32)
    row.resident_pages = torch.full((2, 128), -1, dtype=torch.int32)
    row.mark = torch.zeros(256, dtype=torch.int32)
    row.pos = torch.full((256,), -1, dtype=torch.int32)
    row.meta = [7, 0, 0, 0, 129, 2, 0, 128, 8, 129, 0, 256]
    physical, tokens, _, table, _ = row.plan([128, -1], pages=[[], []])
    assert physical.tolist() == [128, -1] and tokens.numel() == 0 and table.tolist() == [0, 5]
    row.meta[4] = row.meta[9] = 128
    physical, tokens, _, table, _ = row.plan([127, -1], pages=[[], []])
    assert physical.tolist() == [255, -1] and tokens.numel() == 0 and table.tolist() == [0, 5]


def test_corrupt_epoch_lru_and_aliased_state_are_rejected():
    row = Row()
    row.plan([0, 1, -1, -1])
    for bad in (-1, 2**31 - 1):
        row.epoch.fill_(bad)
        with pytest.raises(RuntimeError, match="epoch"):
            row.plan([0, 1, -1, -1])
        assert row.control[0].item() == 0
    row.epoch.zero_()
    row.plan([0, 1, -1, -1])
    row.lru[0] = row.lru[1]
    with pytest.raises(RuntimeError, match="LRU slots"):
        row.plan([0, 1, -1, -1])
    row.lru = row.slots  # Must not pass aliases to the RESTRICT helper.
    with pytest.raises(RuntimeError, match="overlap"):
        row.plan([0, 1, -1, -1])
