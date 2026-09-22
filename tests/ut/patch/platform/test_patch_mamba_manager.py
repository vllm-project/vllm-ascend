# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

from vllm_ascend.patch.platform.patch_mamba_manager import (
    AscendMambaManager,
    _allocate_new_blocks_after_growing_request,
)

# Importing vLLM activates the Ascend platform plugin, which replaces the
# module-level ``MambaManager`` symbol with ``AscendMambaManager``.  Patch the
# actual base class from the MRO so these tests keep exercising the override.
BaseMambaManager = AscendMambaManager.__mro__[1]


def _manager(*, producer: bool):
    manager = AscendMambaManager.__new__(AscendMambaManager)
    manager.is_kv_producer = producer
    manager.block_size = 1024
    manager.kv_cache_group_id = 3
    manager._pending_boundary_state_offloads = []
    return manager


def test_producer_hot_partial_hit_hands_off_exact_source(monkeypatch):
    calls = []

    def add_local(self, request_id, blocks, local_tokens, external_tokens):
        calls.append((request_id, blocks, local_tokens, external_tokens))

    monkeypatch.setattr(BaseMambaManager, "add_local_computed_blocks", add_local)
    manager = _manager(producer=True)
    source = SimpleNamespace(is_null=False, block_id=17)

    manager.add_local_computed_blocks("req", [source], 144, 0)

    assert calls == [("req", [source], 144, 0)]
    assert manager._pending_boundary_state_offloads == [("req", 3, source, 144)]


def test_consumer_partial_hit_does_not_create_producer_handoff(monkeypatch):
    monkeypatch.setattr(
        BaseMambaManager,
        "add_local_computed_blocks",
        lambda self, request_id, blocks, local_tokens, external_tokens: None,
    )
    manager = _manager(producer=False)
    source = SimpleNamespace(is_null=False, block_id=17)

    manager.add_local_computed_blocks("req", [source], 144, 0)

    assert manager._pending_boundary_state_offloads == []


class _BlockPool:
    def __init__(self, batches):
        self.batches = list(batches)

    def get_new_blocks(self, count):
        blocks = self.batches.pop(0)
        assert len(blocks) == count
        return blocks


def _allocation_manager(source, grown_block):
    manager = AscendMambaManager.__new__(AscendMambaManager)
    manager.block_size = 16
    manager.req_to_blocks = {"req": [SimpleNamespace(block_id=1)]}
    manager._partial_hit_reqs = {"req": (1, source)}
    manager._record_new_block_ids = True
    manager.new_block_ids = []
    cow = SimpleNamespace(block_id=3, is_null=False)
    manager.block_pool = _BlockPool([[grown_block], [cow]])
    manager._apply_cow = lambda request_id, idx, old, new: manager.req_to_blocks[request_id].__setitem__(idx, new)
    return manager, cow


def test_partial_hit_cow_runs_after_block_table_growth():
    assert SingleTypeKVCacheManager.allocate_new_blocks is _allocate_new_blocks_after_growing_request
    source = SimpleNamespace(block_id=2, is_null=False)
    manager, cow = _allocation_manager(source, source)

    allocated = _allocate_new_blocks_after_growing_request(manager, "req", 32, 32)

    assert allocated == [cow, source]
    assert manager.req_to_blocks["req"][1] is cow
    assert manager.new_block_ids == [source.block_id, cow.block_id]


def test_stale_partial_hit_registration_skips_cow():
    source = SimpleNamespace(block_id=2, is_null=False)
    replacement = SimpleNamespace(block_id=4, is_null=False)
    manager, _ = _allocation_manager(source, replacement)

    allocated = _allocate_new_blocks_after_growing_request(manager, "req", 32, 32)

    assert allocated == [replacement]
    assert manager.req_to_blocks["req"][1] is replacement
    assert manager.new_block_ids == [replacement.block_id]
