# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager

# Importing vLLM activates the Ascend platform plugin, which replaces the
# module-level ``MambaManager`` symbol with ``AscendMambaManager``.  Patch the
# actual base class from the MRO so these tests keep exercising the override.
BaseMambaManager = AscendMambaManager.__mro__[1]


def _manager(*, producer: bool):
    manager = AscendMambaManager.__new__(AscendMambaManager)
    manager.is_kv_producer = producer
    manager.block_size = 1024
    manager.kv_cache_group_id = 3
    manager._pending_partial_tail_offloads = []
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
    assert manager._pending_partial_tail_offloads == [("req", 3, source, 144)]


def test_consumer_partial_hit_does_not_create_producer_handoff(monkeypatch):
    monkeypatch.setattr(
        BaseMambaManager,
        "add_local_computed_blocks",
        lambda self, request_id, blocks, local_tokens, external_tokens: None,
    )
    manager = _manager(producer=False)
    source = SimpleNamespace(is_null=False, block_id=17)

    manager.add_local_computed_blocks("req", [source], 144, 0)

    assert manager._pending_partial_tail_offloads == []
