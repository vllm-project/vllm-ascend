# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm.v1.hisparse.runtime")

from vllm.v1.hisparse.runtime import HiSparseCacheHandle, HiSparseRuntime

from vllm_ascend.hisparse.flash_compat import install_flash_layer_view_bounds
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(vllm_version_is("0.33.0"), reason="Flash HiSparse uses the verified main pin")


@pytest.mark.parametrize(
    ("supports_flash", "declared_budget", "expected"),
    [(True, 1168, 1168), (True, None, 1024), (False, 1168, 1024)],
)
def test_flash_pool_budget_preserves_generic_fallback(monkeypatch, supports_flash, declared_budget, expected):
    from vllm.v1.core import kv_cache_planning

    from vllm_ascend.patch.platform import patch_flash_hisparse

    config = SimpleNamespace(attention_config=SimpleNamespace(hisparse_config=object()))
    groups = [object()]
    builder = Mock()
    builder._get_pool_bytes_per_block.return_value = declared_budget
    monkeypatch.setattr(patch_flash_hisparse, "supports_glm5_next_hisparse", lambda _: supports_flash)
    monkeypatch.setattr(kv_cache_planning, "_pool_bytes_per_block", lambda *_: 1024)

    assert kv_cache_planning._pool_bytes_per_block_from_builder(builder, config, groups) == expected
    assert builder._get_pool_bytes_per_block.call_count == int(supports_flash)


@pytest.mark.parametrize("supports_flash", [False, True])
def test_connector_uses_terminal_compat_only_for_flash(monkeypatch, supports_flash):
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.connector import HiSparseConnector

    from vllm_ascend.patch.platform import patch_flash_hisparse

    coordinator = object()
    manager = object()
    config = object()
    connector = SimpleNamespace(_vllm_config=config, connector_scheduler=None)
    install = Mock()

    def bind(instance, cache_manager):
        assert instance is connector and cache_manager is manager
        instance.connector_scheduler = SimpleNamespace(coordinator=coordinator)

    monkeypatch.setattr(patch_flash_hisparse, "_ORIGINAL_BIND_KV_CACHE_MANAGER", bind)
    monkeypatch.setattr(patch_flash_hisparse, "supports_glm5_next_hisparse", lambda _: supports_flash)
    monkeypatch.setattr(patch_flash_hisparse, "install_flash_terminal_spill_compat", install)

    HiSparseConnector.bind_kv_cache_manager(connector, manager)
    if supports_flash:
        install.assert_called_once_with(coordinator)
    else:
        install.assert_not_called()


@pytest.mark.parametrize(("block_stride", "attention_pages", "attention_stride"), [(32, 5, 4), (24, 3, 3)])
def test_flash_bounded_view_keeps_storage_on_rebind(block_stride, attention_pages, attention_stride):
    runtime = object.__new__(HiSparseRuntime)
    runtime.kv_dtype = torch.float8_e4m3fn
    runtime.row_width = 8
    handle = HiSparseCacheHandle(runtime)
    install_flash_layer_view_bounds(handle)
    block_table = torch.zeros((1, 3), dtype=torch.int32)
    layout = dict(byte_offset=8, block_stride=block_stride, num_blocks=3, block_size=2)

    for arena in (torch.zeros(137, dtype=torch.uint8), torch.ones(137, dtype=torch.uint8)):
        handle.bind_cache(arena, **layout, block_table=block_table, slot_mapping=block_table[0])
        runtime.bind_hot_cache(arena, **layout, block_table=block_table)

        assert runtime.hot_backing is arena
        for view in (handle.view, runtime.hot):
            assert view.cache.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr()
            assert view.cache.data_ptr() == arena.data_ptr() + 8
            assert view.cache.stride() == (block_stride, 8, 1)
            assert view.attention_cache.shape == (attention_pages, 2, 8)
            assert view.attention_block_stride == attention_stride


def test_legacy_mirror_only_skips_current_rows_for_resident_q6():
    from vllm_ascend.hisparse.host_mirror import HostMirrorConnector, _LegacyDecodeRowMirror

    stage, submit, finish, previous_step = Mock(), Mock(), Mock(), Mock()
    handle = SimpleNamespace(draft_layer=False, submit_layer_mirror=submit)
    worker = SimpleNamespace(
        _stage_row_mirror_mapping=stage,
        _enqueue_host_mirror=finish,
        cache_handles=[handle],
        _group_leaders=[("target", handle)],
    )
    legacy = _LegacyDecodeRowMirror.bind(worker)
    assert _LegacyDecodeRowMirror.bind(worker) is legacy

    def pre_forward(scheduler_output, input_batch, *, attn_metadata):
        worker._stage_row_mirror_mapping(input_batch.num_tokens)
        previous_step()

    connector = HostMirrorConnector(
        SimpleNamespace(_disabled=False, pre_forward=pre_forward),
        worker,
        None,
        None,
        defer_resident_decode_row_mirror=True,
        legacy_decode_row_mirror=legacy,
    )
    scheduled = SimpleNamespace(
        has_sync_kv_loads=True,
        kv_connector_metadata=SimpleNamespace(all_context_pages_resident=True),
    )
    batch = SimpleNamespace(
        has_prefill=False,
        num_tokens=12,
        num_tokens_after_padding=12,
        num_scheduled_tokens=torch.tensor([6, 6]),
    )
    metadata = {"target": SimpleNamespace(num_decode_tokens=12, num_actual_tokens=12)}

    connector.begin_execution(True)
    connector.pre_forward(scheduled, batch, metadata)
    handle.submit_layer_mirror()
    worker._enqueue_host_mirror()
    previous_step.assert_called_once()
    stage.assert_not_called()
    submit.assert_not_called()
    finish.assert_not_called()

    scheduled.kv_connector_metadata.all_context_pages_resident = False
    connector.begin_execution(True)
    connector.pre_forward(scheduled, batch, metadata)
    handle.submit_layer_mirror()
    worker._enqueue_host_mirror()
    assert previous_step.call_count == 2
    stage.assert_called_once_with(12)
    submit.assert_called_once()
    finish.assert_called_once()


@pytest.mark.parametrize("prefix_gap", [False, True])
def test_terminal_spill_retains_device_pages_until_host_publication(prefix_gap):
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.hisparse.coordinator import (
        HiSparseCoordinator,
        _HiSparseRequestState,
        _PendingPublication,
        _PendingSpill,
    )

    from vllm_ascend.hisparse.flash_terminal_spill import install_flash_terminal_spill_compat

    host_pool = BlockPool(8, enable_caching=True, hash_block_size=16)
    gpu_pool = BlockPool(8, enable_caching=True, hash_block_size=16)
    coordinator = object.__new__(HiSparseCoordinator)
    coordinator.host_manager = SimpleNamespace(block_pool=host_pool, block_size=16, kv_cache_group_id=1)
    coordinator.resident_managers = (SimpleNamespace(block_pool=gpu_pool),)
    coordinator.hot_managers = ()
    coordinator.request_states = {}
    coordinator.pending_spills = {}
    coordinator.copies = {}
    coordinator._copy_by_block = {}
    coordinator._owners = {}
    install_flash_terminal_spill_compat(coordinator)

    page_ids = (1, 2) if prefix_gap else (0, 1)
    host_blocks = host_pool.get_new_blocks(page_ids[-1] + 1)
    device_blocks = gpu_pool.get_new_blocks(2)
    request = SimpleNamespace(block_hashes=[bytes([page]) for page in range(len(host_blocks))])
    publication = _PendingPublication(request, 16 * len(host_blocks), len(host_blocks), None, (), host_blocks)
    state = _HiSparseRequestState(publication=publication, pending_pages=dict(zip(page_ids, (1, 2), strict=True)))
    replacement = _HiSparseRequestState()
    coordinator.request_states["reused"] = replacement
    for index, page in enumerate(page_ids):
        host_pool.touch([host_blocks[page]])
        coordinator.pending_spills[index + 1] = _PendingSpill(
            index + 1, ("reused", page), state, host_blocks[page], (device_blocks[index],)
        )

    coordinator.update_spills({1: 2, 2: 2}, {1: 2, 2: 1})
    assert state.publication is publication and coordinator.has_pending_reclamation()
    assert [block.ref_cnt for block in device_blocks] == [1, 1]
    assert not coordinator.copies

    coordinator.update_spills({}, {2: 1})
    assert state.publication is None and not replacement.valid_pages
    assert not coordinator.has_pending_reclamation()
    assert [block.ref_cnt for block in host_blocks + device_blocks] == [0] * (len(host_blocks) + 2)
    expected_copies = {} if prefix_gap else {host.block_hash: (gpu,) for host, gpu in zip(host_blocks, device_blocks)}
    assert coordinator.copies == expected_copies
    if not prefix_gap:
        assert all(block.block_hash is not None for block in host_blocks)
        gpu_pool.get_new_blocks(gpu_pool.get_num_free_blocks())
        assert not coordinator.copies and not coordinator._copy_by_block
