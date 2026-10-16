# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackendImpl,
    mxfp_packet_size_bytes,
    mxfp_paged_cache_views,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import (
    MooncakeConnector,
    MooncakePullConnector,
)
from vllm_ascend.worker import model_runner_v1 as runner_module
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.fixture(autouse=True)
def pull_connector(monkeypatch):
    connector = object.__new__(MooncakePullConnector)
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(runner_module, "get_kv_transfer_group", lambda: connector)
    return connector


def _runner(consumer=True, transfer=True, group_id=0):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    groups = [SimpleNamespace(layer_names=["mamba"]), SimpleNamespace(layer_names=["mamba.other"])]
    groups[group_id] = SimpleNamespace(layer_names=["attention"])
    runner.kv_cache_config = SimpleNamespace(num_blocks=4, kv_cache_groups=groups)
    scale = torch.arange(256, dtype=torch.int32).remainder(251).to(torch.uint8)
    layer = SimpleNamespace(impl=object.__new__(AscendC8MXFPAttentionBackendImpl), v_cache_scale=scale)
    runner.compilation_config = SimpleNamespace(static_forward_context={"attention": layer})
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_consumer=consumer) if transfer else None
    )
    blocks = ([2], [2])
    blocks[group_id][:] = [1]
    runner.requests = {"request": SimpleNamespace(block_ids=blocks)}
    runner.input_batch = SimpleNamespace(req_id_to_index={"request": 0})

    # A reused destination page has five kernel blocks; the transfer writes
    # only its first three. Preserve padding and guards in the byte comparison.
    ratio, offset, guard = 5, 137, 97
    pitch = mxfp_packet_size_bytes(1, 512, 256, 256) + 256
    raw = torch.full((offset + 4 * ratio * pitch + guard,), 69, dtype=torch.int8)
    cache = mxfp_paged_cache_views(raw[offset:-guard], 4 * ratio, 1, 256, 256, 512, page_stride_bytes=pitch)
    runner._fill_c8_mxfp_v_scale_caches({"attention": cache})
    loaded = raw.clone()
    raw[offset + ratio * pitch : offset + 2 * ratio * pitch].fill_(-1)
    raw[offset + ratio * pitch : offset + (ratio + 3) * pitch].copy_(
        loaded[offset + ratio * pitch : offset + (ratio + 3) * pitch]
    )
    return runner, raw, cache, scale, pitch, offset, guard


def _event(kind="new", zeroed=(), sync=False):
    cached = SimpleNamespace(resumed_req_ids={"request"} if kind == "resumed" else set())
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="request")] if kind == "new" else [],
        scheduled_cached_reqs=cached,
        block_table_updates={"request": ([2], [2])} if kind == "replacement" else None,
        new_block_ids_to_zero=list(zeroed),
        has_sync_kv_loads=sync,
    )


def _expected(raw, scale, pitch, offset, guard, page=1):
    expected = raw.clone()
    views = mxfp_paged_cache_views(expected[offset:-guard], 20, 1, 256, 256, 512, page_stride_bytes=pitch)
    views[3][page * 5 : (page + 1) * 5].copy_(scale.view(1, 16, 1, 16, 1))
    return expected


@pytest.mark.parametrize("kind", ["new", "resumed", "replacement"])
@pytest.mark.parametrize("group_id", [0, 1])
def test_partial_load_restores_tail_scales_and_preserves_all_other_bytes(kind, group_id):
    runner, raw, cache, scale, pitch, offset, guard = _runner(group_id=group_id)
    expected = _expected(raw, scale, pitch, offset, guard)
    assert torch.all(cache[3][8:10] == 255)
    event = _event(kind)
    if kind == "replacement":
        event.block_table_updates["request"] = runner.requests["request"].block_ids
    runner._restore_c8_mxfp_loaded_v_scales(event)
    assert torch.equal(raw, expected)


@pytest.mark.parametrize("connector_kind", ["other", "layerwise", "pull_subclass"])
def test_connectors_without_the_exact_pull_admission_contract_are_unchanged(monkeypatch, connector_kind):
    class LayerwiseConnector:
        def wait_for_layer_load(self, layer_name):
            raise AssertionError("Layer loads must not be triggered by scale preparation")

    class PullSubclass(MooncakePullConnector):
        pass

    connectors = {
        "other": object(),
        "layerwise": LayerwiseConnector(),
        "pull_subclass": object.__new__(PullSubclass),
    }
    monkeypatch.setattr(runner_module, "get_kv_transfer_group", lambda: connectors[connector_kind])
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    assert torch.equal(raw, before)


@pytest.mark.parametrize("consumer,transfer", [(False, True), (False, False)])
def test_producer_and_non_pd_execution_are_unchanged(consumer, transfer):
    runner, raw, _, _, _, _, _ = _runner(consumer=consumer, transfer=transfer)
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    assert torch.equal(raw, before)


def test_absent_transfer_group_is_unchanged(monkeypatch):
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: False)
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    assert torch.equal(raw, before)


def test_synchronous_loads_are_not_prepared_before_their_load_finishes():
    runner, raw, _, _, _, _, _ = _runner()
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event(sync=True))
    assert torch.equal(raw, before)


def test_non_c8_execution_does_not_require_scale_or_connector_state():
    NPUModelRunner.__new__(NPUModelRunner)._restore_c8_mxfp_loaded_v_scales(_event())


def test_non_c8_cache_initialization_does_not_require_kv_cache_config():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.compilation_config = SimpleNamespace(static_forward_context={"plain": SimpleNamespace(impl=object())})
    runner._fill_c8_mxfp_v_scale_caches({"plain": (object(), object())})
    assert not runner._c8_mxfp_v_scale_caches


def test_inactive_request_is_not_written():
    runner, raw, _, _, _, _, _ = _runner()
    runner.input_batch.req_id_to_index = {}
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    assert torch.equal(raw, before)


def test_ordinary_decode_does_not_inspect_connector_or_revisit_reassigned_pages(monkeypatch):
    runner, raw, _, _, _, _, _ = _runner()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    raw.fill_(73)
    before = raw.clone()

    def unexpected_lookup():
        raise AssertionError("Ordinary decode must not inspect the connector")

    monkeypatch.setattr(runner_module, "get_kv_transfer_group", unexpected_lookup)
    runner._restore_c8_mxfp_loaded_v_scales(_event("cached"))
    assert torch.equal(raw, before)


def test_normal_zeroing_still_restores_scales_and_is_not_repeated(monkeypatch):
    runner, raw, _, scale, pitch, offset, guard = _runner()
    expected = _expected(raw, scale, pitch, offset, guard)
    runner._zero_block_ids([1])
    assert torch.equal(raw, expected)

    def unexpected_fill(*args):
        raise AssertionError("Already-prepared pages must not be filled twice")

    monkeypatch.setattr(runner_module, "fill_mxfp_v_scale_cache", unexpected_fill)
    runner._restore_c8_mxfp_loaded_v_scales(_event(zeroed=(1,)))


def test_null_and_negative_block_ids_are_not_written():
    runner, raw, _, _, _, _, _ = _runner()
    runner.requests["request"].block_ids = ([0, -1], [2])
    before = raw.clone()
    runner._restore_c8_mxfp_loaded_v_scales(_event())
    assert torch.equal(raw, before)


def test_complete_row_update_takes_precedence_over_the_cached_runner_table():
    runner, raw, _, scale, pitch, offset, guard = _runner()
    raw[offset + 10 * pitch : offset + 15 * pitch].fill_(-1)
    expected = _expected(raw, scale, pitch, offset, guard, page=2)
    runner._restore_c8_mxfp_loaded_v_scales(_event("replacement"))
    assert torch.equal(raw, expected)


def test_shared_pages_are_prepared_once_per_batch(monkeypatch):
    runner, raw, _, scale, pitch, offset, guard = _runner()
    runner.requests["other"] = runner.requests["request"]
    runner.input_batch.req_id_to_index["other"] = 1
    event = _event()
    event.scheduled_new_reqs.append(SimpleNamespace(req_id="other"))
    original = runner_module.fill_mxfp_v_scale_cache
    writes = []

    def record_fill(value_scale, cache):
        writes.append(cache.data_ptr())
        original(value_scale, cache)

    monkeypatch.setattr(runner_module, "fill_mxfp_v_scale_cache", record_fill)
    expected = _expected(raw, scale, pitch, offset, guard)
    runner._restore_c8_mxfp_loaded_v_scales(event)
    assert torch.equal(raw, expected)
    assert len(writes) == 1


def test_compatibility_alias_is_the_supported_pull_class(pull_connector):
    assert MooncakeConnector is MooncakePullConnector
    assert type(pull_connector) is MooncakeConnector
