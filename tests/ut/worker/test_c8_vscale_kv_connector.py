# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_ascend.attention.attention_c8_mxfp import (
    AscendC8MXFPAttentionBackendImpl,
    fill_mxfp_v_scale_cache,
    mxfp_paged_cache_views,
)
from vllm_ascend.worker.v2.kv_connector import C8VScaleKVConnector, kv_connector_tree
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _c8_mxfp_runner(layers: dict, num_blocks: int = 2) -> NPUModelRunner:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(
        num_blocks=num_blocks, kv_cache_groups=[SimpleNamespace(layer_names=list(layers))]
    )
    runner.compilation_config = SimpleNamespace(static_forward_context=layers)
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="mxfp8"),
        compilation_config=runner.compilation_config,
    )
    return runner


def _c8_mxfp_layer(num_kv_heads: int = 2, v_dim: int = 64):
    return SimpleNamespace(
        v_cache_scale=torch.arange(num_kv_heads * v_dim, dtype=torch.int32).remainder(251).to(torch.uint8),
        impl=object.__new__(AscendC8MXFPAttentionBackendImpl),
    )


def _c8_packet(ratio: int, k_dim: int = 64, v_dim: int = 64, num_pages: int = 2):
    """Real packet views with padding, a nonzero offset and registered entries."""
    heads, block = 2, 512
    kernel_blocks = num_pages * ratio
    pitch = heads * block * (k_dim + v_dim + (k_dim + v_dim) // 32) + 256
    raw = torch.full((128 + kernel_blocks * pitch + 128,), 37, dtype=torch.int8)
    views = mxfp_paged_cache_views(raw[128:-128], kernel_blocks, heads, k_dim, v_dim, block, page_stride_bytes=pitch)
    layer = _c8_mxfp_layer(heads, v_dim)
    layer.kv_cache = views
    runner = _c8_mxfp_runner({"attn": layer}, num_blocks=num_pages)
    runner._fill_c8_mxfp_v_scale_caches()
    runner.kv_connector = C8VScaleKVConnector(
        MagicMock(), runner.kv_cache_config, runner.compilation_config.static_forward_context
    )
    return raw, views, layer, runner, pitch


@pytest.mark.parametrize("is_consumer", [False, True])
def test_worker_installs_c8_wrapper_after_kv_cache_initialization(is_consumer):
    from vllm_ascend.worker.worker import NPUWorker

    _, _, _, runner, _ = _c8_packet(1)
    delegate = MagicMock()
    runner.kv_connector = delegate
    events: list[str] = []
    runner.initialize_kv_cache = Mock(side_effect=lambda *_args, **_kwargs: events.append("initialize"))
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = runner
    worker.vllm_config = SimpleNamespace(
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer),
    )
    worker._maybe_get_memory_pool_context = lambda tag: nullcontext()
    runner.kv_cache_config.needs_kv_cache_zeroing = False

    class Pull:
        pass

    group = SimpleNamespace(sub_connectors=[Pull()])
    with (
        patch("vllm_ascend.worker.worker.plan_layerwise_cache_reuse", return_value=None),
        patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized"),
        patch("vllm_ascend.worker.worker.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.worker.worker.get_kv_transfer_group", return_value=group),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector.MooncakePullConnector", Pull),
    ):
        worker.initialize_from_config(runner.kv_cache_config)

    assert events == ["initialize"]
    assert isinstance(runner.kv_connector, C8VScaleKVConnector) is is_consumer
    if is_consumer:
        assert runner.kv_connector._connector is delegate


@pytest.mark.parametrize("ratio", [3, 5])
def test_pull_admission_restores_c8_v_scales_after_partial_page_copy(ratio):
    raw, _, layer, runner, pitch = _c8_packet(ratio)
    # A partial Pull transfer and Mamba copy can overwrite a newly assigned
    # physical page. Its V scales are static and must be repaired before forward.
    raw[128 + ratio * pitch : 128 + 2 * ratio * pitch].fill_(73)
    poisoned = raw.clone()
    expected = raw.clone()
    expected_views = mxfp_paged_cache_views(expected[128:-128], 2 * ratio, 2, 64, 64, 512, page_stride_bytes=pitch)
    fill_mxfp_v_scale_cache(layer.v_cache_scale, expected_views[3][ratio : 2 * ratio])
    output = SimpleNamespace(
        has_sync_kv_loads=False,
        scheduled_new_reqs=[SimpleNamespace(block_ids=([], [1]))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set()),
        new_block_ids_to_zero=[],
    )

    # Page 1 belongs to group 1 here; the C8 cache belongs to group 0.
    runner.kv_connector._restore(output)
    assert torch.equal(raw, poisoned)

    output.scheduled_new_reqs[0].block_ids = ([1],)
    runner.kv_connector._restore(output)

    assert torch.equal(raw, expected)


def test_pull_admission_restores_resumed_pages_but_skips_zeroed_pages():
    _, views, layer, runner, _ = _c8_packet(3)
    views[3][3:6].fill_(73)
    output = SimpleNamespace(
        has_sync_kv_loads=False,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["resumed"], new_block_ids=[([1],)], resumed_req_ids={"resumed"}),
        new_block_ids_to_zero=[1],
    )

    runner.kv_connector._restore(output)
    assert torch.all(views[3][3:6] == 73)

    output.new_block_ids_to_zero = []
    runner.kv_connector._restore(output)
    assert torch.equal(views[3][3, :, :, 0, :, 0].reshape(-1), layer.v_cache_scale)


def test_pull_admission_batches_disjoint_pages_and_ignores_missing_blocks():
    _, views, layer, runner, _ = _c8_packet(3, num_pages=4)
    views[3][3:12].fill_(73)
    output = SimpleNamespace(
        has_sync_kv_loads=False,
        scheduled_new_reqs=[SimpleNamespace(block_ids=None), SimpleNamespace(block_ids=([1, None, 3],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set()),
        new_block_ids_to_zero=[],
    )

    runner.kv_connector._restore(output)

    for block_id in (1, 3):
        assert torch.equal(views[3][block_id * 3, :, :, 0, :, 0].reshape(-1), layer.v_cache_scale)
    assert torch.all(views[3][6:9] == 73)


def test_pull_admission_rejects_out_of_range_page():
    _, _, _, runner, _ = _c8_packet(3)
    output = SimpleNamespace(
        has_sync_kv_loads=False,
        scheduled_new_reqs=[SimpleNamespace(block_ids=([2],))],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], resumed_req_ids=set()),
        new_block_ids_to_zero=[],
    )

    with pytest.raises(IndexError, match="C8 physical block ID out of range: 2"):
        runner.kv_connector._restore(output)


def test_c8_connector_restores_after_delegate_pre_forward():
    calls: list[object] = []
    delegate = MagicMock()
    delegate.pre_forward.side_effect = lambda *args, **kwargs: calls.append("delegate")
    output = SimpleNamespace(has_sync_kv_loads=False)
    connector = C8VScaleKVConnector(delegate, SimpleNamespace(num_blocks=0), {})
    connector._restore = lambda arg: calls.append(("restore", arg))

    connector.pre_forward(output)

    assert calls == ["delegate", ("restore", output)]


def test_c8_connector_releases_cache_views():
    _, _, _, runner, _ = _c8_packet(1)
    connector = runner.kv_connector
    delegate = connector._connector

    assert connector._entries
    assert connector.release_cache_views() is delegate
    assert connector._entries == []


def test_kv_connector_tree_includes_pool_and_mooncake_children():
    mooncake = SimpleNamespace()
    pool = SimpleNamespace(requires_mamba_state_copy_after_load=True)
    nested = SimpleNamespace(sub_connectors=[mooncake, pool])
    root = SimpleNamespace(sub_connectors=[nested])

    connectors = list(kv_connector_tree(root))

    assert connectors == [root, nested, mooncake, pool]
    assert any(getattr(connector, "requires_mamba_state_copy_after_load", False) for connector in connectors)
