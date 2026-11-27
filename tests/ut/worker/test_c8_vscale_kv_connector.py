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
from vllm_ascend.worker.v2.kv_connector import C8VScaleKVConnector
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _packet(ratio: int = 3):
    """Use a padded, offset C8 packet to exercise the real V-scale layout."""
    heads, dim, block, num_blocks = 2, 64, 512, 2
    pitch = heads * block * (2 * dim + 2 * dim // 32) + 256
    raw = torch.full((128 + num_blocks * ratio * pitch + 128,), 37, dtype=torch.int8)
    views = mxfp_paged_cache_views(raw[128:-128], num_blocks * ratio, heads, dim, dim, block, page_stride_bytes=pitch)
    scale = torch.arange(heads * dim, dtype=torch.int32).remainder(251).to(torch.uint8)
    impl = object.__new__(AscendC8MXFPAttentionBackendImpl)
    layer = SimpleNamespace(impl=impl, kv_cache=views, v_cache_scale=scale)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    group = SimpleNamespace(layer_names=["attn"])
    runner.kv_cache_config = SimpleNamespace(num_blocks=num_blocks, kv_cache_groups=[group])
    runner.compilation_config = SimpleNamespace(static_forward_context={"attn": layer})
    runner.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="mxfp8"), compilation_config=runner.compilation_config
    )
    runner._fill_c8_mxfp_v_scale_caches()
    runner.kv_connector = C8VScaleKVConnector(
        MagicMock(), runner.kv_cache_config, runner.compilation_config.static_forward_context
    )
    return raw, views, scale, runner, pitch


def _output(*block_ids, zeroed=(), sync=False):
    return SimpleNamespace(
        has_sync_kv_loads=sync,
        scheduled_new_reqs=[SimpleNamespace(block_ids=blocks) for blocks in block_ids],
        new_block_ids_to_zero=list(zeroed),
    )


@pytest.mark.parametrize("is_consumer", [False, True])
def test_worker_wraps_nested_pull_consumer_after_cache_initialization(is_consumer):
    from vllm_ascend.worker.worker import NPUWorker

    _, _, _, runner, _ = _packet()
    delegate = MagicMock()
    runner.kv_connector = delegate

    def initialize(*_args, **_kwargs):
        assert runner.kv_connector is delegate

    runner.initialize_kv_cache = Mock(side_effect=initialize)
    runner.kv_cache_config.needs_kv_cache_zeroing = False
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_runner = runner
    worker.vllm_config = SimpleNamespace(
        speculative_config=None, kv_transfer_config=SimpleNamespace(is_kv_consumer=is_consumer)
    )
    worker._maybe_get_memory_pool_context = lambda tag: nullcontext()

    class Pull:
        pass

    group = SimpleNamespace(sub_connectors=[Pull()])
    with (
        patch("vllm_ascend.worker.worker.plan_layerwise_cache_reuse", return_value=None),
        patch("vllm_ascend.worker.worker.ensure_kv_transfer_initialized"),
        patch("vllm.distributed.kv_transfer.has_kv_transfer_group", return_value=True),
        patch("vllm.distributed.kv_transfer.get_kv_transfer_group", return_value=group),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector.MooncakePullConnector", Pull),
    ):
        worker.initialize_from_config(runner.kv_cache_config)

    runner.initialize_kv_cache.assert_called_once()
    assert runner.initialize_kv_cache.call_args.args == (runner.kv_cache_config,)
    assert isinstance(runner.kv_connector, C8VScaleKVConnector) is is_consumer
    if is_consumer:
        assert runner.kv_connector._connector is delegate


@pytest.mark.parametrize("ratio", [3, 5])
@pytest.mark.parametrize("sync", [False, True])
def test_pull_restores_only_admitted_nonzeroed_group_page(ratio, sync):
    raw, _, scale, runner, pitch = _packet(ratio)
    raw[128 + ratio * pitch : 128 + 2 * ratio * pitch].fill_(73)
    poisoned = raw.clone()
    expected = raw.clone()
    expected_views = mxfp_paged_cache_views(expected[128:-128], 2 * ratio, 2, 64, 64, 512, page_stride_bytes=pitch)
    fill_mxfp_v_scale_cache(scale, expected_views[3][ratio : 2 * ratio])

    connector = runner.kv_connector
    connector._restore(_output(([], [1])))
    connector._restore(_output(([1],), zeroed=[1]))
    assert torch.equal(raw, poisoned)

    connector._restore(_output(([1],), ([0],), zeroed=[0], sync=sync))
    assert torch.equal(raw, expected)


def test_connector_restores_after_delegate():
    _, _, _, runner, _ = _packet()
    connector = runner.kv_connector
    delegate = connector._connector
    calls: list[object] = []
    delegate.pre_forward.side_effect = lambda *_args, **_kwargs: calls.append("delegate")
    connector._restore = lambda output: calls.append(("restore", output))
    output = _output()

    connector.pre_forward(output)
    assert calls == ["delegate", ("restore", output)]
