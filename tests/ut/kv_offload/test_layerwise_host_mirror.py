# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check stable host snapshots when two logical layers reuse one slot."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p import layerwise_host_mirror as module


@pytest.fixture
def mirror_inputs(monkeypatch):
    names = [f"model.layers.{index}.self_attn.attn" for index in range(2)]
    descriptor = SimpleNamespace(layers=names, size=16, offset=0, layer_stride=8, block_stride=4)
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=[SimpleNamespace(layer_names=names)],
    )
    source = torch.arange(8, dtype=torch.uint8)
    caches = dict.fromkeys(names, source)
    monkeypatch.setattr(
        module, "get_layerwise_kv_cache_specs", lambda _: dict.fromkeys(names, SimpleNamespace(page_size_bytes=4))
    )
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(module.torch, "empty", cpu_empty)
    npu = SimpleNamespace(
        Stream=MagicMock(), Event=MagicMock(), current_stream=MagicMock(), stream=lambda _: nullcontext()
    )
    monkeypatch.setattr(module.torch, "npu", npu, raising=False)
    return names, config, source, caches


def test_snapshot_survives_slot_reuse(mirror_inputs):
    names, config, source, caches = mirror_inputs
    mirror = module.DeepSeekV4LayerwiseHostMirror(config, caches, 2, {1: 0})
    assert mirror.backing.numel() == 64 * 1024
    assert mirror.host_views[names[1]].data_ptr() - mirror.host_views[names[0]].data_ptr() == 8
    mirror.backing.zero_()
    request = SimpleNamespace(block_ids_by_group=[[1]])
    mirror.save_layer(names[0], [request], 2)
    first_event = mirror._copy_done[0]
    mirror.wait_for_layer_reuse(1)
    first_event.synchronize.assert_called_once()
    source.add_(10)
    mirror.save_layer(names[1], [request], 2)
    mirror.wait_for_save()
    assert not mirror._copy_done
    assert mirror.backing[:16].tolist() == [0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 14, 15, 16, 17]


@pytest.mark.parametrize("block_id", [-1, 2])
def test_rejects_invalid_block_id(mirror_inputs, block_id):
    names, config, _, caches = mirror_inputs
    mirror = module.DeepSeekV4LayerwiseHostMirror(config, caches, 2, {})
    with pytest.raises(ValueError, match="out of range"):
        mirror.save_layer(names[0], [SimpleNamespace(block_ids_by_group=[[block_id]])], 2)


@pytest.mark.parametrize("invalid", ["stride", "extent", "storage", "component"])
def test_rejects_invalid_layout(mirror_inputs, invalid):
    names, config, source, caches = mirror_inputs
    if invalid == "stride":
        config.kv_cache_tensors[0].block_stride = 8
    elif invalid == "extent":
        config.kv_cache_tensors[0].offset = 1
    elif invalid == "storage":
        caches[names[0]] = source[:2].clone()
    else:
        caches["extra"] = source
    with pytest.raises(ValueError):
        module.DeepSeekV4LayerwiseHostMirror(config, caches, 2, {})
