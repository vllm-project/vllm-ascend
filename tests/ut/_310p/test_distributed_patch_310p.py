# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from unittest.mock import MagicMock, Mock, create_autospec

import pytest
import torch
import torch.distributed as dist

from vllm_ascend.patch.platform.patch_distributed import communication_adaptation_310p


@pytest.fixture
def native_broadcast(monkeypatch):
    original = create_autospec(dist.distributed_c10d.broadcast)
    for api in (dist, dist.distributed_c10d):
        monkeypatch.setattr(api, "broadcast", original)
        monkeypatch.setattr(api, "all_reduce", Mock())
    communication_adaptation_310p()
    return original


@pytest.mark.parametrize(
    ("kwargs", "global_src", "local_src"),
    [
        ({"group_src": 0}, None, 0),
        ({"src": 2}, 2, None),
        ({}, 0, None),
        ({"src": None, "group_src": None}, 0, None),
    ],
)
def test_310p_broadcast_preserves_cpu_source_rank(native_broadcast, kwargs, global_src, local_src):
    tensor = torch.zeros(1)
    group = object()

    dist.distributed_c10d.broadcast(tensor, group=group, **kwargs)

    native_broadcast.assert_called_once_with(tensor, src=global_src, group=group, async_op=False, group_src=local_src)


@pytest.mark.parametrize("kwargs", [{"group_src": 1}, {"src": 3}])
def test_310p_broadcast_indexes_device_subgroup_by_local_rank(monkeypatch, native_broadcast, kwargs):
    tensor = MagicMock(device=torch.device("meta"))
    group = object()
    peer_tensor = object()

    def all_gather(output, local_tensor, *, group):
        assert local_tensor is tensor
        output[1] = peer_tensor

    resolve_local = Mock(return_value=1)
    monkeypatch.setattr(dist, "get_group_rank", resolve_local)
    monkeypatch.setattr(dist, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(dist, "get_world_size", Mock(return_value=2))
    monkeypatch.setattr(dist, "all_gather", all_gather)
    monkeypatch.setattr(torch, "empty_like", Mock())

    dist.broadcast(tensor, group=group, **kwargs)

    if "src" in kwargs:
        resolve_local.assert_called_once_with(group, 3)
    else:
        resolve_local.assert_not_called()
    tensor.__setitem__.assert_called_once_with(Ellipsis, peer_tensor)
    native_broadcast.assert_not_called()
