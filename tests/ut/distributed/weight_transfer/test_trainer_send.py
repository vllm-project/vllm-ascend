# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.trainer_send import (
    TrainerProcessCoordinator,
    collect_parameter_metadata,
    default_parameter_tensor,
    dispatch_update_info,
)


@dataclass
class _UpdateInfo:
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    packed: bool
    ipc_handles: object


def test_default_parameter_tensor_returns_tensor():
    tensor = torch.zeros(2)

    assert default_parameter_tensor(("weight", tensor)) is tensor


def test_collect_parameter_metadata_stores_reduce_tensor_args_only():
    rebuild_args = (None, None, None, None, None, None, 7, None)
    fake_reduce = MagicMock(return_value=("rebuild_func", rebuild_args))

    with patch("vllm_ascend.distributed.weight_transfer.trainer_send.reduce_tensor", fake_reduce):
        names, dtype_names, shapes, ipc_handles, weight_refs = collect_parameter_metadata(
            iter([("model.weight", torch.ones(2, dtype=torch.float32))]),
            "uuid-0",
        )

    assert names == ["model.weight"]
    assert dtype_names == ["float32"]
    assert shapes == [[2]]
    assert ipc_handles == [{"uuid-0": rebuild_args}]
    assert len(weight_refs) == 1
    fake_reduce.assert_called_once()


def test_coordinator_no_distributed_group_is_rank_zero():
    with patch.object(torch.distributed, "is_initialized", return_value=False):
        assert TrainerProcessCoordinator.is_rank_zero()
        assert TrainerProcessCoordinator.all_gather_and_merge_handles([{"uuid": (1,)}]) == [{"uuid": (1,)}]


def test_dispatch_update_info_callable_mode():
    send_mode = MagicMock()
    args = MagicMock(send_mode=send_mode)
    update_info = _UpdateInfo(["w"], ["float32"], [[1]], False, [{"uuid": (1,)}])

    dispatch_update_info(
        args=args,
        update_info=update_info,
        update_fields={"names": ["w"], "ipc_handles": update_info.ipc_handles},
        ipc_handles=update_info.ipc_handles,
    )

    send_mode.assert_called_once_with(update_info)


def test_dispatch_update_info_http_mode_posts_pickled_handles():
    response = MagicMock()
    args = MagicMock(send_mode="http", url="http://localhost:8000/")
    ipc_handles = [{"uuid": (1, 2, 3)}]
    update_info = _UpdateInfo(["w"], ["float32"], [[1]], False, ipc_handles)

    with patch("vllm_ascend.distributed.weight_transfer.trainer_send.requests.post", return_value=response) as mock_post:
        dispatch_update_info(
            args=args,
            update_info=update_info,
            update_fields={
                "names": ["w"],
                "dtype_names": ["float32"],
                "shapes": [[1]],
                "packed": False,
                "ipc_handles": ipc_handles,
            },
            ipc_handles=ipc_handles,
        )

    mock_post.assert_called_once()
    url = mock_post.call_args.args[0]
    payload = mock_post.call_args.kwargs["json"]
    assert url == "http://localhost:8000/update_weights"
    assert "ipc_handles" not in payload["update_info"]
    assert "ipc_handles_pickled" in payload["update_info"]
    assert payload["update_info"]["names"] == ["w"]
    response.raise_for_status.assert_called_once()


def test_dispatch_update_info_rejects_unknown_send_mode():
    args = MagicMock(send_mode="unsupported")
    update_info = _UpdateInfo(["w"], ["float32"], [[1]], False, [])

    with pytest.raises(ValueError, match="Unsupported weight transfer send_mode"):
        dispatch_update_info(
            args=args,
            update_info=update_info,
            update_fields={"names": ["w"]},
            ipc_handles=[],
        )
