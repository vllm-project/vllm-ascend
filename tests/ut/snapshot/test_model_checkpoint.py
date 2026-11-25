# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.snapshot.model_runner_lifecycle.h2d_copy import _Int8NZCopyStrategy, _W4A8V1NZPackedCopyStrategy
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import restore_state_dict


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.register_buffer("scale", torch.zeros(2))


def test_restore_state_dict_copies_parameters_and_buffers(tmp_path) -> None:
    path = tmp_path / "model.pth"
    torch.save(
        {
            "weight": torch.tensor([1.0, 2.0]),
            "scale": torch.tensor([3.0, 4.0]),
        },
        path,
    )
    model = _Model()

    restore_state_dict(model, str(path), "model")

    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.scale, torch.tensor([3.0, 4.0]))


def test_restore_state_dict_rejects_missing_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Snapshot checkpoint does not exist"):
        restore_state_dict(_Model(), str(tmp_path / "missing.pth"), "model")


@patch(
    "vllm_ascend.snapshot.model_runner_lifecycle.h2d_copy.torch_npu.npu_format_cast",
    side_effect=lambda tensor, _: tensor,
)
def test_w4a8_v1_nz_packed_strategy_copies_bytes(mock_format_cast) -> None:
    src = torch.tensor([[0x01020304, -1]], dtype=torch.int32)
    dst = torch.zeros_like(src)

    _W4A8V1NZPackedCopyStrategy().copy(dst, src)

    torch.testing.assert_close(dst, src)
    mock_format_cast.assert_called_once()


def test_w4a8_v1_nz_packed_strategy_rejects_mismatched_state() -> None:
    dst = torch.zeros((1, 2), dtype=torch.int32)
    strategy = _W4A8V1NZPackedCopyStrategy()

    with pytest.raises(RuntimeError, match="expects int32"):
        strategy.copy(dst, torch.zeros((1, 2), dtype=torch.int16))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        strategy.copy(dst, torch.zeros((2, 2), dtype=torch.int32))


@patch("vllm_ascend.snapshot.model_runner_lifecycle.h2d_copy.torch_npu.npu_format_cast", side_effect=lambda t, _: t)
def test_int8_nz_strategy_stages_noncontiguous_weights(mock_cast) -> None:
    src = torch.arange(6, dtype=torch.int8).reshape(3, 2).T
    dst = torch.zeros(src.shape, dtype=torch.int8)
    _Int8NZCopyStrategy().copy(dst, src)
    torch.testing.assert_close(dst, src)
    staged, _ = mock_cast.call_args.args
    assert staged.is_contiguous()
    assert staged.data_ptr() != dst.data_ptr()


def test_int8_nz_strategy_preserves_broadcast_and_conversion() -> None:
    dst = torch.zeros(3, dtype=torch.int8)
    _Int8NZCopyStrategy().copy(dst, torch.tensor([2], dtype=torch.int32))
    torch.testing.assert_close(dst, torch.full_like(dst, 2))
