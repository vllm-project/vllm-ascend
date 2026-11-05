# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops.triton.query_gather_prep as query_prep


@pytest.mark.parametrize(
    "width,offset,step",
    [(192, 128, 1), (128, 0, 2), (63, 0, 1)],
    ids=["projection-tail", "feature-stride", "masked-tail"],
)
def test_query_gather_launches_masked_kernel_for_legal_views(monkeypatch, width, offset, step):
    kernel = MagicMock()
    monkeypatch.setattr(query_prep, "_q_gather_prep_head_major_kernel", kernel)
    monkeypatch.setattr(query_prep, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(query_prep, "get_vectorcore_num", lambda: 8)
    monkeypatch.setattr(query_prep, "is_950", lambda: False)
    query = torch.empty(3, 12, 512)
    tail = torch.empty(3, 12, width)[..., offset::step]

    output = query_prep.prep_query_head_major(query, tail)

    launch = kernel.__getitem__.return_value
    launch.assert_called_once()
    args, kwargs = launch.call_args
    assert args[0] is query
    assert args[1] is tail
    assert args[2] is output
    assert args[3:8] == (3, 12, 512, tail.shape[-1], 512 + tail.shape[-1])
    assert args[8:11] == query.stride()
    assert args[11:14] == tail.stride()
    assert kwargs["BLOCK_R"] == 64
    assert output.shape == (12, 3, 512 + tail.shape[-1])
    assert output.dtype == query.dtype
    assert output.is_contiguous()
