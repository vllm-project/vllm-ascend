# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2 import dp_utils as ascend_dp_utils


def test_eager_padding_uses_only_existing_cpu_metadata(monkeypatch):
    desc = BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=3, num_reqs=1)
    counts = torch.tensor([3, 7], dtype=torch.int32, device="cpu")
    monkeypatch.setattr(ascend_dp_utils, "oproj_tp_enable", lambda: True)
    upstream = MagicMock(return_value=(desc, counts))
    monkeypatch.setattr(ascend_dp_utils, "upstream_dispatch_cg_and_sync_dp", upstream)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("unexpected scalar extraction")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected transfer")),
        patch.object(torch.Tensor, "to", side_effect=AssertionError("unexpected transfer")),
        patch.object(dp_utils.dist, "all_reduce", side_effect=AssertionError("unexpected collective")),
    ):
        padded, across_dp = ascend_dp_utils.dispatch_cg_and_sync_dp(None, 1, 3, None, 2, 0)
    upstream.assert_called_once_with(None, 1, 3, None, 2, 0)
    assert padded.num_tokens == 7
    assert across_dp is counts
    assert counts.tolist() == [7, 7]


@pytest.mark.parametrize("tp_component", [None, "oproj", "embedding", "mlp", "lmhead"])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("counts", [(3, 7), (0, 7), (0, 0), (3,)])
def test_finegrained_tp_eager_dp_padding(tp_component, mode, counts, monkeypatch):
    for component in ("oproj", "embedding", "mlp", "lmhead"):
        monkeypatch.setattr(ascend_dp_utils, f"{component}_tp_enable", lambda name=component: name == tp_component)

    def all_reduce(tensor, group):
        tensor[0] = torch.tensor(counts, dtype=torch.int32)
        tensor[1].fill_(mode.value)

    collective = MagicMock(side_effect=all_reduce)
    monkeypatch.setattr(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(dp_utils.dist, "all_reduce", collective)
    manager = MagicMock()
    manager.dispatch.side_effect = lambda num_reqs, num_tokens, *args, **kwargs: BatchExecutionDescriptor(
        cg_mode=mode, num_tokens=num_tokens, num_reqs=num_reqs, num_active_loras=2
    )
    desc, across_dp = ascend_dp_utils.dispatch_cg_and_sync_dp(
        manager,
        1,
        counts[0],
        None,
        len(counts),
        0,
        need_eager=mode == CUDAGraphMode.NONE,
        num_active_loras=2,
    )

    assert collective.call_count == int(len(counts) > 1)
    if len(counts) == 1 or not any(counts):
        assert across_dp is None
        assert desc.num_tokens == counts[0]
        return
    should_pad = mode != CUDAGraphMode.NONE or tp_component is not None
    assert desc.num_tokens == (max(counts) if should_pad else counts[0])
    assert desc.num_reqs == 1
    assert desc.num_active_loras == 2
    assert desc.cg_mode == mode
    assert across_dp.tolist() == ([max(counts)] * len(counts) if should_pad else list(counts))
