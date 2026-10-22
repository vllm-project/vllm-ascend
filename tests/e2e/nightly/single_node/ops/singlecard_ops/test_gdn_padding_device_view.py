# SPDX-License-Identifier: Apache-2.0
"""A5 validation of device-only graph-padding boundary normalization."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.ops.gdn_attn_builder import _remove_spec_graph_padding_queries


class ForbidHostUpload(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default and args[0].device.type == "cpu":
            raise AssertionError("Padding normalization must not upload a CPU tensor")
        return func(*args, **kwargs)


@pytest.mark.parametrize("live", [1, 2, 5, 6, 7])
def test_device_padding_preserves_real_queries_and_shared_metadata(live):
    capacity = 8
    cpu = torch.arange(capacity + 1, dtype=torch.int32) * 6
    device = cpu.to("npu")
    metadata = SimpleNamespace(
        num_reqs=capacity,
        seq_lens_cpu_upper_bound=torch.tensor([262144] * live + [0] * (capacity - live)),
        query_start_loc_cpu=cpu,
        query_start_loc=device,
    )
    metadata.replace = lambda **changes: SimpleNamespace(**{**vars(metadata), **changes})
    drafts = torch.tensor([5] * live + [-1] * (capacity - live))
    torch.npu.synchronize()
    with ForbidHostUpload():
        result = _remove_spec_graph_padding_queries(metadata, drafts)
    torch.npu.synchronize()
    expected = torch.arange(capacity + 1, dtype=torch.int32).clamp(max=live) * 6
    torch.testing.assert_close(result.query_start_loc.cpu(), expected)
    torch.testing.assert_close(result.query_start_loc_cpu, expected)
    torch.testing.assert_close(device.cpu(), cpu)
    assert result.query_start_loc.device.type == "npu"
    assert result.num_actual_tokens == live * 6
