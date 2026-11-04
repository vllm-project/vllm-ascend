# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import vllm.distributed.device_communicators.cuda_communicator as cuda_comm_mod

from vllm_ascend.worker.v2 import utils as v2_utils


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda", "npu"),
        ("cuda:2", "npu:2"),
        (torch.device("cuda"), "npu"),
        (torch.device("cuda:3"), "npu:3"),
        ("cpu", "cpu"),
        (torch.device("cpu"), "cpu"),
        ("meta", "meta"),
        (None, None),
    ],
)
def test_empty_mapping_persists_and_is_idempotent(device, expected):
    original_empty = MagicMock(return_value="allocation")
    fake_torch = SimpleNamespace(cuda=MagicMock(), npu=MagicMock(), empty=original_empty, device=torch.device)
    with patch.object(v2_utils, "torch", fake_torch):
        with v2_utils.torch_cuda_wrapper():
            wrapped = fake_torch.empty
        with v2_utils.torch_cuda_wrapper():
            assert fake_torch.empty is wrapped
        assert fake_torch.empty(2, device=device, dtype=torch.uint8) == "allocation"
        original_empty.assert_called_once()
        kwargs = original_empty.call_args.kwargs
        actual_device = kwargs["device"]
        assert (None if actual_device is None else str(actual_device)) == expected
        assert kwargs["dtype"] == torch.uint8


def test_empty_mapping_preserves_allocation_errors():
    fake_torch = SimpleNamespace(empty=MagicMock(side_effect=RuntimeError("allocation failed")), device=torch.device)
    with patch.object(v2_utils, "torch", fake_torch):
        v2_utils._wrap_torch_empty()
        with pytest.raises(RuntimeError, match="allocation failed"):
            fake_torch.empty(2, device="cuda")


def test_v2_utils_context_managers_switch_and_restore():
    fake_cuda = MagicMock()
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    original_comm = cuda_comm_mod.CudaCommunicator
    npu_cls = object()

    with (
        patch.object(
            v2_utils, "torch", SimpleNamespace(cuda=fake_cuda, npu=fake_npu, empty=MagicMock(), device=torch.device)
        ),
        patch.object(v2_utils, "breakable_cudagraph", MagicMock()),
        patch.object(v2_utils, "weak_ref_workspaces") as weak_ref,
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
        patch.object(
            v2_utils,
            "get_ascend_config",
            return_value=SimpleNamespace(ascend_compilation_config=SimpleNamespace(enable_super_kernel=False)),
        ),
        patch.object(v2_utils.logger, "info_once", create=True),
        patch.object(v2_utils.logger, "debug"),
        patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator",
            npu_cls,
        ),
    ):
        with v2_utils.torch_cuda_wrapper():
            assert fake_cuda.Event is fake_npu.Event
            assert fake_cuda.graph is v2_utils.torch_npu_graph_wrapper
            assert fake_cuda.is_current_stream_capturing is fake_npu.is_current_stream_capturing
            assert v2_utils.breakable_cudagraph.weak_ref_tensor is v2_utils.weak_ref_tensor
        # Mapping is process-wide and must survive runner init so GPU V2
        # load_model / PrefetchOffloader can query capture state on NPU.
        assert fake_cuda.is_current_stream_capturing is fake_npu.is_current_stream_capturing

        with v2_utils.communicator_switch():
            assert cuda_comm_mod.CudaCommunicator is npu_cls
        assert cuda_comm_mod.CudaCommunicator is original_comm

        with v2_utils.torch_npu_graph_wrapper("capture"):
            pass
        weak_ref.assert_any_call("graph")
        weak_ref.assert_any_call("draft")
        assert weak_ref.call_count == 2
