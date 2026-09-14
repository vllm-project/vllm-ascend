# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import vllm.distributed.device_communicators.cuda_communicator as cuda_comm_mod

from vllm_ascend.worker.v2 import utils as v2_utils


def test_v2_utils_context_managers_switch_and_restore():
    fake_cuda = MagicMock()
    fake_npu = MagicMock()
    fake_npu.graph.return_value = nullcontext()
    original_comm = cuda_comm_mod.CudaCommunicator
    npu_cls = object()

    with (
        patch.object(v2_utils, "torch", SimpleNamespace(cuda=fake_cuda, npu=fake_npu)),
        patch.object(v2_utils, "breakable_cudagraph", MagicMock()),
        patch.object(v2_utils, "weak_ref_workspaces") as weak_ref,
        patch.object(v2_utils, "get_graph_params", return_value="graph"),
        patch.object(v2_utils, "get_draft_graph_params", return_value="draft"),
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
            assert v2_utils.breakable_cudagraph.weak_ref_tensor is v2_utils.weak_ref_tensor

        with v2_utils.communicator_switch():
            assert cuda_comm_mod.CudaCommunicator is npu_cls
        assert cuda_comm_mod.CudaCommunicator is original_comm

        with v2_utils.torch_npu_graph_wrapper("capture"):
            pass
        weak_ref.assert_any_call("graph")
        weak_ref.assert_any_call("draft")
        assert weak_ref.call_count == 2
