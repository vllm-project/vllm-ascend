# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock, call

import pytest
import torch
import torch_npu
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)

from vllm_ascend.distributed.eplb.eplb_communicator import AscendGlooEplbCommunicator
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts, EplbExpertTensorList
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.base import AscendMoEScheme
from vllm_ascend.quantization.nz import NzCastSpec
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ


@pytest.fixture
def communicator(monkeypatch):
    monkeypatch.setattr(EplbCommunicator, "_log_initialized", lambda self: None)
    return AscendGlooEplbCommunicator(cpu_group=MagicMock())


def test_communicator_reuses_upstream_gloo_staging(communicator):
    assert isinstance(communicator, TorchDistGlooStagedEplbCommunicator)
    assert communicator.needs_profile_buffer_reservation is False


def test_execute_uses_group_local_peer_ranks(communicator, monkeypatch):
    send_tensor = torch.arange(2)
    recv_tensor = torch.zeros(2)
    communicator.add_send([send_tensor], dst_rank=1, expert_id=3)
    communicator.add_recv([recv_tensor], src_rank=0, expert_id=3)
    staging_tensors = [torch.empty_like(send_tensor), torch.empty_like(recv_tensor)]
    monkeypatch.setattr(
        communicator,
        "_acquire_staging_buffer",
        MagicMock(side_effect=staging_tensors),
    )
    p2p_op = MagicMock(side_effect=lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr("vllm_ascend.distributed.eplb.eplb_communicator.P2POp", p2p_op)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    current_stream = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: current_stream)

    communicator.execute()

    assert p2p_op.call_args_list == [
        call(torch.distributed.isend, staging_tensors[0], group=communicator._cpu_group, group_peer=1),
        call(torch.distributed.irecv, staging_tensors[1], group=communicator._cpu_group, group_peer=0),
    ]


def test_pinned_staging_buffers_are_reused_between_transfers(communicator, monkeypatch):
    allocated = [object(), object()]
    empty_like = MagicMock(side_effect=allocated)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    first_transfer: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
    assert communicator._acquire_staging_buffer(tensor, first_transfer) is allocated[0]
    assert communicator._acquire_staging_buffer(tensor, first_transfer) is allocated[1]

    second_transfer: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
    assert communicator._acquire_staging_buffer(tensor, second_transfer) is allocated[0]
    assert empty_like.call_count == 2
    empty_like.assert_called_with(tensor, device="cpu", pin_memory=True)


def test_execute_uses_current_device_stream_when_stream_is_unset(communicator, monkeypatch):
    tensor = torch.zeros(1)
    communicator.add_send([tensor], dst_rank=0, expert_id=3)
    monkeypatch.setattr(communicator, "_acquire_staging_buffer", lambda *_args: tensor)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    current_stream = MagicMock()
    monkeypatch.setattr(torch.accelerator, "current_stream", lambda: current_stream)

    communicator.execute()

    current_stream.synchronize.assert_called_once_with()


@pytest.mark.parametrize(
    "device_type,npu_format",
    [("cpu", ACL_FORMAT_FRACTAL_ND), ("npu", ACL_FORMAT_FRACTAL_ND), ("npu", ACL_FORMAT_FRACTAL_NZ)],
)
def test_execute_converts_nz_receives_on_device(communicator, monkeypatch, device_type, npu_format):
    recv_tensor = MagicMock()
    recv_tensor.device.type = device_type
    staging_tensor = MagicMock()
    nz_tensor = MagicMock()
    communicator.add_recv([recv_tensor], src_rank=0, expert_id=3)
    monkeypatch.setattr(communicator, "_acquire_staging_buffer", lambda *_args: staging_tensor)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    monkeypatch.setattr(torch.accelerator, "current_stream", MagicMock())
    get_format = MagicMock(return_value=npu_format)
    format_cast = MagicMock(return_value=nz_tensor)
    monkeypatch.setattr(torch_npu, "get_npu_format", get_format, raising=False)
    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)

    communicator.execute()

    if device_type == "npu" and npu_format == ACL_FORMAT_FRACTAL_NZ:
        staging_tensor.to.assert_called_once_with(recv_tensor.device, non_blocking=True)
        format_cast.assert_called_once_with(staging_tensor.to.return_value, ACL_FORMAT_FRACTAL_NZ)
        recv_tensor.copy_.assert_called_once_with(nz_tensor, non_blocking=True)
    else:
        staging_tensor.to.assert_not_called()
        format_cast.assert_not_called()
        recv_tensor.copy_.assert_called_once_with(staging_tensor, non_blocking=True)
    if device_type == "cpu":
        get_format.assert_not_called()


def _staged_recv(monkeypatch, npu_format, cast_spec=None, expert_buffer=None):
    """Drive one receive into a buffer tensor that reports ``npu_format``.

    The communicator learns each buffer's cast at construction, the way the
    HIXL one learns the weights, and upstream hands add_recv() those exact
    buffer tensors.
    """
    if expert_buffer is None:
        destination = MagicMock()
        destination.device.type = "npu"
        expert_buffer = [EplbExpertTensorList([destination], cast_spec)]
    else:
        destination = expert_buffer[0][0]
    communicator = AscendGlooEplbCommunicator(cpu_group=MagicMock(), expert_buffer=expert_buffer)
    staging_tensor = MagicMock()
    communicator.add_recv([destination], src_rank=0, expert_id=3)
    monkeypatch.setattr(communicator, "_acquire_staging_buffer", lambda *_args: staging_tensor)
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.P2POp",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.distributed.eplb.eplb_communicator.batch_isend_irecv",
        lambda _ops: [],
    )
    monkeypatch.setattr(torch.accelerator, "current_stream", MagicMock())
    monkeypatch.setattr(torch_npu, "get_npu_format", MagicMock(return_value=npu_format), raising=False)
    format_cast = MagicMock()
    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)
    return communicator, destination, staging_tensor, format_cast


def test_execute_rebuilds_a_c0_variant_with_the_destination_cast(monkeypatch):
    """FRACTAL_NZ_C0_16 destinations need the cast that produced them.

    Converting to plain FRACTAL_NZ yields a different storage description, and
    copy_ refuses internal-format tensors whose descriptions disagree, so the
    hardcoded conversion made every parameterized expert weight unreachable.
    """
    spec = NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch.bfloat16)
    communicator, destination, staging_tensor, format_cast = _staged_recv(monkeypatch, 50, spec)

    communicator.execute()

    format_cast.assert_called_once_with(
        staging_tensor.to.return_value,
        ACL_FORMAT_FRACTAL_NZ,
        **spec.cast_kwargs(),
    )
    destination.copy_.assert_called_once_with(format_cast.return_value, non_blocking=True)


def test_execute_rejects_an_internal_format_whose_cast_is_unknown(monkeypatch):
    """Name the missing contract instead of failing inside copy_.

    Without the parameters the conversion lands on plain FRACTAL_NZ and copy_
    reports a storage-description mismatch, which says nothing about the quant
    method that has to report its cast.
    """
    communicator, destination, _, format_cast = _staged_recv(monkeypatch, 50, None)

    with pytest.raises(RuntimeError, match="get_eplb_weight_views"):
        communicator.execute()

    format_cast.assert_not_called()
    destination.copy_.assert_not_called()


def test_the_cast_travels_from_the_quant_method_to_the_receive_buffers():
    """The buffers are what the receive path writes into, so they carry the cast.

    A quant method hands the cast over on the view it exports; the layer keeps
    it through get_expert_weights(), the upstream buffer allocation inherits it
    through empty_like, and the communicator finds it by buffer identity. The
    adapter sits in the middle of that chain in production.
    """
    spec = NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch.bfloat16)

    class _Scheme(AscendMoEScheme):
        def get_weight(self, *args, **kwargs):
            return {}

        def get_dynamic_quant_param(self, *args, **kwargs):
            return {}

        def apply(self, *args, **kwargs):
            raise AssertionError("Exporting EPLB weight views runs no forward pass.")

        def get_eplb_weight_views(self, layer):
            experts = [torch.ones(3, 4, dtype=torch.int8) for _ in range(2)]
            return [EplbExpertTensorList(experts, spec)]

    routed_experts = AscendRoutedExperts.__new__(AscendRoutedExperts)
    routed_experts.local_num_experts = 2
    routed_experts.quant_method = AscendFusedMoEMethod(_Scheme(), MagicMock())
    views = list(routed_experts.get_expert_weights())

    # Upstream allocates one buffer set from the first layer's views.
    expert_buffer = [torch.empty_like(view) for view in views]
    communicator = AscendGlooEplbCommunicator(cpu_group=MagicMock(), expert_buffer=expert_buffer)

    for destination in expert_buffer[0]:
        assert communicator._destination_cast_kwargs(destination) == spec.cast_kwargs()


def test_every_layer_must_agree_with_the_shared_receive_buffers():
    """One buffer set serves every layer, so a layer cast differently cannot work."""
    spec = NzCastSpec(customize_dtype=torch.float8_e4m3fn)
    expert_buffer = [EplbExpertTensorList([MagicMock()], spec)]
    divergent_layer = [EplbExpertTensorList([MagicMock()])]

    with pytest.raises(ValueError, match="NZ cast layout has to match"):
        AscendGlooEplbCommunicator(
            cpu_group=MagicMock(),
            expert_buffer=expert_buffer,
            all_expert_weights=[divergent_layer],
        )
