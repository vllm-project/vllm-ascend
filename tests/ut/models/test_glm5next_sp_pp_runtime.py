# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for native local-row PP buffers and the host wire protocol."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next.model import (
    Glm5NextSPPPBuffer,
    Glm5NextSPPPError,
    Glm5NextSPPPRuntime,
    Glm5NextSPPPSpec,
)


def _runtime(tp_rank=0):
    return Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, tp_rank, 8, 3), torch.device("cpu"))


def _payload(spec, num_tokens):
    return Glm5NextSPPPBuffer.allocate(spec, num_tokens, torch.device("cpu")).active


def test_spec_is_immutable_and_does_not_hold_runtime_state():
    spec = _runtime().spec
    with pytest.raises(FrozenInstanceError):
        spec.tp_rank = 1
    assert set(spec.__dict__) == {"tp_size", "tp_rank", "hidden_size", "residual_streams", "dtype"}


@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5, 15, 16])
def test_active_views_keep_local_capacity_addresses_and_upstream_slicing(num_tokens):
    runtime = _runtime()
    tensors = _payload(runtime.spec, 16)
    runtime.bind_input(tensors)
    backing = dict(runtime.input_buffer.backing)
    addresses = {key: tensor.data_ptr() for key, tensor in backing.items()}
    assert all(tensor.shape[0] == 4 for tensor in backing.values())

    for total in (1, num_tokens, 16, 3, num_tokens):
        runtime.prepare_inputs(total)
        rows = runtime.spec.local_rows(total)
        # Original upstream [:T] operates on ordinary tensors already sized t.
        upstream_input = tensors[:total]
        assert type(upstream_input) is IntermediateTensors
        assert all(
            type(tensor) is torch.Tensor and tensor.shape[0] == rows for tensor in upstream_input.tensors.values()
        )
        assert {key: tensor.data_ptr() for key, tensor in tensors.tensors.items()} == addresses
        assert all(tensor.shape[0] == 4 for tensor in backing.values())


def test_receive_copy_only_writes_local_rows_and_cannot_broadcast():
    runtime = _runtime()
    tensors = _payload(runtime.spec, 16)
    runtime.bind_input(tensors)
    runtime.begin_step((1, 5, 4, 0))
    source = _payload(runtime.spec, 5)
    for tensor in source.tensors.values():
        tensor.fill_(7)
    runtime.validate_wire(source.tensors, (1, 5, 4, 0))
    runtime.prepare_inputs(5)
    copied = {key: value[:5].copy_(source.tensors[key][:5]) for key, value in tensors.tensors.items()}
    assert all(value.shape[0] == 2 and torch.all(value == 7) for value in copied.values())
    # A [1,...] source would broadcast in copy_; protocol rejects it first.
    bad_source = {key: value[:1] for key, value in source.tensors.items()}
    with pytest.raises(Glm5NextSPPPError, match="shape or dtype"):
        runtime.validate_wire(bad_source, (1, 5, 4, 0))
    runtime.end_step()


@pytest.mark.parametrize("sender_tokens,receiver_tokens", [(5, 6), (1, 4), (9, 12)])
def test_equal_local_rows_do_not_hide_physical_token_mismatch(sender_tokens, receiver_tokens):
    runtime = _runtime()
    assert runtime.spec.local_rows(sender_tokens) == runtime.spec.local_rows(receiver_tokens)
    runtime.begin_step((1, sender_tokens, 4, 0))
    with pytest.raises(Glm5NextSPPPError, match="physical token mismatch"):
        runtime.prepare_inputs(receiver_tokens)
    runtime.end_step()


@pytest.mark.parametrize(
    "header",
    [None, (), (1, 5, 4), (True, 5, 4, 0), (2, 5, 4, 0), (1, 5, 8, 0), (1, 5, 4, 1), (1, 0, 4, 0)],
)
def test_malformed_or_wrong_lane_header_is_rejected(header):
    with pytest.raises(Glm5NextSPPPError):
        _runtime().validate_header(header)


def test_wire_requires_exact_keys_shape_and_dtype():
    runtime = _runtime()
    header = (1, 5, 4, 0)
    tensors = _payload(runtime.spec, 5).tensors
    runtime.validate_wire(tensors, header)
    for bad in (
        {**tensors, "unexpected": torch.empty(2)},
        {key: tensor for key, tensor in tensors.items() if key != "comb"},
        {**tensors, "post": tensors["post"].to(torch.bfloat16)},
        {**tensors, "residual": tensors["residual"][:, :, :1]},
    ):
        with pytest.raises(Glm5NextSPPPError):
            runtime.validate_wire(bad, header)


def test_dummy_and_finished_execution_cannot_publish_stale_transport_tokens():
    runtime = _runtime()
    runtime.prepare_inputs(9)
    with pytest.raises(Glm5NextSPPPError, match="no prepared real"):
        runtime.make_header()
    runtime.begin_step()
    with pytest.raises(Glm5NextSPPPError, match="already active"):
        runtime.begin_step()
    runtime.prepare_inputs(5)
    assert runtime.make_header() == (1, 5, 4, 0)
    runtime.end_step()
    runtime.prepare_inputs(13)
    with pytest.raises(Glm5NextSPPPError, match="no prepared real"):
        runtime.make_header()
    runtime.begin_step()
    with pytest.raises(Glm5NextSPPPError, match="no prepared real"):
        runtime.make_header()
    runtime.end_step()


def test_capture_input_container_shares_storage_but_not_active_view_dictionary():
    runtime = _runtime()
    tensors = _payload(runtime.spec, 16)
    runtime.bind_input(tensors)
    capture = runtime.input_buffer.fork()
    runtime.prepare_inputs(1)
    capture.activate(13)
    assert tensors["hidden_states"].shape[0] == 1
    assert capture.active["hidden_states"].shape[0] == 4
    assert tensors["hidden_states"].data_ptr() == capture.active["hidden_states"].data_ptr()
    with pytest.raises(Glm5NextSPPPError, match="capacity"):
        capture.activate(17)


@pytest.mark.parametrize("mixed", [False, True])
def test_first_stage_wire_checks_runner_device_without_receive_buffer(mixed):
    runtime = _runtime()
    assert runtime.input_buffer is None
    tensors = _payload(runtime.spec, 5).tensors
    runtime.validate_wire(tensors, (1, 5, 4, 0))
    keys = ("post",) if mixed else tuple(tensors)
    bad = {key: tensor.to("meta") if key in keys else tensor for key, tensor in tensors.items()}
    with pytest.raises(Glm5NextSPPPError, match="device mismatch"):
        runtime.validate_wire(bad, (1, 5, 4, 0))


@pytest.mark.parametrize("mixed", [False, True])
def test_binding_rejects_buffers_on_a_different_or_mixed_device(mixed):
    runtime = _runtime()
    tensors = _payload(runtime.spec, 16).tensors
    keys = ("post",) if mixed else tuple(tensors)
    bad = {key: tensor.to("meta") if key in keys else tensor for key, tensor in tensors.items()}
    with pytest.raises(Glm5NextSPPPError, match="device"):
        runtime.bind_input(IntermediateTensors(bad))
    assert runtime.input_buffer is None


def test_device_identity_normalizes_cpu_index_but_requires_accelerator_index():
    spec = Glm5NextSPPPSpec(4, 0, 8, 3)
    runtime = Glm5NextSPPPRuntime(spec, torch.device("cpu:0"))
    assert runtime.device == torch.device("cpu")
    runtime.validate_wire(_payload(spec, 5).tensors, (1, 5, 4, 0))
    with pytest.raises(ValueError, match="explicitly indexed"):
        Glm5NextSPPPRuntime(spec, torch.device("cuda"))
