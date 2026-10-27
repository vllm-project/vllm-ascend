# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for local-row PP buffers and model-state activation."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next.model import Glm5NextSPPPBuffer, Glm5NextSPPPError, Glm5NextSPPPSpec
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState, Glm5NextSPPPModelState


def _spec():
    return Glm5NextSPPPSpec(4, 0, 8, 3)


def _buffer(num_tokens=16):
    return Glm5NextSPPPBuffer.allocate(_spec(), num_tokens, torch.device("cpu"))


def _state(spec, tensors):
    state = Glm5NextSPPPModelState.__new__(Glm5NextSPPPModelState)
    state._sp_pp_capture_layout = None
    state.bind_sp_pp_inputs(spec, tensors)
    return state


def test_spec_is_immutable():
    spec = _spec()
    with pytest.raises(FrozenInstanceError):
        spec.tp_rank = 1


def test_allocated_local_buffers_have_the_mhc_shapes_and_dtypes():
    buffer = _buffer(5)
    assert buffer.capacity == 2
    expected = {
        "hidden_states": ((2, 8), torch.bfloat16),
        "residual": ((2, 3, 8), torch.bfloat16),
        "post": ((2, 3, 1), torch.float32),
        "comb": ((2, 3, 3), torch.float32),
    }
    assert set(buffer.active.tensors) == set(expected)
    for key, (shape, dtype) in expected.items():
        tensor = buffer.active[key]
        assert type(tensor) is torch.Tensor
        assert tensor.shape == shape
        assert tensor.dtype == dtype


@pytest.mark.parametrize("num_tokens", [1, 3, 4, 5, 15, 16])
def test_active_views_keep_capacity_addresses_and_upstream_slicing(num_tokens):
    buffer = _buffer()
    active = buffer.active
    addresses = {key: tensor.data_ptr() for key, tensor in buffer.backing.items()}

    for total in (1, num_tokens, 16, 3, num_tokens):
        buffer.activate(total)
        rows = (total + 3) // 4
        # Native runner [:T] must keep ordinary tensors already sized to local rows.
        upstream_input = active[:total]
        assert type(upstream_input) is IntermediateTensors
        assert all(
            type(tensor) is torch.Tensor and tensor.shape[0] == rows for tensor in upstream_input.tensors.values()
        )
        assert buffer.active is active
        assert {key: tensor.data_ptr() for key, tensor in active.tensors.items()} == addresses
        assert buffer.capacity == 4
        assert all(tensor.shape[0] == 4 for tensor in buffer.backing.values())


def test_native_receive_copy_writes_only_active_local_rows():
    buffer = _buffer()
    source = _buffer(5).active
    for tensor in buffer.backing.values():
        tensor.fill_(-1)
    for tensor in source.tensors.values():
        tensor.fill_(7)
    buffer.activate(5)

    # This is the native runner's destination[:T].copy_(source[:T]) operation.
    copied = IntermediateTensors(
        {key: value[:5].copy_(source.tensors[key][:5]) for key, value in buffer.active.tensors.items()}
    )
    assert all(value.shape[0] == 2 and torch.all(value == 7) for value in copied.tensors.values())
    assert all(torch.all(value[2:] == -1) for value in buffer.backing.values())


@pytest.mark.parametrize("num_tokens", [0, -1, 17])
def test_invalid_activation_does_not_replace_active_views(num_tokens):
    buffer = _buffer()
    buffer.activate(5)
    original_views = buffer.active.tensors
    with pytest.raises(Glm5NextSPPPError, match="positive|capacity"):
        buffer.activate(num_tokens)
    assert buffer.active.tensors is original_views


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "dtype", "rows", "empty", "device"])
def test_binding_rejects_invalid_buffer_layouts(fault):
    tensors = dict(_buffer().active.tensors)
    if fault == "missing":
        del tensors["comb"]
    elif fault == "extra":
        tensors["extra"] = torch.empty(4)
    elif fault == "shape":
        tensors["residual"] = tensors["residual"][:, :, :1]
    elif fault == "dtype":
        tensors["post"] = tensors["post"].bfloat16()
    elif fault == "rows":
        tensors["hidden_states"] = tensors["hidden_states"][:1]
    elif fault == "empty":
        tensors = {key: tensor[:0] for key, tensor in tensors.items()}
    else:
        tensors["post"] = tensors["post"].to("meta")
    with pytest.raises(Glm5NextSPPPError):
        Glm5NextSPPPBuffer(_spec(), IntermediateTensors(tensors))


def test_fork_shares_storage_with_an_independent_active_container():
    buffer = _buffer()
    capture = buffer.fork()
    assert capture.active is not buffer.active
    assert capture.active.tensors is not buffer.active.tensors
    buffer.activate(1)
    capture.activate(13)
    assert buffer.active["hidden_states"].shape[0] == 1
    assert capture.active["hidden_states"].shape[0] == 4
    assert all(buffer.active[key].data_ptr() == capture.active[key].data_ptr() for key in buffer.backing)
    capture.active["hidden_states"].fill_(3)
    assert torch.all(buffer.active["hidden_states"] == 3)


def test_model_state_activates_inputs_before_parent_preparation(monkeypatch):
    buffer = _buffer()
    state = _state(buffer.spec, buffer.active)
    calls = []
    result = {"existing_hybrid_inputs": object()}

    def parent_prepare(self, input_batch, req_states):
        assert req_states is result
        calls.append(buffer.active["hidden_states"].shape[0])
        return result

    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_inputs", parent_prepare)
    for total in (1, 13, 3, 16):
        assert state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=total), result) is result
    assert calls == [1, 4, 1, 4]
    assert state.sp_pp_input_buffer is not None
    assert state.sp_pp_input_buffer.active is buffer.active
    assert state.sp_pp_input_buffer.capacity == 4
    with pytest.raises(Glm5NextSPPPError, match="capacity"):
        state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=17), result)
    assert calls == [1, 4, 1, 4]


def test_first_stage_model_state_preserves_absent_receive_inputs(monkeypatch):
    spec = _spec()
    state = _state(spec, None)
    calls = []
    result = {"first_stage_inputs": object()}

    def parent_prepare(self, input_batch, req_states):
        calls.append((input_batch.num_tokens_after_padding, req_states))
        return result

    monkeypatch.setattr(AscendMambaHybridModelState, "prepare_inputs", parent_prepare)
    for total in (1, 13):
        assert state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=total), "kept") is result
    assert state.sp_pp_input_buffer is None
    assert calls == [(1, "kept"), (13, "kept")]
