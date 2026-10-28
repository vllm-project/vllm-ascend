# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2 import aclgraph_utils as graph_mod
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.pp_transport import PPTokenShardBuffer


def _state(monkeypatch):
    state = AscendModelState.__new__(AscendModelState)
    state.prepare_engram_inputs = lambda *args: {}
    state.prepare_engram_dummy_inputs = lambda *args: {}
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    return state


def test_model_state_activates_shards_before_parent_input_preparation(monkeypatch):
    state = _state(monkeypatch)
    state.pp_input_buffer = PPTokenShardBuffer(
        IntermediateTensors({"hidden_states": torch.zeros(4, 2), "aux": torch.zeros(8, 2)}),
        ("hidden_states",),
        2,
    )

    def prepare_parent(self, input_batch, req_states):
        assert self.pp_num_tokens == 3
        assert self.pp_input_buffer.active["hidden_states"].shape == (2, 2)
        assert self.pp_input_buffer.active["aux"].shape == (3, 2)
        return {"prepared": True}

    monkeypatch.setattr(DefaultModelState, "prepare_inputs", prepare_parent)
    assert state.prepare_inputs(SimpleNamespace(num_tokens_after_padding=3), None) == {"prepared": True}


def _manager(state):
    active = IntermediateTensors(
        {
            "hidden_states": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "residual": torch.zeros(4, 1, 2),
            "aux": torch.zeros(8, 2),
            "topk": torch.zeros(8, 4, dtype=torch.int64),
        }
    )
    input_buffer = PPTokenShardBuffer(active, ("hidden_states", "residual"), 2)
    input_buffer.activate(1)
    state.pp_input_buffer = input_buffer
    manager = graph_mod.ModelAclGraphManager.__new__(graph_mod.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(pp_sharded_tensor_keys=("hidden_states", "residual"), model_state=state)
    manager.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
    manager.capture_sizes = [1, 3, 8]
    manager.is_last_pp_rank = False
    manager.intermediate_tensors = None
    manager.update_stream = object()
    return manager


class _GrowingResidualBank(nn.Module):
    def forward(self, positions, intermediate_tensors):
        hidden = intermediate_tensors["hidden_states"]
        assert hidden.shape[0] == (positions.shape[0] + 1) // 2
        return IntermediateTensors(
            {
                "hidden_states": hidden + 1,
                "residual": hidden[:, None, :].expand(-1, 3, -1),
                "aux": intermediate_tensors["aux"] + 2,
                "topk": intermediate_tensors["topk"] + 3,
            }
        )


def test_capture_and_replay_preserve_mixed_layout_and_outgoing_bank_shape(monkeypatch):
    state = _state(monkeypatch)
    manager = _manager(state)
    previous_layout = Mock()
    state.pp_capture_layout = previous_layout
    monkeypatch.setattr(graph_mod, "communicator_switch", nullcontext)
    monkeypatch.setattr(graph_mod, "_EXTRA_CTX", SimpleNamespace())
    monkeypatch.setattr(
        graph_mod, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)

    def capture_parent(self, model, model_state, input_buffers, intermediate_tensors, *args, **kwargs):
        # Exercise the core's full-token slices and persistent output copies.
        for num_tokens in (8, 3, 1, 8):
            model_state.prepare_dummy_inputs(1, num_tokens)
            output = model(positions=torch.arange(num_tokens), intermediate_tensors=intermediate_tensors[:num_tokens])
            for key, tensor in output.tensors.items():
                self.intermediate_tensors[key][:num_tokens].copy_(tensor)
            assert self.intermediate_tensors["aux"].shape[0] == num_tokens
            assert self.intermediate_tensors["topk"].shape[0] == num_tokens

    monkeypatch.setattr(ModelCudaGraphManager, "capture", capture_parent)
    manager.capture(_GrowingResidualBank(), state, None, state.pp_input_buffer.active, None, [], None)

    assert state.pp_capture_layout is previous_layout
    previous_layout.assert_not_called()
    assert state.pp_input_buffer.active["hidden_states"].shape == (1, 2)
    assert manager.pp_output_buffer.backing["residual"].shape == (4, 3, 2)
    assert manager.pp_output_buffer.backing["topk"].dtype == torch.int64
    pointer = manager.pp_output_buffer.backing["hidden_states"].data_ptr()

    state.attn_metadata = {}
    monkeypatch.setattr(graph_mod, "set_current_vllm_config", lambda *args: nullcontext())
    monkeypatch.setattr(graph_mod, "_get_graph_update_backend", lambda *args: object())
    monkeypatch.setattr(graph_mod, "use_updatable_graph", lambda *args: False)
    manager.model_runner.attn_groups = []
    manager._graph_relay = lambda backend, desc, num_tokens, metadata: manager.intermediate_tensors[:num_tokens]
    for num_tokens, shard_rows in ((3, 2), (1, 1), (8, 4)):
        output = manager.run_fullgraph(SimpleNamespace(num_tokens=num_tokens))
        assert output["hidden_states"].shape == (shard_rows, 2)
        assert output["residual"].shape == (shard_rows, 3, 2)
        assert output["aux"].shape == (num_tokens, 2)
        assert output["hidden_states"].data_ptr() == pointer
        torch.testing.assert_close(output["hidden_states"], torch.arange(shard_rows * 2).reshape(shard_rows, 2) + 1.0)

    manager.graphs = {"captured": object()}
    manager._graphs_captured = True
    capture_descs = {CUDAGraphMode.FULL: [SimpleNamespace(num_tokens=8)]}
    manager._capture_descs = capture_descs
    manager.breakable_cg_runner = None
    output_buffer = manager.pp_output_buffer
    pointers = {key: tensor.data_ptr() for key, tensor in output_buffer.backing.items()}
    manager.release_graphs()
    assert not manager.graphs
    assert not manager._graphs_captured
    assert manager._capture_descs is capture_descs
    assert manager.needs_capture()
    assert manager.pp_output_buffer is output_buffer
    assert manager.intermediate_tensors is output_buffer.active
    manager.capture(_GrowingResidualBank(), state, None, state.pp_input_buffer.active, None, [], None)
    assert manager.pp_output_buffer is output_buffer
    assert {key: tensor.data_ptr() for key, tensor in output_buffer.backing.items()} == pointers
    assert output_buffer.backing["residual"].shape == (4, 3, 2)


def test_capture_failure_restores_layout_callback(monkeypatch):
    state = _state(monkeypatch)
    manager = _manager(state)
    previous_layout = Mock()
    state.pp_capture_layout = previous_layout
    state.device_metadata = SimpleNamespace(finish=Mock())
    monkeypatch.setattr(graph_mod, "communicator_switch", nullcontext)

    def fail_capture(*args, **kwargs):
        state.prepare_dummy_inputs(1, 3)
        raise RuntimeError("capture failed")

    monkeypatch.setattr(ModelCudaGraphManager, "capture", fail_capture)
    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(_GrowingResidualBank(), state, None, None, None, [], None)

    assert state.pp_capture_layout is previous_layout
    state.device_metadata.finish.assert_called_once()
