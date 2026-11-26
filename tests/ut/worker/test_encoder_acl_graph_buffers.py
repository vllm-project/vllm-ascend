# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU storage/replay contracts, without simulating NPU streams or FIA kernels."""

import gc
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig, EncoderItemSpec

from vllm_ascend.worker import encoder_acl_graph as acl


class RecordingGraph(TorchDispatchMode):
    """Record real CPU tensor operations, including the captured output copy.

    This deliberately models only tensor data flow, not device graph execution.
    Replaying a Python forward alone would miss a copy outside capture.
    """

    def __init__(self):
        super().__init__()
        self.ops = []
        self.replays = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        self.ops.append((func, args, kwargs, result))
        return result

    @torch.inference_mode()
    def replay(self):
        self.replays += 1
        for func, args, kwargs, captured in self.ops:
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor) and result.data_ptr() != captured.data_ptr():
                captured.copy_(result)


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(acl.torch.npu, "NPUGraph", RecordingGraph)

    @contextmanager
    def graph_context(graph, pool):
        with graph:
            yield

    monkeypatch.setattr(acl.torch.npu, "graph", graph_context)
    monkeypatch.setattr(acl, "update_encoder_graph_params", MagicMock())
    monkeypatch.setattr(acl, "weak_ref_workspaces", MagicMock())
    model = MagicMock()
    model.get_encoder_cudagraph_config.return_value = EncoderCudaGraphConfig(
        modalities=["image"], buffer_keys=["pixels", "scale"], out_hidden_size=2
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[4, 8],
            encoder_cudagraph_max_vision_items_per_batch=1,
            encoder_cudagraph_max_frames_per_batch=None,
            encoder_cudagraph_max_token_budget=8,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(get_limit_per_prompt=lambda _: 0, mm_encoder_tp_mode="tensor")
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    def prepare(budget, *args):
        return SimpleNamespace(values={"pixels": torch.full((budget, 2), float(budget)), "scale": torch.tensor(budget)})

    model.prepare_encoder_cudagraph_capture_inputs.side_effect = prepare
    model.encoder_cudagraph_forward.side_effect = lambda values, path: values["pixels"] + values["scale"]
    model.prepare_encoder_cudagraph_replay_buffers.side_effect = lambda kwargs, *args: SimpleNamespace(values=kwargs)
    model.get_encoder_cudagraph_item_specs.side_effect = lambda kwargs: [
        EncoderItemSpec(input_size=len(kwargs["pixels"]), output_tokens=len(kwargs["pixels"]))
    ]
    mgr = acl.EncoderAclGraphManager(config, torch.device("cpu"), torch.float32, model)
    acl.set_encoder_graph_params(mgr.path_token_budgets)
    yield mgr
    mgr.clear()


def capture_pair(mgr, path="default", axis_keys=()):
    mgr._capture_budget_graph(8, path, axis_keys)
    mgr._capture_budget_graph(4, path, axis_keys)
    graphs = mgr._get_graph_set(path)
    return [graphs[(budget, axis_keys) if axis_keys else budget] for budget in (8, 4)]


def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def test_compatible_buffers_share_storage_and_keep_probe_contents(manager):
    large, small = capture_pair(manager)
    assert same_storage(large.input_buffers["pixels"], small.input_buffers["pixels"])
    assert small.input_buffers["pixels"].shape == (4, 2)
    assert torch.all(small.input_buffers["pixels"] == 4)
    assert not same_storage(large.input_buffers["scale"], small.input_buffers["scale"])
    assert large.input_buffers["scale"].item() == 8
    assert same_storage(large.output_buffer, small.output_buffer)
    assert small.output_buffer.shape == (4, 2)
    assert manager._shared_input_keys[("default", ())] == {"pixels"}
    for meta in (large, small):
        assert meta.graph.ops[-1][0] == torch.ops.aten.copy_.default
        assert meta.graph.ops[-1][1][0] is meta.output_buffer


@pytest.mark.parametrize("path,axes", [("other", ()), ("default", ("other-axis",))])
def test_path_and_axis_groups_do_not_alias(manager, path, axes):
    original, _ = capture_pair(manager)
    isolated, _ = capture_pair(manager, path, axes)
    assert not same_storage(original.input_buffers["pixels"], isolated.input_buffers["pixels"])
    assert not same_storage(original.output_buffer, isolated.output_buffer)
    assert (
        acl.get_encoder_graph_params().handles[acl._graph_key(8)]
        is not acl.get_encoder_graph_params().handles[acl._graph_key(8, path, axes)]
    )


@pytest.mark.parametrize("kind", ["shape", "dtype", "missing", "larger", "scalar"])
def test_incompatible_inputs_keep_own_allocation(manager, kind):
    original = manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect

    def prepare(budget, *args):
        values = original(budget, *args).values
        if budget == 8:
            if kind != "missing":
                values["extra"] = torch.ones(8, 2)
        else:
            values["extra"] = {
                "shape": lambda: torch.ones(4, 3),
                "dtype": lambda: torch.ones(4, 2, dtype=torch.float64),
                "missing": lambda: torch.ones(4, 2),
                "larger": lambda: torch.ones(9, 2),
                "scalar": lambda: torch.tensor(4),
            }[kind]()
        return SimpleNamespace(values=values)

    manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect = prepare
    large, small = capture_pair(manager)
    assert "extra" not in manager._shared_input_keys[("default", ())]
    if kind != "missing":
        assert not same_storage(large.input_buffers["extra"], small.input_buffers["extra"])
    assert same_storage(large.input_buffers["pixels"], small.input_buffers["pixels"])


def test_larger_output_keeps_previous_storage_live(manager):
    def forward(values, path):
        pixels = values["pixels"]
        if len(pixels) == 8:
            return pixels + 1
        return torch.ones(9, 2)

    manager.model.encoder_cudagraph_forward.side_effect = forward
    large, small = capture_pair(manager)
    assert not same_storage(large.output_buffer, small.output_buffer)
    gc.collect()
    large.graph.replay()
    # Small capture refilled only the shared input prefix.
    assert torch.all(large.output_buffer[:4] == 5)
    assert torch.all(large.output_buffer[4:] == 9)
    assert large.output_buffer.shape == (8, 2)


def test_alternating_replays_refresh_shared_outputs_and_padding(manager):
    large, small = capture_pair(manager)
    for budget, fill in [(8, 3), (4, 7), (8, 11)]:
        rows = budget - 1
        result = manager._run_budget_graph(
            {"pixels": torch.full((rows, 2), float(fill)), "scale": torch.tensor(fill)}, budget
        )
        assert torch.all(result[:rows] == fill * 2)
        assert torch.all(result[rows:] == fill)
    assert manager.graph_hits == 3
    assert large.graph.replays == 2
    assert small.graph.replays == 1


@pytest.mark.parametrize("budget", [4, 8])
def test_shared_input_cannot_be_omitted(manager, budget):
    capture_pair(manager)
    manager._get_item_specs = lambda _: [object()]
    with pytest.raises(RuntimeError, match="replay must provide"):
        manager._run_budget_graph({"scale": torch.tensor(1)}, budget)
    assert manager.graph_hits == 0


def test_oversize_replay_misses_before_any_copy(manager):
    _, small = capture_pair(manager)
    before = small.input_buffers["pixels"].clone()
    eager_output = torch.ones(5, 2)
    manager.model.encoder_eager_forward.return_value = eager_output
    result = manager._run_budget_graph({"pixels": torch.ones(5, 2), "scale": torch.tensor(10)}, 4)
    assert result is eager_output
    assert manager.graph_misses == 1
    assert small.graph.replays == 0
    assert torch.equal(before, small.input_buffers["pixels"])
    acl.update_encoder_graph_params.assert_not_called()


def test_nonshared_missing_input_keeps_per_budget_value(manager):
    capture_pair(manager)
    output = manager._run_budget_graph({"pixels": torch.ones(4, 2)}, 4)
    assert torch.all(output == 5)


def test_execute_clones_output_before_next_replay(manager):
    capture_pair(manager)
    manager.model.select_encoder_cudagraph_items.side_effect = lambda kwargs, indices: dict(kwargs)

    def postprocess(outputs, indices, counts, result, *, clone, batch_mm_kwargs):
        assert clone
        result[0] = outputs["default"][: counts[0]].clone()

    manager.model.postprocess_encoder_output.side_effect = postprocess
    first = manager._execute_local({"pixels": torch.ones(4, 2), "scale": torch.tensor(2)})[0]
    second = manager._execute_local({"pixels": torch.ones(8, 2), "scale": torch.tensor(5)})[0]
    assert torch.all(first == 3)
    assert torch.all(second == 6)
    assert not same_storage(first, second)


def test_clear_releases_backing_and_recapture_uses_new_storage(manager):
    large, small = capture_pair(manager)
    backing = weakref.ref(manager._shared_output_buffers[("default", ())])
    del large, small
    manager.clear()
    gc.collect()
    assert backing() is None
    assert not manager._shared_input_buffers
    assert not manager._shared_output_buffers
    assert not manager._shared_input_keys
    assert acl.get_encoder_graph_params() is None
    assert manager.update_stream is None
    acl.set_encoder_graph_params(manager.path_token_budgets)
    capture_pair(manager)
    assert manager._shared_output_buffers


def test_partial_failure_cleans_state_and_allows_recapture(manager):
    prepare = manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect

    def fail_small(budget, *args):
        if budget == 4:
            raise RuntimeError("small capture failed")
        return prepare(budget, *args)

    manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect = fail_small
    with pytest.raises(RuntimeError, match="small capture failed"):
        manager.capture(graph_pool=object())
    assert all(not graphs for graphs in manager.budget_graphs.values())
    assert not manager._shared_input_buffers
    assert not manager._shared_output_buffers
    assert not manager._shared_input_keys
    assert not manager.is_captured()
    assert acl.get_encoder_graph_params() is None
    manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect = prepare
    manager.capture(graph_pool=object())
    assert manager.is_captured()
    manager.capture(graph_pool=object())
    assert len(manager._get_graph_set()) == 2


def test_probe_does_not_record_fia_tasks_and_capture_is_largest_first(manager):
    states = []

    def forward(values, path):
        context = acl.get_encoder_forward_context()
        states.append((len(values["pixels"]), context.capturing))
        return values["pixels"] + values["scale"]

    manager.model.encoder_cudagraph_forward.side_effect = forward
    manager.capture(graph_pool=object())
    assert states == [(8, False), (8, True), (4, False), (4, True)]
    assert not acl.get_encoder_forward_context().capturing


@pytest.mark.parametrize("custom", [False, True])
def test_replay_padding_contract(manager, custom):
    prepare = manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect

    def with_lengths(budget, *args):
        values = prepare(budget, *args).values
        values["cu_seqlens"] = torch.arange(budget + 1, dtype=torch.int32)
        return SimpleNamespace(values=values)

    def pad_lengths(dst, src):
        dst.fill_(99)
        dst[: len(src)].copy_(src)

    manager.model.prepare_encoder_cudagraph_capture_inputs.side_effect = with_lengths
    if custom:
        manager.config.padding_logics["cu_seqlens"] = pad_lengths
    large, _ = capture_pair(manager)
    manager._run_budget_graph(
        {"pixels": torch.ones(6, 2), "scale": torch.tensor(1), "cu_seqlens": torch.tensor([0, 2, 6])},
        8,
    )
    lengths = large.input_buffers["cu_seqlens"]
    assert lengths[:3].tolist() == [0, 2, 6]
    expected_tail = 99 if custom else (6 if acl._ENCODER_SUPPORTS_PADDING_LOGIC else 0)
    assert torch.all(lengths[3:] == expected_tail)


def test_execute_falls_back_when_replay_exceeds_capture(manager):
    _, small = capture_pair(manager)
    manager.model.select_encoder_cudagraph_items.side_effect = lambda kwargs, indices: dict(kwargs)
    manager.model.prepare_encoder_cudagraph_replay_buffers.side_effect = lambda *args: SimpleNamespace(
        values={"pixels": torch.ones(5, 2)}
    )
    eager_output = torch.full((4, 2), 17.0)
    manager.model.encoder_eager_forward.return_value = eager_output

    def postprocess(outputs, indices, counts, result, **kwargs):
        result[0] = outputs["default"].clone()

    manager.model.postprocess_encoder_output.side_effect = postprocess
    result = manager._execute_local({"pixels": torch.ones(4, 2)})
    assert torch.equal(result[0], eager_output)
    assert small.graph.replays == 0
    manager.model.encoder_eager_forward.assert_called_once()
