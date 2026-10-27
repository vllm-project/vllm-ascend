# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker integration: local shards, lazy receives and default-path isolation."""

import gc
import importlib
import sys
import weakref
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next.model import (
    Glm5NextSPPPBuffer,
    Glm5NextSPPPSpec,
)
from vllm_ascend.worker.v2 import model_runner as runner_module


@pytest.fixture
def worker_module(monkeypatch):
    # Import the real worker while isolating its unused native sleep allocator.
    # This CPU transport suite never loads or calls a device kernel.
    native = ModuleType("vllm_ascend.vllm_ascend_C")
    for name in ("init_module", "python_create_and_map", "python_unmap_and_release"):
        setattr(native, name, Mock())
    with monkeypatch.context() as boundary:
        boundary.setitem(sys.modules, native.__name__, native)
        return importlib.import_module("vllm_ascend.worker.worker")


@pytest.fixture
def transport(monkeypatch, worker_module):
    group = SimpleNamespace(
        is_first_rank=False,
        is_last_rank=False,
        irecv_tensor_dict=Mock(),
        isend_tensor_dict=Mock(return_value=[]),
    )
    tp_group = object()
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(worker_module, "get_ascend_config", lambda: SimpleNamespace(msmonitor_use_daemon=False))
    monkeypatch.setattr(worker_module, "enable_sp", lambda config: config.sp)
    worker = worker_module.NPUWorker.__new__(worker_module.NPUWorker)
    worker.log_memory_stats = Mock()
    worker._pp_send_work = []
    worker.profiler = None
    worker.use_v2_model_runner = True
    worker.vllm_config = SimpleNamespace(sp=True, parallel_config=SimpleNamespace(distributed_executor_backend="mp"))
    worker.model_runner = SimpleNamespace(is_pooling_model=False, execute_model=Mock())
    return worker, group, tp_group


def _spec():
    return Glm5NextSPPPSpec(4, 2, 8, 3)


def _states(spec, total=5):
    return Glm5NextSPPPBuffer.allocate(spec, total, torch.device("cpu")).active


def _core_group(monkeypatch, direction="send", fail_after=None):
    """Run real Core posting/retention; replace only distributed device I/O."""
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0 if direction == "send" else 1,
        ranks=[0, 1],
        device_group=object(),
        cpu_group=object(),
        use_cpu_custom_send_recv=False,
        is_first_rank=direction == "send",
        is_last_rank=False,
        references=[],
        waits=[],
        posts=0,
        complete=False,
    )

    class Handle:
        def __init__(self, backing=None):
            self.backing = backing

        def wait(self):
            group.waits.append(id(self))

        def is_completed(self):
            return group.complete

    def send_metadata(metadata, dst):
        backing = torch.empty(8, dtype=torch.uint8)
        handle = Handle(backing)
        group.references.extend((weakref.ref(backing), weakref.ref(handle)))
        return handle

    def post(tensor, **kwargs):
        group.posts += 1
        if fail_after is not None and group.posts > fail_after:
            raise RuntimeError(f"{direction} failed after {fail_after} posts")
        handle = Handle()
        group.references.extend((weakref.ref(tensor), weakref.ref(handle)))
        return handle

    group.isend_object = send_metadata
    metadata, _ = parallel_state._split_tensor_dict(_states(_spec()).tensors)
    group.recv_object = lambda src: metadata
    group._should_use_all_gather = lambda *args: False
    group._reap_completed_isends = lambda: parallel_state.GroupCoordinator._reap_completed_isends(group)
    group.isend_tensor_dict = lambda *args, **kwargs: parallel_state.GroupCoordinator.isend_tensor_dict(
        group, *args, **kwargs
    )
    group.irecv_tensor_dict = lambda *args, **kwargs: parallel_state.GroupCoordinator.irecv_tensor_dict(
        group, *args, **kwargs
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "isend", post)
    monkeypatch.setattr(torch.distributed, "irecv", post)
    return group


@pytest.mark.parametrize("use_v2,sp", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("forward", [False, True])
def test_other_models_preserve_transport_objects_and_groups(transport, use_v2, sp, forward):
    worker, group, tp_group = transport
    worker.use_v2_model_runner = use_v2
    worker.vllm_config.sp = sp
    received = {"hidden_states": torch.ones(2, 4)}
    callbacks, handles = [Mock()], [Mock()]
    group.irecv_tensor_dict.return_value = received, handles, callbacks
    output = IntermediateTensors({"hidden_states": torch.zeros(2, 4)})
    output.kv_connector_output = None

    def execute(scheduled, intermediate):
        assert intermediate.tensors is received if forward else intermediate is None
        return output

    worker.model_runner.execute_model.side_effect = execute
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=int(forward))) is None
    expected_group = None if use_v2 and sp else tp_group
    if forward:
        group.irecv_tensor_dict.assert_called_once_with(all_gather_group=expected_group)
        handles[0].wait.assert_called_once()
        callbacks[0].assert_called_once()
    else:
        group.irecv_tensor_dict.assert_not_called()
    assert group.isend_tensor_dict.call_args.args[0] is output.tensors
    assert group.isend_tensor_dict.call_args.kwargs == {"all_gather_group": expected_group}
    assert "_glm5next_sp_pp_inflight" not in vars(worker)
    assert "_glm5next_sp_pp_send_refs" not in vars(worker)


@pytest.mark.parametrize("first,last", [(True, False), (False, False), (False, True)])
def test_local_shards_use_native_lazy_receive_and_wait_before_next_forward(transport, first, last):
    worker, group, _ = transport
    group.is_first_rank, group.is_last_rank = first, last
    spec = worker.model_runner._glm5next_sp_pp_spec = _spec()
    received = _states(spec).tensors
    handle, callback, old_send = Mock(), Mock(), Mock()
    group.irecv_tensor_dict.return_value = received, [handle], [callback]
    worker._pp_send_work = [old_send]
    output = None if last else _states(spec)

    def execute(scheduled, intermediate):
        old_send.wait.assert_called_once()
        if first:
            assert intermediate is None
        else:
            handle.wait.assert_not_called()
            assert intermediate.tensors is received
            handle.wait.assert_called_once()
            callback.assert_called_once()
            assert all(value.shape[0] == 2 for value in received.values())
        return output

    worker.model_runner.execute_model.side_effect = execute
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=5)) is None
    if last:
        group.isend_tensor_dict.assert_not_called()
    else:
        assert output is not None
        group.isend_tensor_dict.assert_called_once_with(output.tensors, all_gather_group=None)
        assert set(output.tensors) == set(spec.states)
    assert "_glm5next_sp_pp_inflight" not in vars(worker)
    assert "_glm5next_sp_pp_send_refs" not in vars(worker)


def test_core_owns_successful_send_buffers_until_completion_without_worker_refs(monkeypatch):
    group = _core_group(monkeypatch)
    handles = group.isend_tensor_dict(_states(_spec()).tensors, all_gather_group=None)
    assert group._pending_isends[0][0] is handles
    gc.collect()
    assert all(reference() is not None for reference in group.references)
    group._reap_completed_isends()
    assert len(group._pending_isends) == 1
    group.complete = True
    group._reap_completed_isends()
    assert not group._pending_isends
    # Drop the caller's handles too; Core already finished all work.
    del handles
    gc.collect()
    assert all(reference() is None for reference in group.references)


@pytest.mark.parametrize("failure", ["model", "lazy_receive"])
def test_target_idle_and_nonposting_errors_preserve_native_behavior(transport, worker_module, monkeypatch, failure):
    worker, group, _ = transport
    worker.model_runner._glm5next_sp_pp_spec = _spec()
    exit_worker = Mock()
    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    worker.model_runner.execute_model.return_value = None
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) is None
    group.irecv_tensor_dict.assert_not_called()
    error = RuntimeError("ordinary execution failure")
    handle = Mock()
    group.irecv_tensor_dict.return_value = _states(_spec()).tensors, [handle], []
    if failure == "lazy_receive":
        handle.wait.side_effect = error
        worker.model_runner.execute_model.side_effect = lambda scheduled, intermediate: intermediate.tensors
    else:
        group.is_first_rank = True
        worker.model_runner.execute_model.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=1))
    assert caught.value is error
    exit_worker.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_graph_factory_selects_private_class_only_for_capability(monkeypatch, enabled):
    runner = SimpleNamespace(_glm5next_sp_pp_spec=_spec() if enabled else None)
    original = runner_module.vllm_model_runner.ModelCudaGraphManager
    generic = Mock(return_value="generic")
    private = Mock(return_value="private")
    monkeypatch.setattr(runner_module, "ModelAclGraphManager", generic)
    from vllm_ascend.worker.v2 import aclgraph_utils as sp_pp_graph

    monkeypatch.setattr(sp_pp_graph, "Glm5NextSPPPGraphManager", private)
    with runner_module.graph_manager_wrapper(runner):
        result = runner_module.vllm_model_runner.ModelCudaGraphManager(
            "config", "device", "mode", 1, lora_capture_cases=[0], varlen_decode=True, ubatch_runner=None
        )
    selected, skipped = (private, generic) if enabled else (generic, private)
    assert result == ("private" if enabled else "generic")
    selected.assert_called_once_with(
        "config", "device", "mode", 1, runner, lora_capture_cases=[0], varlen_decode=True, ubatch_runner=None
    )
    skipped.assert_not_called()
    assert runner_module.vllm_model_runner.ModelCudaGraphManager is original


@pytest.mark.parametrize("enabled,first", [(False, True), (False, False), (True, True), (True, False)])
def test_load_model_binds_only_declared_target_layout(monkeypatch, enabled, first):
    from vllm.config.compilation import CUDAGraphMode

    warmup_calls = {}
    for module_name, names in (
        (
            "vllm_ascend.model_executor.warmup.early_kernel_warmup",
            ("join_early_kernel_warmup", "start_early_kernel_warmup"),
        ),
        ("vllm_ascend.model_executor.warmup.nz_warmup", ("join_nz_warm_thread", "start_nz_warm_thread")),
    ):
        module = ModuleType(module_name)
        for name in names:
            warmup_calls[name] = Mock()
            setattr(module, name, warmup_calls[name])
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(runner_module.GPUModelRunner, "load_model", Mock())
    monkeypatch.setattr(runner_module, "get_pd_dspark_aux_layer_ids", lambda config: ())
    runner = runner_module.NPUModelRunner.__new__(runner_module.NPUModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)
    runner.prompt_logprobs_worker = object()
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import Glm5NextSPPPModelState

    runner.model_state = Glm5NextSPPPModelState.__new__(Glm5NextSPPPModelState) if enabled else SimpleNamespace()
    runner.model = SimpleNamespace()
    spec = Glm5NextSPPPSpec(4, 2, 8, 3)
    runner.intermediate_tensors = None if first else Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu")).active
    original_inputs = runner.intermediate_tensors
    if enabled:
        runner.model._glm5next_sp_pp_spec = spec
    runner.load_model()
    for call in warmup_calls.values():
        call.assert_called_once()
    assert isinstance(runner.model_state, (Glm5NextSPPPModelState, SimpleNamespace))
    assert runner.model_state.prompt_logprobs_worker is runner.prompt_logprobs_worker
    assert runner.intermediate_tensors is original_inputs
    if enabled:
        assert isinstance(runner.model_state, Glm5NextSPPPModelState)
        assert runner._glm5next_sp_pp_spec is spec
        buffer = runner.model_state.sp_pp_input_buffer
        if first:
            assert buffer is None
        else:
            assert buffer is not None
            assert buffer.active is original_inputs
            buffer.activate(1)
            assert original_inputs["hidden_states"].shape[0] == 1
            buffer.activate(13)
            assert original_inputs["hidden_states"].shape[0] == 4
    else:
        assert runner._glm5next_sp_pp_spec is None
        assert not hasattr(runner.model_state, "sp_pp_input_buffer")


@pytest.mark.parametrize("direction", ["receive", "send"])
@pytest.mark.parametrize("posted_count", [1, 2])
@pytest.mark.parametrize("logging_fails", [False, True])
def test_partial_core_posting_exits_with_original_frame_buffers_alive(
    transport, worker_module, monkeypatch, direction, posted_count, logging_fails
):
    worker, _, _ = transport
    worker.model_runner._glm5next_sp_pp_spec = _spec()
    group = _core_group(monkeypatch, direction, posted_count)
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
    worker.model_runner.execute_model.return_value = _states(_spec())

    class FatalExit(BaseException):
        pass

    def exit_worker(code):
        assert code != 0
        error = sys.exc_info()[1]
        if logging_fails:
            assert isinstance(error, RuntimeError) and str(error) == "logging failed"
            error = error.__context__
        assert type(error) is RuntimeError
        assert str(error) == f"{direction} failed after {posted_count} posts"
        gc.collect()
        assert len(group.references) == 2 * posted_count + (2 if direction == "send" else 0)
        assert all(reference() is not None for reference in group.references)
        trace = error.__traceback__
        method = "isend_tensor_dict" if direction == "send" else "irecv_tensor_dict"
        while trace is not None and trace.tb_frame.f_code.co_name != method:
            trace = trace.tb_next
        assert trace is not None
        assert len(trace.tb_frame.f_locals["handles"]) == posted_count + (direction == "send")
        assert not getattr(group, "_pending_isends", None)
        assert group.waits == []
        raise FatalExit

    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    monkeypatch.setattr(
        worker_module.logger, "exception", Mock(side_effect=RuntimeError("logging failed") if logging_fails else None)
    )
    with pytest.raises(FatalExit):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
    assert group.posts == posted_count + 1
    assert "_glm5next_sp_pp_inflight" not in vars(worker)
    assert "_glm5next_sp_pp_send_refs" not in vars(worker)


@pytest.mark.parametrize("direction", ["receive", "send"])
def test_other_models_preserve_original_posting_exception(transport, worker_module, monkeypatch, direction):
    worker, group, _ = transport
    group.is_first_rank = direction == "send"
    error = RuntimeError(f"ordinary {direction} failure")
    post = group.irecv_tensor_dict if direction == "receive" else group.isend_tensor_dict
    post.side_effect = error
    worker.model_runner.execute_model.return_value = IntermediateTensors({"hidden_states": torch.ones(2, 4)})
    exit_worker = Mock()
    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    with pytest.raises(RuntimeError) as caught:
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    assert caught.value is error
    exit_worker.assert_not_called()
    post.assert_called_once()
    assert "_glm5next_sp_pp_inflight" not in vars(worker)
    assert "_glm5next_sp_pp_send_refs" not in vars(worker)


@pytest.mark.parametrize("logging_fails", [False, True])
def test_partial_metadata_send_retains_both_serialized_buffers_until_exit(
    transport, worker_module, monkeypatch, logging_fails
):
    worker, _, _ = transport
    worker.model_runner._glm5next_sp_pp_spec = _spec()
    worker.model_runner.execute_model.return_value = _states(_spec())
    group = _core_group(monkeypatch, "send", fail_after=1)
    group.isend_object = lambda obj, dst: parallel_state.GroupCoordinator.isend_object(group, obj, dst)
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
    original_post = torch.distributed.isend
    metadata_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def post(tensor, **kwargs):
        metadata_refs.append(weakref.ref(tensor))
        return original_post(tensor, **kwargs)

    monkeypatch.setattr(torch.distributed, "isend", post)

    class FatalExit(BaseException):
        pass

    def exit_worker(code):
        assert code != 0
        error = sys.exc_info()[1]
        if logging_fails:
            assert isinstance(error, RuntimeError) and str(error) == "logging failed"
            error = error.__context__
        assert type(error) is RuntimeError
        assert str(error) == "send failed after 1 posts"
        gc.collect()
        assert len(metadata_refs) == 2
        assert all(reference() is not None for reference in [*metadata_refs, *group.references])
        trace = error.__traceback__
        while trace is not None and trace.tb_frame.f_code.co_name != "isend_object":
            trace = trace.tb_next
        assert trace is not None
        assert len(trace.tb_frame.f_locals["works"]) == 1
        assert len(trace.tb_frame.f_locals["retained"]) == 2
        assert not getattr(group, "_pending_isends", None)
        assert group.waits == []
        raise FatalExit

    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    monkeypatch.setattr(
        worker_module.logger, "exception", Mock(side_effect=RuntimeError("logging failed") if logging_fails else None)
    )
    with pytest.raises(FatalExit):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
    assert group.posts == 2
