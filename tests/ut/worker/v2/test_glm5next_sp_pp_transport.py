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
from vllm.sequence import IntermediateTensors

from vllm_ascend.models.glm5next.model import (
    SP_PP_HEADER_KEY,
    Glm5NextSPPPBuffer,
    Glm5NextSPPPError,
    Glm5NextSPPPRuntime,
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


def _runtime():
    runtime = Glm5NextSPPPRuntime(Glm5NextSPPPSpec(4, 2, 8, 3), torch.device("cpu"))
    runtime.bind_input(Glm5NextSPPPBuffer.allocate(runtime.spec, 16, torch.device("cpu")).active)
    return runtime


def _states(runtime, total=5):
    return Glm5NextSPPPBuffer.allocate(runtime.spec, total, torch.device("cpu")).active


@pytest.mark.parametrize("use_v2,sp", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("forward", [False, True])
def test_other_models_preserve_transport_objects_and_groups(transport, use_v2, sp, forward):
    worker, group, tp_group = transport
    worker.use_v2_model_runner = use_v2
    worker.vllm_config.sp = sp
    received = {"hidden_states": torch.ones(2, 4)}
    callbacks = [Mock()]
    handles = [Mock()]
    group.irecv_tensor_dict.return_value = received, handles, callbacks
    output = IntermediateTensors({"hidden_states": torch.zeros(2, 4)})
    output.kv_connector_output = None

    def execute(scheduled, intermediate):
        if forward:
            assert intermediate.tensors is received
        else:
            assert intermediate is None
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
    sent = group.isend_tensor_dict.call_args
    assert sent.args[0] is output.tensors
    assert sent.kwargs == {"all_gather_group": expected_group}
    assert SP_PP_HEADER_KEY not in sent.args[0]
    assert "_glm5next_sp_pp_send_refs" not in vars(worker)


@pytest.mark.parametrize("first,last", [(True, False), (False, False), (False, True)])
def test_local_wire_header_and_send_lifetime(transport, first, last):
    worker, group, _ = transport
    group.is_first_rank, group.is_last_rank = first, last
    runtime = worker._glm5next_sp_pp_runtime = _runtime()
    received = dict(_states(runtime).tensors)
    received[SP_PP_HEADER_KEY] = (1, 5, 4, 2)
    handle, callback = Mock(), Mock()
    group.irecv_tensor_dict.return_value = received, [handle], [callback]
    output = None if last else _states(runtime)
    old_send = Mock()
    worker._pp_send_work = [old_send]
    worker._glm5next_sp_pp_send_refs = ("old",)

    def execute(scheduled, intermediate):
        old_send.wait.assert_called_once()
        assert worker._glm5next_sp_pp_send_refs is None
        runtime.prepare_inputs(5)
        if first:
            assert intermediate is None
        else:
            # Header validation must not eagerly wait for device receives.
            handle.wait.assert_not_called()
            assert set(intermediate.tensors) == set(runtime.spec.states)
            handle.wait.assert_called_once()
            callback.assert_called_once()
            assert all(value.shape[0] == 2 for value in intermediate.tensors.values())
        return output

    worker.model_runner.execute_model.side_effect = execute
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3)) is None
    assert SP_PP_HEADER_KEY in received  # Receive metadata was removed from a separate dictionary.
    if last:
        group.isend_tensor_dict.assert_not_called()
    else:
        assert output is not None
        wire = group.isend_tensor_dict.call_args.args[0]
        assert wire is not output.tensors
        assert wire[SP_PP_HEADER_KEY] == (1, 5, 4, 2)
        assert SP_PP_HEADER_KEY not in output.tensors
        assert worker._glm5next_sp_pp_send_refs == (output, wire)
        assert group.isend_tensor_dict.call_args.kwargs == {"all_gather_group": None}
        assert all(wire[key] is output[key] for key in runtime.spec.states)
    assert worker._glm5next_sp_pp_inflight == []
    with pytest.raises(Glm5NextSPPPError, match="no prepared real execution"):
        runtime.make_header()


@pytest.mark.parametrize("fault", ["missing", "lane", "shape", "dtype", "keys", "tokens"])
def test_protocol_failures_exit_before_copy_with_live_receive_refs(transport, worker_module, monkeypatch, fault):
    worker, group, _ = transport
    runtime = worker._glm5next_sp_pp_runtime = _runtime()
    received = dict(_states(runtime).tensors)
    received[SP_PP_HEADER_KEY] = (1, 5, 4, 2)
    if fault == "missing":
        del received[SP_PP_HEADER_KEY]
    elif fault == "lane":
        received[SP_PP_HEADER_KEY] = (1, 5, 4, 1)
    elif fault == "shape":
        received["hidden_states"] = torch.empty(5, 8, dtype=torch.bfloat16)
    elif fault == "dtype":
        received["post"] = received["post"].bfloat16()
    elif fault == "keys":
        received["unsupported_aux"] = torch.empty(2, 8)
    handle = Mock()
    group.irecv_tensor_dict.return_value = received, [handle], []

    def execute(scheduled, intermediate):
        runtime.prepare_inputs(6)  # Same local t as T=5; header T must still reject it.
        pytest.fail("A mismatched physical T reached the receive copy.")

    worker.model_runner.execute_model.side_effect = execute

    class FatalExit(BaseException):
        pass

    def exit_worker(code):
        assert code != 0
        assert worker._glm5next_sp_pp_inflight[0] is received
        handle.wait.assert_not_called()
        raise FatalExit

    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    monkeypatch.setattr(worker_module.logger, "exception", Mock())
    with pytest.raises(FatalExit):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    group.isend_tensor_dict.assert_not_called()
    if fault != "tokens":
        worker.model_runner.execute_model.assert_not_called()


def test_idle_and_ordinary_errors_do_not_enter_fatal_protocol(transport, worker_module, monkeypatch):
    worker, group, _ = transport
    runtime = worker._glm5next_sp_pp_runtime = _runtime()
    exit_worker = Mock()
    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    worker.model_runner.execute_model.return_value = None
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) is None
    group.irecv_tensor_dict.assert_not_called()
    group.isend_tensor_dict.assert_not_called()
    group.is_first_rank = True
    worker.model_runner.execute_model.side_effect = RuntimeError("unrelated failure")
    with pytest.raises(RuntimeError, match="unrelated failure"):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=1))
    exit_worker.assert_not_called()
    with pytest.raises(Glm5NextSPPPError, match="no prepared real execution"):
        runtime.make_header()


@pytest.mark.parametrize("enabled", [False, True])
def test_graph_factory_selects_private_class_only_for_capability(monkeypatch, enabled):
    runner = SimpleNamespace(_glm5next_sp_pp_runtime=_runtime() if enabled else None)
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
def test_load_model_binds_only_declared_target_runtime(monkeypatch, enabled, first):
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
    runner.model_state = SimpleNamespace()
    runner.model = SimpleNamespace()
    spec = Glm5NextSPPPSpec(4, 2, 8, 3)
    runner.intermediate_tensors = None if first else Glm5NextSPPPBuffer.allocate(spec, 16, torch.device("cpu")).active
    original_inputs = runner.intermediate_tensors
    if enabled:
        runner.model._glm5next_sp_pp_spec = spec
    runner.load_model()
    for call in warmup_calls.values():
        call.assert_called_once()
    assert runner.model_state.prompt_logprobs_worker is runner.prompt_logprobs_worker
    assert runner.intermediate_tensors is original_inputs
    if enabled:
        runtime = runner._glm5next_sp_pp_runtime
        assert runtime.spec is spec
        assert runner.model_state.sp_pp_runtime is runtime
        if first:
            assert runtime.input_buffer is None
        else:
            assert runtime.input_buffer.active is original_inputs
            runtime.prepare_inputs(1)
            assert original_inputs["hidden_states"].shape[0] == 1
            runtime.prepare_inputs(13)
            assert original_inputs["hidden_states"].shape[0] == 4
    else:
        assert runner._glm5next_sp_pp_runtime is None
        assert not hasattr(runner.model_state, "sp_pp_runtime")


def test_model_normalization_error_exits_with_receive_refs_retained(transport, worker_module, monkeypatch):
    worker, group, _ = transport
    runtime = worker._glm5next_sp_pp_runtime = _runtime()
    received = dict(_states(runtime).tensors)
    received[SP_PP_HEADER_KEY] = (1, 5, 4, 2)
    handle, callback = Mock(), Mock()
    group.irecv_tensor_dict.return_value = received, [handle], [callback]

    def execute(scheduled, intermediate):
        runtime.prepare_inputs(5)
        # A model-boundary error occurs after the ordinary lazy receive wait.
        assert set(intermediate.tensors) == set(runtime.spec.states)
        raise Glm5NextSPPPError("GLM SP+PP state does not match the local physical token count.")

    worker.model_runner.execute_model.side_effect = execute

    class FatalExit(BaseException):
        pass

    def exit_worker(code):
        assert code != 0
        assert worker._glm5next_sp_pp_inflight[0] is received
        handle.wait.assert_called_once()
        callback.assert_called_once()
        raise FatalExit

    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    monkeypatch.setattr(worker_module.logger, "exception", Mock())
    with pytest.raises(FatalExit):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    group.isend_tensor_dict.assert_not_called()


@pytest.mark.parametrize("direction", ["receive", "send"])
@pytest.mark.parametrize("posted_count", [1, 2])
def test_partial_posting_failure_exits_with_frame_owned_buffers_alive(
    transport, worker_module, monkeypatch, direction, posted_count
):
    worker, group, _ = transport
    runtime = worker._glm5next_sp_pp_runtime = _runtime()
    group.is_first_rank = direction == "send"
    references: list[weakref.ReferenceType[object]] = []
    waits = []

    class PostedHandle:
        def wait(self):
            waits.append(self)

    def partial_post(*args, **kwargs):
        # These objects have no owner outside this failing PP API frame. Like
        # Core, it has not yet returned its handles or registered pending sends.
        metadata_buffer = torch.empty(8, dtype=torch.uint8)
        partial_buffers = [torch.empty(2, 8) for _ in range(posted_count)]
        posted_handles = [PostedHandle() for _ in range(posted_count)]
        references.extend(weakref.ref(value) for value in [metadata_buffer, *partial_buffers, *posted_handles])
        raise RuntimeError(f"{direction} failed after {posted_count} posts")

    post = group.irecv_tensor_dict if direction == "receive" else group.isend_tensor_dict
    post.side_effect = partial_post

    def execute(scheduled, intermediate):
        runtime.prepare_inputs(5)
        return _states(runtime)

    worker.model_runner.execute_model.side_effect = execute

    class FatalExit(BaseException):
        pass

    def exit_worker(code):
        assert code != 0
        error = sys.exc_info()[1]
        assert isinstance(error, Glm5NextSPPPError)
        assert f"{direction} posting failed" in str(error)
        cause = error.__cause__
        assert type(cause) is RuntimeError
        assert str(cause) == f"{direction} failed after {posted_count} posts"
        gc.collect()
        assert len(references) == 1 + 2 * posted_count
        assert all(reference() is not None for reference in references)
        traceback = cause.__traceback__
        while traceback is not None and traceback.tb_frame.f_code.co_name != "partial_post":
            traceback = traceback.tb_next
        assert traceback is not None
        assert len(traceback.tb_frame.f_locals["posted_handles"]) == posted_count
        assert len(traceback.tb_frame.f_locals["partial_buffers"]) == posted_count
        assert waits == []
        raise FatalExit

    monkeypatch.setattr(worker_module.os, "_exit", exit_worker)
    monkeypatch.setattr(worker_module.logger, "exception", Mock())
    with pytest.raises(FatalExit):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=3))
    post.assert_called_once()
    if direction == "receive":
        worker.model_runner.execute_model.assert_not_called()
        group.isend_tensor_dict.assert_not_called()
    else:
        group.irecv_tensor_dict.assert_not_called()


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
