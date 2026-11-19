# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

import vllm_ascend.worker.metadata_async as module
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask


@pytest.fixture
def env(monkeypatch):
    state = SimpleNamespace(calls=[], capturing=False, step=1, next_event=0)

    class Stream:
        def __init__(self, name="metadata"):
            self.name = name

        def wait_event(self, event):
            state.calls.append((self.name, "wait", event.name))

        def wait_stream(self, stream):
            state.calls.append((self.name, "wait_stream", stream.name))

    class Event:
        def __init__(self):
            self.name = f"event-{state.next_event}"
            state.next_event += 1

        def record(self, stream):
            state.calls.append((stream.name, "record", self.name))

        def wait(self, stream):
            state.calls.append((stream.name, "external_wait", self.name))

        def reset(self, stream):
            state.calls.append((stream.name, "reset", self.name))

    @contextmanager
    def stream_context(stream):
        previous, state.current = state.current, stream
        try:
            yield
        finally:
            state.current = previous

    state.current = state.main = Stream("main")
    state.context = SimpleNamespace(
        batch_descriptor=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        is_padding=SimpleNamespace(ndim=1, shape=(6,)),
    )
    monkeypatch.setattr(torch.npu, "Stream", Stream)
    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "ExternalEvent", Event, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: state.current)
    monkeypatch.setattr(torch.npu, "stream", stream_context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: state.capturing)
    monkeypatch.setattr(module, "get_forward_context", lambda: state.context)
    return state


def _config():
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        model_config=SimpleNamespace(architecture="DeepseekV41ForCausalLM"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        lora_config=None,
    )


def _factory(config, vllm_config=None, **overrides):
    options = dict(packed_cache=False, is_a3=True)
    options.update(overrides)
    return module.create_async_target_metadata(config, _config() if vllm_config is None else vllm_config, **options)


def _build(env, frontiers=None, *, fail=None):
    if frontiers is None:
        frontiers = (
            (DeviceMetadataStage.ATTENTION, 0),
            (DeviceMetadataStage.ATTENTION, 1),
            (DeviceMetadataStage.ATTENTION, 2),
            (DeviceMetadataStage.INDEXER, 1),
            (DeviceMetadataStage.INDEXER, 2),
        )

    class Provider:
        tasks = ()
        entries = 0

        @contextmanager
        def defer_device_metadata(self, *, in_graph):
            assert not in_graph, "A3 producer kernels must stay outside the model graph"
            self.entries += 1
            yield

        def take_device_metadata_tasks(self):
            tasks, self.tasks = self.tasks, ()
            return tasks

    provider = Provider()
    resource = SimpleNamespace()

    def build(**kwargs):
        step = env.step
        provider.tasks = tuple(
            DeviceMetadataTask(
                stage,
                lambda s=stage, g=group: env.calls.append(("task", s, g, env.current.name, step)),
                group,
            )
            for stage, group in frontiers
        )
        if fail is not None:
            raise fail
        return {"cache": resource}

    group = SimpleNamespace(get_metadata_builder=lambda _: provider)
    return provider, resource, build, dict(attn_groups=[[group]], num_input_tokens=6)


@pytest.mark.parametrize("config", [{}, {"enabled": False}])
def test_disabled_does_not_inspect_model_or_allocate_events(env, config):
    assert _factory(config, SimpleNamespace(), is_a3=False, packed_cache=True) is None
    assert env.next_event == 0


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        True,
        {"enabled": 0},
        {"enabled": 1},
        {"enabled": "true"},
        {"enabled": True, "mode": "M1"},
        {"enabled": True, "mode": "M2"},
        {"enabled": False, "mode": "M1"},
        {"enabled": True, "extra": 1},
    ],
)
def test_configuration_is_strict(env, config):
    with pytest.raises(ValueError):
        _factory(config)
    assert env.next_event == 0


@pytest.mark.parametrize("option", ["packed_cache", "is_a3"])
def test_unsupported_device_and_layout_rejected(env, option):
    with pytest.raises(ValueError):
        _factory({"enabled": True}, **{option: option != "is_a3"})


@pytest.mark.parametrize(
    "field,value",
    [("pipeline_parallel_size", 2), ("decode_context_parallel_size", 2)],
)
def test_parallel_configuration_rejected(env, field, value):
    config = _config()
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match="LoRA, PP or DCP"):
        _factory({"enabled": True}, config)


@pytest.mark.parametrize("case", ["model", "lora"])
def test_unsupported_model_or_lora_rejected(env, case):
    config = _config()
    if case == "model":
        config.model_config.architecture = "OtherModel"
    else:
        config.lora_config = object()
    with pytest.raises(ValueError):
        _factory({"enabled": True}, config)


@pytest.mark.parametrize("role", [None, "kv_consumer", "kv_producer", "kv_both"])
@pytest.mark.parametrize("pcp_size", [1, 4])
def test_factory_accepts_kv_roles_and_pcp(env, role, pcp_size):
    config = _config()
    config.kv_transfer_config = SimpleNamespace(kv_role=role) if role else None
    config.parallel_config.prefill_context_parallel_size = pcp_size
    assert isinstance(_factory({"enabled": True}, config), module.AsyncTargetDeviceMetadata)


def test_sas_and_qli_run_on_metadata_stream_each_step(env):
    metadata = _factory({"enabled": True})
    provider, resource, build, kwargs = _build(env)
    for step in (1, 2):
        env.step = step
        env.calls.clear()
        assert metadata.run_build(build, **kwargs)["cache"] is resource
        assert resource.device_metadata_executor is metadata.executor
        tasks = [call for call in env.calls if call[0] == "task"]
        assert len(tasks) == 5
        assert all(call[-1] == step for call in tasks)
        assert all(call[3] == "metadata" for call in tasks)
        metadata.begin_forward()
        metadata.executor.wait(DeviceMetadataStage.ATTENTION, 1)
        before = len(env.calls)
        metadata.executor.wait(DeviceMetadataStage.ATTENTION, 1)
        assert len(env.calls) == before
        with pytest.raises(RuntimeError, match="unregistered"):
            metadata.executor.wait(DeviceMetadataStage.INDEXER, 99)
        metadata.finish()
        assert not metadata.executor.submission_in_flight
        assert env.calls[-1][:2] == ("main", "record")
    assert provider.entries == 2


def test_inputs_ready_and_previous_consumer_fence_precede_producer(env):
    metadata = module.AsyncTargetDeviceMetadata()
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs)
    metadata.finish()
    reusable = env.calls[-1][2]
    env.calls.clear()
    metadata.run_build(build, **kwargs)
    first_task = next(i for i, call in enumerate(env.calls) if call[0] == "task")
    assert env.calls[0][:2] == ("main", "record")
    assert ("metadata", "wait", env.calls[0][2]) in env.calls[:first_task]
    assert ("metadata", "wait", reusable) in env.calls[:first_task]
    metadata.finish()


@pytest.mark.parametrize("sizes", [[], [0], [-1], [True], [1.0]])
def test_capture_sizes_require_actual_positive_token_counts(env, sizes):
    with pytest.raises(ValueError):
        module.AsyncTargetDeviceMetadata().configure_capture_sizes(sizes)


def test_capture_binding_and_replay_do_not_repeat_wait_reset(env):
    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6, 12])
    metadata.configure_capture_sizes([12, 6])
    _, _, build, kwargs = _build(env)
    kwargs["full_graph_mode"] = True
    metadata.run_build(build, **kwargs)
    metadata.begin_forward()  # NONE warmup uses the padded input length.
    env.capturing = True
    metadata.finish()
    resets = [c for c in env.calls if c[1] == "reset"]
    assert len(resets) == 5  # Dummy paths retire even unconsumed groups.
    env.capturing = False
    metadata.run_build(build, **kwargs)
    metadata.validate_replay(6)
    with pytest.raises(ValueError, match="Replay bucket"):
        metadata.validate_replay(12)
    metadata.finish_replay()
    assert [c for c in env.calls if c[1] == "reset"] == resets
    with pytest.raises(ValueError, match="Cannot change"):
        metadata.configure_capture_sizes([6])


def test_full_descriptor_and_none_warmup_must_agree(env):
    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6, 12])
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    env.context.batch_descriptor = BatchDescriptor(num_tokens=6)
    metadata.begin_forward()
    env.context.batch_descriptor = None
    env.context.is_padding = SimpleNamespace(ndim=1, shape=(12,))
    with pytest.raises(ValueError, match="bucket mismatch"):
        metadata.begin_forward()
    assert not metadata.executor.submission_in_flight
    with pytest.raises(RuntimeError, match="recreate"):
        metadata.run_build(build, **kwargs)


def test_replay_requires_capture_and_current_submission(env):
    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6])
    _, _, build, kwargs = _build(env)
    with pytest.raises(RuntimeError, match="valid graph-external"):
        metadata.validate_replay(6)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    with pytest.raises(RuntimeError, match="not captured"):
        metadata.validate_replay(6)
    metadata.abort()


@pytest.mark.parametrize("kind", ["duplicate", "empty", "compressor-only"])
def test_invalid_frontiers_are_rejected_before_submission(env, kind):
    metadata = module.AsyncTargetDeviceMetadata()
    if kind == "duplicate":
        frontiers = [(DeviceMetadataStage.INDEXER, 1)] * 2
    elif kind == "empty":
        frontiers = []
    else:
        frontiers = [(DeviceMetadataStage.COMPRESSOR, 1)]
    _, _, build, kwargs = _build(env, frontiers)
    metadata.configure_capture_sizes([6])
    with pytest.raises(RuntimeError):
        metadata.run_build(build, **kwargs, full_graph_mode=True)
    assert not metadata.executor.submission_in_flight


def test_non_full_step_without_async_tasks_runs_synchronous_tasks(env):
    metadata = module.AsyncTargetDeviceMetadata()
    _, resource, build, kwargs = _build(env, [(DeviceMetadataStage.COMPRESSOR, 0)])
    assert metadata.run_build(build, **kwargs)["cache"] is resource
    assert resource.device_metadata_executor is None
    assert env.calls == [("task", DeviceMetadataStage.COMPRESSOR, 0, "main", env.step)]
    metadata.begin_forward()
    metadata.finish()
    assert not metadata.executor.submission_in_flight

    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs)
    assert metadata.executor.submission_in_flight
    metadata.finish()


def test_same_bucket_requires_stable_frontiers_and_single_submission(env):
    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6])
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    with pytest.raises(RuntimeError, match="not been retired"):
        metadata.run_build(build, **kwargs)
    metadata.finish()
    _, _, other, kwargs = _build(env, [(DeviceMetadataStage.INDEXER, 99)])
    with pytest.raises(RuntimeError, match="frontiers changed"):
        metadata.run_build(other, **kwargs, full_graph_mode=True)


def test_duplicate_provider_is_only_entered_once(env):
    metadata = module.AsyncTargetDeviceMetadata()
    provider, _, build, kwargs = _build(env)
    kwargs["attn_groups"] *= 2
    metadata.run_build(build, **kwargs)
    assert provider.entries == 1
    metadata.finish()


def test_build_failure_drains_tasks_and_invalidates_state(env):
    metadata = module.AsyncTargetDeviceMetadata()
    original = ValueError("build failed")
    provider, _, build, kwargs = _build(env, fail=original)
    with pytest.raises(ValueError) as error:
        metadata.run_build(build, **kwargs)
    assert error.value is original
    assert provider.tasks == ()
    with pytest.raises(RuntimeError, match="recreate"):
        metadata.run_build(build, **kwargs)


def test_partial_producer_failure_joins_stream_not_unrecorded_event(env):
    metadata = module.AsyncTargetDeviceMetadata()
    _, _, build, kwargs = _build(env)
    original = ValueError("producer failed")

    def submit(tasks, descriptor):
        metadata.executor._submission_in_flight = True
        raise original

    metadata.executor.submit = submit
    with pytest.raises(ValueError) as error:
        metadata.run_build(build, **kwargs)
    assert error.value is original
    assert ("main", "wait_stream", "metadata") in env.calls
    assert not any(c[1] in ("wait", "external_wait") for c in env.calls)
    assert not metadata.executor.submission_in_flight


def test_capture_failure_defers_cleanup_and_preserves_original_error(env, monkeypatch):
    metadata = module.AsyncTargetDeviceMetadata()
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs)
    env.capturing = True
    module.finish_metadata(metadata, failed=True)
    assert metadata.executor.submission_in_flight
    assert not any(c[1] == "wait_stream" for c in env.calls)
    env.capturing = False
    module.finish_metadata(metadata, failed=True)
    assert not metadata.executor.submission_in_flight
    assert ("main", "wait_stream", "metadata") in env.calls

    def bad_cleanup():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(metadata, "abort", bad_cleanup)
    original = ValueError("forward failed")
    with pytest.raises(ValueError) as error:
        try:
            raise original
        finally:
            module.finish_metadata(metadata, failed=True)
    assert error.value is original


def test_existing_a5_finish_dispatch_is_unchanged(env):
    calls = []
    legacy = SimpleNamespace(finish=lambda: calls.append("finish"), finish_replay=lambda: calls.append("replay"))
    module.finish_metadata(None)
    module.finish_metadata(legacy)
    module.finish_metadata(legacy, replay=True)
    module.finish_metadata(legacy, failed=True)
    assert calls == ["finish", "replay", "finish"]


@pytest.mark.parametrize("failure", [None, "begin", "forward"])
def test_actual_model_wrapper_retires_success_and_failure(env, monkeypatch, failure):
    from vllm_ascend.worker.v2 import aclgraph_utils as graph

    metadata = module.AsyncTargetDeviceMetadata()
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs)
    monkeypatch.setattr(graph, "get_forward_context", lambda: env.context)
    original = ValueError("model execution failed")

    def fail():
        raise original

    if failure == "begin":
        monkeypatch.setattr(metadata, "begin_forward", fail)
    forward = fail if failure == "forward" else lambda: "output"
    wrapper = graph.ModelWithContext(forward, device_metadata=metadata)
    if failure is None:
        assert wrapper() == "output"
        assert not metadata._failed
    else:
        with pytest.raises(ValueError) as error:
            wrapper()
        assert error.value is original
        assert metadata._failed
    assert not metadata.executor.submission_in_flight


@pytest.mark.parametrize("failure", [False, True])
def test_actual_replay_wrapper_orders_validation_and_release(env, monkeypatch, failure):
    from vllm_ascend.worker.v2 import aclgraph_utils as graph

    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6])
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    env.capturing = True
    metadata.finish()
    env.capturing = False
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    resets = [call for call in env.calls if call[1] == "reset"]
    manager = graph.ModelAclGraphManager.__new__(graph.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(
        attn_groups=[], model_state=SimpleNamespace(device_metadata=metadata, attn_metadata={})
    )
    manager.vllm_config, manager.update_stream = object(), object()
    monkeypatch.setattr(graph, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(graph, "_get_graph_update_backend", lambda *_: object())
    monkeypatch.setattr(graph, "use_updatable_graph", lambda *_: True)
    original = ValueError("graph replay failed")

    def replay(*args):
        assert metadata.executor.submission_in_flight
        if failure:
            raise original
        return "output"

    manager._updatable_graph_replay = replay
    desc = SimpleNamespace(num_tokens=6)
    if failure:
        with pytest.raises(ValueError) as error:
            manager.run_fullgraph(desc)
        assert error.value is original
        assert metadata._failed
    else:
        assert manager.run_fullgraph(desc) == "output"
        assert not metadata._failed
    assert not metadata.executor.submission_in_flight
    assert [call for call in env.calls if call[1] == "reset"] == resets


@pytest.mark.parametrize("legacy_lora_argument", [False, True])
def test_actual_capture_failure_invalidates_a3_only(env, monkeypatch, legacy_lora_argument):
    from vllm_ascend.worker.v2 import aclgraph_utils as graph

    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6])
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    manager = graph.ModelAclGraphManager.__new__(graph.ModelAclGraphManager)
    manager.capture_sizes = [6]
    monkeypatch.setattr(graph, "communicator_switch", nullcontext)
    monkeypatch.setattr(graph, "vllm_version_is", lambda version: legacy_lora_argument)
    original = ValueError("input preparation failed")
    pcp_manager = object()

    def fail(*args, **kwargs):
        assert kwargs["pcp_manager"] is pcp_manager
        assert ("has_lora" in kwargs) == legacy_lora_argument
        if legacy_lora_argument:
            assert kwargs["has_lora"] is False
        raise original

    monkeypatch.setattr(graph.ModelCudaGraphManager, "capture", fail)
    with pytest.raises(ValueError) as error:
        manager.capture(
            torch.nn.Identity(),
            SimpleNamespace(device_metadata=metadata),
            None,
            None,
            None,
            [],
            None,
            pcp_manager=pcp_manager,
        )
    assert error.value is original
    assert not metadata.executor.submission_in_flight
    with pytest.raises(RuntimeError, match="recreate"):
        metadata.run_build(build, **kwargs)


def test_actual_replay_rejects_wrong_bucket_before_submitting_graph(env, monkeypatch):
    from vllm_ascend.worker.v2 import aclgraph_utils as graph

    metadata = module.AsyncTargetDeviceMetadata()
    metadata.configure_capture_sizes([6])
    _, _, build, kwargs = _build(env)
    metadata.run_build(build, **kwargs, full_graph_mode=True)
    manager = graph.ModelAclGraphManager.__new__(graph.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(
        attn_groups=[], model_state=SimpleNamespace(device_metadata=metadata, attn_metadata={})
    )
    manager.vllm_config, manager.update_stream = object(), object()
    monkeypatch.setattr(graph, "set_current_vllm_config", lambda *_: nullcontext())
    monkeypatch.setattr(graph, "_get_graph_update_backend", lambda *_: object())
    manager._updatable_graph_replay = lambda *_: pytest.fail("graph must not be submitted")
    with pytest.raises(ValueError, match="Replay bucket"):
        manager.run_fullgraph(SimpleNamespace(num_tokens=12))
    assert metadata._failed
    assert not metadata.executor.submission_in_flight
