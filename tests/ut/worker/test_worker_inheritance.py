# SPDX-License-Identifier: Apache-2.0
"""Behavior checks for NPUWorker inheritance and derived Workers."""

import importlib
import inspect
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

# Remove an entry and its NPUWorker stub together only after validating the
# inherited contract.
UNVERIFIED_GPU_WORKER_METHODS = (
    "checkpoint_prepare",
    "checkpoint_restore",
    "_scoped_allocator_max_split",
    "handle_ft_command",
    "_compile_or_warm_up_model",
    "_maybe_activate_jit_monitor",
    "_get_cudagraph_capture_context",
    "_set_draft_weight_update_target",
    "get_compilation_match_table",
    "get_encoder_timing_stats",
    "annotate_profile",
    "randomize_dummy_inputs",
    "save_sharded_state",
    "save_tensorized_model",
    "start_draft_weight_update",
    "elastic_ep_execute",
)


@pytest.mark.parametrize("name", UNVERIFIED_GPU_WORKER_METHODS)
def test_unverified_interface_rejects_calls_without_worker_state(name):
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_ascend.worker.worker import NPUWorker

    worker = object.__new__(NPUWorker)
    descriptor = inspect.getattr_static(NPUWorker, name)
    assert type(descriptor) is type(inspect.getattr_static(GPUWorker, name))
    if isinstance(descriptor, property):
        with pytest.raises(NotImplementedError, match=name):
            getattr(worker, name)
        return
    method = getattr(worker, name)
    required = [p for p in inspect.signature(method).parameters.values() if p.default is inspect.Parameter.empty]
    args = [object() for p in required if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kwargs = {p.name: object() for p in required if p.kind is p.KEYWORD_ONLY}
    with pytest.raises(NotImplementedError, match=name):
        method(*args, **kwargs)


def test_weight_update_hook_rejects_draft_without_worker_state():
    from vllm_ascend.worker.worker import NPUWorker

    worker = object.__new__(NPUWorker)
    with pytest.raises(NotImplementedError, match="Draft weight updates"):
        worker._start_weight_update(is_draft=True)


def test_shutdown_weight_transfer_engine_is_inherited():
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_ascend.worker.worker import NPUWorker

    # Releasing the trainer connection needs no NPU state, so it is inherited.
    assert NPUWorker.shutdown_weight_transfer_engine is GPUWorker.shutdown_weight_transfer_engine


@pytest.mark.parametrize(
    "name",
    ("reset_mm_cache", "compute_weight_checksums", "reset_weights"),
)
def test_model_accessors_are_inherited(name):
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_ascend.worker.worker import NPUWorker

    # Every supported runner exposes get_model()/reset_mm_cache(), so these
    # worker RPCs need no Ascend override.
    assert getattr(NPUWorker, name) is getattr(GPUWorker, name)


@pytest.mark.parametrize("drafter", ["absent", "draft-model", "other-speculator"])
def test_get_draft_model_reports_the_runner_drafter(drafter):
    """This RPC used to raise on Ascend; it now reports the runner's drafter.

    Bind the real upstream runner implementation so both branches of its
    isinstance() gate run: a configured DraftModelSpeculator yields its draft
    module, while no drafter (or a non-draft speculator) yields None.
    """
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    from vllm_ascend.worker.worker import NPUWorker

    draft_model = torch.nn.Linear(2, 2) if drafter == "draft-model" else None
    # The runner only reads isinstance()/attribute state from this slot, so the
    # parameterization mixes a mock with plain objects on purpose. Annotate the
    # union up front so mypy does not pin it to the first MagicMock branch.
    speculator: object | None
    if drafter == "draft-model":
        mock_speculator = MagicMock(spec=DraftModelSpeculator)
        mock_speculator.model = draft_model
        speculator = mock_speculator
    elif drafter == "other-speculator":
        speculator = object()  # e.g. a speculator that carries no draft model
    else:
        speculator = None

    runner = SimpleNamespace(speculator=speculator)
    runner.get_draft_model = MethodType(GPUModelRunner.get_draft_model, runner)

    worker = object.__new__(NPUWorker)
    worker.model_runner = runner

    assert worker.get_draft_model() is draft_model


@pytest.mark.parametrize("configured", [False, True])
def test_shutdown_weight_transfer_engine_releases_trainer_connection(configured):
    from vllm_ascend.worker.worker import NPUWorker

    # Keep the engine in a local so the optional type narrows for mypy.
    engine = Mock() if configured else None
    worker = object.__new__(NPUWorker)
    worker._weight_update_active = False
    worker.weight_transfer_engine = engine

    worker.shutdown_weight_transfer_engine()

    if engine is not None:
        engine.shutdown.assert_called_once_with()


def test_shutdown_weight_transfer_engine_rejects_active_update():
    from vllm_ascend.worker.worker import NPUWorker

    worker = object.__new__(NPUWorker)
    worker._weight_update_active = True
    worker.weight_transfer_engine = Mock()

    with pytest.raises(RuntimeError, match="during a weight update"):
        worker.shutdown_weight_transfer_engine()

    worker.weight_transfer_engine.shutdown.assert_not_called()


@pytest.fixture
def worker_constructor(monkeypatch):
    """Mock external setup while retaining both real Worker constructors."""
    import vllm.ir
    from vllm.model_executor.layers import linear
    from vllm.v1.worker import gpu_worker

    import vllm_ascend
    from vllm_ascend import logger as ascend_logger
    from vllm_ascend import utils
    from vllm_ascend.worker import worker as worker_module

    events = []

    def record(name):
        return Mock(side_effect=lambda *args, **kwargs: events.append(name))

    monkeypatch.setattr(utils, "adapt_patch", record("patches"))
    kda = ModuleType("vllm_ascend.patch.worker.patch_kda_recoverssm")
    kda.patch_kda_recoverssm = record("kda")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, kda.__name__, kda)
    monkeypatch.setattr(vllm_ascend, "ops", SimpleNamespace(register_dummy_fusion_op=record("ops")))
    monkeypatch.setattr(worker_module.envs_ascend, "COMPILE_CUSTOM_KERNELS", True)
    monkeypatch.setattr(
        worker_module, "get_current_hardware_profile", Mock(return_value=Mock(supports=lambda _: False))
    )
    for name in ("register_ascend_customop", "check_ascend_device_type"):
        monkeypatch.setattr(worker_module, name, record(name))
    ascend_config = SimpleNamespace(ascend_log_path="/test/ascend/log")

    def init_config(config):
        events.append("init_ascend_config")
        return ascend_config

    monkeypatch.setattr(worker_module, "init_ascend_config", Mock(side_effect=init_config))
    file_logging = record("logging")
    monkeypatch.setattr(ascend_logger, "configure_ascend_file_logging", file_logging)
    monkeypatch.setattr(
        worker_module,
        "get_ascend_config",
        Mock(
            return_value=SimpleNamespace(
                ascend_compilation_config=SimpleNamespace(
                    enable_npugraph_ex=False, enable_static_kernel=False, enable_super_kernel=False
                )
            )
        ),
    )
    sleep_manager = SimpleNamespace()

    def make_sleep_manager(config, worker, getter):
        events.append("sleep_manager")
        sleep_manager.worker = worker
        sleep_manager.get_runner = getter
        return sleep_manager

    monkeypatch.setattr(worker_module, "SleepWakeupManager", Mock(side_effect=make_sleep_manager))
    monkeypatch.setattr(linear, "WEIGHT_LOADER_V2_SUPPORTED", ["UnquantizedLinearMethod", "Other"])
    monkeypatch.setattr(vllm.ir, "set_default_torch_wrap", Mock())
    monkeypatch.setattr(gpu_worker.torch, "set_float32_matmul_precision", Mock())
    monkeypatch.setattr(gpu_worker, "validate_worker_profiler_config", record("validate_profiler"))
    executor_module = ModuleType("vllm.distributed.elastic_ep.elastic_execute")
    executor_module.ElasticEPScalingExecutor = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, executor_module.__name__, executor_module)
    original_parent_init = gpu_worker.Worker.__init__
    parent_calls = Mock()

    def parent_init(worker, *args, **kwargs):
        events.append("parent")
        parent_calls(worker, *args, **kwargs)
        original_parent_init(worker, *args, **kwargs)

    monkeypatch.setattr(gpu_worker.Worker, "__init__", parent_init)
    return SimpleNamespace(
        module=worker_module,
        parent_calls=parent_calls,
        events=events,
        ascend_config=ascend_config,
        file_logging=file_logging,
        sleep_manager=sleep_manager,
        executor_factory=executor_module.ElasticEPScalingExecutor,
    )


def construct_worker(setup, worker_class=None, cache_dtype="auto"):
    import torch

    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float16),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        parallel_config=SimpleNamespace(enable_fault_tolerance=False),
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_default=Mock())),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=False),
        profiler_config=SimpleNamespace(profiler="torch"),
        quant_config=object(),
    )
    for name in (
        "lora_config",
        "load_config",
        "scheduler_config",
        "device_config",
        "speculative_config",
        "observability_config",
        "kv_transfer_config",
    ):
        setattr(config, name, None)
    worker_class = worker_class or setup.module.NPUWorker
    worker = worker_class(config, local_rank=2, rank=0, distributed_init_method="env://", is_driver_worker=True)
    setup.parent_calls.assert_called_once_with(
        worker,
        vllm_config=config,
        local_rank=2,
        rank=0,
        distributed_init_method="env://",
        is_driver_worker=True,
    )
    setup.executor_factory.assert_called_once_with(worker)
    setup.file_logging.assert_called_once_with(setup.ascend_config.ascend_log_path)
    assert worker.vllm_config is config
    assert worker.model_config is config.model_config
    assert worker.parallel_config is config.parallel_config
    assert worker.rank == config.parallel_config.rank == 0
    assert worker.local_rank == 2
    assert worker.distributed_init_method == "env://"
    assert worker.is_driver_worker is True
    assert worker.weight_transfer_engine is None
    assert worker._weight_update_active is False
    assert worker._weight_update_is_draft is False
    assert worker.profiler is None
    assert worker._pp_send_work == []
    assert worker._sleep_mode_backend is None
    assert worker.worker_sentinel is None
    assert worker.profiler_config is config.profiler_config
    assert worker.device is None
    assert worker.model_runner is None
    assert worker.sleep_wakeup_manager is setup.sleep_manager
    assert setup.sleep_manager.worker is worker
    assert setup.sleep_manager.get_runner() is None
    assert worker.npugraph_memory_bytes == 0
    assert worker._kvpp_cache_allocation_plan is None
    assert setup.events == [
        "patches",
        "kda",
        "ops",
        "register_ascend_customop",
        "init_ascend_config",
        "logging",
        "check_ascend_device_type",
        "parent",
        "validate_profiler",
        "sleep_manager",
    ]
    return worker


@pytest.mark.parametrize("cache_dtype", ["auto", "float32"])
def test_real_npu_worker_constructor_initializes_parent_and_npu_state(worker_constructor, cache_dtype):
    import torch

    worker = construct_worker(worker_constructor, cache_dtype=cache_dtype)
    assert worker.cache_dtype == (torch.float16 if cache_dtype == "auto" else torch.float32)


def test_310p_parent_constructor_restores_triton_availability(worker_constructor, monkeypatch):
    """The 310P lanes run Model Runner V2 with Triton uninstalled."""
    from vllm.v1.worker import gpu_worker

    from vllm_ascend._310p.worker_310p import NPUWorker310

    monkeypatch.setattr(gpu_worker, "HAS_TRITON", False)
    construct_worker(worker_constructor, NPUWorker310)
    assert gpu_worker.HAS_TRITON is False


def test_parent_constructor_failure_restores_triton_availability(worker_constructor, monkeypatch):
    from vllm.v1.worker import gpu_worker

    monkeypatch.setattr(gpu_worker, "HAS_TRITON", False)
    gpu_worker.validate_worker_profiler_config.side_effect = RuntimeError("parent initialization failed")

    with pytest.raises(RuntimeError, match="parent initialization failed"):
        construct_worker(worker_constructor)

    assert gpu_worker.HAS_TRITON is False


def test_310p_constructor_device_runner_and_export(worker_constructor, monkeypatch):
    from vllm_ascend._310p import worker_310p

    worker = construct_worker(worker_constructor, worker_310p.NPUWorker310)
    assert worker_310p.NPUWorker310.__bases__ == (worker_constructor.module.NPUWorker,)
    device, runner = object(), SimpleNamespace(model=object())
    init_device = Mock(return_value=device)
    monkeypatch.setattr(worker, "_init_device", init_device)
    workspace = Mock()
    monkeypatch.setattr(worker_310p, "init_workspace_manager", workspace)
    compile_mode = Mock()
    monkeypatch.setattr(worker_310p.torch_npu.npu, "set_compile_mode", compile_mode)
    runner_module = ModuleType("vllm_ascend._310p.worker.v2.model_runner")
    runner_module.NPUModelRunner310V2 = Mock(return_value=runner)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    worker.init_device()
    init_device.assert_called_once_with()
    compile_mode.assert_called_once_with(jit_compile=False)
    workspace.assert_called_once_with(device, num_ubatches=1)
    runner_module.NPUModelRunner310V2.assert_called_once_with(worker.vllm_config, device)
    assert worker.model_runner is runner

    loader_module = ModuleType("vllm_ascend._310p.sharded_state_loader_310p")
    loader_module.ShardedStateLoader310 = loader = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, loader_module.__name__, loader_module)
    worker.save_sharded_state("output", pattern="part-{rank}", max_size=1024)
    loader.save_model.assert_called_once_with(runner.model, "output", pattern="part-{rank}", max_size=1024)
    loader.generate_quant_description.assert_called_once_with(runner.model, "output", worker.vllm_config.quant_config)


@pytest.fixture
def xlite_worker_module(monkeypatch):
    """Mock the optional native runner without leaking an imported Worker."""
    import vllm_ascend.xlite as package

    name = "vllm_ascend.xlite.xlite_worker"
    previous_module = sys.modules.get(name)
    previous_attribute = getattr(package, "xlite_worker", None)
    runner_module = ModuleType("vllm_ascend.xlite.xlite_model_runner")
    runner_module.XliteModelRunnerV2 = Mock(__name__="XliteModelRunnerV2")  # type: ignore[attr-defined]
    with monkeypatch.context() as scoped:
        scoped.setitem(sys.modules, runner_module.__name__, runner_module)
        try:
            yield importlib.import_module(name)
        finally:
            if previous_module is None:
                sys.modules.pop(name, None)
                if previous_attribute is None:
                    if hasattr(package, "xlite_worker"):
                        delattr(package, "xlite_worker")
                else:
                    package.xlite_worker = previous_attribute


@pytest.mark.parametrize("xlite", [False, True], ids=["default", "xlite"])
def test_init_device_selects_runner_and_orders_registration(
    worker_constructor, monkeypatch, xlite_worker_module, xlite
):
    xlite_worker = xlite_worker_module
    device, runner = object(), object()
    worker = construct_worker(worker_constructor, xlite_worker.XliteWorker if xlite else None)
    worker.vllm_config.is_mm_encoder_only = False
    if xlite:
        assert xlite_worker.XliteWorker.__bases__ == (worker_constructor.module.NPUWorker,)
        # Runner selection is the hook's job; the init_device skeleton stays inherited.
        assert "init_device" not in vars(xlite_worker.XliteWorker)
        assert "_make_model_runner" in vars(xlite_worker.XliteWorker)
        factory = Mock(return_value=runner, __name__="XliteModelRunnerV2")
        monkeypatch.setattr(xlite_worker, "XliteModelRunnerV2", factory)
    else:
        factory = Mock(return_value=runner, __name__="NPUModelRunner")
        runner_module = ModuleType("vllm_ascend.worker.v2.model_runner")
        runner_module.NPUModelRunner = factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    init_device = Mock(return_value=device)
    fla, workspace, usage = Mock(), Mock(), Mock()
    ordered = Mock()
    for name, mock in (
        ("init_device", init_device),
        ("register_fla", fla),
        ("init_workspace", workspace),
        ("make_runner", factory),
        ("report_usage", usage),
    ):
        ordered.attach_mock(mock, name)
    monkeypatch.setattr(worker, "_init_device", init_device)
    monkeypatch.setattr(worker_constructor.module, "register_ascend_fla_customop", fla)
    monkeypatch.setattr(worker_constructor.module, "init_workspace_manager", workspace)
    monkeypatch.setattr(worker_constructor.module, "report_usage_stats", usage)
    worker.init_device()
    assert ordered.mock_calls == [
        call.init_device(),
        call.register_fla(),
        call.init_workspace(device, 1),
        call.make_runner(worker.vllm_config, device),
        call.report_usage(worker.vllm_config),
    ]
    assert worker.model_runner is runner
    if xlite:
        assert worker.xlite_runner is runner
