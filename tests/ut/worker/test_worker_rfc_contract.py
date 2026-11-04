# SPDX-License-Identifier: Apache-2.0
"""Portable Worker contracts; execute production method bodies without NPU imports.

Run directly with unittest to bypass the NPU-wide pytest conftest. Full worker
integration tests remain in test_worker_v1.py and test_worker_v2.py.
"""

import ast
import copy
import importlib.util
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, NoneType, SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "vllm_ascend/worker/worker.py"


def worker_method(name, *, source=WORKER, class_name="NPUWorker", **bindings):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next((node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if node is None:
        # Source-only local validation and the installed pinned vLLM in CI.
        source = ROOT.parent / "worker-upstream/vllm/v1/worker/gpu_worker.py"
        if not source.exists():
            spec = importlib.util.find_spec("vllm")
            source = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu_worker.py"
        parent = next(
            n
            for n in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(n, ast.ClassDef) and n.name == "Worker"
        )
        node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method = copy.deepcopy(node)
    method.name = "_contract_method"
    method.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    scope = dict(bindings)
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), scope)
    return scope["_contract_method"]


def contract_class(names, **bindings):
    source = ROOT.parent / "worker-upstream/vllm/v1/worker/gpu_worker.py"
    if not source.exists():
        spec = importlib.util.find_spec("vllm")
        source = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu_worker.py"
    parent = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    local = next(
        n
        for n in ast.parse(WORKER.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "NPUWorker"
    )
    for cls, base in ((parent, "object"), (local, "Worker")):
        cls.bases = [ast.Name(id=base, ctx=ast.Load())]
        cls.body = [copy.deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        for method in cls.body:
            method.decorator_list = []
        if not cls.body:
            cls.body = [ast.Pass()]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), parent, local],
        type_ignores=[],
    )
    scope = dict(bindings)
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), scope)
    return scope["NPUWorker"]


class WorkerAlignmentContract(unittest.TestCase):
    def test_reload_context_and_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                events = []

                @contextmanager
                def config_context(config, events=events):
                    events.append(config)
                    try:
                        yield
                    finally:
                        events.append("exit")

                runner = MagicMock()

                def reload(events=events, fail=fail, **kwargs):
                    self.assertEqual(events, ["config"])
                    self.assertEqual(kwargs, {"weights_path": "checkpoint"})
                    if fail:
                        raise ValueError("load failed")

                runner.reload_weights.side_effect = reload
                worker = SimpleNamespace(vllm_config="config", model_runner=runner)
                method = worker_method("reload_weights", set_current_vllm_config=config_context)
                if fail:
                    with self.assertRaisesRegex(ValueError, "load failed"):
                        method(worker, weights_path="checkpoint")
                else:
                    method(worker, weights_path="checkpoint")
                self.assertEqual(events, ["config", "exit"])

    def test_pp_device_handles_only(self):
        class Intermediate:
            tensors = {"hidden": "buffer"}
            kv_connector_output = None

        class Output:
            pass

        for v2 in (False, True):
            for sp in (False, True):
                for forward in (False, True):
                    with self.subTest(v2=v2, sp=sp, forward=forward):
                        pp, tp = MagicMock(), object()
                        pp.is_first_rank = False
                        pp.is_last_rank = False
                        pp.irecv_tensor_dict.return_value = ({"hidden": "input"}, [], [])
                        metadata, device, previous = MagicMock(), MagicMock(), MagicMock()
                        pp.isend_tensor_dict.return_value = [metadata, device]
                        runner = MagicMock()
                        runner.execute_model.return_value = Intermediate()
                        worker = SimpleNamespace(
                            log_memory_stats=MagicMock(),
                            _pp_send_work=[previous],
                            use_v2_model_runner=v2,
                            model_runner=runner,
                            profiler=None,
                            vllm_config=SimpleNamespace(
                                parallel_config=SimpleNamespace(distributed_executor_backend="mp")
                            ),
                        )
                        bindings = dict(
                            get_pp_group=lambda pp=pp: pp,
                            get_tp_group=lambda tp=tp: tp,
                            get_ascend_config=lambda: SimpleNamespace(msmonitor_use_daemon=False),
                            enable_sp=lambda _, sp=sp: sp,
                            AsyncIntermediateTensors=lambda *a, **kw: "received",
                            IntermediateTensors=Intermediate,
                            ModelRunnerOutput=Output,
                            AsyncModelRunnerOutput=Output,
                            NoneType=NoneType,
                            nullcontext=nullcontext,
                        )
                        for hook in (
                            "_before_execute_model",
                            "annotate_profile",
                            "_get_pp_all_gather_group",
                            "_after_pp_send",
                        ):
                            setattr(worker, hook, MethodType(worker_method(hook, **bindings), worker))
                        execute = worker_method("execute_model", **bindings)
                        execute(worker, SimpleNamespace(total_num_scheduled_tokens=int(forward)))
                        previous.wait.assert_called_once_with()
                        self.assertEqual(worker._pp_send_work, [device])
                        metadata.wait.assert_not_called()
                        group = None if v2 and sp else tp
                        pp.isend_tensor_dict.assert_called_once_with(Intermediate.tensors, all_gather_group=group)
                        if forward:
                            pp.irecv_tensor_dict.assert_called_once_with(all_gather_group=group)
                        else:
                            pp.irecv_tensor_dict.assert_not_called()

    def test_config_identity_and_rejection(self):
        for runner_version in ("v1", "v2"):
            with self.subTest(runner=runner_version):
                old = SimpleNamespace(value=1)
                config = SimpleNamespace(model_config=old, load_config=old)
                worker = SimpleNamespace(vllm_config=config, model_runner=SimpleNamespace())

                def replace_config(config, overrides):
                    return SimpleNamespace(**(vars(config) | overrides))

                update = worker_method("update_config", update_config=replace_config)
                update(worker, {"model_config": {"value": 2}, "load_config": {"value": 3}})
                for name in ("model_config", "load_config"):
                    self.assertIs(getattr(worker, name), getattr(config, name))
                    self.assertIs(getattr(worker.model_runner, name), getattr(config, name))
                    self.assertIsNot(getattr(config, name), old)
                with self.assertRaisesRegex(ValueError, "cache_config"):
                    update(worker, {"cache_config": {}})

    def test_load_dummy_weights_rejected_before_allocation(self):
        load = worker_method("load_model")
        with self.assertRaisesRegex(NotImplementedError, "elastic-EP"):
            load(SimpleNamespace(), load_dummy_weights=True)

    def test_dummy_contract_and_release_compatibility(self):
        for release in (False, True):
            for dp_size in (1, 2):
                runner = SimpleNamespace(uniform_decode_query_len=4, _dummy_run=MagicMock())
                worker = SimpleNamespace(
                    log_memory_stats=MagicMock(),
                    model_runner=runner,
                    vllm_config=object(),
                    parallel_config=SimpleNamespace(data_parallel_size=dp_size),
                )
                execute = worker_method(
                    "execute_dummy_batch",
                    set_current_vllm_config=lambda _: contextmanager(lambda: iter([None]))(),
                    vllm_version_is=lambda _, release=release: release,
                    envs=SimpleNamespace(VLLM_RANDOMIZE_DP_DUMMY_INPUTS=True),
                )
                execute(worker)
                expected = {"uniform_decode": True, "skip_gdn_state_update": True}
                if not release:
                    expected["randomize_inputs"] = dp_size > 1
                runner._dummy_run.assert_called_once_with(4, **expected)

    def test_parent_constructor_and_elastic_boundary(self):
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
        self.assertEqual([ast.unparse(base) for base in cls.bases], ["GPUWorker"])
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        calls = [ast.unparse(n.func) for n in ast.walk(constructor) if isinstance(n, ast.Call)]
        self.assertIn("super().__init__", calls)
        self.assertNotIn("WorkerBase.__init__", calls)
        scope = {}
        path = ROOT / "vllm_ascend/worker/worker_interfaces.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        executor = scope["UnsupportedElasticEPExecutor"]()
        with self.assertRaisesRegex(NotImplementedError, "elastic"):
            executor.execute("load_model")
        executor.shutdown()

    def test_unsupported_interfaces_raise(self):
        for name in (
            "checkpoint_prepare",
            "checkpoint_restore",
            "handle_ft_command",
            "save_tensorized_model",
            "_compile_or_warm_up_model",
            "_get_cudagraph_capture_context",
            "start_draft_weight_update",
        ):
            with self.subTest(interface=name), self.assertRaisesRegex(NotImplementedError, "Ascend"):
                worker_method(name)(SimpleNamespace())

    def test_shutdown_order_and_eplb_failure(self):
        class EPLB:
            def __init__(self, events, fail):
                self.events, self.fail = events, fail

            def close(self):
                self.events.append("eplb")
                if self.fail:
                    raise RuntimeError("EPLB still running")

        for fail in (False, True):
            events = []

            def resource(name, events=events):
                return SimpleNamespace(shutdown=lambda events=events: events.append(name))

            worker = SimpleNamespace(
                model_runner=resource("runner"),
                profiler=resource("profiler"),
                weight_transfer_engine=resource("engine"),
                elastic_ep_executor=resource("elastic"),
                _pp_send_work=[SimpleNamespace(wait=lambda events=events: events.append("pp"))],
            )
            worker.model_runner.eplb_state = EPLB(events, fail)
            cls = contract_class(
                {"shutdown"},
                AscendEplbState=EPLB,
                current_platform=SimpleNamespace(is_cuda_alike=lambda: False),
                gc=SimpleNamespace(unfreeze=lambda events=events: events.append("gc")),
                ensure_kv_transfer_shutdown=lambda events=events: events.append("kv"),
                ensure_ec_transfer_shutdown=lambda events=events: events.append("ec"),
                CaMemAllocator=SimpleNamespace(
                    instance=SimpleNamespace(release_pools=lambda events=events: events.append("pools"))
                ),
            )
            instance = cls()
            instance.__dict__.update(vars(worker))
            worker = instance
            shutdown = cls.shutdown
            if fail:
                with self.assertRaisesRegex(RuntimeError, "EPLB"):
                    shutdown(worker)
                self.assertEqual(events, ["eplb"])
            else:
                shutdown(worker)
                self.assertEqual(
                    events, ["eplb", "pp", "gc", "kv", "ec", "profiler", "engine", "elastic", "runner", "pools"]
                )

    def test_forbidden_overrides(self):
        scope = {}
        path = ROOT / "vllm_ascend/worker/worker_interfaces.py"
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        for file, class_name in (
            (WORKER, "NPUWorker"),
            (ROOT / "vllm_ascend/_310p/worker_310p.py", "NPUWorker310"),
            (ROOT / "vllm_ascend/xlite/xlite_worker.py", "XliteWorker"),
        ):
            cls = next(
                n
                for n in ast.parse(file.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == class_name
            )
            methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
            self.assertFalse(methods.intersection(scope["INHERITED_WORKER_METHODS"]), class_name)

    def test_inherited_weight_update_state_machine(self):
        from contextlib import nullcontext

        names = {
            "_check_weight_transfer_engine",
            "_check_nz_disabled",
            "_start_weight_update",
            "start_weight_update",
            "update_weights",
            "finish_weight_update",
            "shutdown_weight_transfer_engine",
        }
        cls = contract_class(
            names,
            set_current_vllm_config=lambda _: nullcontext(),
            get_ascend_config=lambda: SimpleNamespace(weight_nz_mode=0),
        )
        worker = cls()
        worker.weight_transfer_engine = MagicMock()
        worker._weight_update_active = False
        worker._weight_update_is_draft = False
        worker.model_runner = MagicMock()
        worker.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_index=1, world_size=2))
        worker.rank = 1
        worker.start_weight_update()
        with self.assertRaisesRegex(RuntimeError, "already active"):
            worker.start_weight_update()
        payload = [{"rank": rank} for rank in range(4)]
        worker.update_weights(payload)
        worker.weight_transfer_engine.update_weights.assert_called_once_with(payload[3])
        worker.weight_transfer_engine.finish_weight_update.side_effect = RuntimeError("retryable finish")
        with self.assertRaisesRegex(RuntimeError, "retryable"):
            worker.finish_weight_update()
        self.assertTrue(worker._weight_update_active)
        worker.weight_transfer_engine.finish_weight_update.side_effect = None
        worker.finish_weight_update()
        self.assertFalse(worker._weight_update_active)
        worker.model_runner.reset_lora_state.assert_called_once_with()
        worker.weight_transfer_engine.reset_weight_update_target.assert_called_once_with()
        worker.start_weight_update()
        worker.weight_transfer_engine.update_weights.side_effect = ValueError("bad chunk")
        with self.assertRaisesRegex(ValueError, "bad chunk"):
            worker.update_weights({})
        self.assertFalse(worker._weight_update_active)
        self.assertEqual(worker.weight_transfer_engine.reset_weight_update_target.call_count, 2)
        worker.weight_transfer_engine.start_weight_update.side_effect = RuntimeError("failed start")
        with self.assertRaisesRegex(RuntimeError, "failed start"):
            worker.start_weight_update()
        self.assertFalse(worker._weight_update_active)
        worker.shutdown_weight_transfer_engine()
        worker.weight_transfer_engine.shutdown.assert_called_once_with()

    def test_nz_validation_precedes_engine_mutation(self):
        from contextlib import nullcontext

        cls = contract_class(
            {"_check_weight_transfer_engine", "_check_nz_disabled", "_start_weight_update", "start_weight_update"},
            set_current_vllm_config=lambda _: nullcontext(),
            get_ascend_config=lambda: SimpleNamespace(weight_nz_mode=1),
        )
        worker = cls()
        worker.weight_transfer_engine = MagicMock()
        worker.vllm_config = object()
        worker._weight_update_active = False
        with self.assertRaisesRegex(ValueError, "FRACTAL_NZ"):
            worker.start_weight_update()
        worker.weight_transfer_engine.start_weight_update.assert_not_called()

    def test_camem_backend_levels_tags_and_failure_state(self):
        path = ROOT / "vllm_ascend/device_allocator/sleep_mode_backend.py"
        cls = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))

        class Base:
            def __init__(self):
                self._state = "RUNNING"

        allocator, transfer = MagicMock(), MagicMock()
        allocator.pointer_to_data = {1: SimpleNamespace(is_asleep=True)}
        scope = {
            "SleepModeBackend": Base,
            "CaMemAllocator": SimpleNamespace(get_instance=lambda: allocator),
            "global_te": transfer,
        }
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
        backend = scope["CaMemSleepBackend"]()
        for level in (1, 2):
            backend.suspend(level)
            allocator.sleep.assert_called_with(offload_tags=("weights",) if level == 1 else ())
            self.assertEqual(backend._state, "SUSPENDED")
        transfer.reset_mock()
        backend.resume(["weights"])
        transfer.reregister_buffer.assert_not_called()
        self.assertEqual(backend._state, "SUSPENDED")
        allocator.pointer_to_data[1].is_asleep = False
        backend.resume(["kv_cache"])
        transfer.reregister_buffer.assert_called_once_with()
        self.assertEqual(backend._state, "RUNNING")
        backend.discard(("kv_cache",))
        allocator.discard.assert_called_once_with(("kv_cache",))
        self.assertEqual(backend._state, "SUSPENDED")
        with self.assertRaisesRegex(ValueError, "1 or 2"):
            backend.suspend(3)
        allocator.wake_up.side_effect = ValueError("allocation failed")
        with self.assertRaisesRegex(ValueError, "allocation failed"):
            backend.resume(None)
        self.assertEqual(backend._state, "SUSPENDED")

    def test_v1_connector_output_and_v2_boundary(self):
        empty = SimpleNamespace(kv_connector_output=None)
        handle = worker_method("_after_pp_send", EMPTY_MODEL_RUNNER_OUTPUT=empty, copy=copy)
        worker = SimpleNamespace(use_v2_model_runner=False)
        self.assertIsNone(handle(worker, SimpleNamespace(kv_connector_output=None)))
        pending = SimpleNamespace(finished_sending=False, finished_recving=False)
        self.assertIs(handle(worker, SimpleNamespace(kv_connector_output=pending)), empty)
        for sending, receiving in ((True, False), (False, True), (True, True)):
            connector = SimpleNamespace(finished_sending=sending, finished_recving=receiving)
            result = handle(worker, SimpleNamespace(kv_connector_output=connector))
            self.assertIsNot(result, empty)
            self.assertIs(result.kv_connector_output, connector)
            self.assertIsNone(empty.kv_connector_output)
        worker.use_v2_model_runner = True
        self.assertIsNone(handle(worker, SimpleNamespace()))

    def test_monitor_and_profiler_steps_on_empty_batch(self):
        worker = SimpleNamespace(log_memory_stats=MagicMock(), profiler=MagicMock())
        monitor = MagicMock()
        before = worker_method(
            "_before_execute_model", get_ascend_config=lambda: SimpleNamespace(msmonitor_use_daemon=True), dp=monitor
        )
        before(worker, SimpleNamespace(total_num_scheduled_tokens=0))
        worker.log_memory_stats.assert_called_once_with()
        monitor.step.assert_called_once_with()
        annotate = worker_method("annotate_profile", nullcontext=nullcontext)
        with annotate(worker, object()):
            worker.profiler.step.assert_called_once_with()

    def test_inherited_pooling_and_non_pp_outputs(self):
        class Output:
            pass

        runner = MagicMock()
        pooled = Output()
        runner.execute_model.return_value = None
        runner.pool.return_value = pooled
        runner.is_pooling_model = True
        pp = SimpleNamespace(is_first_rank=True)
        worker = SimpleNamespace(
            _pp_send_work=[],
            _before_execute_model=MagicMock(),
            annotate_profile=lambda _: nullcontext(),
            model_runner=runner,
            use_v2_model_runner=True,
        )
        execute = worker_method(
            "execute_model",
            get_pp_group=lambda: pp,
            ModelRunnerOutput=Output,
            AsyncModelRunnerOutput=Output,
            NoneType=NoneType,
        )
        self.assertIs(execute(worker, SimpleNamespace(total_num_scheduled_tokens=0)), pooled)
        runner.pool.assert_called_once_with()
        runner.reset_mock()
        runner.execute_model.return_value = pooled
        worker.use_v2_model_runner = False
        self.assertIs(execute(worker, SimpleNamespace(total_num_scheduled_tokens=1)), pooled)
        runner.pool.assert_not_called()

    def test_runner_factories_cover_device_variants(self):
        import builtins

        class V1:
            def __init__(self, config, device):
                self.config, self.device = config, device

        class V2(V1):
            pass

        class Encoder(V1):
            pass

        classes = {
            "vllm_ascend.worker.v2.model_runner": SimpleNamespace(NPUModelRunner=V2),
            "vllm.v1.worker.mm_encoder_model_runner": SimpleNamespace(MMEncoderModelRunner=Encoder),
            "vllm_ascend.worker.v2.utils": SimpleNamespace(torch_cuda_wrapper=nullcontext),
            "vllm_ascend._310p.worker.v2.model_runner": SimpleNamespace(NPUModelRunner310V2=V2),
            "vllm_ascend.worker.v1._310p.model_runner_310p": SimpleNamespace(NPUModelRunner310=V1),
            "vllm_ascend.worker.v1.xlite.xlite_model_runner": SimpleNamespace(XliteModelRunner=V1),
        }

        def import_factory(name, *args, **kwargs):
            return classes[name] if name in classes else builtins.__import__(name, *args, **kwargs)

        for file, name in (
            (WORKER, "NPUWorker"),
            (ROOT / "vllm_ascend/_310p/worker_310p.py", "NPUWorker310"),
            (ROOT / "vllm_ascend/xlite/xlite_worker.py", "XliteWorker"),
        ):
            for v2 in (False, True):
                worker = SimpleNamespace(
                    use_v2_model_runner=v2, vllm_config=SimpleNamespace(is_mm_encoder_only=False), device="npu:0"
                )
                factory = worker_method(
                    "_make_model_runner",
                    source=file,
                    class_name=name,
                    __builtins__=dict(vars(builtins), __import__=import_factory),
                    NPUModelRunner=V1,
                    XliteModelRunnerV2=V2,
                    logger=MagicMock(),
                )
                runner = factory(worker)
                self.assertIs(type(runner), V2 if v2 else V1)
                self.assertIs(runner.config, worker.vllm_config)
                self.assertEqual(runner.device, "npu:0")
                if name == "XliteWorker":
                    self.assertIs(worker.xlite_runner, runner)
                if name == "NPUWorker" and v2:
                    worker.vllm_config.is_mm_encoder_only = True
                    self.assertIs(type(factory(worker)), Encoder)

    def test_workspace_constructs_runner_after_device_selection(self):
        runner = object()
        workspace, usage = MagicMock(), MagicMock()
        worker = SimpleNamespace(
            device="npu:2", rank=0, vllm_config=object(), _make_model_runner=MagicMock(return_value=runner)
        )
        initialize = worker_method(
            "_init_workspace_and_model_runner", init_workspace_manager=workspace, report_usage_stats=usage
        )
        initialize(worker)
        workspace.assert_called_once_with("npu:2", num_ubatches=1)
        self.assertIs(worker.model_runner, runner)
        usage.assert_called_once_with(worker.vllm_config)

    def test_execution_decorators(self):
        cls = next(
            n for n in ast.parse(WORKER.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker"
        )
        for name in ("execute_model", "sample_tokens"):
            node = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
            if node is None:
                source = ROOT.parent / "worker-upstream/vllm/v1/worker/gpu_worker.py"
                if not source.exists():
                    spec = importlib.util.find_spec("vllm")
                    source = Path(next(iter(spec.submodule_search_locations))) / "v1/worker/gpu_worker.py"
                parent = next(
                    n
                    for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.ClassDef) and n.name == "Worker"
                )
                node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(
                [ast.unparse(d) for d in node.decorator_list], ["torch.inference_mode()", "with_gpu_sync_check"]
            )


if __name__ == "__main__":
    unittest.main()
