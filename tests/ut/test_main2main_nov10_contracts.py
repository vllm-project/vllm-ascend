# SPDX-License-Identifier: Apache-2.0
"""Execute release/main adapter boundaries without loading an NPU runtime."""

from __future__ import annotations

import ast
import builtins
import hashlib
import unittest
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]


def load_nodes(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), *body],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace


class TestNov10Contracts(unittest.TestCase):
    def test_platform_bootstrap_does_not_import_vllm_config_from_utils(self):
        source = ROOT / "vllm_ascend/utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports: list[ast.stmt] = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]

        def bootstrap_import(name, *args, **kwargs):
            if name == "vllm.config" or name.startswith("vllm.config."):
                raise ImportError("vllm.config re-enters current_platform during platform bootstrap")
            return MagicMock()

        ns = {"__builtins__": {**vars(builtins), "__import__": bootstrap_import}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=imports, type_ignores=[])), str(source), "exec"), ns)

    def test_lookup_protocol_import_selects_its_own_version_constants(self):
        source = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/lookup_key.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        branch = next(n for n in tree.body if isinstance(n, ast.If) and "TYPE_CHECKING or" in ast.unparse(n.test))
        names = ("LOOKUP_MSG", "RESET_MSG", "RESP_ERR", "RESP_OK")
        legacy = SimpleNamespace(**{name: f"release-{name}".encode() for name in names})
        main = SimpleNamespace(**{name: f"main-{name}".encode() for name in names})
        for release, checking in ((True, False), (False, False), (True, True)):
            loader = MagicMock(return_value=legacy)
            imports: list[str] = []

            def native_import(name, *args, import_log=imports, **kwargs):
                import_log.append(name)
                self.assertEqual(name, "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol")
                return main

            ns = {
                "TYPE_CHECKING": checking,
                "vllm_version_is": lambda _, selected_release=release: selected_release,
                "import_module": loader,
                "__builtins__": {**vars(builtins), "__import__": native_import},
            }
            exec(
                compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])), str(source), "exec"), ns
            )
            expected = legacy if release and not checking else main
            self.assertEqual([ns[name] for name in names], [getattr(expected, name) for name in names])
            if release and not checking:
                loader.assert_called_once_with("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol")
                self.assertEqual(imports, [])
            else:
                loader.assert_not_called()
                self.assertEqual(len(imports), 1)

    def test_weight_loaders_import_legacy_context_only_on_release_and_restore_on_error(self):
        for engine_file, class_name in (
            ("hccl_engine.py", "HCCLWeightTransferEngine"),
            ("npu_ipc_engine.py", "NPUIPCWeightTransferEngine"),
        ):
            tree = ast.parse(
                (ROOT / "vllm_ascend/distributed/weight_transfer" / engine_file).read_text(encoding="utf-8")
            )
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            receive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "receive_weights")
            module = ast.Module(
                body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), receive],
                type_ignores=[],
            )
            for release in (True, False):
                with self.subTest(engine=engine_file, release=release):
                    state = {"enabled": True}

                    @contextmanager
                    def disable_check(scope=state):
                        scope["enabled"] = False
                        try:
                            yield
                        finally:
                            scope["enabled"] = True

                    loader = MagicMock(return_value=SimpleNamespace(disable_mtp_completeness_check=disable_check))
                    weights = [("w", object())]

                    def load(received, expected=weights, scope=state, selected_release=release):
                        self.assertIs(received, expected)
                        self.assertEqual(scope["enabled"], not selected_release)
                        raise RuntimeError("load failed")

                    params = {"packed": True, "packed_buffer_size_bytes": 16, "packed_num_buffers": 2}
                    engine = SimpleNamespace(
                        packed=True,
                        device=SimpleNamespace(index=1),
                        model_update_group=object(),
                        model=MagicMock(load_weights=load),
                        _wire_params=lambda _, selected=params: selected,
                        _last_hccl_update_info={},
                        _packed_importer=None,
                    )
                    info = SimpleNamespace(
                        names=["w"], shapes=[[1]], dtype_names=["float32"], ipc_handles={}, tensor_sizes=[4]
                    )
                    ns: dict[str, Any] = {
                        "vllm_version_is": lambda _, selected_release=release: selected_release,
                        "nullcontext": nullcontext,
                        "import_module": loader,
                        "npu_generate_uuid": lambda _: "node-1",
                        "packed_npu_ipc_consumer": lambda expected=weights, **_: expected,
                        "packed_broadcast_consumer": lambda expected=weights, **kwargs: kwargs["post_unpack_func"](
                            expected
                        ),
                        "torch": SimpleNamespace(npu=SimpleNamespace(device=lambda _: nullcontext())),
                    }
                    exec(compile(ast.fix_missing_locations(module), engine_file, "exec"), ns)
                    with self.assertRaisesRegex(RuntimeError, "load failed"):
                        ns["receive_weights"](engine, info)
                    self.assertTrue(state["enabled"])
                    if release:
                        loader.assert_called_once_with("vllm.model_executor.model_loader.mtp_validation")
                    else:
                        loader.assert_not_called()

    def test_new_async_kv_limit_rejects_copied_schedulers_before_requests_can_stall(self):
        for release in (True, False):
            ns = load_nodes(
                "vllm_ascend/platform.py",
                {"_copied_scheduler_mode", "_validate_copied_scheduler_async_kv_limit"},
                {"vllm_version_is": lambda _, selected_release=release: selected_release},
            )
            validate = ns["_validate_copied_scheduler_async_kv_limit"]
            for mode in ("native", "recompute", "dyntra", "profiling", "balance"):
                extension = SimpleNamespace(
                    recompute_scheduler_enable=mode == "recompute",
                    dyntra_lb_config=SimpleNamespace(enabled=mode == "dyntra"),
                    profiling_chunk_config=SimpleNamespace(enabled=mode == "profiling"),
                    enable_balance_scheduling=mode == "balance",
                )
                config = SimpleNamespace(scheduler_config=SimpleNamespace())
                ascend_config = SimpleNamespace(scheduler_config=extension)
                # Old release configs and existing fixtures do not have the new field.
                validate(config, ascend_config)
                for limit in (None, 1, 8):
                    config.scheduler_config.max_async_kv_loads_per_step = limit
                    with self.subTest(release=release, mode=mode, limit=limit):
                        if not release and mode != "native" and limit is not None:
                            with self.assertRaisesRegex(ValueError, "Leave this limit unset"):
                                validate(config, ascend_config)
                        else:
                            validate(config, ascend_config)
                        self.assertEqual(config.scheduler_config.max_async_kv_loads_per_step, limit)

    def test_compiler_supports_main_registration_and_release_hash(self):
        flags = SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False, enable_super_kernel=True)
        for release in (True, False):
            with self.subTest(release=release):

                class Parent:
                    def __init_subclass__(cls, selected_release=release):
                        if not selected_release and "compute_hash" in cls.__dict__:
                            raise TypeError("main rejects the removed compiler hook")

                ns = load_nodes(
                    "vllm_ascend/compilation/compiler_interface.py",
                    {"AscendCompiler"},
                    {
                        "CompilerInterface": Parent,
                        "vllm_version_is": lambda _, selected_release=release: selected_release,
                        "get_ascend_config": lambda: SimpleNamespace(ascend_compilation_config=flags),
                        "torch_npu": SimpleNamespace(__version__="test-version"),
                        "logger": MagicMock(),
                    },
                )
                compiler = ns["AscendCompiler"]()
                config = object()
                factors = compiler.compile_factors(config)
                self.assertIs(compiler.vllm_config, config)
                self.assertEqual(
                    factors,
                    {
                        "torch_npu_version": "test-version",
                        "enable_npugraph_ex": True,
                        "enable_static_kernel": False,
                        "enable_super_kernel": True,
                    },
                )
                if release:
                    self.assertEqual(
                        compiler.compute_hash(config), hashlib.sha256(str(factors).encode()).hexdigest()[:10]
                    )
                else:
                    self.assertNotIn("compute_hash", type(compiler).__dict__)

    def test_config_hash_uses_native_contract_and_keeps_mutable_copies_uncached(self):
        for release in (True, False):
            config = SimpleNamespace(value=1, compute_hash=lambda: "release")
            native = SimpleNamespace(
                hash_config=lambda c: str(c.value),
                hash_config_cached=MagicMock(return_value="cached"),
            )

            def import_config(name, *args, selected=native, **kwargs):
                if name == "vllm.config":
                    return SimpleNamespace(utils=selected)
                return builtins.__import__(name, *args, **kwargs)

            ns = load_nodes(
                "vllm_ascend/utils.py",
                {"get_config_hash"},
                {
                    "vllm_version_is": lambda _, selected_release=release: selected_release,
                    "config_utils": native,
                    "__builtins__": {
                        **vars(builtins),
                        "__import__": import_config,
                    },
                },
            )
            get_hash = ns["get_config_hash"]
            self.assertEqual(get_hash(config), "release" if release else "1")
            config.value = 2
            self.assertEqual(get_hash(config), "release" if release else "2")
            self.assertEqual(get_hash(config, cached=True), "release" if release else "cached")
            self.assertEqual(native.hash_config_cached.call_count, 0 if release else 1)

    def test_scheduler_payload_metrics_empty_prefill_decode_and_encoder(self):
        @dataclass
        class Details:
            num_prefill_requests: int = 0
            num_prefill_tokens: int = 0
            num_decode_requests: int = 0
            num_decode_tokens: int = 0
            num_encoder_inputs: int = 0
            num_encoder_output_tokens: int = 0

        for release in (True, False):
            for logging, profiling in ((False, False), (True, False), (False, True)):
                for scheduled in ({}, {"prefill": 6, "decode": 2}):
                    with self.subTest(release=release, logging=logging, profiling=profiling, scheduled=scheduled):
                        running, new_data = ["decode"], ["prefill"]
                        resumed: list[str] = []
                        blocks: dict[str, Any] = {}
                        calls: list[tuple[Any, ...]] = []

                        def legacy(run, resume, tokens, spec, new_blocks, call_log=calls):
                            call_log.append((run, resume, tokens, spec, new_blocks))
                            return "cached"

                        def main(run, tokens, spec, new_blocks, iteration_details=None, call_log=calls):
                            call_log.append((run, tokens, spec, new_blocks))
                            if iteration_details is not None and "decode" in tokens:
                                iteration_details.num_decode_requests = 1
                                iteration_details.num_decode_tokens = tokens["decode"]
                            return "cached"

                        updates = object()
                        scheduler = SimpleNamespace(
                            log_stats=logging,
                            observability_config=SimpleNamespace(enable_logging_iteration_details=True),
                            vllm_config=SimpleNamespace(
                                profiler_config=SimpleNamespace(profiler="torch" if profiling else None)
                            ),
                            max_num_encoder_input_tokens=32,
                            _make_cached_request_data=legacy if release else main,
                            _get_encoder_input_updates=MagicMock(return_value=updates),
                        )
                        ns = load_nodes(
                            "vllm_ascend/core/scheduler_output.py",
                            {"make_cached_request_data"},
                            {
                                "vllm_version_is": lambda _, selected_release=release: selected_release,
                                "IterationDetails": Details,
                            },
                        )
                        inputs = {"prefill": [0, 1]} if scheduled else {}
                        cached, metadata = ns["make_cached_request_data"](
                            scheduler,
                            running,
                            resumed,
                            scheduled,
                            {},
                            blocks,
                            new_data,
                            inputs,
                            22 if scheduled else 32,
                        )
                        self.assertEqual(cached, "cached")
                        self.assertIs(calls[0][0], running)
                        if release:
                            self.assertEqual(metadata, {})
                            self.assertIs(calls[0][1], resumed)
                            scheduler._get_encoder_input_updates.assert_not_called()
                        else:
                            self.assertIs(metadata["encoder_input_updates"], updates)
                            scheduler._get_encoder_input_updates.assert_called_once_with(scheduled, new_data, inputs)
                            details = metadata["iteration_details"]
                            if logging or profiling:
                                self.assertEqual(details, Details(1, 6, 1, 2, 2, 10) if scheduled else Details())
                            else:
                                self.assertIsNone(details)

    def test_main_engram_retains_slotless_launch_and_empty_input(self):
        class Parent:
            pass

        class Buffer:
            shape = (0,)

            def new_empty(self, shape, **kwargs):
                return shape

        ns = load_nodes(
            "vllm_ascend/models/deepseek_v41/engram/hash_state.py",
            {"AscendNgramHashState"},
            {
                "NgramHashState": Parent,
                "vllm_version_is": lambda _: False,
                "torch": SimpleNamespace(int32="int32"),
            },
        )
        state = ns["AscendNgramHashState"]()
        state.swa_cache_module = SimpleNamespace(block_size=128)
        state.multipliers = SimpleNamespace(shape=(3, 4))
        state.primes = SimpleNamespace(shape=(3, 2))
        self.assertFalse(state.use_slot_cache)
        self.assertEqual(state.block_size, 128)
        self.assertEqual(state.forward(Buffer(), None, None, None, None, None), (0, 3, 6))

    def test_capture_guard_only_calls_release_monitor(self):
        for release in (True, False):
            monitor = SimpleNamespace(validate_cudagraph_capturing_enabled=MagicMock())
            ns = load_nodes(
                "vllm_ascend/compilation/acl_graph.py",
                {"validate_cudagraph_capturing_enabled"},
                {
                    "vllm_version_is": lambda _, selected_release=release: selected_release,
                    "monitor": monitor,
                },
            )
            ns["validate_cudagraph_capturing_enabled"]()
            self.assertEqual(monitor.validate_cudagraph_capturing_enabled.call_count, int(release))

    def test_idle_dp_uses_full_speculative_query_length_on_main(self):
        tree = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute_dummy_batch")
        module = ast.Module(body=[method], type_ignores=[])
        for release in (True, False):
            ns: dict[str, Any] = {
                "vllm_version_is": lambda _, selected_release=release: selected_release,
                "set_current_vllm_config": lambda _: nullcontext(),
                "runtime_guard_idle_step": lambda fn: fn,
            }
            exec(compile(ast.fix_missing_locations(module), "<worker dummy batch>", "exec"), ns)
            worker = SimpleNamespace(
                log_memory_stats=MagicMock(),
                vllm_config=object(),
                model_runner=SimpleNamespace(
                    decode_query_len=5,
                    _dummy_run=MagicMock(),
                ),
            )
            ns["execute_dummy_batch"](worker)
            worker.model_runner._dummy_run.assert_called_once_with(
                1 if release else 5, uniform_decode=True, skip_gdn_state_update=True
            )


if __name__ == "__main__":
    unittest.main()
