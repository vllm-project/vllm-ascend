"""CPU integration checks for MoonEP transport selection, without loading NPU operators."""

import ast
import importlib.util
import io
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
COMM = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"
NEW = "MOONEP_TRANSPORT_TYPE"
OLD = "MOONEP_DISABLE_UDMA"


class TransportTypeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("transport_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        cls = next(
            n for n in ast.parse(COMM.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "_MoonEPRuntime"
        )
        self.methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("_initialize_locked", "_ensure_initialized")
        ]
        balance = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py").read_text())
        self.overlap_fn = next(
            n for n in balance.body if isinstance(n, ast.FunctionDef) and n.name == "udma_prefetch_overlap_requested"
        )

    def harness(self, mode="dispatch", effective=None, capability_error=False, ep=8, capacity=4096):
        # Execute the complete real initializer against host-only Buffer/SHMEM
        # doubles. No torch import, NPU allocation, collective or operator runs.
        events, contexts = [], []
        group = SimpleNamespace(device_group=object(), world_size=ep, rank_in_group=0)
        state = SimpleNamespace(_w13_src=object(), publish=Mock(side_effect=lambda: events.append("publish")))

        def buffer(*args, **kwargs):
            events.append("buffer")
            flags = dict(
                planning=False,
                dispatch=kwargs.get("use_udma", False),
                combine=kwargs.get("use_udma", False),
                prefetch=kwargs.get("use_udma", False),
            )
            if effective is not None:
                flags.update(effective)
            ctx = {f"_{name}_use_udma": value for name, value in flags.items()}
            contexts.append(ctx)
            return SimpleNamespace(_require_ctx=lambda: ctx)

        constructor = Mock(side_effect=buffer)
        init_ipport = Mock(side_effect=lambda **kw: events.append("ipport"))
        init_uid = Mock(side_effect=lambda **kw: events.append("uid"))
        validate = Mock(side_effect=RuntimeError("direct-mode required") if capability_error else None)
        ready = Mock(return_value=(3, 2, 2))
        shmem = SimpleNamespace(
            is_initialized=lambda: False,
            init_with_ip_port=init_ipport,
            init_with_buffer=init_uid,
            validate_udma_prefetch_overlap=validate,
            require_udma_prefetch_overlap=ready,
        )
        ns = {
            "os": os,
            "envs": self.envs,
            "get_ep_group": lambda: group,
            "get_moonep_tokens_capacity": lambda: capacity,
            "_registry_keys": lambda: [(256, 6, 4096, 32)],
            "_moonep_pool_capacities": lambda *a, **kw: [128, 4096],
            "_moonep_slot_count": lambda *a: 32,
            "_registry_lock": threading.Lock(),
            "_moonep_layer_registry": [object()],
            "_warmup_aicpu_custom_scheduler": lambda: events.append("warmup"),
            "torch": SimpleNamespace(
                zeros=Mock(return_value=object()),
                bfloat16=object(),
                float32=object(),
                int32=object(),
                npu=SimpleNamespace(
                    current_device=lambda: 0, empty_cache=Mock(), mem_get_info=lambda dev: (2**30, 2**31)
                ),
            ),
        }
        exec(compile(ast.Module(body=self.methods, type_ignores=[]), str(COMM), "exec"), ns)
        overlap_ns = {"_PREFETCH_OVERLAP": mode}
        exec(compile(ast.Module(body=[self.overlap_fn], type_ignores=[]), str(COMM), "exec"), overlap_ns)
        modules = {
            "ascend_moonep": SimpleNamespace(Buffer=constructor, ShmemRuntime=shmem),
            "ascend_moonep.buffer_c": SimpleNamespace(symmetric_descriptors=lambda: []),
            "vllm.config": SimpleNamespace(
                get_current_vllm_config=lambda: SimpleNamespace(
                    parallel_config=SimpleNamespace(
                        data_parallel_master_ip="127.0.0.1", data_parallel_size=8, data_parallel_size_local=8
                    )
                )
            ),
            "vllm_ascend.ops.fused_moe.moonep_balance": SimpleNamespace(
                _build_slot_states=lambda *a: [state],
                udma_prefetch_overlap_requested=overlap_ns["udma_prefetch_overlap_requested"],
            ),
        }
        runtime = SimpleNamespace(_buffers={}, _pads={}, _lock=threading.Lock(), _ready=False, _fail=None)
        runtime._initialize_locked = lambda: ns["_initialize_locked"](runtime)

        def run(environment, ensured=False):
            output = io.StringIO()
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.dict(sys.modules, modules),
                redirect_stdout(output),
            ):
                if ensured:
                    ns["_ensure_initialized"](runtime)
                else:
                    runtime._initialize_locked()
            return output.getvalue()

        return SimpleNamespace(
            run=run,
            events=events,
            constructor=constructor,
            contexts=contexts,
            ipport=init_ipport,
            uid=init_uid,
            validate=validate,
            ready=ready,
            state=state,
            runtime=runtime,
            torch=ns["torch"],
        )

    def test_default_mte_is_central_and_old_name_is_not_registered(self):
        self.assertIn(NEW, self.envs.env_variables)
        self.assertNotIn(OLD, self.envs.env_variables)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(getattr(self.envs, NEW), "mte")
            with self.assertRaises(AttributeError):
                getattr(self.envs, OLD)
        h = self.harness()
        output = h.run({})
        self.assertFalse(h.ipport.call_args.kwargs["enable_udma"])
        self.assertIn("planning=mte dispatch=mte combine=mte prefetch=mte", output)

    def test_both_transports_reach_every_buffer_and_both_bootstrap_interfaces(self):
        for transport in ("mte", "udma"):
            for bootstrap in ("ipport", "uid"):
                with self.subTest(transport=transport, bootstrap=bootstrap):
                    h = self.harness()
                    output = h.run({NEW: transport, "MOONEP_SHMEM_BOOTSTRAP_TYPE": bootstrap})
                    self.assertEqual(h.constructor.call_count, 2)
                    expected = transport == "udma"
                    for call in h.constructor.call_args_list:
                        self.assertEqual(call.kwargs.get("use_udma", False), expected)
                    selected = h.ipport if bootstrap == "ipport" else h.uid
                    self.assertEqual(selected.call_args.kwargs["enable_udma"], expected)
                    self.assertIn(f"planning=mte dispatch={transport} combine={transport} prefetch={transport}", output)
                    self.assertEqual(h.events, ["buffer", "buffer", "warmup", bootstrap, "publish"])

    def test_old_flag_values_are_ignored_including_old_udma_request(self):
        for raw in ("0", "1", "", "invalid"):
            with self.subTest(raw=raw):
                h = self.harness()
                h.run({OLD: raw})
                self.assertFalse(h.ipport.call_args.kwargs["enable_udma"])
                self.assertFalse(h.ipport.call_args.kwargs["prefetch_overlap"])

    def test_new_value_wins_over_contradictory_old_flag(self):
        for transport, old in (("mte", "0"), ("udma", "1")):
            with self.subTest(transport=transport):
                h = self.harness()
                h.run({NEW: transport, OLD: old})
                self.assertEqual(h.ipport.call_args.kwargs["enable_udma"], transport == "udma")

    def test_invalid_exact_values_rejected_before_allocation_or_initialization(self):
        for raw in ("", "MTE", "UDMA", " mte", "udma ", "mte\n", "0", "1", "auto", "urma"):
            with self.subTest(raw=raw):
                h = self.harness()
                with self.assertRaisesRegex(ValueError, NEW + " must be mte or udma"):
                    h.run({NEW: raw, OLD: "1"})
                self.assertEqual(h.events, [])
                h.torch.zeros.assert_not_called()
                h.validate.assert_not_called()
                h.ipport.assert_not_called()
                h.uid.assert_not_called()

    def test_overlap_qp_requested_only_for_udma_and_non_none_modes(self):
        for transport in ("mte", "udma"):
            for mode in ("none", "dispatch", "dispatch|home_gemm1"):
                with self.subTest(transport=transport, mode=mode):
                    h = self.harness(mode=mode)
                    h.run({NEW: transport})
                    expected = transport == "udma" and mode != "none"
                    self.assertEqual(h.ipport.call_args.kwargs["prefetch_overlap"], expected)
                    self.assertEqual(h.validate.call_count, int(expected))
                    self.assertEqual(h.ready.call_count, int(expected))
                    self.assertTrue(all(c["udma_prefetch_overlap_ready"] == expected for c in h.contexts))

    def test_udma_capability_failure_does_not_fall_back_or_publish(self):
        h = self.harness(capability_error=True)
        with self.assertRaisesRegex(RuntimeError, "direct-mode required"):
            h.run({NEW: "udma"})
        h.ipport.assert_not_called()
        h.uid.assert_not_called()
        h.ready.assert_not_called()
        h.state.publish.assert_not_called()

    def test_mte_does_not_probe_udma_capability(self):
        h = self.harness(capability_error=True)
        h.run({NEW: "mte"})
        h.validate.assert_not_called()
        h.ready.assert_not_called()
        h.state.publish.assert_called_once_with()

    def test_effective_backend_mismatch_is_rejected_before_publish(self):
        for transport, effective in (
            ("udma", {"dispatch": False}),
            ("mte", {"prefetch": True}),
            ("udma", {"planning": True}),
        ):
            with self.subTest(transport=transport, effective=effective):
                h = self.harness(effective=effective)
                with self.assertRaisesRegex(RuntimeError, "MoonEP backend mismatch"):
                    h.run({NEW: transport})
                h.state.publish.assert_not_called()

    def test_config_read_once_and_not_re_read_after_successful_initialization(self):
        h = self.harness()
        getter = Mock(return_value="mte")
        with patch.dict(self.envs.env_variables, {NEW: getter}):
            h.run({}, ensured=True)
            h.run({NEW: "invalid"}, ensured=True)
        getter.assert_called_once_with()
        self.assertTrue(h.runtime._ready)
        self.assertEqual(h.constructor.call_count, 2)

    def test_invalid_configuration_preserves_failed_init_guard(self):
        h = self.harness()
        with self.assertRaisesRegex(ValueError, NEW):
            h.run({NEW: "auto"}, ensured=True)
        with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
            h.run({NEW: "mte"}, ensured=True)
        self.assertFalse(h.runtime._ready)
        self.assertEqual(h.events, [])

    def test_ep_or_capacity_rejection_precedes_transport_read(self):
        for ep, capacity in ((1, 4096), (8, 0)):
            with self.subTest(ep=ep, capacity=capacity):
                h = self.harness(ep=ep, capacity=capacity)
                getter = Mock(side_effect=AssertionError("Rejected init must not read transport"))
                with patch.dict(self.envs.env_variables, {NEW: getter}), self.assertRaises(RuntimeError):
                    h.run({})
                getter.assert_not_called()

    def test_one_central_read_and_no_old_runtime_alias(self):
        tree = ast.parse(COMM.read_text())
        reads = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "envs" and n.attr == NEW
        ]
        self.assertEqual(len(reads), 1)
        for path in (COMM, ROOT / "vllm_ascend/envs.py"):
            strings = {
                n.value
                for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            }
            self.assertNotIn(OLD, strings)

    def test_generic_launcher_defaults_mte_and_drops_old_transport_entry(self):
        script = (ROOT / "scripts/dsv4flash_service/service/start_service_mode.sh").read_text()
        self.assertIn("export MOONEP_TRANSPORT_TYPE=${MOONEP_TRANSPORT_TYPE:-mte}", script)
        self.assertNotIn(OLD, script)


if __name__ == "__main__":
    unittest.main()
