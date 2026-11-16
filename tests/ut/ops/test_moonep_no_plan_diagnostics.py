"""CPU checks for removed Slot/plan probes and preserved real prefetch wiring."""

import ast
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_moonep_no_early_prefetch as prefetch_cases
import test_moonep_no_host_sync as dispatch_cases
import torch

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "vllm_ascend/ops/fused_moe"
REMOVED = (
    "MOONEP_SLOT_STATS",
    "MOONEP_PLAN_STATS",
    "MOONEP_EARLY_PLAN",
    "MOONEP_DUMP_PLAN",
    "MOONEP_DUMP_PLAN_N",
    "MOONEP_DEV_BASE",
)
ALL_DIAGNOSTICS = (
    "MOONEP_TIME_STATS",
    "MOONEP_TIME_STATS_EVERY",
    "MOONEP_MLP_SYNC",
    "MOONEP_PF_TIMING",
    "MOONEP_PACE_SYNC",
    "MOONEP_CAP_TRACE",
    "ASCEND_MOONEP_TIME_STATS",
    "ASCEND_MOONEP_TIME_STATS_EVERY",
    "ASCEND_MOONEP_TRACE",
    "VLLM_ASCEND_MOE_PAR_STATS",
    "VLLM_ASCEND_MOE_PAR_STATS_DIR",
    "VLLM_ASCEND_MOE_PAR_STATS_TAG",
    "VLLM_ASCEND_MOE_PAR_FLUSH_EVERY",
    "MOONEP_BALANCE_STATS",
    "MOONEP_BALANCE_STATS_EVERY",
    *REMOVED[:-1],
)


def layers_function():
    source = OPS / "moonep_comm.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_early_layers")
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), ns)
    return ns["_early_layers"]


def runner_tail(events, fail_init=False):
    path = ROOT / "vllm_ascend/worker/v2/model_runner.py"
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    # V2 initializes MoonEP inside the guarded model-load scope.
    block = next(
        n
        for n in ast.walk(method)
        if isinstance(n, ast.If)
        and any(
            isinstance(child, ast.ImportFrom) and child.module == "vllm_ascend.ops.fused_moe.moonep_comm"
            for child in n.body
        )
    )
    invoke = ast.FunctionDef(
        name="invoke",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=block.body,
        decorator_list=[],
    )
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[invoke], type_ignores=[])), str(path), "exec"), ns)
    model = object()

    def initialize():
        events.append("initialize")
        if fail_init:
            raise RuntimeError("initialization failed")

    def install(actual):
        assert actual is model
        events.append("real_prefetch_hook")

    modules = {
        "vllm_ascend.ops.fused_moe.moonep_comm": SimpleNamespace(ensure_runtime_initialized=initialize),
        "vllm_ascend.ops.fused_moe.moonep_balance": SimpleNamespace(maybe_install_pf_stream=install),
    }
    return ns["invoke"], SimpleNamespace(model=model), modules


class NoPlanDiagnosticsTests(unittest.TestCase):
    def test_all_twenty_diagnostics_and_dev_base_have_no_runtime_reads_or_registration(self):
        self.assertEqual(len(ALL_DIAGNOSTICS), 20)
        for package in (ROOT / "vllm_ascend", ROOT.parent / "ascend-moonep/ascend_moonep"):
            for path in package.rglob("*.py"):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertNotIn(node.value, (*ALL_DIAGNOSTICS, "MOONEP_DEV_BASE"))
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(node.attr, (*ALL_DIAGNOSTICS, "MOONEP_DEV_BASE"))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr in ("get", "getenv") and node.args:
                            self.assertFalse(isinstance(node.args[0], ast.Constant) and node.args[0].value == "TAG")

    def test_prediction_helpers_and_diagnostic_state_are_removed_but_shared_enumerator_remains(self):
        tree = ast.parse((OPS / "moonep_comm.py").read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for name in (
            "_early_route",
            "maybe_install_early_router_probe",
            "_plan_on_predicted_routing",
            "early_predicted_plan_ctx",
            "early_plan_fallback",
            "_dump_early_plan",
            "_dump_plan_stats",
            "_dump_plan",
            "_dump_slot_stats",
        ):
            self.assertNotIn(name, names)
        self.assertIn("_early_layers", names)
        identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        for name in (
            "_SLOT_STATS",
            "_PLAN_STATS",
            "_EARLY",
            "_PRED_PLAN_BUF",
            "_DUMP_PLAN_DIR",
            "_DUMP_PLAN_N",
            "_DUMP_PLAN_I",
            "_DUMP_PLAN_FAIL",
        ):
            self.assertNotIn(name, identifiers)

    def check_dispatch(self, transport, rank=0):
        for value in ("0", "1", "invalid"):
            case = dispatch_cases.NoHostCountSyncTests().make_dispatch(transport)
            fn, dispatcher, inputs, shard, weights, scales, calls, modules = case
            fn.__globals__["get_ep_group"] = lambda: SimpleNamespace(world_size=2, rank_in_group=rank)
            with ExitStack() as stack:
                stack.enter_context(
                    patch.dict(os.environ, {**dict.fromkeys(REMOVED, value), "TAG": "preserved-run-label"})
                )
                stack.enter_context(patch.dict(sys.modules, modules))
                for name in ("item", "cpu", "tolist"):
                    stack.enter_context(patch.object(torch.Tensor, name, side_effect=AssertionError("No plan D2H")))
                for name in ("histc", "bincount", "save"):
                    stack.enter_context(patch.object(torch, name, side_effect=AssertionError("No diagnostic work")))
                result = fn(dispatcher, inputs)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
            self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0] if rank == 0 else [0, 0, 1, 0])
            self.assertEqual(result.group_list_type, 1)
            self.assertEqual(calls[0]["num_tokens"], inputs.hidden_states.shape[0])
            self.assertEqual(result.combine_metadata.source_topk_weights_sk.dtype, torch.float32)
            self.assertTrue(torch.equal(result.combine_metadata.w_trim, weights))
            if transport == "mxfp8":
                self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())

    def test_bf16_dispatch_ignores_old_probe_flags_and_keeps_counts_and_weight_snapshot(self):
        self.check_dispatch("bf16")

    def test_mxfp8_dispatch_ignores_old_probe_flags_and_keeps_scale_pair(self):
        self.check_dispatch("mxfp8")

    def test_dev_base_no_longer_changes_real_ep_rank_home_group_selection(self):
        self.check_dispatch("bf16", rank=1)

    def test_shared_enumerator_supports_existing_model_layer_paths(self):
        discover = layers_function()
        layers = [object(), object()]
        for path in ("model.layers", "layers", "transformer.layers", "model.model.layers"):
            model = SimpleNamespace()
            current = model
            parts = path.split(".")
            for part in parts[:-1]:
                nested = SimpleNamespace()
                setattr(current, part, nested)
                current = nested
            setattr(current, parts[-1], layers)
            self.assertEqual(discover(model), layers)

    def test_shared_enumerator_handles_missing_and_empty_containers(self):
        discover = layers_function()
        self.assertEqual(discover(object()), [])
        self.assertEqual(discover(SimpleNamespace(model=SimpleNamespace(layers=[]))), [])
        layer = object()
        self.assertEqual(discover(SimpleNamespace(model=SimpleNamespace(layers=[]), layers=[layer])), [layer])

    def test_real_prefetch_installer_uses_preserved_enumerator_without_a_router_probe(self):
        case = prefetch_cases.NoEarlyPrefetchTests()
        # Reuse the existing real installer extraction; only layer discovery is supplied.
        ns = case.installer("dispatch")
        hooks = []
        weight = object()
        layer = SimpleNamespace(
            mlp=SimpleNamespace(experts=SimpleNamespace(w13_weight=weight)),
            register_forward_pre_hook=lambda hook, **kwargs: hooks.append(hook),
        )
        model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
        modules = {
            "vllm_ascend.ops.fused_moe": SimpleNamespace(moonep_comm=SimpleNamespace(_early_layers=layers_function())),
            "vllm_ascend.ascend_config": SimpleNamespace(get_ascend_config=lambda: SimpleNamespace(enable_moonep=True)),
            "vllm.distributed.parallel_state": SimpleNamespace(get_ep_group=lambda: SimpleNamespace(world_size=8)),
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, dict.fromkeys(REMOVED, "1")):
            ns["maybe_install_pf_stream"](model)
            ns["maybe_install_pf_stream"](model)
            self.assertEqual(len(hooks), 1)
            hooks[0](layer, (), {})
        self.assertEqual(ns["_PF"]["cur_key"], id(weight))

    def test_model_runner_initializes_before_installing_only_real_prefetch_hook(self):
        for value in ("0", "1", "invalid"):
            events = []
            invoke, runner, modules = runner_tail(events)
            with patch.dict(sys.modules, modules), patch.dict(os.environ, dict.fromkeys(REMOVED, value)):
                invoke(runner)
            self.assertEqual(events, ["initialize", "real_prefetch_hook"])

    def test_failed_runtime_initialization_does_not_install_prefetch_hook(self):
        events = []
        invoke, runner, modules = runner_tail(events, fail_init=True)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "initialization failed"):
            invoke(runner)
        self.assertEqual(events, ["initialize"])

    def test_generic_launchers_remove_slot_stats_and_keep_official_profile_and_normal_paths(self):
        for name in ("start_service_mode.sh", "start_service_mode_dp4.sh"):
            service = ROOT / "scripts/dsv4flash_service/service"
            text = (service / name).read_text()
            if name.endswith("_dp4.sh"):
                self.assertIn("export DP_SIZE=${DP_SIZE:-4}", text)
                self.assertIn('exec "$HERE/start_service_mode.sh" "$@"', text)
                text = (service / "start_service_mode.sh").read_text()
            self.assertNotIn("SLOT_STATS", text)
            self.assertIn("--profiler-config", text)
            self.assertIn('-m vllm.entrypoints.cli.main serve "$MODEL"', text)
            self.assertIn('exec "$VLLM_PYTHON" -I -B', text)
            self.assertIn('tee -a -- "$LOG_FILE"', text)
            self.assertIn('"$OUT_DIR/profiling"', text)
            self.assertIn('--data-parallel-size "${DP_SIZE:-8}"', text)


if __name__ == "__main__":
    unittest.main()
