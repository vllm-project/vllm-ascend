"""CPU regressions for load-observer removal; native routing and EPLB remain."""
import ast
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch
import test_moonep_no_host_sync as dispatch_cases


ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT.parent / "ascend-moonep"
REMOVED = ("VLLM_ASCEND_MOE_PAR_STATS", "VLLM_ASCEND_MOE_PAR_STATS_DIR",
           "VLLM_ASCEND_MOE_PAR_STATS_TAG", "VLLM_ASCEND_MOE_PAR_FLUSH_EVERY",
           "MOONEP_BALANCE_STATS", "MOONEP_BALANCE_STATS_EVERY")
FUSED = ("routed_experts.py",)


def environment(value):
    return {**dict.fromkeys(REMOVED, value), "MOONEP_SLOT_STATS": "0"}


def run_eplb(filename, values, group_type=1, *, enabled=True, heat=True,
             multi_stage=False, iterations=1):
    path = ROOT / "vllm_ascend/ops/fused_moe" / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                and ast.unparse(n.test) ==
                "self.dynamic_eplb and _EXTRA_CTX.eplb_heat_collection_status")
    layer = SimpleNamespace(dynamic_eplb=enabled, multi_stage=multi_stage,
                            moe_load=torch.zeros((2, 4) if multi_stage else (4,), dtype=torch.int32 if multi_stage else torch.int64),
                            load_counter=torch.tensor([0], dtype=torch.int64), num_iter=2)
    ns = {"torch": torch, "self": layer,
          "_EXTRA_CTX": SimpleNamespace(eplb_heat_collection_status=heat),
          "fused_experts_results": SimpleNamespace(expert_tokens=values, group_list_type=group_type)}
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")
    for _ in range(iterations):
        exec(code, ns)
    return layer


class NoLoadDiagnosticsTests(unittest.TestCase):
    def test_removed_environment_keys_cannot_be_read_or_parse_invalid_settings(self):
        spec = importlib.util.spec_from_file_location("load_diagnostic_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, environment("invalid")):
            spec.loader.exec_module(envs)
            for key in REMOVED:
                self.assertNotIn(key, envs.env_variables)
                with self.assertRaises(AttributeError):
                    getattr(envs, key)

    def test_collector_histogram_callbacks_and_imports_are_removed_from_runtime(self):
        self.assertFalse((ROOT / "vllm_ascend/ops/fused_moe/moe_par_stats.py").exists())
        forbidden = {"_route_histogram", "_note_route_counts", "_note_compute_load",
                     "_dump_balance_stats", "_BALANCE_STATS_ON", "_BALANCE_STATS", "_BALANCE_EVERY"}
        for source in (ROOT / "vllm_ascend").rglob("*.py"):
            text = source.read_text()
            self.assertNotIn("moe_par_stats", text, str(source))
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.assertNotIn(node.value, REMOVED)
                if isinstance(node, ast.Name):
                    self.assertNotIn(node.id, forbidden)

    def check_dispatch(self, transport):
        for value in ("0", "1", "invalid"):
            case = dispatch_cases.NoHostCountSyncTests().make_dispatch(transport)
            fn, dispatcher, inputs, shard, weights, scales, calls, modules = case
            buffer, _ = fn.__globals__["_select_buffer_for_step"](0, 0, 0, 0)
            name = "dispatch_mxfp8" if transport == "mxfp8" else "dispatch"
            original = getattr(buffer, name)
            captured = []
            def launch(*args, **kwargs):
                captured.append(args)
                return original(*args, **kwargs)
            setattr(buffer, name, launch)
            compat = fn.__globals__["_scratch_for"]()["tpe_compat"]
            old_weights = inputs.topk_weights.clone()
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, environment(value)))
                stack.enter_context(patch.dict(sys.modules, modules))
                for name in ("histc", "bincount", "save"):
                    stack.enter_context(patch.object(torch, name, side_effect=AssertionError("No load observer")))
                for name in ("item", "cpu"):
                    stack.enter_context(patch.object(torch.Tensor, name, side_effect=AssertionError("No load D2H")))
                result = fn(dispatcher, inputs)
            self.assertIs(captured[0][3], compat)
            self.assertTrue(torch.equal(compat, torch.zeros_like(compat)))
            self.assertEqual(result.group_list.tolist(), [2, 1, 1, 0])
            self.assertEqual(result.group_list_type, 1)
            self.assertEqual(result.hidden_states.data_ptr(), shard.data_ptr())
            metadata = result.combine_metadata
            self.assertEqual(metadata.source_topk_weights_sk.dtype, torch.float32)
            self.assertTrue(torch.equal(metadata.source_topk_weights_sk[:2, :2], old_weights))
            self.assertTrue(torch.equal(metadata.w_trim, weights))
            self.assertEqual(calls[0]["num_tokens"], 2)
            if transport == "mxfp8":
                self.assertEqual(result.dynamic_scale.data_ptr(), scales.data_ptr())
            else:
                self.assertTrue(calls[0]["zero_copy"])

    def test_bf16_real_counts_compatibility_and_fp32_route_snapshot_survive(self):
        self.check_dispatch("bf16")

    def test_mxfp8_real_counts_scale_pair_and_valid_token_limit_survive(self):
        self.check_dispatch("mxfp8")

    def test_native_eplb_counts_still_accumulate_for_both_runners(self):
        counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        for name in FUSED:
            for value in ("0", "1", "invalid"):
                with patch.dict(os.environ, environment(value)):
                    layer = run_eplb(name, counts, iterations=2)
                self.assertTrue(torch.equal(layer.moe_load, counts * 2))

    def test_native_eplb_prefix_mode_still_converts_to_counts(self):
        counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        for name in FUSED:
            with patch.dict(os.environ, environment("1")):
                layer = run_eplb(name, counts.cumsum(0), group_type=0)
            self.assertTrue(torch.equal(layer.moe_load, counts))

    def test_native_eplb_multi_stage_window_counter_and_wrap_remain(self):
        counts = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        for name in FUSED:
            with patch.dict(os.environ, environment("invalid")):
                layer = run_eplb(name, counts, multi_stage=True, iterations=3)
            self.assertTrue(torch.equal(layer.moe_load, torch.stack((counts * 2, counts))))
            self.assertEqual(layer.load_counter.tolist(), [3])

    def test_disabled_eplb_or_heat_collection_does_not_access_missing_counts(self):
        for name in FUSED:
            for enabled, heat in ((False, True), (True, False), (False, False)):
                layer = run_eplb(name, None, enabled=enabled, heat=heat)
                self.assertTrue(torch.equal(layer.moe_load, torch.zeros(4, dtype=torch.int64)))

    def test_enabled_eplb_still_rejects_missing_counts(self):
        for name in FUSED:
            with self.assertRaisesRegex(AssertionError, "expert_tokens"):
                run_eplb(name, None)

    def test_planning_preserves_compatibility_schema_but_launches_from_real_topk(self):
        path = SDK / "ascend_moonep/ops.py"
        source = ast.parse(path.read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "PlanningOp")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute")
        records, launches = [], []
        ns = {"torch": torch, "_collect_inputs": lambda *a, **kw: records.append(kw),
              "launch_planning_op": lambda *a: launches.append(a),
              "_num_valid_tokens": lambda ctx: ctx["num_valid_tokens"]}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
        ctx = dict(rank=0, R=2, S=2, K=2, E=4, B=2, NvS=4,
                   planning_workspace=torch.zeros(8, dtype=torch.int32), num_valid_tokens=1)
        plan = SimpleNamespace(dst=torch.empty(4, dtype=torch.int32),
                               experts_to_copy=torch.empty(2, 2, dtype=torch.int32),
                               dup_groups=torch.empty(4, 3, dtype=torch.int32),
                               dup_loffs=torch.empty(4, dtype=torch.int32),
                               dup_counts=torch.empty(2, dtype=torch.int32))
        op = SimpleNamespace(sym_workspace=lambda: SimpleNamespace(addr=123, length=456), num_sms=8)
        topk = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
        cu = torch.empty(6, dtype=torch.int32)
        for fill in (0, 99):
            compatibility = torch.full((4,), fill, dtype=torch.int32)
            ns["compute"](op, ctx, topk, compatibility, cu, plan)
            schema = records[-1]["tensors"][1]
            self.assertEqual(schema[0], "tokens_per_expert")
            self.assertIs(schema[1], compatibility)
            self.assertEqual(schema[2:], ([4], torch.int32))
            self.assertIs(launches[-1][0], topk)
            self.assertFalse(any(arg is compatibility for arg in launches[-1]))
            self.assertEqual(launches[-1][11], 1)

    def test_generic_launchers_remove_load_controls_and_keep_official_profiling(self):
        service = ROOT / "scripts/dsv4flash_service/service"
        for name in ("start_service_mode.sh", "start_service_mode_dp4.sh"):
            text = (service / name).read_text()
            for key in REMOVED:
                self.assertNotIn(key, text)
            for marker in ("$DUMP_DIR", "$BALANCE", "$MOE_PAR", "moe-par-ledger", "moonep-balance"):
                self.assertNotIn(marker, text)
            self.assertIn("--profiler-config", text)
            self.assertIn('vllm serve "$MODEL"', text)
            self.assertIn("ASCEND_RT_VISIBLE_DEVICES", text)
            self.assertIn('tee -a "$LOG_FILE"', text)


if __name__ == "__main__":
    unittest.main()
