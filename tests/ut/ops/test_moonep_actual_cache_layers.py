"""CPU checks for cache sizing from the loaded, registered MoE layers."""
import ast
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import test_moonep_slot_cache_stream as cache_cases


ROOT = Path(__file__).resolve().parents[3]
BALANCE = ROOT / "vllm_ascend/ops/fused_moe/moonep_balance.py"
COMM = ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py"


class ActualCacheLayersTests(unittest.TestCase):
    def setUp(self):
        self.allocations = []
        self.events = []
        self.fail_cache = False
        self.layout_queries = []
        ns = {
            "dataclass": dataclass, "torch": torch, "threading": threading,
            "envs": SimpleNamespace(MOONEP_REDUNDANT_EXPERT_SLOT_COUNT=None),
            "_CACHED_EXPERT_COUNT": 2, "MoEWeights": SimpleNamespace,
            "_packed_alias": lambda home, slot, *args: home,
        }

        def infer(layer, hidden):
            self.layout_queries.append(layer)
            if layer.invalid:
                raise ValueError("Invalid loaded layout")
            return layer.F

        ns["_infer_F"] = infer
        ns["_payload_specs"] = lambda layer, hidden, width: layer.specs
        names = {"_LayerWeightLayout", "_layer_weight_layout", "_slot_pool_key",
                 "_build_slot_states", "_SharedSlots", "_LayerWeightState"}
        body = [node for node in ast.parse(BALANCE.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(
            body=[future, *body], type_ignores=[])), str(BALANCE), "exec"), ns)
        self.ns = ns

        def allocate(shape, dtype, dev):
            if self.fail_cache and len(shape) == 4:
                raise RuntimeError("Injected NPU cache allocation OOM")
            self.allocations.append(tuple(shape))
            self.events.append("allocate")
            return torch.zeros(shape, dtype=dtype), None, None

        def projection(home, slot, experts, slots, rank, dev):
            return torch.empty((experts + slots, *home.shape[1:]), dtype=home.dtype)

        self.fake_buffer = SimpleNamespace(
            create_vmm_physical=allocate,
            create_full_projection_tensor=projection,
            attach_local_source_vas=lambda *args: None,
        )
        modules = patch.dict(sys.modules, {"ascend_moonep.buffer_c": self.fake_buffer})
        modules.start()
        self.addCleanup(modules.stop)

    def layer(self, width=2, quantized=False, invalid=False):
        specs = [SimpleNamespace(name="w13", carrier=(2, 2 * width)),
                 SimpleNamespace(name="w2", carrier=(width, 2))]
        if quantized:
            specs = [specs[0], SimpleNamespace(name="w13_scale", carrier=(1, 2)),
                     specs[1], SimpleNamespace(name="w2_scale", carrier=(1, 2))]
        return SimpleNamespace(
            w13_weight=torch.zeros(2, 2, 2 * width), w2_weight=torch.zeros(2, width, 2),
            F=width, specs=specs, invalid=invalid,
        )

    def entries(self, layers):
        return [dict(layer=layer, E=16, H=2) for layer in layers]

    def build(self, layers):
        return self.ns["_build_slot_states"](self.entries(layers), 0, 0)

    def pools(self):
        return list(self.ns["_SharedSlots"]._pools.values())

    def test_eighty_layers_allocate_eighty_and_have_valid_indices(self):
        states = self.build([self.layer() for _ in range(80)])
        pool = self.pools()[0]
        self.assertEqual(pool["cache_layers"], 80)
        self.assertEqual(pool["cache"]["w13"].shape, (80, 2, 2, 4))
        self.assertEqual([s.layer_index for s in states], list(range(80)))
        self.assertEqual([s.cache_layer_index for s in states], list(range(80)))
        for state in states:
            state.caches["w13"][state.cache_layer_index, 0].fill_(state.layer_index)
        self.assertEqual(pool["cache"]["w13"][79, 0, 0, 0].item(), 79)

    def test_interleaved_shapes_use_dense_pool_local_indices(self):
        states = self.build([self.layer(2), self.layer(3), self.layer(2),
                             self.layer(3), self.layer(2)])
        self.assertEqual([s.layer_index for s in states], [0, 1, 2, 3, 4])
        self.assertEqual([s.cache_layer_index for s in states], [0, 0, 1, 1, 2])
        self.assertEqual([p["cache_layers"] for p in self.pools()], [3, 2])
        self.assertIs(states[0].slots["w13"], states[2].slots["w13"])
        self.assertIsNot(states[0].slots["w13"], states[1].slots["w13"])

    def test_payload_geometry_separates_cache_pools(self):
        states = self.build([self.layer(), self.layer(quantized=True), self.layer()])
        self.assertEqual([p["cache_layers"] for p in self.pools()], [2, 1])
        self.assertEqual(set(states[1].caches), {"w13", "w13_scale", "w2", "w2_scale"})
        self.assertEqual(states[1].caches["w13_scale"].shape, (1, 2, 1, 2))

    def test_duplicate_and_empty_registry_entries_do_not_add_capacity(self):
        first, second = self.layer(), self.layer()
        entries = self.entries([first, first]) + [dict(layer=None), {}] + self.entries([second])
        states = self.ns["_build_slot_states"](entries, 0, 0)
        self.assertEqual(len(states), 2)
        self.assertEqual(len(self.layout_queries), 2)
        self.assertEqual(self.pools()[0]["cache_layers"], 2)

    def test_empty_registry_allocates_nothing(self):
        self.assertEqual(self.ns["_build_slot_states"]([{}, dict(layer=None)], 0, 0), [])
        self.assertEqual(self.allocations, [])
        self.assertEqual(self.pools(), [])

    def test_cache_disabled_allocates_only_existing_slots_and_homes(self):
        self.ns["_CACHED_EXPERT_COUNT"] = 0
        states = self.build([self.layer(), self.layer()])
        self.assertEqual(self.pools()[0]["cache"], {})
        self.assertTrue(all(state.caches == {} for state in states))
        self.assertTrue(all(len(shape) == 3 for shape in self.allocations))
        self.assertEqual(len(self.allocations), 2 + 2 * 2)

    def test_same_pool_capacity_reuses_all_cache_and_slot_allocations(self):
        state = self.build([self.layer()])[0]
        before = list(self.allocations)
        layout = self.ns["_layer_weight_layout"](state._layer, 16, 2)
        pool = self.ns["_SharedSlots"].get(16, 2, 2, 2, 0, 0, layout.specs, cache_layers=1)
        self.assertIs(pool["cache"]["w13"], state.caches["w13"])
        self.assertEqual(self.allocations, before)

    def test_changed_capacity_rejected_without_resize(self):
        first = self.layer()
        self.build([first])
        before = list(self.allocations)
        with self.assertRaisesRegex(RuntimeError, "count changed"):
            self.build([first, self.layer()])
        self.assertEqual(self.allocations, before)
        self.assertEqual(self.pools()[0]["cache_layers"], 1)

    def test_invalid_pool_capacity_rejected_before_allocation(self):
        specs = self.layer().specs
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "registered layer"):
                self.ns["_SharedSlots"].get(16, 2, 2, 2, 0, 0, specs, cache_layers=count)
        self.assertEqual(self.allocations, [])

    def test_invalid_cache_index_rejected_before_allocation(self):
        layer = self.layer()
        layout = self.ns["_layer_weight_layout"](layer, 16, 2)
        for index in (-1, 2):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "outside"):
                self.ns["_LayerWeightState"](
                    layer, layout, 0, 0, layer_index=9,
                    cache_layer_index=index, cache_layer_count=2)
        self.assertEqual(self.allocations, [])

    def test_wrong_cache_shape_rejected_during_state_construction(self):
        layer = self.layer()
        state = self.build([layer])[0]
        pool = self.pools()[0]
        pool["cache"]["w13"] = torch.empty(2, 2, 2, 4)
        before = list(self.allocations)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.ns["_LayerWeightState"](
                layer, self.ns["_layer_weight_layout"](layer, 16, 2), 0, 0,
                layer_index=0, cache_layer_index=0, cache_layer_count=1)
        self.assertEqual(self.allocations, before)
        self.assertIsNot(pool["cache"]["w13"], state.caches["w13"])

    def test_old_environment_values_do_not_control_capacity(self):
        for old in ("1", "64", "1000", "invalid", ""):
            with self.subTest(old=old), patch.dict(os.environ, {"MOONEP_SLOT_CACHE_LAYERS": old}):
                self.ns["_SharedSlots"]._pools.clear()
                states = self.build([self.layer(), self.layer(), self.layer()])
                self.assertEqual(states[0].caches["w13"].shape[0], 3)

    def test_logical_bytes_follow_pool_counts_and_payload_geometry(self):
        self.ns["_CACHED_EXPERT_COUNT"] = 3
        states = self.build([self.layer(2), self.layer(3), self.layer(2)])
        actual = sum(t.numel() * t.element_size() for p in self.pools() for t in p["cache"].values())
        # Two width-2 layers plus one width-3 layer; all payload carriers are BF16 bytes.
        expected = 3 * (2 * (2 * 4 + 2 * 2) + (2 * 6 + 3 * 2)) * 2
        self.assertEqual(actual, expected)
        old_reserved = 64 * 3 * ((2 * 4 + 2 * 2) + (2 * 6 + 3 * 2)) * 2
        self.assertLess(actual, old_reserved)
        self.assertEqual(len(states), 3)

    def test_layout_failure_is_reported_before_any_cache_or_home_allocation(self):
        with self.assertRaisesRegex(ValueError, "Invalid loaded layout"):
            self.build([self.layer(), self.layer(invalid=True)])
        self.assertEqual(self.allocations, [])

    def test_allocator_failure_is_propagated_without_reducing_cache(self):
        self.fail_cache = True
        with self.assertRaisesRegex(RuntimeError, "cache allocation OOM"):
            self.build([self.layer(), self.layer()])
        self.assertEqual(self.ns["_CACHED_EXPERT_COUNT"], 2)
        self.assertEqual(self.pools(), [])

    def test_cache_rows_are_independent_while_temporary_slots_are_shared(self):
        first, second = self.build([self.layer(), self.layer()])
        self.assertIs(first.caches["w13"], second.caches["w13"])
        self.assertIs(first.slots["w13"], second.slots["w13"])
        first.caches["w13"][first.cache_layer_index].fill_(11)
        second.caches["w13"][second.cache_layer_index].fill_(22)
        self.assertEqual(first.caches["w13"][:, 0, 0, 0].tolist(), [11., 22.])
        first.cache_experts[0] = 7
        self.assertEqual(second.cache_experts, [-1, -1])

    def test_main_and_side_stream_copy_with_pool_index_not_global_layer_id(self):
        for overlap in ("none", "dispatch"):
            with self.subTest(overlap=overlap):
                case = cache_cases.CacheStreamTests(methodName="runTest")
                case.setUp()
                self.addCleanup(case.doCleanups)
                case.ns["_PREFETCH_OVERLAP"] = overlap
                case.state.layer_index = 79  # Log identity intentionally exceeds this pool's size.
                case.state.cache_layer_index = 0
                case.state.cache_next = 1
                plan = cache_cases.Plan(torch.tensor([[1, 2]]))
                if overlap == "dispatch":
                    case.hook(plan)
                self.assertEqual(case.forward(plan).flatten().tolist(), [11., 22.])
                self.assertEqual(case.state.caches["w13"][0, 1].item(), 22.)

    def test_every_layout_is_known_before_first_allocation(self):
        layers = [self.layer(2), self.layer(3), self.layer(2)]
        original = self.fake_buffer.create_vmm_physical

        def checked(*args):
            self.assertEqual(self.layout_queries, layers)
            return original(*args)

        self.fake_buffer.create_vmm_physical = checked
        self.build(layers)

    def test_registry_snapshot_and_allocation_precede_shmem_and_publication(self):
        tree = ast.parse(COMM.read_text())
        runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MoonEPRuntime")
        fn = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == "_initialize_locked")
        calls = {n.func.id: n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        builder_line = calls["_build_slot_states"]
        warmup_line = calls["_warmup_aicpu_custom_scheduler"]
        shmem_lines = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr in {"init_with_ip_port", "init_with_buffer"}]
        publish_line = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute) and n.func.attr == "publish")
        self.assertLess(builder_line, warmup_line)
        self.assertTrue(all(builder_line < line < publish_line for line in shmem_lines))
        # Runtime's existing ready/failure guard keeps all sizing and allocation one-shot.
        ensure = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == "_ensure_initialized")
        ns = {}
        exec(compile(ast.Module(body=[ensure], type_ignores=[]), str(COMM), "exec"), ns)
        initialize = Mock()
        obj = SimpleNamespace(_lock=threading.Lock(), _ready=False, _fail=None, _initialize_locked=initialize)
        ns["_ensure_initialized"](obj)
        ns["_ensure_initialized"](obj)
        initialize.assert_called_once()

    def test_removed_key_has_no_runtime_reference(self):
        for path in (BALANCE, COMM, ROOT / "vllm_ascend/envs.py"):
            tree = ast.parse(path.read_text())
            self.assertFalse(any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                                 and "MOONEP_SLOT_CACHE_LAYERS" in n.value for n in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
