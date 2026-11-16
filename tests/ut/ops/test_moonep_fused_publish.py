# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for publication selection and combine ordering."""
import ast
import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(path, class_name, method_name, namespace):
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    tree = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[method_name]


class Metadata(SimpleNamespace):
    pass


class DispatcherPublishTest(unittest.TestCase):
    def invoke(self, masked=True, counts=None, source_weights=True):
        source = torch.randn(6 if masked else 3, 4, dtype=torch.bfloat16)
        if masked:
            source[3:].fill_(float("nan"))
        calls = []

        class FakeBuffer:
            def combine_weighted(self, **kwargs):
                calls.append(kwargs)
                return torch.ones(2, 4, dtype=torch.bfloat16), None, None

        namespace = {"torch": torch, "time": time, "MoonEPCombineMetadata": Metadata}
        fn = load_method(ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py",
                         "TokenDispatcherWithMoonEP", "token_combine", namespace)
        metadata = Metadata(num_tokens=2, buffer=FakeBuffer(), tail_masked=masked,
                            group_list=counts, source_topk_weights_sk=(
                                torch.ones(2, 8) if source_weights else None),
                            plan=object(), hidden_shape=(2, 4))
        obj = SimpleNamespace(_tstats=lambda *args: None)
        result = fn(obj, source, metadata)
        self.assertEqual(tuple(result.shape), (2, 4))
        return source, calls[0]

    def test_fused_passes_raw_tensor_and_device_counts(self):
        class Counts:
            def sum(self):
                raise AssertionError("Fused path must not launch a standalone sum")

        counts = Counts()
        with patch.object(torch, "where", side_effect=AssertionError("No temporary clean matrix")):
            source, call = self.invoke(counts=counts)
        self.assertIs(call["hidden_nvsh"], source)
        self.assertIs(call["row_counts"], counts)
        self.assertTrue(call["fused_publish"])
        self.assertEqual(source.dtype, torch.bfloat16)

    def test_nan_tail_is_delegated_to_native_publication(self):
        counts = torch.tensor([1, 2], dtype=torch.int64)
        source, call = self.invoke(counts=counts)
        self.assertIs(call["hidden_nvsh"], source)
        self.assertTrue(torch.isnan(source[3:]).all())
        self.assertIs(call["row_counts"], counts)
        self.assertTrue(call["fused_publish"])

    def test_old_zero_cannot_select_temporary_mask_or_copy(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_MOONEP_FUSED_PUBLISH": "0"}):
            with patch.object(torch, "where", side_effect=AssertionError("No where")), patch.object(
                torch, "arange", side_effect=AssertionError("No row-id mask")
            ), patch.object(torch, "zeros", side_effect=AssertionError("No clean buffer")):
                source, call = self.invoke(counts=torch.tensor([1, 2], dtype=torch.int64))
        self.assertIs(call["hidden_nvsh"], source)
        self.assertTrue(call["fused_publish"])

    def test_missing_source_weights_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires source-rank route weights"):
            self.invoke(counts=torch.tensor([1, 2], dtype=torch.int64), source_weights=False)

    def test_trimmed_fused_input_uses_its_shape(self):
        source, call = self.invoke(masked=False)
        self.assertIs(call["hidden_nvsh"], source)
        self.assertEqual(source.shape[0], 3)
        self.assertIsNone(call["row_counts"])
        self.assertTrue(call["fused_publish"])

    def test_missing_masked_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires counts"):
            self.invoke(counts=None)

    def test_environment_registration_is_removed(self):
        spec = importlib.util.spec_from_file_location("publish_envs", ROOT / "vllm_ascend/envs.py")
        envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(envs)
        key = "VLLM_ASCEND_MOONEP_FUSED_PUBLISH"
        self.assertNotIn(key, envs.env_variables)
        for value in (None, "0", "1", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ):
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
                self.assertFalse(hasattr(envs, key))


class BufferPublishTest(unittest.TestCase):
    def setUp(self):
        sdk = ROOT.parent / "ascend-moonep"
        spec = importlib.util.spec_from_file_location(
            "publish_geometry", sdk / "ascend_moonep/combine_publish.py"
        )
        self.geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.geometry)
        self.order = []
        self.sym = torch.empty(8, 4, dtype=torch.bfloat16)
        owner = self
        class Publisher:
            def publish(self, source, destination, counts):
                owner.geometry.publish_geometry(source, destination, counts)
                owner.order.append("publish")
                rows = source.shape[0] if counts is None else int(counts.sum())
                rows = max(0, min(rows, source.shape[0]))
                destination.zero_()
                destination[:rows].copy_(source[:rows])
        class Plan:
            N, R, K = 8, 1, 2
        self.plan = Plan()
        self.weights = torch.ones(4, 8)
        self.ctx = {
            "S": 4, "H": 4, "K": 2, "R": 1, "hidden_buf_local": self.sym,
            "kernels": {"combine": SimpleNamespace(compute_weighted=self.compute)},
        }
        self.obj = SimpleNamespace(
            _require_ctx=lambda: self.ctx, _ensure_shmem_views=lambda: None,
            _combine_publisher=Publisher(),
        )
        namespace = {"torch": torch, "MoonEPCommPlan": Plan,
                     "cross_rank_barrier": lambda *args: self.order.append("barrier")}
        self.fn = load_method(sdk / "ascend_moonep/api.py", "Buffer", "combine_weighted", namespace)

    def compute(self, ctx, output, plan, weights):
        self.order.append("weighted")
        self.assertIs(plan, self.plan)
        self.assertIs(weights, self.weights)
        output.zero_()

    def test_publish_precedes_existing_barrier_and_weighted_kernel(self):
        source = torch.ones(8, 4, dtype=torch.bfloat16)
        source[5:].fill_(float("nan"))
        self.fn(self.obj, self.plan, source, self.weights, num_tokens=4,
                fused_publish=True, row_counts=torch.tensor([2, 3], dtype=torch.int64))
        self.assertEqual(self.order, ["publish", "barrier", "weighted"])
        self.assertTrue(torch.equal(self.sym[5:], torch.zeros(3, 4, dtype=torch.bfloat16)))

    def test_trimmed_prefix_zero_extends(self):
        source = torch.ones(3, 4, dtype=torch.bfloat16)
        self.fn(self.obj, self.plan, source, self.weights, num_tokens=4, fused_publish=True)
        self.assertTrue(torch.equal(self.sym[:3], source))
        self.assertTrue(torch.equal(self.sym[3:], torch.zeros(5, 4, dtype=torch.bfloat16)))

    def test_zero_copy_contract_is_not_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    zero_copy=True, fused_publish=True)
        self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4, zero_copy=True)
        self.assertEqual(self.order, ["barrier", "weighted"])

    def test_legacy_shape_and_fused_count_contracts(self):
        with self.assertRaisesRegex(ValueError, "Legacy weighted"):
            self.fn(self.obj, self.plan, self.sym[:3], self.weights, num_tokens=4)
        with self.assertRaisesRegex(ValueError, "row_counts requires"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    row_counts=torch.ones(3, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "int64"):
            self.fn(self.obj, self.plan, self.sym, self.weights, num_tokens=4,
                    fused_publish=True, row_counts=torch.ones(3, dtype=torch.int32))


    def test_sdk_default_copy_path_remains_available(self):
        source = torch.ones(8, 4, dtype=torch.bfloat16)
        self.fn(self.obj, self.plan, source, self.weights, num_tokens=4)
        self.assertTrue(torch.equal(self.sym, source))
        self.assertEqual(self.order, ["barrier", "weighted"])

    def test_publish_failure_has_no_barrier_compute_or_copy_fallback(self):
        def fail(*args):
            raise RuntimeError("Fused publish library failed")

        self.obj._combine_publisher = SimpleNamespace(publish=fail)
        self.sym.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "library failed"):
            self.fn(self.obj, self.plan, torch.ones(8, 4, dtype=torch.bfloat16),
                    self.weights, num_tokens=4, fused_publish=True)
        self.assertEqual(self.order, [])
        self.assertTrue(torch.isnan(self.sym).all())

    def test_vllm_to_sdk_preserves_weighted_order_and_cleans_nan_tail(self):
        source = torch.ones(8, 4, dtype=torch.bfloat16)
        source[3:].fill_(float("nan"))
        buffer = SimpleNamespace(combine_weighted=lambda **kwargs: self.fn(self.obj, **kwargs))
        metadata = Metadata(
            num_tokens=2, buffer=buffer, tail_masked=True,
            group_list=torch.tensor([1, 2], dtype=torch.int64),
            source_topk_weights_sk=self.weights, plan=self.plan, hidden_shape=(2, 4))
        namespace = {"torch": torch, "time": time, "MoonEPCombineMetadata": Metadata}
        combine = load_method(ROOT / "vllm_ascend/ops/fused_moe/moonep_comm.py",
                              "TokenDispatcherWithMoonEP", "token_combine", namespace)
        output = combine(SimpleNamespace(_tstats=lambda *args: None), source, metadata)
        self.assertEqual(tuple(output.shape), (2, 4))
        self.assertEqual(self.order, ["publish", "barrier", "weighted"])
        self.assertTrue(torch.equal(self.sym[:3], source[:3]))
        self.assertTrue(torch.equal(self.sym[3:], torch.zeros_like(self.sym[3:])))


class PublisherDependencyTest(unittest.TestCase):
    def setUp(self):
        sdk = ROOT.parent / "ascend-moonep"
        spec = importlib.util.spec_from_file_location(
            "publisher_dependency", sdk / "ascend_moonep/combine_publish.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_missing_library_is_an_explicit_error(self):
        with patch.object(self.module.Path, "is_file", return_value=False):
            with patch.object(self.module.ctypes, "CDLL") as load:
                with self.assertRaisesRegex(RuntimeError, "library is not installed"):
                    self.module.CombinePublisher()
        load.assert_not_called()

    def test_abi_mismatch_is_an_explicit_error(self):
        library = Mock()
        library.moonep_publish_abi.return_value = 999
        with patch.object(self.module.Path, "is_file", return_value=True), patch.object(
            self.module.ctypes, "CDLL", return_value=library
        ):
            with self.assertRaisesRegex(RuntimeError, "ABI mismatch"):
                self.module.CombinePublisher()

    def test_valid_abi_uses_the_unified_device_library(self):
        library = Mock()
        library.moonep_publish_abi.return_value = 1
        with patch.object(self.module.Path, "is_file", return_value=True), patch.object(
            self.module.ctypes, "CDLL", return_value=library
        ) as load:
            publisher = self.module.CombinePublisher()
        expected = (ROOT.parent / "ascend-moonep/ascend_moonep/lib/libascend_moonep_kernels.so").resolve()
        self.assertEqual(publisher.library_path, expected)
        load.assert_called_once_with(str(expected))


if __name__ == "__main__":
    unittest.main()
