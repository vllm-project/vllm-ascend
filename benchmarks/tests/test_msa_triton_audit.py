# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of diagnostic transparency, not Triton/NPU correctness."""

import json
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.msa_triton_audit import TritonAudit


class TestTritonAudit(unittest.TestCase):
    def test_records_metadata_preserves_hooks_and_returns_and_restores(self):
        calls = []
        hooks = SimpleNamespace(
            pre=lambda **kw: calls.append(("pre", kw["key"])), post=lambda **kw: calls.append(("post", kw["key"]))
        )

        def kernel(raw, width):
            return raw, width

        kernel.__module__ = "vllm_ascend.ops.triton.msa.msa_m3_index_postprocess"

        class Jit:
            fn = staticmethod(kernel)
            arg_names = ["raw", "width"]

            def run(self, *args, **kwargs):
                hooks.pre(key="real-key")
                hooks.post(key="real-key")
                return SimpleNamespace(hash="binary-key", original_args=args, original_kwargs=kwargs)

        original_run, old_pre, old_post = Jit.run, hooks.pre, hooks.post
        value = object()
        with tempfile.TemporaryDirectory() as directory:
            audit = TritonAudit(directory).install(Jit, hooks, "pre", "post")
            try:
                for _ in range(2):
                    result = Jit().run(value, 33, grid=(1,))
                    self.assertIs(result.original_args[0], value)
                    self.assertEqual(result.original_args[1], 33)
                    self.assertEqual(result.original_kwargs, {"grid": (1,)})
                audit.mark_phase("inference")
                Jit().run(value, 33, grid=(1,))
            finally:
                audit.close()
            records = [json.loads(line) for line in audit.path.read_text().splitlines()]
        self.assertIs(Jit.run, original_run)
        self.assertIs(hooks.pre, old_pre)
        self.assertIs(hooks.post, old_post)
        self.assertEqual(len(calls), 6)
        self.assertEqual(sum(row["event"] == "shape" for row in records), 1)
        self.assertEqual(sum(row["event"] == "binary_seen" for row in records), 1)
        self.assertEqual(records[-1]["unique_shapes"], 1)
        self.assertEqual(sum(row["count"] for row in records[-1]["calls"]), 3)

    def test_existing_skip_return_and_kernel_exception_are_preserved(self):
        hooks = SimpleNamespace(pre=lambda **_: True)

        def kernel():
            pass

        kernel.__module__ = "vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5"

        class Jit:
            fn = staticmethod(kernel)
            arg_names = []

            def run(self):
                if hooks.pre(key="skipped"):
                    raise ValueError("original skip")

        original = Jit.run
        with tempfile.TemporaryDirectory() as directory:
            audit = TritonAudit(directory).install(Jit, hooks, "pre")
            try:
                with self.assertRaisesRegex(ValueError, "original skip"):
                    Jit().run()
            finally:
                audit.close()
        self.assertIs(Jit.run, original)

    def test_compiling_bypasses_all_observation_and_preserves_run_contract(self):
        compiler = ModuleType("torch.compiler")
        compiler.is_compiling = Mock(return_value=True)
        torch = ModuleType("torch")
        torch.compiler = compiler
        calls = []
        error = ValueError("original compile failure")

        class Result:
            @property
            def hash(self):
                raise AssertionError("Compiled results must not be inspected")

        result = Result()

        class Jit:
            @property
            def fn(self):
                raise AssertionError("Compiled kernel metadata must not be inspected")

            @property
            def arg_names(self):
                raise AssertionError("Compiled argument metadata must not be inspected")

            def run(self, value, *, fail=False):
                calls.append((value, fail))
                if fail:
                    raise error
                return result

        original = Jit.run
        value = object()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"torch": torch, "torch.compiler": compiler}),
        ):
            audit = TritonAudit(directory).install(Jit, SimpleNamespace(pre=None), "pre")
            try:
                with (
                    patch.object(audit, "relevant", side_effect=AssertionError("Observed compiled kernel")),
                    patch.object(audit, "describe", side_effect=AssertionError("Observed compiled input")),
                    patch.object(audit, "emit", side_effect=AssertionError("Wrote during compilation")),
                ):
                    self.assertIs(Jit().run(value), result)
                    with self.assertRaises(ValueError) as caught:
                        Jit().run(value, fail=True)
                    self.assertIs(caught.exception, error)
                self.assertFalse(audit.counts)
                self.assertFalse(audit.shapes)
                self.assertFalse(audit.binaries)
            finally:
                audit.close()
            records = [json.loads(line) for line in audit.path.read_text().splitlines()]
        self.assertIs(Jit.run, original)
        self.assertEqual(calls, [(value, False), (value, True)])
        self.assertEqual(compiler.is_compiling.call_count, 2)
        self.assertEqual([row["event"] for row in records], ["installed", "summary"])

    def test_symbolic_metadata_serializes_without_reading_or_changing_values(self):
        compiler = ModuleType("torch.compiler")
        compiler.is_compiling = Mock(return_value=False)
        torch = ModuleType("torch")
        torch.compiler = compiler

        class Symbol:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return self.name

            def __int__(self):
                raise AssertionError("Symbolic metadata must not be concretized")

        storage = bytearray(b"original input")

        class TensorMetadata:
            shape = (Symbol("s0"), 4)
            dtype = "float32"
            device = "cpu"

            def stride(self):
                return (Symbol("s1"), 1)

            def __getattribute__(self, name):
                if name in {"item", "tolist", "numpy", "cpu", "data", "_storage"}:
                    raise AssertionError("Tensor values must not be read")
                return object.__getattribute__(self, name)

        value = TensorMetadata()
        value._storage = storage
        shape = value.shape

        def kernel(raw):
            return raw

        kernel.__module__ = "vllm_ascend.ops.triton.msa.msa_m3_index_postprocess"

        class Jit:
            fn = staticmethod(kernel)
            arg_names = ["raw"]

            def run(self, raw):
                return raw

        original = Jit.run
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"torch": torch, "torch.compiler": compiler}),
        ):
            audit = TritonAudit(directory).install(Jit, SimpleNamespace(pre=None), "pre")
            try:
                self.assertIs(Jit().run(value), value)
                self.assertIs(Jit().run(value), value)
            finally:
                audit.close()
            records = [json.loads(line) for line in audit.path.read_text().splitlines()]
        self.assertIs(Jit.run, original)
        self.assertIs(value.shape, shape)
        self.assertIs(object.__getattribute__(value, "_storage"), storage)
        self.assertEqual(storage, bytearray(b"original input"))
        shapes = [row for row in records if row["event"] == "shape"]
        self.assertEqual(len(shapes), 1)
        self.assertEqual(
            shapes[0]["arguments"]["raw"],
            {"shape": ["s0", 4], "stride": ["s1", 1], "dtype": "float32", "device": "cpu"},
        )
        self.assertEqual(records[-1]["unique_shapes"], 1)
        self.assertEqual(sum(row["count"] for row in records[-1]["calls"]), 2)


if __name__ == "__main__":
    unittest.main()
