# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; native QLI and Triton execution require NPU regression."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_a3_qli_local_candidates import Tensor

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class TestDcpContract(unittest.TestCase):
    def test_global_bounds_local_lengths_and_shard_parameters_are_distinct(self):
        calls = []
        indices, scores = object(), object()
        native = lambda *a, **kw: calls.append((a, kw)) or (indices, scores, None)
        fake_torch = SimpleNamespace(
            Tensor=Tensor,
            int8="int8",
            float16="float16",
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_quant_lightning_indexer_dcp=native)),
        )
        namespace = dict(
            torch=fake_torch, uses_native_int8_sparse_adaptor=lambda _: True, get_current_hardware_profile=lambda: None
        )
        call = load_function(
            "vllm_ascend/device/device_op.py", "indexer_select_dcp_candidates", namespace, "BaseDeviceAdaptor"
        )
        q = Tensor((5, 64, 128), "int8")
        k = Tensor((7, 768, 1, 128), "int8", stride0=100000)
        scale = Tensor((7, 768, 1, 1), "float16")
        starts = Tensor((3,), "int32", data=[0, 2, 5])
        local = Tensor((2,), "int32", data=[0, 768])
        global_lengths = Tensor((2,), "int32", data=[1, 20000])
        normalize = SimpleNamespace(normalize_dcp_candidates=lambda i, s, loc: (i, s))
        with patch.dict(sys.modules, {"vllm_ascend.ops.triton.sharded_indexer": normalize}):
            result = call(
                q,
                Tensor((5, 64), "bfloat16"),
                k,
                scale,
                Tensor((2, 7), "int32"),
                starts,
                local,
                global_lengths,
                Tensor((5, 64, 1), "float16"),
                cp_world=16,
                cp_rank=7,
                cp_interleave=768,
                qli_metadata=object(),
            )
        self.assertEqual(result, (indices, scores))
        args, kw = calls[0]
        self.assertIs(args[1], k)
        self.assertEqual(args[2].dtype, "float16")
        self.assertIs(kw["seqused_k"], local)
        self.assertIs(kw["causal_seqlens_kv_global"], global_lengths)
        for key, value in dict(
            mask_mode=3,
            enable_dcp=True,
            cp_world=16,
            cp_rank=7,
            cp_kv_cache_interleave_size=768,
            return_value=1,
            candidate_mode=3,
        ).items():
            self.assertEqual(kw[key], value)

    def test_normalization_reuses_fp32_storage_and_rejects_old_bridge(self):
        launches = []

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args: launches.append((grid, args))

        namespace = dict(
            torch=SimpleNamespace(float32="float32"),
            triton=SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d),
            _normalize_dcp_candidates=Kernel(),
        )
        normalize = load_function("vllm_ascend/ops/triton/sharded_indexer.py", "normalize_dcp_candidates", namespace)
        indices = Tensor((5, 1, 2048), "int32")
        scores = Tensor((5, 1, 2048), "float32")
        starts = SimpleNamespace(numel=lambda: 3)
        result = normalize(indices, scores, starts)
        self.assertIs(result[0], indices)
        self.assertIs(result[1], scores)
        self.assertEqual(launches[0][0], (5, 8))
        self.assertIs(launches[0][1][1], scores)
        with self.assertRaisesRegex(TypeError, "native FP32"):
            normalize(indices, Tensor((5, 1, 2048), "bfloat16"), starts)
        normalize(Tensor((0, 1, 2048), "int32"), Tensor((0, 1, 2048), "float32"), starts)
        self.assertEqual(len(launches), 1)


if __name__ == "__main__":
    unittest.main()
