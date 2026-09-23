# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
"""CPU-only contract tests; no torch_npu, CANN runtime or device is loaded.

Run directly with unittest (or pytest --confcutdir=tests/ut/ops) to avoid
repository-wide device discovery. Compile the real adapters against CPU ATen,
replacing only ACLNN execution; this does not validate kernel execution.
"""

import ast
import regex as re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "csrc/attention/lightning_indexer"
FP32 = ROOT / "csrc/attention/lightning_indexer_fp32"


def source(path):
    assert path.is_file(), path
    return path.read_text()


def schema(name):
    binding = source(ROOT / "csrc/torch_binding.cpp")
    match = re.search(r'ops.def\(\s*("' + name + r"\(.*?)\s*\);", binding, re.S)
    assert match, name
    return "".join(ast.literal_eval(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', match[1]))


class TestLightningIndexerFp32Source(unittest.TestCase):
    def test_schema_preserves_defaults_and_keywords(self):
        old = schema("npu_lightning_indexer")
        new = schema("npu_lightning_indexer_fp32")
        self.assertEqual(old, new.replace("npu_lightning_indexer_fp32", "npu_lightning_indexer"))
        self.assertIn("*, Tensor?", old)
        self.assertIn("bool return_value=False", old)
        self.assertIn('str layout_query="BSND", str layout_key="BSND"', old)
        self.assertIn("int sparse_count=2048, int sparse_mode=3", old)

    def test_distinct_cann_and_aclnn_contract(self):
        definition = source(FP32 / "op_host/lightning_indexer_fp32_def.cpp")
        values = definition.split('Output("sparse_values")')[1].split("this->Attr")[0]
        self.assertIn("OP_ADD(LightningIndexerFp32)", definition)
        self.assertEqual(values.count("ge::DT_FLOAT"), 4)
        infer = source(FP32 / "op_host/lightning_indexer_fp32_infershape.cpp")
        self.assertIn("SetOutputDataType(0, ge::DT_INT32)", infer)
        self.assertIn("SetOutputDataType(1, ge::DT_FLOAT)", infer)
        api = source(FP32 / "op_host/op_api/aclnn_lightning_indexer_fp32.cpp")
        self.assertIn("TensorHolder(sparseValuesOut, ACL_FLOAT", api)
        self.assertIn("aclnnInnerLightningIndexerFp32GetWorkspaceSize", api)
        self.assertIn("aclnnInnerLightningIndexerFp32(workspace", api)

    def test_supported_target_and_packaging(self):
        definition = source(FP32 / "op_host/lightning_indexer_fp32_def.cpp")
        self.assertEqual(re.findall(r'AddConfig\("([^" ]+)"', definition), ["ascend910b"])
        cmake = source(FP32 / "op_host/CMakeLists.txt")
        self.assertIn('if (NOT "ascend910b" IN_LIST ASCEND_COMPUTE_UNIT)', cmake)
        self.assertIn("lightning_indexer_fp32_depends attention/lightning_indexer", cmake)
        build = source(ROOT / "csrc/build_aclnn.sh")
        self.assertEqual(build.count('"lightning_indexer_fp32"'), 1)
        a2 = build.split('elif [[ "$SOC_VERSION" =~ ^ascend910b ]]')[1].split("elif")[0]
        self.assertIn('"lightning_indexer_fp32"', a2)
        entry = source(FP32 / "op_kernel/lightning_indexer_fp32.cpp")
        self.assertIn("#if __CCE_AICORE__ != 220", entry)
        self.assertIn("#error", entry)
        self.assertNotIn("arch35/", entry)
        self.assertEqual(entry.count("DT_W_FLAG, float);"), 2)
        tiling = source(LEGACY / "op_host/lightning_indexer_tiling.cpp")
        self.assertIn("fp32Scores_ && socVersion_ != platform_ascendc::SocVersion::ASCEND910B", tiling)
        self.assertIn("valuesOutType_ != (fp32Scores_ ? ge::DT_FLOAT : inputQType_)", tiling)
        self.assertIn("LIInfoParser parser(context, true)", source(FP32 / "op_host/lightning_indexer_fp32_tiling.cpp"))

    def test_all_publication_paths_have_matching_width(self):
        common = source(LEGACY / "op_kernel/lightning_indexer_common.h")
        self.assertIn("typename SCORE_T = K_T", common)
        vector = source(LEGACY / "op_kernel/arch22/lightning_indexer_service_vector.h")
        kernel = source(LEGACY / "op_kernel/arch22/lightning_indexer_kernel.h")
        for text in [vector, kernel]:
            self.assertIn("GlobalTensor<SCORE_T> valueOutGm", text)
            self.assertIn("std::is_same<SCORE_T, float>::value, uint32_t, uint16_t", text)
            self.assertIn("negInf = 0xFF800000U", text)
            self.assertNotIn("GlobalTensor<uint16_t> valueOut", text)
        self.assertEqual(vector.count("if constexpr (!std::is_same<SCORE_T, float>::value)"), 2)
        self.assertEqual(vector.count("Cast(valueULocal1"), 2)
        self.assertIn("sizeof(SCORE_T)), 0, 0}", vector)
        self.assertIn(
            "dataCopyOutyParams.blockLen = copyCount * sizeof(T)",
            source(LEGACY / "op_kernel/arch22/lightning_indexer_vector.h"),
        )
        self.assertIn("(__gm__ SCORE_T *)sparseValues", kernel)

    def test_legacy_dtype_and_registration(self):
        adapter = source(LEGACY / "lightning_indexer_torch_adpt.h")
        self.assertEqual(adapter.count("query.options().dtype(query.dtype())"), 2)
        self.assertIn("EXEC_NPU_CMD(aclnnLightningIndexer,", adapter)
        self.assertIn(
            "SetOutputDataType(1, context->GetInputDataType(QUERY_INDEX))",
            source(LEGACY / "op_host/lightning_indexer_infershape.cpp"),
        )


class TestLightningIndexerFp32Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("CPU C++ compiler is unavailable")
        cls.build = tempfile.TemporaryDirectory(prefix=".li-fp32-test-", dir=ROOT)
        cls.addClassCleanup(cls.build.cleanup)
        build = Path(cls.build.name)
        cpp = build / "adapter.cpp"
        registrations = []
        for name in ["npu_lightning_indexer", "npu_lightning_indexer_fp32"]:
            registrations.append(f'ops.def(R"schema({schema(name)})schema");')
            registrations.append(f'ops.impl("{name}", torch::kCPU, &vllm_ascend::{name});')
        registrations.append(
            'ops.impl("npu_lightning_indexer_fp32", torch::kMeta, &vllm_ascend::npu_lightning_indexer_fp32_meta);'
        )
        cpp.write_text(
            "#include <ATen/ATen.h>\n#include <torch/library.h>\n#include <torch/types.h>\n"
            "#define EXEC_NPU_CMD(...) do {} while (0)\n"
            '#include "csrc/attention/lightning_indexer/lightning_indexer_torch_adpt.h"\n'
            '#include "csrc/attention/lightning_indexer_fp32/lightning_indexer_fp32_torch_adpt.h"\n'
            "TORCH_LIBRARY(li_fp32_host_test, ops) {\n" + "\n".join(registrations) + "\n}\n"
        )
        lib = build / "adapter.so"
        command = [
            compiler,
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-O0",
            "-g0",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            f"-I{ROOT}",
            *[f"-I{p}" for p in include_paths()],
            str(cpp),
            *[f"-L{p}" for p in library_paths()],
            "-ltorch_cpu",
            "-ltorch",
            "-lc10",
            "-o",
            str(lib),
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        torch.ops.load_library(str(lib))

    def check_outputs(self, device, return_value, dtype, layout, check_legacy=True):
        if layout == "BSND":
            qshape, kshape, expected = (2, 3, 4, 128), (2, 9, 1, 128), (2, 3, 1, 2048)
            key_layout = "BSND"
        elif layout == "TND":
            qshape, kshape, expected = (3, 4, 128), (9, 1, 128), (3, 1, 2048)
            key_layout = "TND"
        else:
            qshape, kshape, expected = (3, 4, 128), (2, 64, 1, 128), (3, 1, 2048)
            layout, key_layout = "TND", "PA_BSND"
        query = torch.empty(qshape, dtype=dtype, device=device)
        key = torch.empty(kshape, dtype=dtype, device=device)
        weights = torch.empty(qshape[:-1], dtype=dtype, device=device)
        indices, values = torch.ops.li_fp32_host_test.npu_lightning_indexer_fp32(
            query, key, weights, layout_query=layout, layout_key=key_layout, return_value=return_value
        )
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(tuple(indices.shape), expected)
        self.assertEqual(tuple(values.shape), expected if return_value else (0,))
        self.assertEqual(values.device, query.device)
        if device == "cpu" and check_legacy:
            _, legacy = torch.ops.li_fp32_host_test.npu_lightning_indexer(
                query, key, weights, layout_query=layout, layout_key=key_layout, return_value=return_value
            )
            self.assertEqual(legacy.dtype, dtype)
            self.assertEqual(legacy.shape, values.shape)

    def test_cpu_allocation_and_legacy(self):
        for dtype in [torch.float16, torch.bfloat16]:
            for layout in ["BSND", "TND", "PA_BSND"]:
                for enabled in [False, True]:
                    with self.subTest(dtype=dtype, layout=layout, enabled=enabled):
                        self.check_outputs("cpu", enabled, dtype, layout)

    def test_meta(self):
        for dtype in [torch.float16, torch.bfloat16]:
            for layout in ["BSND", "TND", "PA_BSND"]:
                for enabled in [False, True]:
                    with self.subTest(dtype=dtype, layout=layout, enabled=enabled):
                        self.check_outputs("meta", enabled, dtype, layout)

    def test_fake(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            self.check_outputs("cpu", True, torch.bfloat16, "TND", check_legacy=False)

    def test_default_return_and_keyword_only(self):
        query = torch.empty((1, 2, 1, 128), dtype=torch.bfloat16, device="meta")
        indices, values = torch.ops.li_fp32_host_test.npu_lightning_indexer_fp32(query, query, query)
        self.assertEqual(tuple(indices.shape), (1, 2, 1, 2048))
        self.assertEqual(tuple(values.shape), (0,))
        self.assertEqual(values.dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            torch.ops.li_fp32_host_test.npu_lightning_indexer_fp32(query, query, query, None)

    def test_invalid_sparse_count(self):
        query = torch.empty((2, 1, 128), dtype=torch.bfloat16, device="meta")
        with self.assertRaisesRegex(RuntimeError, "sparse count should be greater than 0"):
            torch.ops.li_fp32_host_test.npu_lightning_indexer_fp32(
                query, query, query, layout_query="TND", layout_key="TND", sparse_count=0
            )


if __name__ == "__main__":
    unittest.main()
