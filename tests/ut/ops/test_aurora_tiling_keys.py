# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""Check the compiled template matrix without requiring CANN or an NPU.

The C++ preprocessor selects the real architecture branch and expands the
real headers. The stub only exposes macro arguments for enumeration; this is
not a CANN compilation or a numerical operator test.
"""

import ast
import itertools
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TILING_HEADERS = (
    REPO_ROOT / "csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla_template_tiling_key.h",
    REPO_ROOT / "csrc/attention/quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_template_tiling_key.h",
    REPO_ROOT / "csrc/attention/compressor/op_kernel/arch32/compressor_template_tiling_key.h",
    REPO_ROOT / "csrc/attention/compressor/op_kernel/arch35/compressor_template_tiling_key.h",
)
TEMPLATE_ARGUMENT_STUB = """
#define ASCENDC_TPL_ARGS_DECL(op, ...) declaration = [__VA_ARGS__]
#define ASCENDC_TPL_BOOL_DECL(name, ...) [#name, 1, [__VA_ARGS__]]
#define ASCENDC_TPL_DTYPE_DECL(name, ...) [#name, "dtype", [__VA_ARGS__]]
#define ASCENDC_TPL_UINT_DECL(name, bits, kind, ...) [#name, bits, [__VA_ARGS__]]
#define ASCENDC_TPL_SEL(...) selection = [__VA_ARGS__]
#define ASCENDC_TPL_ARGS_SEL(...) [__VA_ARGS__]
#define ASCENDC_TPL_BOOL_SEL(name, ...) [__VA_ARGS__]
#define ASCENDC_TPL_DTYPE_SEL(name, ...) [__VA_ARGS__]
#define ASCENDC_TPL_UINT_SEL(name, kind, ...) [__VA_ARGS__]
#define ASCENDC_TPL_TILING_STRUCT_SEL(...)
"""

COMPRESSOR_KERNEL_STUB = """
#pragma once
#include <cstdint>
#define __global__
#define __aicore__
#define __gm__
#define REGISTER_TILING_DEFAULT(...)
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define GET_TILING_DATA_WITH_STRUCT(type, name, ...) type name;
namespace optiling { struct CompressorTilingData {}; }
namespace Compressor {
struct TPipe {};
enum class X_LAYOUT : uint8_t { BSH = 0, TH = 1 };
enum class X_DTYPE : uint8_t { BF16 = 0, FP16 = 1 };
enum class COFF : uint8_t { DISABLE = 1, OVERLAP = 2 };
enum class ROTARY_MODE : uint8_t { HALF = 1, INTERLEAVE = 2 };
enum class CACHE_MODE : uint8_t { CONTINUOUS = 1, CYCLE = 2 };
enum class ROPE_DTYPE : uint8_t { SAME_AS_X = 0, FP32 = 1 };
enum class TEMPLATE_ID : uint8_t {
    NORMAL = 0, EMPTY_X = 1, FULL_LOAD = 2
};
template<auto... Args> struct COMPType {};
template<class T, int Tag> struct Kernel {
    Kernel(...) {
        static_assert(Tag == EXPECTED_KERNEL, "Unexpected computation template instantiated");
    }
    void Init(...) {}
    void Process() {}
};
template<class T> using CompressorKernel = Kernel<T, 0>;
#if __CCE_AICORE__ == 220
template<class T> using CompressorKernelPerf = Kernel<T, 1>;
#endif
template<class T> using CompressorKernelFullLoad = Kernel<T, 2>;
}
"""


class AuroraTilingKeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("A C++ preprocessor is required to inspect template selections")
        cls.matrices = {}
        cls.declarations = {}
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / "ascendc/host_api/tiling/template_argument.h"
            stub.parent.mkdir(parents=True)
            stub.write_text(TEMPLATE_ARGUMENT_STUB)
            for header, arch in itertools.product(TILING_HEADERS, (None, 220, 310)):
                compressor_arch = {"arch32": 220, "arch35": 310}.get(header.parent.name)
                if compressor_arch is not None and arch != compressor_arch:
                    continue
                args = [compiler, "-E", "-P", "-x", "c++", "-I", directory]
                if arch is not None:
                    args.append(f"-D__CCE_AICORE__={arch}")
                output = subprocess.check_output([*args, str(header)], text=True, timeout=30)
                assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(output).body}
                op = "compressor" if compressor_arch is not None else header.parents[1].name
                cls.declarations[op, arch] = assignments["declaration"]
                cls.matrices[op, arch] = [
                    key for selection in assignments["selection"] for key in itertools.product(*selection)
                ]

    def test_compressor_retains_model_dispatch_and_key_encoding(self):
        declaration = [
            ["X_LAYOUT", 1, [0, 1]],
            ["X_DTYPE", 4, [0, 1]],
            ["COFF", 2, [1, 2]],
            ["ROTARY_MODE", 2, [1, 2]],
            ["CACHE_MODE", 2, [1, 2]],
            ["TEMPLATE_ID", 2, [0, 1, 2]],
        ]
        for arch in (220, 310):
            # V4 ratio 4 overlaps (coff=2), ratio 128 does not (coff=1).
            # TH selects PERF on A2/A3 and NORMAL on A5; EMPTY_X is required
            # on both. FULL_LOAD requires BSH and is unreachable in the model.
            template_ids = (0, 1)
            expected = {
                (1, 0, coff, 2, 1, template_id) for coff, template_id in itertools.product((1, 2), template_ids)
            }
            with self.subTest(arch=arch):
                self.assertEqual(set(self.matrices["compressor", arch]), expected)
                self.assertEqual(len(self.matrices["compressor", arch]), 4)
                self.assertEqual(
                    self.declarations["compressor", arch],
                    declaration,
                )

    def test_compressor_dtype_registration_matches_selected_templates(self):
        root = REPO_ROOT / "csrc/attention/compressor/op_host"
        source = (root / "compressor_def.cpp").read_text()
        for arch in ("arch32", "arch35"):
            with self.subTest(arch=arch):
                rows = []
                for line in source.splitlines():
                    if 'Input("' in line or 'Output("' in line:
                        name = line.split('"')[1]
                    elif ".DataType" in line:
                        rows.append((name, line.split("{", 1)[1].split("}", 1)[0]))
                self.assertEqual(len(rows), 14)
                expected = {
                    "x": "ge::DT_BF16, ge::DT_FLOAT16",
                    "wkv": "ge::DT_BF16, ge::DT_FLOAT16",
                    "wgate": "ge::DT_BF16, ge::DT_FLOAT16",
                    "cmp_kv": "ge::DT_BF16, ge::DT_FLOAT16",
                    "state_cache": "ge::DT_FLOAT",
                    "ape": "ge::DT_FLOAT",
                    "norm_weight": "ge::DT_FLOAT",
                    "rope_sin": "ge::DT_FLOAT",
                    "rope_cos": "ge::DT_FLOAT",
                    "state_block_table": "ge::DT_INT32",
                    "cu_seqlens": "ge::DT_INT32",
                    "seqused": "ge::DT_INT32",
                    "start_pos": "ge::DT_INT32",
                }
                for name, dtypes in rows:
                    self.assertEqual(dtypes.strip(), expected[name])
                host = (root / arch / "compressor_tiling.h").read_text()
                host_expected = expected.copy()
                if arch == "arch35":
                    for name in ("x", "wkv", "wgate", "cmp_kv"):
                        host_expected[name] = "ge::DT_BF16"
                for name, dtype in host_expected.items():
                    self.assertRegex(host, rf"\{{{name.upper()}_NAME,\s*\{{{dtype}\}}\}}")
                self.assertRegex(
                    host,
                    r"\{X_NAME,\s*\{COMPRESSOR_DIM_NUM_2"
                    + (r",\s*COMPRESSOR_DIM_NUM_3" if arch == "arch32" else "")
                    + r"\}\}",
                )
                expected_modes = {
                    "ROTARY_MODE": "1, 2" if arch == "arch32" else "2",
                    "CACHE_MODE": "1",
                }
                for mode, values in expected_modes.items():
                    definitions = [
                        line for line in host.splitlines() if line.startswith(f"const std::vector<int> {mode} ")
                    ]
                    self.assertGreaterEqual(len(definitions), 1)
                    # arch32 carries a DAY0 branch first; the final definition
                    # is the normal production branch selected without DAY0_SCOPE.
                    self.assertRegex(definitions[-1], rf"\{{\s*{values}\s*\}}")

    def test_compressor_empty_input_does_not_instantiate_computation(self):
        compiler = shutil.which("c++")
        source = (REPO_ROOT / "csrc/attention/compressor/op_kernel/compressor.cpp").read_text()
        for arch in (220, 310):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                arch_dir = root / ("arch32" if arch == 220 else "arch35")
                arch_dir.mkdir()
                (arch_dir / "stub.h").write_text(COMPRESSOR_KERNEL_STUB)
                # Only the target architecture's headers exist: crossing the
                # architecture branch fails compilation instead of going unnoticed.
                for name in (
                    "compressor_kernel.h",
                    "compressor_kernel_perf.h",
                    "compressor_kernel_full_load.h",
                ):
                    (arch_dir / name).write_text('#include "stub.h"\n')
                for key in self.matrices["compressor", arch]:
                    with self.subTest(arch=arch, key=key):
                        expected_kernel = -1 if key[5] == 1 else int(arch == 220)
                        invocation = ", ".join(map(str, key))
                        arguments = ", ".join(["nullptr"] * 16)
                        (root / "entry.cpp").write_text(
                            source + f"\nvoid instantiate() {{ compressor<{invocation}>({arguments}); }}\n"
                        )
                        result = subprocess.run(
                            [
                                compiler,
                                "-std=c++17",
                                "-fsyntax-only",
                                f"-D__CCE_AICORE__={arch}",
                                f"-DEXPECTED_KERNEL={expected_kernel}",
                                str(root / "entry.cpp"),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sparse_mla_covers_model_dispatch_on_each_architecture(self):
        # DoOpTiling derives flags from hardware, template mode and local head
        # count. A5 CSA can enable address vectorization depending on the KV
        # block shape and UB capacity; both paths must stay compiled.
        for arch in (220, 310):
            expected = set()
            for ratio, heads, batch_consistency in itertools.product((0, 1, 2), (1, 2, 4, 8, 16, 32, 64, 128), (0, 1)):
                mode = 0 if ratio == 0 else 2
                split_g = int(arch == 310 and heads > 64)
                head_ratio_one = int(arch == 220 and mode == 2 and heads == 1)
                vectorize_flags = (0, 1) if arch == 310 and mode == 2 else (0,)
                for vectorize in vectorize_flags:
                    expected.add((0, 1, 2, mode, split_g, head_ratio_one, batch_consistency, vectorize))
            with self.subTest(arch=arch):
                selected = self.matrices["sparse_flash_mla", arch]
                self.assertEqual(set(selected), expected)
                self.assertEqual(len(selected), 6 if arch == 220 else 12)
                self.assertEqual(len(selected), len(set(selected)))

    def test_sparse_mla_architectures_exclude_each_others_specializations(self):
        a2a3 = set(self.matrices["sparse_flash_mla", 220])
        a5 = set(self.matrices["sparse_flash_mla", 310])
        self.assertTrue(all(key[4] == 0 and key[7] == 0 for key in a2a3))
        self.assertTrue(all(key[5] == 0 for key in a5))
        self.assertEqual(len(a2a3 & a5), 4)
        host = self.matrices["sparse_flash_mla", None]
        self.assertEqual(set(host), a2a3 | a5)
        self.assertEqual(len(host), 14)

    def test_sparse_mla_scope_and_a2a3_qli_quantization(self):
        for arch in (None, 220, 310):
            with self.subTest(arch=arch):
                for key in self.matrices["sparse_flash_mla", arch]:
                    self.assertEqual(key[1:3], (1, 2))  # TND / PA_BBND
                    self.assertIn(key[3], (0, 2))  # SWA / CSA
                if arch != 310:
                    # A2/A3: INT8 Q/K, INT32 output, paged attention, TND / PA_BBND.
                    self.assertEqual(
                        self.matrices["quant_lightning_indexer_v2", arch],
                        [
                            (2, 2, 3, 1, 0, 2),
                            (2, 2, 3, 1, 1, 2),
                            (2, 2, 3, 0, 0, 0),
                            (2, 2, 3, 0, 1, 1),
                        ],
                    )

    def test_qli_a5_retains_full_dtype_and_layout_matrix(self):
        # A5's general QLI supports FP8/MXFP8, HiFloat8, MXFP4 and INT8.
        # FP8 and MXFP8 share a dtype key and dispatch by runtime quant_mode.
        expected = {
            (dtype, dtype, 3, paged, q_layout, k_layout)
            for dtype, (paged, q_layout, k_layout) in itertools.product(
                (36, 34, 40, 2), ((1, 0, 2), (1, 1, 2), (0, 0, 0), (0, 1, 1))
            )
        }
        selected = self.matrices["quant_lightning_indexer_v2", 310]
        self.assertEqual(set(selected), expected)
        self.assertEqual(len(selected), 16)

    def test_key_argument_order_widths_and_values_stay_unchanged(self):
        # Keep the declaration, including unused values: changing its encoding
        # can change the numeric keys shared by host tiling and kernel lookup.
        expected = [
            ["FLASH_DECODE", 1, [0, 1]],
            ["LAYOUT_T", 4, [0, 1]],
            ["KV_LAYOUT_T", 4, [0, 1, 2]],
            ["TEMPLATE_MODE", 4, [0, 1, 2, 3, 4]],
            ["SPLIT_G", 1, [0, 1]],
            ["HEAD_RATIO_ONE", 1, [0, 1]],
            ["BATCH_CONSISTENCY", 1, [0, 1]],
            ["IS_VEC_S2PHYADDR", 1, [0, 1]],
        ]
        for arch in (None, 220, 310):
            with self.subTest(arch=arch):
                self.assertEqual(self.declarations["sparse_flash_mla", arch], expected)
                qk_types = [36, 34, 40, 2] if arch == 310 else [2]
                self.assertEqual(
                    self.declarations["quant_lightning_indexer_v2", arch],
                    [
                        ["DT_Q", "dtype", qk_types],
                        ["DT_K", "dtype", qk_types],
                        ["DT_OUT", "dtype", [3]],
                        ["PAGE_ATTENTION", 1, [1, 0]],
                        ["Q_LAYOUT_T", 4, [0, 1]],
                        ["K_LAYOUT_T", 4, [0, 1, 2]],
                    ],
                )


if __name__ == "__main__":
    unittest.main()
