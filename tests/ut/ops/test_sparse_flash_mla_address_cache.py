# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile actual host guard and kernel dispatch with CPU stubs, without CANN.

These checks do not compile AscendC vector intrinsics or replace NPU tests.
Run directly with unittest to avoid importing torch-backed test fixtures.
"""

import itertools
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP = ROOT / "csrc/attention/sparse_flash_mla"


class SparseFlashMlaAddressCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("A C++17 compiler is required")

    def compile(self, source, *defines):
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "test.cpp"
            entry.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [self.compiler, "-std=c++17", "-fsyntax-only", *defines, "-I", str(OP / "op_host"), str(entry)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_whole_pool_element_offset_bounds(self):
        # Hand-calculated boundaries from PR 15596, plus adversarial overflow
        # inputs. Compile static_asserts against the extracted production guard.
        host = (OP / "op_host/sparse_flash_mla_tiling.cpp").read_text(encoding="utf-8")
        start = host.index("constexpr uint64_t addressLimit =")
        end_marker = "vectorizeFlag = static_cast<uint32_t>(addressFits);"
        end = host.index(end_marker, start) + len(end_marker)
        guard = host[start:end]
        # Compile the production guard verbatim, with only its input struct stubbed.
        self.compile(
            r"""
#include <cstdint>
#include <climits>
struct Info {
    uint64_t n2Size, cmpKvHeadDim, cmpKvStride0, cmpKvPhysicalBlockCount;
};
constexpr bool CsaV0AddressCacheFitsInt32(int32_t blockSize, uint64_t kvHeads,
                                       uint64_t headDim, uint64_t stride, uint64_t blocks) {
    const int32_t cmpBlockSize = blockSize;
    const Info inputs{kvHeads, headDim, stride, blocks};
    const Info* tilingInfo = &inputs;
    uint32_t vectorizeFlag = 1;
"""
            + guard
            + r"""
    return vectorizeFlag != 0;
}
static_assert(CsaV0AddressCacheFitsInt32(64, 1, 512, 32768, 65536));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 32768, 65537));
static_assert(CsaV0AddressCacheFitsInt32(128, 1, 512, 65536, 32768));
static_assert(!CsaV0AddressCacheFitsInt32(128, 1, 512, 65536, 32769));
static_assert(CsaV0AddressCacheFitsInt32(64, 1, 512, 65536, 32768));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 65536, 32769));
static_assert(CsaV0AddressCacheFitsInt32(64, 1, 512, 32769, 65534));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 32769, 65535));
static_assert(CsaV0AddressCacheFitsInt32(1, 1, 1, 1, 2147483648ULL));
static_assert(!CsaV0AddressCacheFitsInt32(1, 1, 1, 1, 2147483649ULL));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 0, 1));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 32768, 0));
static_assert(!CsaV0AddressCacheFitsInt32(64, 0, 512, 32768, 1));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 0, 32768, 1));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 2147483648ULL, 1));
static_assert(!CsaV0AddressCacheFitsInt32(64, 1, 512, 32768, UINT64_MAX));
"""
        )

    def test_a2a3_entry_propagates_cache_flag_for_both_dtypes(self):
        source = (OP / "op_kernel/sparse_flash_mla.cpp").read_text(encoding="utf-8")
        source = re.sub(r"^#include .*$", "", source, flags=re.MULTILINE)
        common = (OP / "op_kernel/arch22/sparse_flash_mla_common_arch22.h").read_text(encoding="utf-8")
        # Use the real SMLAType declaration; the stubs implement only Init/Process.
        start = common.index("template <typename Q_T")
        smlatype = common[start : common.index("};", start) + 2]
        stub = r"""
#include <cstdint>
#define __global__
#define __aicore__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define GET_TILING_DATA_WITH_STRUCT(type, name, ...) type name;
#define DT_FLOAT16 1
#define DT_BF16 2
#define CSA_TEMPLATE 2
#define SWA_TEMPLATE 0
using half = short;
struct bfloat16_t {};
namespace AscendC {
struct TPipe {};
inline uint8_t* GetUserWorkspace(uint8_t* p) { return p; }
}
namespace optiling { namespace detail { struct SparseFlashMlaTilingData {}; } }
namespace SMLAKernel {
enum class SMLA_LAYOUT { BSND=0, TND=1, PA_BBND=2 };
"""
        stub += (
            smlatype
            + r"""
// Omission of the entry-point cache flag must fail this assertion.
template<class T> struct SparseFlashMlaCsa {
    static_assert(T::isVecS2PhyAddr == EXPECTED_CACHE);
    static_assert(T::headRatioOne == EXPECTED_HEAD);
    void Init(...) {} void Process() {}
};
template<class T> struct SparseFlashMlaSwa {
    static_assert(!T::isVecS2PhyAddr);
    void Init(...) {} void Process() {}
};
}
"""
        )
        for dtype, mode, head, cache in itertools.product((1, 2), (0, 2), (0, 1), (0, 1)):
            if mode == 0 and (head or cache):
                continue
            with self.subTest(dtype=dtype, mode=mode, head=head, cache=cache):
                key = f"0, 1, 2, {mode}, 0, {head}, 0, {cache}"
                invocation = f"\nvoid instantiate() {{ sparse_flash_mla<{key}>({', '.join(['nullptr'] * 22)}); }}\n"
                self.compile(
                    stub + source + invocation,
                    "-D__CCE_AICORE__=220",
                    f"-DORIG_DTYPE_Q={dtype}",
                    f"-DORIG_DTYPE_ORI_KV={dtype}",
                    f"-DORIG_DTYPE_ATTN_OUT={dtype}",
                    f"-DEXPECTED_CACHE={cache}",
                    f"-DEXPECTED_HEAD={head}",
                )


if __name__ == "__main__":
    unittest.main()
