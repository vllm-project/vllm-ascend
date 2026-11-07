# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile and exercise the real host planner without CANN or an NPU.

These checks establish task coverage, H ownership, workspace bounds and fallback
selection. A synchronous CPU API shim also executes production kernel control
flow against an independent reference; it does not validate NPU pipes or APIs.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AttnResParallelPlanTest(unittest.TestCase):
    def test_legacy_ub_budget_and_resident_boundary(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required to exercise the legacy UB planner")
        root = Path(__file__).resolve().parents[3]
        source = r"""
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_legacy_plan.h"
using namespace AttnResFwd;
int main() {
    LegacyUbConfig config{true, true, false, false, false};
    assert(LegacyUbBytes(7168, 2, config) == 176704);
    assert(LegacyMaxResidentRows(7168, 192*1024, config) == 3);
    assert(LegacyResidentRowLimit(7168, 192*1024, config, true) == 3);
    config.regBase = true;
    assert(LegacyUbBytes(7168, 6, config) == 233824);
    assert(LegacyMaxResidentRows(7168, 248*1024, config) == 7);
    assert(LegacyMaxResidentRows(7168, 256*1024, config) == 7);
    assert(LegacyResidentRowLimit(7168, 248*1024, config, true) == 4);
    assert(LegacyResidentRowLimit(7168, 256*1024, config, true) == 4);
    assert(LegacyMaxResidentRows(0, 256*1024, config) == 0);
    assert(LegacyMaxResidentRows(7168, LEGACY_UB_RESERVE_BYTES, config) == 0);
    for (uint32_t hidden : {1U, 15U, 16U, 17U, 1537U, 7168U, 8193U, 32769U})
    for (uint32_t rows : {1U, 2U, 5U, 6U, 7U, 9U, 65U, 128U, 129U, 2049U})
    for (bool fused : {false, true})
    for (bool backward : {false, true})
    for (bool reg : {false, true})
    for (bool cached : {false, true}) {
        LegacyUbConfig c{true, fused, backward, reg, cached};
        const auto bytes = LegacyUbBytes(hidden, rows, c);
        assert(LegacyMaxResidentRows(hidden, bytes, c) == rows);
        const auto admitted = LegacyResidentRowLimit(hidden, bytes, c, fused);
        assert(admitted <= rows);
        if (admitted) assert(LegacyUbBytes(hidden, admitted, c) <= bytes);
        assert(LegacyMaxResidentRows(hidden, bytes-1, c) == rows-1);
        c.cacheOutputNorm = !cached;
        const uint64_t rowBytes = ((uint64_t(hidden)+15)/16)*32;
        const auto changed = LegacyUbBytes(hidden, rows, c);
        assert(cached ? bytes-changed == rowBytes : changed-bytes == rowBytes);
        c.resident = false;
        assert(LegacyUbBytes(hidden, rows, c) >= LEGACY_UB_RESERVE_BYTES);
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "legacy_ub.cpp"
            binary = Path(directory) / "legacy_ub.exe"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root), str(cpp), "-o", str(binary)],
                check=True,
                timeout=180,
            )
            subprocess.run([str(binary)], check=True, timeout=60)

    def test_production_kernel_with_cpu_api_shim(self):
        self._run_production_kernel_with_cpu_api_shim(False)

    def test_a5_kernel_control_flow_with_cpu_api_shim(self):
        self._run_production_kernel_with_cpu_api_shim(True)

    def _run_production_kernel_with_cpu_api_shim(self, regbase):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required to exercise the kernel with CPU API stubs")
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "kernel_tiling").mkdir()
            (temporary / "kernel_tiling/kernel_tiling.h").write_text("#pragma once\n", encoding="utf-8")
            (temporary / "kernel_operator.h").write_text(
                '#include "tests/ut/ops/attn_res_fwd_cpu_stubs.h"\n', encoding="utf-8"
            )
            binary = temporary / "cooperative_kernel.exe"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-pthread",
                    "-ffp-contract=off",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    *(["-DATTN_RES_TEST_REGBASE=1"] if regbase else []),
                    "-I",
                    str(root),
                    "-I",
                    str(temporary),
                    str(root / "tests/ut/ops/attn_res_fwd_cpu_kernel.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                timeout=180,
            )
            subprocess.run([str(binary)], check=True, timeout=90)

    def test_native_workspace_api_rejects_malformed_inputs(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required to exercise native API validation")
        root = Path(__file__).resolve().parents[3]
        # Run the production ACLNN wrapper with metadata-only stubs. Rejection
        # must happen before the mocked backend launcher is reached.
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            headers = [
                "aclnn/aclnn_base.h",
                "aclnn_util.h",
                "opdev/make_op_executor.h",
                "opdev/op_def.h",
                "opdev/op_dfx.h",
                "opdev/op_executor.h",
                "opdev/tensor_view_utils.h",
            ]
            for name in headers:
                header = temporary / name
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text('#include "tests/ut/ops/attn_res_fwd_native_stubs.h"\n', encoding="utf-8")
            binary = temporary / "native_validation.exe"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Wno-unused-parameter",
                    "-Werror",
                    "-I",
                    str(root),
                    "-I",
                    str(temporary),
                    str(root / "tests/ut/ops/attn_res_fwd_native_validation.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                timeout=180,
            )
            subprocess.run([str(binary)], check=True, timeout=60)

    def test_reject_unrepresentable_metadata_and_epsilon(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required to exercise metadata validation")
        root = Path(__file__).resolve().parents[3]
        # These formerly wrapped in Host before the UB check; no huge allocation
        # is needed to reproduce the unsafe narrowing/alignment arithmetic.
        source = r"""
#include <cassert>
#include <limits>
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_parallel_plan.h"
#include "csrc/attention/attn_res_fwd/attn_res_fwd_validation.h"
using namespace AttnResFwd;
int main() {
    constexpr int64_t u32max = std::numeric_limits<uint32_t>::max();
    assert(AreShapeFieldsRepresentable(1, 9, 7168));
    assert(!AreShapeFieldsRepresentable(-1, 9, 7168));
    assert(!AreShapeFieldsRepresentable(1, -1, 7168));
    assert(!AreShapeFieldsRepresentable(1, 9, -1));
    assert(!AreShapeFieldsRepresentable(u32max + 1, 9, 7168));
    assert(!AreShapeFieldsRepresentable(1, u32max + 1, 7168));
    assert(!AreShapeFieldsRepresentable(1, 1, u32max));
    const int64_t overflowingCapacity = std::numeric_limits<int64_t>::max() / 7168 + 1;
    assert(!IsValidBankExtent(0, overflowingCapacity, 7168, 0, 0, -1));
    assert(IsValidBankExtent(0, 2, 512, 2, 1024, -1));
    assert(AreCandidateRowsRepresentable(0));
    assert(!AreCandidateRowsRepresentable(u32max));
    assert(IsValidBankExtent(1, 9, 7168, 8, 9 * 7168, 8));
    assert(IsValidBankExtent(4, 9, 7168, 8, 11 * 7168, -1));
    assert(!IsValidBankExtent(1, 9, 7168, 8, 9 * 7168 - 1, 8));
    assert(!IsValidBankExtent(1, 9, 7168, 8, 9 * 7168, 7));
    assert(!IsValidBankExtent(1, 9, 7168, 8, -1, -1));
    assert(!IsValidBankExtent(1, u32max, 1, u32max, u32max, -1));
    assert(!IsValidBankExtent(1, u32max, 1, 8, u32max, int64_t{1} << 31));
    assert(!IsValidBankExtent(u32max, 8, 1, 8, std::numeric_limits<int64_t>::max(), -1));
    assert(IsValidNormEpsilon(1e-5));
    assert(IsValidNormEpsilon(0, true));
    assert(!IsValidNormEpsilon(0));
    assert(!IsValidNormEpsilon(-1));
    assert(!IsValidNormEpsilon(std::numeric_limits<double>::quiet_NaN()));
    assert(!IsValidNormEpsilon(std::numeric_limits<double>::infinity()));
    assert(!IsValidNormEpsilon(1e300));
    assert(!IsValidNormEpsilon(std::numeric_limits<double>::denorm_min()));
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "metadata_validation.cpp"
            binary = Path(directory) / "metadata_validation.exe"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root), str(cpp), "-o", str(binary)],
                check=True,
                timeout=180,
            )
            subprocess.run([str(binary)], check=True, timeout=60)

    def test_host_plans_cover_rows_and_output_without_overlap(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++ compiler is required to exercise the host planner")
        root = Path(__file__).resolve().parents[3]
        source = r"""
#include <cassert>
#include <cstdint>
#include <vector>
#include "csrc/attention/attn_res_fwd/op_host/attn_res_fwd_parallel_plan.h"
using namespace AttnResFwd;
int main() {
    for (uint32_t cores : {1U, 4U, 8U, 40U, 48U, 56U, 64U})
    for (uint32_t tokens : {0U, 1U, 4U, 8U, 10U, 11U, 12U, 16U, 20U, 21U, 24U, 28U, 29U, 32U, 40U, 128U})
    for (uint32_t rows : {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 17U, 65U, 128U, 129U})
    for (uint32_t hidden : {0U, 65U, 511U, 512U, 1537U, 7168U, 32768U})
    for (uint64_t ub : {0ULL, 192ULL * 1024, 248ULL * 1024})
    for (bool regBase : {false, true})
    for (bool norm : {false, true}) {
        auto unsupported = MakeParallelPlan(tokens, rows, hidden, cores, ub, norm, false);
        assert(!unsupported.usedCores && !unsupported.workspaceBytes);
        auto p = MakeParallelPlan(tokens, rows, hidden, cores, ub, norm, true, regBase);
        const uint64_t tile = ((uint64_t(hidden) + 63) / 64) * 64;
        const bool eligible = tokens > 0 && tokens <= cores / COOPERATIVE_CORE_GAIN &&
            rows >= 2 && rows <= COOPERATIVE_MAX_ROWS && hidden >= COOPERATIVE_MIN_HIDDEN &&
            ub > 0;
        if (!eligible) assert(!p.usedCores);
        if (!p.usedCores) {
            assert(!p.tileElems && !p.workspaceBytes);
            continue;
        }
        assert(!p.residentGroup && p.usedCores <= cores);
        if (!p.rowParallel) {
            assert(p.scoreSplits > 1 && p.outputSplits > 0);
            const uint64_t tasks = uint64_t(tokens) * rows * p.scoreSplits;
            std::vector<unsigned> owners(tasks, 0);
            for (unsigned core = 0; core < p.usedCores; ++core)
                for (uint64_t task = core; task < tasks; task += p.usedCores) ++owners[task];
            for (auto n : owners) assert(n == 1);
            for (auto splits : {p.scoreSplits, p.outputSplits}) {
                unsigned end = 0;
                const unsigned units = (hidden + 63) / 64;
                for (unsigned slice = 0; slice < splits; ++slice) {
                    unsigned start = (slice * (units / splits) + std::min(slice, units % splits)) * 64;
                    unsigned count = std::min((units / splits + (slice < units % splits)) * 64, hidden - start);
                    assert(start == end && count > 0 && count <= p.tileElems);
                    end += count;
                }
                assert(end == hidden);
            }
            assert(tokens * p.outputSplits <= p.usedCores);
            assert(p.workspaceBytes == (2 * tasks + (norm ? tokens * p.outputSplits : 0)) * 32);
            continue;
        }
        assert(RowParallelUbBytes(rows, tile, regBase) <= ub);
        assert(p.scoreSplits == 1 && p.outputSplits == 1);
        assert(ShouldUseCooperativeKernel(p) && p.tokenGroupSize != 1);
        const uint64_t tasks = uint64_t(tokens) * rows;
        if (p.tokenGroupSize) {
            assert(p.usedCores == tokens * p.tokenGroupSize);
            assert(p.cachedRows < (rows + p.tokenGroupSize - 1) / p.tokenGroupSize);
            assert(TokenGroupUbBytes(rows, tile, regBase) + uint64_t(p.cachedRows) * tile * 2 <= ub);
            if (p.tokenGroupSize > 1 && tokens <= cores / COOPERATIVE_LOW_TOKEN_OCCUPANCY)
                assert((rows + p.tokenGroupSize - 1) / p.tokenGroupSize == (tasks + cores - 1) / cores);
            // Removing a worker must increase the critical score-wave count;
            // otherwise the group pays for a redundant worker.
            const auto waves = (rows + p.tokenGroupSize - 1) / p.tokenGroupSize;
            assert((rows + p.tokenGroupSize - 2) / (p.tokenGroupSize - 1) > waves);
        } else {
            assert(p.usedCores == std::min<uint64_t>(cores, tasks));
            assert(!p.cachedRows);
        }
        assert(p.tileElems == tile && p.workspaceBytes == (p.tokenGroupSize == 1 ? 0 : tasks * COOPERATIVE_SLOT_BYTES));
        std::vector<unsigned> scoreOwners(tasks, 0), outputOwners(tokens, 0), outputsPerCore(p.usedCores, 0);
        for (unsigned core = 0; core < p.usedCores; ++core) {
            const auto record = [&](uint32_t token, uint32_t position) {
                const uint32_t residual = position == 0 ? rows - 1 : position - 1;
                const uint64_t slot = uint64_t(token) * rows + residual;
                ++scoreOwners[slot];
                if (p.tokenGroupSize != 1) assert((slot + 1) * COOPERATIVE_SLOT_BYTES <= p.workspaceBytes);
            };
            if (p.tokenGroupSize) {
                for (uint32_t pos = core % p.tokenGroupSize; pos < rows; pos += p.tokenGroupSize)
                    record(core / p.tokenGroupSize, pos);
            } else {
                for (uint64_t task = core; task < tasks; task += p.usedCores) record(task / rows, task % rows);
            }
            for (unsigned token = 0; token < tokens; ++token)
                if ((p.tokenGroupSize ? uint64_t(token) * p.tokenGroupSize :
                     uint64_t(token) * p.usedCores / tokens) == core) {
                    ++outputOwners[token];
                    ++outputsPerCore[core];
                }
        }
        for (auto owners : scoreOwners) assert(owners == 1);
        for (auto owners : outputOwners) assert(owners == 1);
        for (auto count : outputsPerCore) assert(count <= 1);
    }
    assert(COOPERATIVE_ROW_PADDING_BYTES % COOPERATIVE_SLOT_BYTES == 0);
    // Both layouts fit whole rows, but only 7168 needs the new allocation.
    assert(RowParallelUbBytes(9, 7168) - RowParallelUbBytes(9, 7104) ==
           64 * (4 * sizeof(uint16_t) + 3 * sizeof(float)) + COOPERATIVE_ROW_PADDING_BYTES);
    const uint64_t paddedBudget = RowParallelUbBytes(9, 7168);
    assert(MakeParallelPlan(1, 9, 7168, 40, paddedBudget, true, true).rowParallel);
    assert(!MakeParallelPlan(1, 9, 7168, 40, paddedBudget - 1, true, true).rowParallel);
    assert(MakeParallelPlan(1, 9, 7168, 40, 192*1024, true, true).usedCores == 9);
    assert(MakeParallelPlan(4, 9, 7168, 40, 192*1024, true, true).usedCores == 36);
    // B=16 keeps full-H output ownership with multiple score workers per token.
    for (bool regBase : {false, true}) {
        const uint32_t cores = regBase ? 56 : 40;
        auto batch16 = MakeParallelPlan(16, 9, 7168, cores, 192*1024, true, true, regBase);
        assert(batch16.rowParallel && batch16.usedCores == (regBase ? 48U : 32U));
        assert(batch16.scoreSplits == 1 && batch16.outputSplits == 1);
        assert(batch16.workspaceBytes == uint64_t(16 * 9) * COOPERATIVE_SLOT_BYTES);
        assert(MakeParallelPlan(cores / 2, 17, 7168, cores, 192*1024, true, true, regBase).rowParallel);
        assert(!MakeParallelPlan(cores / 2 + 1, 9, 7168, cores, 192*1024, true, true, regBase).usedCores);
    }
    for (uint32_t n = 1; n <= 8; ++n) {
        auto a2 = MakeParallelPlan(16, n + 1, 7168, 40, 192*1024, true, true, false);
        auto a5 = MakeParallelPlan(16, n + 1, 7168, 56, 192*1024, true, true, true);
        assert(a2.rowParallel);
        assert(bool(a5.rowParallel));
        assert(MakeParallelPlan(8, n + 1, 7168, 56, 192*1024, true, true, true).rowParallel);
    }
    // Every target case must stay on a new cross-core scoring path.
    // Whole-row high-occupancy plans use uniform groups rather than falling
    // back to a one-worker token plan or the legacy kernels.
    for (bool regBase : {false, true}) {
        const uint32_t cores = regBase ? 56 : 40;
        const uint64_t ub = (regBase ? 248ULL : 192ULL) * 1024;
        for (uint32_t n = 1; n <= 8; ++n) {
            auto candidate = MakeParallelPlan(16, n + 1, 7168, cores, ub, true, true, regBase);
            assert(ShouldUseCooperativeKernel(candidate));
            assert(candidate.usedCores > 16 && candidate.tokenGroupSize > 1);
            for (uint32_t tokens : {1U, 4U, 8U}) {
                auto parallel = MakeParallelPlan(tokens, n + 1, 7168, cores, ub, true, true, regBase);
                assert(ShouldUseCooperativeKernel(parallel));
            }
        }
        auto split = MakeParallelPlan(4, 9, 32768, cores, ub, true, true, regBase);
        assert(split.scoreSplits > 1 && ShouldUseCooperativeKernel(split));
    }
    assert(!ShouldUseCooperativeKernel(ParallelPlan{}));
    // Equality between saved score rows and duplicate weight rows is declined.
    auto group = MakeParallelPlan(16, 8, 7168, 56, 192*1024, true, true, true);
    assert(group.usedCores == 48 && group.tokenGroupSize == 3 && group.cachedRows == 2);
    auto threeRows = MakeParallelPlan(16, 3, 7168, 56, 248*1024, true, true, true);
    assert(threeRows.usedCores == 48 && threeRows.tokenGroupSize == 3 && threeRows.cachedRows == 0);
    assert(MakeParallelPlan(1, 9, 32768, 40, 192*1024, true, true).scoreSplits > 1);
    auto large = MakeParallelPlan(4, 9, 32768, 40, 192*1024, true, true);
    assert(large.usedCores && uint64_t(4*9) * large.scoreSplits > large.usedCores);
}

"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "parallel_plan.cpp"
            binary = Path(directory) / "parallel_plan.exe"
            cpp.write_text(source, encoding="utf-8")
            subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(root), str(cpp), "-o", str(binary)],
                check=True,
                timeout=180,
            )
            subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
