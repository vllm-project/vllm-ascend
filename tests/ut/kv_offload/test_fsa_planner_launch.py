# SPDX-License-Identifier: Apache-2.0
"""Check the FSA planner's host enqueue/return ordering without an NPU."""

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PLANNER = Path(__file__).resolve().parents[3] / "csrc/attention/fused_sparse_attention_overlap/device_planner"

RUNTIME = """
#include <cassert>
#include <cstdint>
#include <string>
#define __host__
#define __aicore__
#include "fsa-lru-geometry-a3.h"

using aclrtStream = void*;
using aclError = int;
constexpr int ACL_SUCCESS = 0, ACL_ERROR_INVALID_PARAM = 1, ACL_RT_THREAD_LEVEL = 0;
static int pending;
static char failing;
static std::string launched;
static int aclrtGetLastError(int level) {
    assert(level == ACL_RT_THREAD_LEVEL);
    int result = pending;
    pending = ACL_SUCCESS;
    return result;
}
static void enqueue(char stage) {
    launched += stage;
    if (stage == failing) pending = 9000 + stage;
}
template <class... Args> void fsa_lru_abi_map(Args...) { enqueue('m'); }
template <class... Args> void fsa_lru_abi_plan(Args...) { enqueue('p'); }
template <class... Args> void fsa_lru_abi_publish(Args...) { enqueue('u'); }
"""

CHECK = """
int main() {
    int buffer;
    auto plan = [&] {
        return fsa_lru_plan_async_v5(&buffer, &buffer, &buffer, &buffer, &buffer, &buffer,
            &buffer, &buffer, &buffer, 1, 1, 1, 8, 1, 1, 192 * 1024, 8, nullptr);
    };
    auto publish = [&] { return fsa_lru_publish_async_v4(&buffer, &buffer, &buffer, 1, 1, 1, nullptr); };
    auto check = [](auto submit, int prior, char failure, int expected, const char* launches) {
        pending = prior;
        failing = failure;
        launched.clear();
        assert(submit() == expected);
        assert(launched == launches);
        assert(pending == ACL_SUCCESS);
    };
    check(plan, 37, 0, 37, "");
    check(plan, 0, 'm', 9000 + 'm', "m");
    check(plan, 0, 'p', 9000 + 'p', "mp");
    check(plan, 0, 0, ACL_SUCCESS, "mp");
    check(publish, 37, 0, 37, "");
    check(publish, 0, 'u', 9000 + 'u', "u");
    check(publish, 0, 0, ACL_SUCCESS, "u");
}
"""


class TestFsaPlannerLaunch(unittest.TestCase):
    def test_enqueue_failures(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("A C++ compiler is required for the native planner host entries")

        source = (PLANNER / "fsa-mapped-recurrent-planner-a3-abi.asc").read_text(encoding="utf-8")
        host = source[source.index("int fsa_lru_plan_async_v5(") :]
        # Ordinary C++ functions stand in for the three asynchronous enqueue operations.
        host = re.sub(r"<<<[^<>]+>>>", "", host)
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "launch.cpp"
            exe = Path(directory) / ("launch.exe" if os.name == "nt" else "launch")
            cpp.write_text(RUNTIME + host + CHECK, encoding="utf-8")
            subprocess.run([compiler, "-std=c++17", "-I", str(PLANNER), str(cpp), "-o", str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    unittest.main()
