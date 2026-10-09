# SPDX-License-Identifier: Apache-2.0
"""Source and host-stub checks for the FwdH launcher/header contract.

These tests need neither vLLM nor CANN. Host-stub compilation checks dispatch
and argument counts only; it does not compile or execute the device kernels.
"""

import shutil
import subprocess
import unittest
from pathlib import Path
from typing import ClassVar

import regex as re

ROOT = Path(__file__).resolve().parents[3]
OP_KERNEL = ROOT / "csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/op_kernel"
LAUNCHER = OP_KERNEL / "chunk_gated_delta_rule_fwd_h_vllm.cpp"
OP_DEF = OP_KERNEL.parent / "op_host/chunk_gated_delta_rule_fwd_h_vllm_def.cpp"


def entry_parameters() -> list[str]:
    definition = OP_DEF.read_text(encoding="utf-8")
    inputs = re.findall(r'this->Input\("([^"\n]+)"\)', definition)
    outputs = re.findall(r'this->Output\("([^"\n]+)"\)', definition)
    return [*inputs, *outputs, "workspace", "tiling"]


# Match the declarations actually present in each architecture header.
HOST_STUB = """
#define __global__
#define __aicore__
#define __gm__
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define KERNEL_TASK_TYPE(...)
#define TILING_KEY_IS(...) true
using GM_ADDR = void*;
using half = float;
using bfloat16_t = double;
struct ChunkGatedDeltaRuleFwdHVllmTilingData {
    int dataType;
    int stateDataType;
    int gDataType;
    bool useGk;
};
namespace AscendC {
inline GM_ADDR GetUserWorkspace(GM_ADDR workspace) { return workspace; }
}
namespace Catlass::Gemm::Kernel {
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ != 200)
struct GDNFwdHTileShapes128 {};
struct GDNFwdHTileShapes256 {};
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
template<class Input, class Gate, class State, class Workspace,
         class TileShapes = GDNFwdHTileShapes128, bool KGated = false,
         bool ScalarGated = true, bool UseExp2 = false>
#else
template<class Input, class Gate, class State, class Workspace,
         class TileShapes = GDNFwdHTileShapes128, bool KGated = false>
#endif
struct GDNFwdHKernel {
    void Init(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
              GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR) {}
    void Process() {}
};
#else
template<class Input, class Gate, class State, class Workspace>
struct GDNFwdHKernel {
    void Init(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR,
              GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR) {}
    void Process() {}
};
#endif
}
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#define CATLASS_UNIFIED_CORE 1
#endif
"""


def render_host_stub(source):
    # Quoted includes are checked separately against the real source tree.
    body = re.sub(r'^#include\s+"[^"\n]+"\s*$', "", source, flags=re.MULTILINE)
    arguments = ", ".join("nullptr" for _ in entry_parameters())
    wrapper = f"\nvoid generated_wrapper() {{ chunk_gated_delta_rule_fwd_h_vllm({arguments}); }}\n"
    return HOST_STUB + body + wrapper


class FwdHLauncherContractTests(unittest.TestCase):
    source: ClassVar[str]
    compiler: ClassVar[str | None]

    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.compiler = shutil.which("g++") or shutil.which("clang++")

    def compile_source(self, source, architecture):
        if self.compiler is None:
            self.skipTest("g++ or clang++ required for host-stub syntax checks")
        command = [self.compiler, "-std=c++17", "-fsyntax-only", "-x", "c++"]
        if architecture is not None:
            command.append(f"-D__CCE_AICORE__={architecture}")
        return subprocess.run(
            [*command, "-"],
            input=render_host_stub(source),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
            check=False,
        )

    def test_real_headers_match_architecture_contracts(self):
        for arch, template_count, init_count in (("arch20", 4, 12), ("arch22", 6, 13), ("arch35", 8, 13)):
            with self.subTest(arch=arch):
                source = (OP_KERNEL / arch / "gemm/kernel/gdn_fwd_h_kernel.hpp").read_text(encoding="utf-8")
                match = re.search(r"template\s*<([^>]+)>\s*class\s+GDNFwdHKernel", source)
                assert match is not None, f"Missing GDNFwdHKernel template in {arch}"
                self.assertEqual(len(match.group(1).split(",")), template_count)
                init = re.search(r"\bvoid\s+Init\((.*?)\)", source, re.DOTALL)
                assert init is not None, f"Missing GDNFwdHKernel.Init in {arch}"
                self.assertEqual(len(init.group(1).split(",")), init_count)
                self.assertEqual(bool(re.search(r"\bgk\b", init.group(1))), arch != "arch20")

    def test_architecture_includes_exist(self):
        includes = re.findall(r'^#include\s+"(arch[^"\n]+)"', self.source, re.MULTILINE)
        self.assertTrue(includes)
        for include in includes:
            self.assertTrue((OP_KERNEL / include).is_file(), include)

    def preprocess_source(self, architecture):
        if self.compiler is None:
            self.skipTest("g++ or clang++ required for host-stub preprocessing")
        command = [self.compiler, "-std=c++17", "-E", "-P", "-x", "c++"]
        if architecture is not None:
            command.append(f"-D__CCE_AICORE__={architecture}")
        result = subprocess.run(
            [*command, "-"],
            input=render_host_stub(self.source),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=20,
            check=True,
        )
        return result.stdout

    def test_entry_abi_for_each_architecture(self):
        for architecture in (200, 220, 310, None):
            with self.subTest(architecture=architecture):
                source = self.preprocess_source(architecture)
                entries = re.findall(r"\bvoid\s+chunk_gated_delta_rule_fwd_h_vllm\((.*?)\)", source, re.DOTALL)
                self.assertEqual(len(entries), 1)
                names = [argument.strip().split()[-1] for argument in entries[0].split(",")]
                self.assertEqual(names, entry_parameters())

    def test_tile_shape_dispatch_is_non_310p_only(self):
        source = self.preprocess_source(200)
        self.assertNotIn("GDNFwdHTileShapes", source)
        self.assertNotIn("ChunkGatedDeltaRuleFwdHVllmDispatch", source)
        for architecture in (220, 310, None):
            self.assertIn("ChunkGatedDeltaRuleFwdHVllmDispatch", self.preprocess_source(architecture))

    def test_host_stub_syntax_for_all_existing_architecture_routes(self):
        for architecture in (200, 220, 310, None):
            with self.subTest(architecture=architecture):
                result = self.compile_source(self.source, architecture)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_host_stub_rejects_six_parameter_regression(self):
        mutated = self.source.replace("float, float, workspaceType>", "float, float, workspaceType, int, false>")
        self.assertNotEqual(mutated, self.source)
        result = self.compile_source(mutated, 200)
        self.assertNotEqual(result.returncode, 0, "Negative control unexpectedly compiled")
        self.assertRegex(result.stderr, r"wrong number of template arguments|too many template arguments")

    def test_host_stub_rejects_missing_optional_input_slot(self):
        state_parameter = f"GM_ADDR {entry_parameters()[5]}"
        mutated = self.source.replace(f"GM_ADDR gk, {state_parameter}", state_parameter, 1)
        self.assertNotEqual(mutated, self.source)
        result = self.compile_source(mutated, 200)
        self.assertNotEqual(result.returncode, 0, "Missing OpDef input slot unexpectedly compiled")
        self.assertRegex(result.stderr, r"too many arguments")

    def test_initial_state_seed_is_310p_only(self):
        binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
        start = binding.index("#ifdef ASCEND_PLATFORM_310P\n    // 310P NZ h contract:")
        end = binding.index("#endif\n    bool save_new_value_", start)
        self.assertIn(".permute({0, 1, 4, 2, 3, 5})", binding[start:end])


if __name__ == "__main__":
    unittest.main()
