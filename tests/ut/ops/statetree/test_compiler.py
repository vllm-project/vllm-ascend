# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from vllm_ascend.ops.statetree import compiler


class TestStateTreeCompiler(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        compiler.build_kernel.cache_clear()
        self.addCleanup(compiler.build_kernel.cache_clear)

    def test_resolves_checkout_and_packaged_sources(self):
        module = self.root / "vllm_ascend/ops/statetree/compiler.py"
        checkout = self.root / "csrc/statetree/statetree.asc"
        packaged = module.parent / "kernels/statetree.asc"
        with patch.object(compiler, "__file__", str(module)):
            with self.assertRaises(FileNotFoundError):
                compiler.get_kernel_source()
            checkout.parent.mkdir(parents=True)
            checkout.write_text("// checkout source")
            self.assertEqual(compiler.get_kernel_source(), checkout)
            packaged.parent.mkdir(parents=True)
            packaged.write_text("// wheel source")
            self.assertEqual(compiler.get_kernel_source(), packaged)

    def test_build_cache_and_decode_abi(self):
        source = self.root / "statetree.asc"
        source.write_text("// test source")
        library = MagicMock()

        def compile_source(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_bytes(b"test library")

        with (
            patch.object(compiler, "get_kernel_source", return_value=source),
            patch.object(compiler.shutil, "which", return_value="bisheng"),
            patch.object(compiler.vllm_envs, "VLLM_CACHE_ROOT", str(self.root / "cache")),
            patch.object(compiler.subprocess, "run", side_effect=compile_source) as run,
            patch.object(compiler.ctypes, "CDLL", return_value=library),
        ):
            first = compiler.build_kernel()
            self.assertIs(first, compiler.build_kernel())
            compiler.build_kernel.cache_clear()
            compiler.build_kernel()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(library.statetree_decode_launch.argtypes, [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 3)
            source.write_text("// changed source")
            compiler.build_kernel.cache_clear()
            compiler.build_kernel()
            self.assertEqual(run.call_count, 2)

    def test_failed_compilation_does_not_publish_partial_library(self):
        source = self.root / "statetree.asc"
        source.write_text("// test source")

        def fail_compile(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_bytes(b"incomplete")
            raise subprocess.CalledProcessError(1, command)

        with (
            patch.object(compiler, "get_kernel_source", return_value=source),
            patch.object(compiler.shutil, "which", return_value="bisheng"),
            patch.object(compiler.vllm_envs, "VLLM_CACHE_ROOT", str(self.root / "cache")),
            patch.object(compiler.subprocess, "run", side_effect=fail_compile),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            compiler.build_kernel()
        self.assertEqual(list(self.root.rglob("*.so")), [])
        self.assertEqual(list(self.root.rglob("*.tmp")), [])
