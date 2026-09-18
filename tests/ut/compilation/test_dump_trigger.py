"""Unit tests for the stage-4 W4 dump one-liner on the inductor track.

Covers (stage design/stage4/02_设计方案.md §四-1, probe T0b-7):
  1. non-empty ``-cc.debug_dump_path`` → ``TORCH_COMPILE_DEBUG=1`` setdefault
     + info log pointing at the vLLM cache ``inductor_cache/`` artifacts
  2. user pre-set ``TORCH_COMPILE_DEBUG`` wins (setdefault semantics)
  3. empty ``debug_dump_path`` → nothing set
  4. red line: the hook never touches the cache-dir envs (upstream
     initialize_cache hard-sets TORCHINDUCTOR_CACHE_DIR, V-W4-A4) and never
     sets VLLM_DEBUG_DUMP_PATH (depyf 0.20.0 is incompatible with torch
     2.13 — setting it mounts depyf and crashes kernel loading)

Config-level only: no NPU device is initialized.
"""

import logging
import os
from pathlib import Path
from types import SimpleNamespace

from vllm.config import CompilationConfig

from tests.ut.base import TestBase

# Env vars the dump trigger may read/write plus the ones the late hook
# writes; saved/restored around every test.
_DUMP_ENV_VARS = (
    "TORCH_COMPILE_DEBUG",
    "TORCHINDUCTOR_NPU_BACKEND",
    "VLLM_USE_STANDALONE_COMPILE",
    "VLLM_USE_AOT_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT",
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "TORCH_COMPILE_DISABLE",
    "VLLM_ASCEND_STRICT_INDUCTOR_CONFIG",
    # Red-line vars: must stay untouched (asserted absent after the hook).
    "VLLM_DEBUG_DUMP_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


class DumpTestBase(TestBase):
    def setUp(self):
        super().setUp()
        self._saved_env = {name: os.environ.get(name) for name in _DUMP_ENV_VARS}
        for name in _DUMP_ENV_VARS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        super().tearDown()

    @staticmethod
    def _make_stubs(track: bool = True, debug_dump_path=None):
        compilation_config = CompilationConfig(backend="inductor", debug_dump_path=debug_dump_path) if track else CompilationConfig(debug_dump_path=debug_dump_path)
        vllm_config = SimpleNamespace(compilation_config=compilation_config)
        return vllm_config


class TestDumpTrigger(DumpTestBase):
    def test_non_empty_dump_path_enables_torch_compile_debug(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(debug_dump_path=Path("/tmp/va_dump"))
        with self.assertLogs("vllm", level=logging.INFO) as logs:
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        joined = "\n".join(logs.output)
        self.assertIn("debug_dump_path", joined)
        self.assertIn("inductor_cache/", joined)
        self.assertEqual(os.environ["TORCH_COMPILE_DEBUG"], "1")

    def test_user_torch_compile_debug_wins(self):
        """setdefault semantics: an explicit user value is never clobbered."""
        from vllm_ascend.platform import NPUPlatform

        os.environ["TORCH_COMPILE_DEBUG"] = "1"  # any pre-set value wins
        vllm_config = self._make_stubs(debug_dump_path=Path("/tmp/va_dump"))
        NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertEqual(os.environ["TORCH_COMPILE_DEBUG"], "1")
        os.environ["TORCH_COMPILE_DEBUG"] = "0"  # even an explicit off wins
        NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertEqual(os.environ["TORCH_COMPILE_DEBUG"], "0")

    def test_empty_dump_path_sets_nothing(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(debug_dump_path=None)
        NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertNotIn("TORCH_COMPILE_DEBUG", os.environ)

    def test_track_off_sets_nothing_even_with_dump_path(self):
        """Red line (总纲 §〇): track off = zero perturbation."""
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(track=False, debug_dump_path=Path("/tmp/va_dump"))
        NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertNotIn("TORCH_COMPILE_DEBUG", os.environ)

    def test_depyf_and_cache_dirs_are_never_touched(self):
        """T0b-7 / V-W4-A4: VLLM_DEBUG_DUMP_PATH would mount depyf (incompatible
        with torch 2.13) and the cache dirs are hard-set by upstream
        initialize_cache — the hook must not write any of them."""
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(debug_dump_path=Path("/tmp/va_dump"))
        NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertNotIn("VLLM_DEBUG_DUMP_PATH", os.environ)
        self.assertNotIn("TORCHINDUCTOR_CACHE_DIR", os.environ)
        self.assertNotIn("TRITON_CACHE_DIR", os.environ)
