"""Unit tests for the stage-4 #66 dual env guards on the inductor track.

Covers (stage design/stage4/02_设计方案.md §三-3):
  1. ``VLLM_USE_AOT_COMPILE=1`` explicit → warning_once (runs but the saved
     AOT artifact re-load is unverified, stage-4 probe T0b-6), never raise
  2. ``TORCH_COMPILE_DISABLE=1`` → warning_once (upstream already forced the
     compilation mode to NONE, so the track has no effect)

Both guards read the raw os.environ value and only fire on the track.
Config-level only: no NPU device is initialized.
"""

import logging
import os
from types import SimpleNamespace

from vllm.config import CompilationConfig

from tests.ut.base import TestBase

# Env vars the guards read plus the ones the late hook writes; saved/restored
# around every test.
_GUARD_ENV_VARS = (
    "VLLM_USE_AOT_COMPILE",
    "TORCH_COMPILE_DISABLE",
    "TORCHINDUCTOR_NPU_BACKEND",
    "VLLM_USE_STANDALONE_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT",
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "TORCH_COMPILE_DEBUG",
    "VLLM_ASCEND_STRICT_INDUCTOR_CONFIG",
)


class GuardTestBase(TestBase):
    def setUp(self):
        super().setUp()
        self._saved_env = {name: os.environ.get(name) for name in _GUARD_ENV_VARS}
        for name in _GUARD_ENV_VARS:
            os.environ.pop(name, None)
        self._clear_warning_once_cache()

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._clear_warning_once_cache()
        super().tearDown()

    @staticmethod
    def _clear_warning_once_cache() -> None:
        # warning_once dedupes messages through an lru_cache; clear it so each
        # test observes its own warning regardless of suite order.
        from vllm.logger import _print_warning_once

        _print_warning_once.cache_clear()

    @staticmethod
    def _make_stubs(track: bool = True):
        compilation_config = CompilationConfig(backend="inductor") if track else CompilationConfig()
        # Mirror the post-early-hook state: _apply_inductor_track_defaults
        # already pinned the combo keys off before the late hook runs in the
        # real flow, so the normalizer stays silent on these stubs.
        compilation_config.inductor_compile_config["combo_kernels"] = False
        compilation_config.inductor_compile_config["benchmark_combo_kernel"] = False
        vllm_config = SimpleNamespace(compilation_config=compilation_config)
        return vllm_config


class TestAotCompileGuard(GuardTestBase):
    def test_explicit_aot_1_warns_once_and_keeps_running(self):
        """Explicit AOT on the track: warn (artifact re-load unverified,
        T0b-6) but never raise — the env value is kept."""
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        vllm_config = self._make_stubs()
        with self.assertLogs("vllm", level=logging.WARNING) as logs:
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        joined = "\n".join(logs.output)
        self.assertIn("VLLM_USE_AOT_COMPILE=1", joined)
        self.assertIn("T0b-6", joined)
        # The guard does not raise and does not clobber the explicit value.
        self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "1")
        self.assertEqual(os.environ.get("TORCHINDUCTOR_NPU_BACKEND"), "triton_experimental")

    def test_unset_aot_pins_default_without_warning(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs()
        with self.assertNoLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertEqual(os.environ["VLLM_USE_AOT_COMPILE"], "0")

    def test_explicit_aot_0_does_not_warn(self):
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_AOT_COMPILE"] = "0"
        vllm_config = self._make_stubs()
        with self.assertNoLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)


class TestTorchCompileDisableGuard(GuardTestBase):
    def test_disable_1_warns_once(self):
        """TORCH_COMPILE_DISABLE=1: upstream already forced the mode to NONE
        (vllm/config/vllm.py) — the track is inert, warning points there."""
        from vllm_ascend.platform import NPUPlatform

        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        vllm_config = self._make_stubs()
        with self.assertLogs("vllm", level=logging.WARNING) as logs:
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        joined = "\n".join(logs.output)
        self.assertIn("TORCH_COMPILE_DISABLE=1", joined)
        self.assertIn("no effect", joined)
        # A warning, not a raise: the hook still completes.
        self.assertEqual(os.environ.get("TORCHINDUCTOR_NPU_BACKEND"), "triton_experimental")

    def test_disable_unset_does_not_warn(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs()
        with self.assertNoLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertNotIn("TORCH_COMPILE_DISABLE", os.environ)


class TestGuardsAreTrackScoped(GuardTestBase):
    def test_track_off_never_warns(self):
        """Red line (总纲 §〇): track off = zero perturbation — neither guard
        fires and no env is written."""
        from vllm_ascend.platform import NPUPlatform

        os.environ["VLLM_USE_AOT_COMPILE"] = "1"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        vllm_config = self._make_stubs(track=False)
        with self.assertNoLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        for name in _GUARD_ENV_VARS:
            if name in ("VLLM_USE_AOT_COMPILE", "TORCH_COMPILE_DISABLE"):
                continue  # the explicit user values stay untouched
            self.assertNotIn(name, os.environ, name)
