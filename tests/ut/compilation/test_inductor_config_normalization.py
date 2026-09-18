"""Unit tests for ``-cc.inductor_compile_config`` normalization (stage-4 W3).

Covers (stage design/stage4/02_设计方案.md §三, decisions D3/D4):
  1. unknown keys warn + drop (upstream compile_fx would raise AttributeError
     at first piece compile); ``VLLM_ASCEND_STRICT_INDUCTOR_CONFIG=1``
     restores the strict raise
  2. the correctness-dangerous ``split_reductions`` fails fast with guidance
  3. track-pinned-off keys (shape_padding / layout_optimization /
     coordinate_descent_tuning / combo_kernels / benchmark_combo_kernel)
     truthy warn + override False
  4. ``npu_backend`` is a legal key (path (1) per-compile backend selection)

Config-level only: no NPU device is initialized.
"""

import logging
import os
from types import SimpleNamespace

from vllm.config import CompilationConfig

from tests.ut.base import TestBase

# Env vars the normalizer reads and the ones the late hook writes when the
# integration tests call it; saved/restored around every test.
_NORM_ENV_VARS = (
    "VLLM_ASCEND_STRICT_INDUCTOR_CONFIG",
    "TORCHINDUCTOR_NPU_BACKEND",
    "VLLM_USE_STANDALONE_COMPILE",
    "VLLM_USE_AOT_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT",
    "VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE",
    "VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "TORCH_COMPILE_DEBUG",
    "TORCH_COMPILE_DISABLE",
)

# Keys the triton_experimental activation pins off (warn + override False).
_PINNED_OFF_KEYS = (
    "shape_padding",
    "layout_optimization",
    "coordinate_descent_tuning",
    "combo_kernels",
    "benchmark_combo_kernel",
)


class NormalizationTestBase(TestBase):
    def setUp(self):
        super().setUp()
        self._saved_env = {name: os.environ.get(name) for name in _NORM_ENV_VARS}
        for name in _NORM_ENV_VARS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        super().tearDown()

    @staticmethod
    def _make_stubs(inductor_compile_config=None):
        compilation_config = CompilationConfig(backend="inductor")
        if inductor_compile_config is not None:
            compilation_config.inductor_compile_config = dict(inductor_compile_config)
        vllm_config = SimpleNamespace(compilation_config=compilation_config)
        return vllm_config


class TestLegalKeySet(NormalizationTestBase):
    def test_keys_are_fetched_from_live_torch_config(self):
        """D4: the whitelist is generated from the live torch config, never
        hardcoded — known torch keys (dotted sub-config keys included) and
        npu_backend must be members."""
        import torch._inductor.config as torch_inductor_config

        from vllm_ascend.platform import _legal_inductor_config_keys

        legal_keys = _legal_inductor_config_keys()
        live_keys = set(torch_inductor_config.get_config_copy())
        # A few stable representatives, including a dotted sub-config key.
        for key in ("shape_padding", "split_reductions", "cpp.dynamic_threads"):
            self.assertIn(key, legal_keys, key)
            self.assertIn(key, live_keys, key)
        # Unioned unconditionally: torch_npu injects it as a legal per-compile
        # key only after activation (patch-timing sensitive, V-W3-extra(2)).
        self.assertIn("npu_backend", legal_keys)


class TestUnknownKeys(NormalizationTestBase):
    def test_unknown_key_is_dropped_with_warning(self):
        from vllm_ascend.platform import _normalize_inductor_config

        config = {"bogus_key": 1, "shape_padding": False}
        with self.assertLogs("vllm", level=logging.WARNING) as logs:
            result = _normalize_inductor_config(config)
        self.assertEqual(result, {"shape_padding": False})
        joined = "\n".join(logs.output)
        self.assertIn("bogus_key", joined)
        self.assertIn("VLLM_ASCEND_STRICT_INDUCTOR_CONFIG=1", joined)

    def test_strict_mode_raises_on_unknown_key(self):
        """Escape hatch: strict mode restores the upstream raise semantics."""
        from vllm_ascend.platform import _normalize_inductor_config

        os.environ["VLLM_ASCEND_STRICT_INDUCTOR_CONFIG"] = "1"
        with self.assertRaises(ValueError) as ctx:
            _normalize_inductor_config({"bogus_key": 1, "another": 2})
        message = str(ctx.exception)
        self.assertIn("bogus_key", message)
        self.assertIn("another", message)
        self.assertIn("VLLM_ASCEND_STRICT_INDUCTOR_CONFIG", message)

    def test_known_torch_keys_pass_through(self):
        from vllm_ascend.platform import _normalize_inductor_config

        config = {"cpp.dynamic_threads": True, "enable_auto_functionalized_v2": False}
        result = _normalize_inductor_config(config)
        self.assertEqual(result, config)

    def test_npu_backend_is_legal(self):
        """Path (1): torch_npu's patch makes npu_backend a legal per-compile
        key — it must survive normalization."""
        from vllm_ascend.platform import _normalize_inductor_config

        config = {"npu_backend": "triton_experimental"}
        self.assertEqual(_normalize_inductor_config(config), config)

    def test_none_is_a_noop(self):
        from vllm_ascend.platform import _normalize_inductor_config

        self.assertIsNone(_normalize_inductor_config(None))


class TestDangerousKeys(NormalizationTestBase):
    def test_split_reductions_truthy_fails_fast(self):
        """Correctness key: fail fast with guidance (precedent: the
        standalone raise in _setup_inductor_track_envs)."""
        from vllm_ascend.platform import _normalize_inductor_config

        for value in (True, 2):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    _normalize_inductor_config({"split_reductions": value})
                message = str(ctx.exception)
                self.assertIn("split_reductions", message)
                self.assertIn("Remove the key", message)

    def test_split_reductions_falsy_is_kept(self):
        from vllm_ascend.platform import _normalize_inductor_config

        config = {"split_reductions": False}
        self.assertEqual(_normalize_inductor_config(config), config)

    def test_pinned_off_keys_truthy_warn_and_override_false(self):
        from vllm_ascend.platform import _normalize_inductor_config

        for key in _PINNED_OFF_KEYS:
            with self.subTest(key=key):
                config = {key: True}
                with self.assertLogs("vllm", level=logging.WARNING) as logs:
                    result = _normalize_inductor_config(config)
                self.assertEqual(result, {key: False})
                self.assertIn(key, "\n".join(logs.output))

    def test_pinned_off_keys_falsy_are_kept_silent(self):
        from vllm_ascend.platform import _normalize_inductor_config

        config = {key: False for key in _PINNED_OFF_KEYS}
        with self.assertNoLogs("vllm", level=logging.WARNING):
            result = _normalize_inductor_config(dict(config))
        self.assertEqual(result, config)


class TestHookIntegration(NormalizationTestBase):
    """The late hook wires the normalizer into the track (D3: platform-hook
    dict rewrite, cpu.py style)."""

    def test_hook_normalizes_inductor_compile_config(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(
            inductor_compile_config={"bogus_key": 1, "shape_padding": True, "npu_backend": "triton_experimental"}
        )
        with self.assertLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        self.assertEqual(
            vllm_config.compilation_config.inductor_compile_config,
            {"shape_padding": False, "npu_backend": "triton_experimental"},
        )

    def test_hook_raises_on_split_reductions(self):
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs(inductor_compile_config={"split_reductions": True})
        with self.assertRaises(ValueError):
            NPUPlatform._setup_inductor_track_envs(vllm_config)

    def test_hook_overrides_vllm_combo_kernels_default(self):
        """Defense in depth: a fresh CompilationConfig(backend="inductor")
        carries vLLM's combo_kernels=True default; without the early hook
        (e.g. direct late-hook calls) the normalizer is the layer that pins
        it off."""
        from vllm_ascend.platform import NPUPlatform

        vllm_config = self._make_stubs()
        self.assertTrue(vllm_config.compilation_config.inductor_compile_config["combo_kernels"])
        with self.assertLogs("vllm", level=logging.WARNING):
            NPUPlatform._setup_inductor_track_envs(vllm_config)
        cc = vllm_config.compilation_config
        self.assertFalse(cc.inductor_compile_config["combo_kernels"])
        self.assertFalse(cc.inductor_compile_config["benchmark_combo_kernel"])


class TestSingleSizeOverrideWarning(TestBase):
    """Stage-4 W3 leftover: single-size compile_sizes silently overwrites
    user-set max_autotune / coordinate_descent_tuning (V-W3-extra①)."""

    def _cfg(self, sizes, cfg):
        from types import SimpleNamespace

        from vllm_ascend.platform import _warn_single_size_key_override

        return _warn_single_size_key_override(
            SimpleNamespace(compile_sizes=sizes, inductor_compile_config=cfg)
        )

    def test_warns_on_collision(self):
        with self.assertLogs("vllm", level=logging.WARNING) as cm:
            self._cfg([1, 8], {"max_autotune": True})
        self.assertTrue(any("unconditionally overwrite" in m for m in cm.output))

    def test_silent_without_sizes_or_overlap(self):
        with self.assertNoLogs("vllm", level=logging.WARNING):
            self._cfg(None, {"max_autotune": True})
            self._cfg([1], {"size_asserts": True})
