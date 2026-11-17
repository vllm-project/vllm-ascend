# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise profiler patching without importing the accelerator stack."""

import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

PATCH_PATH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_worker_profiler.py"


class TestWorkerProfilerPatch(unittest.TestCase):
    def setUp(self):
        names = (
            "vllm",
            "vllm.config",
            "vllm.config.profiler",
            "vllm.profiler",
            "vllm.profiler.wrapper",
            "vllm.platforms",
            "vllm.v1",
            "vllm.v1.worker",
            "vllm.v1.worker.gpu_worker",
        )
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            if any(other.startswith(name + ".") for other in names):
                module.__path__ = []
            if "." in name:
                parent, attribute = name.rsplit(".", 1)
                setattr(modules[parent], attribute, module)
        self.original = Mock()
        self.platform = SimpleNamespace(device_type="npu")
        self.wrapper = modules["vllm.profiler.wrapper"]
        self.worker = modules["vllm.v1.worker.gpu_worker"]
        self.wrapper.validate_worker_profiler_config = self.original  # type: ignore[attr-defined]
        self.worker.validate_worker_profiler_config = self.original  # type: ignore[attr-defined]
        modules["vllm.config.profiler"].ProfilerConfig = SimpleNamespace  # type: ignore[attr-defined]
        modules["vllm.platforms"].current_platform = self.platform  # type: ignore[attr-defined]
        # Model the parent's cached from-import before the patch is loaded.
        exec("def construct(config):\n    validate_worker_profiler_config(config)\n", self.worker.__dict__)
        with patch.dict(sys.modules, modules):
            self.scope = runpy.run_path(str(PATCH_PATH))

    def test_parent_cached_validator_accepts_npu_torch_profiling(self):
        for config in (None, SimpleNamespace(profiler=None), SimpleNamespace(profiler="torch")):
            with self.subTest(config=config):
                self.worker.construct(config)
        self.assertIs(self.worker.validate_worker_profiler_config, self.wrapper.validate_worker_profiler_config)
        self.original.assert_not_called()

    def test_unsupported_npu_profilers_raise(self):
        for profiler in ("cuda", "proton", "unknown"):
            with self.subTest(profiler=profiler), self.assertRaisesRegex(ValueError, "torch NPU profiler"):
                self.worker.construct(SimpleNamespace(profiler=profiler))
        self.original.assert_not_called()

    def test_other_platforms_keep_upstream_validation(self):
        config = SimpleNamespace(profiler="torch")
        for device_type in ("cpu", "cuda", "xpu"):
            with self.subTest(device_type=device_type):
                self.platform.device_type = device_type
                self.worker.construct(config)
                self.original.assert_called_once_with(config)
                self.original.reset_mock()

    def test_other_platform_errors_propagate(self):
        self.platform.device_type = "cuda"
        self.original.side_effect = ValueError("upstream rejected activities")
        with self.assertRaisesRegex(ValueError, "upstream rejected activities"):
            self.worker.construct(SimpleNamespace(profiler="torch"))


if __name__ == "__main__":
    unittest.main()
