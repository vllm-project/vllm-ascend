# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestVitMropeSetupPatch(unittest.TestCase):
    def test_launch_grid_matches_block_size(self):
        target = "vllm.model_executor.layers.rotary_embedding.mrope_vit_setup"
        models = (
            "vllm.models.minimax_m3.common.vision_tower",
            "vllm_ascend.models._vllm_minimax_m3_common_vision_tower",
        )
        for n in (1, 16, 17, 32, 33, 65):
            for loaded_models in ((), models[:1], models[1:], models):
                with self.subTest(n=n, loaded_models=loaded_models):
                    modules = {}
                    names = [target, "vllm.triton_utils", "vllm.utils.torch_utils", "vllm_ascend.models"]
                    names.extend(loaded_models)
                    for name in names:
                        parts = name.split(".")
                        for end in range(1, len(parts) + 1):
                            key = ".".join(parts[:end])
                            if key not in modules:
                                modules[key] = types.ModuleType(key)
                                modules[key].__path__ = []
                            if end > 1:
                                setattr(modules[".".join(parts[: end - 1])], parts[end - 1], modules[key])
                    torch = MagicMock()
                    modules["torch"] = torch
                    kernel = MagicMock()
                    upstream = modules[target]
                    original = MagicMock(name="original_vit_mrope_setup")
                    upstream.vit_mrope_setup = original
                    for model in loaded_models:
                        modules[model].vit_mrope_setup = original
                    vars(upstream)["_vit_mrope_setup_kernel"] = kernel
                    vars(modules["vllm.triton_utils"])["triton"] = types.SimpleNamespace(
                        cdiv=lambda x, y: (x + y - 1) // y,
                        next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
                    )
                    copy = MagicMock()
                    vars(modules["vllm.utils.torch_utils"])["async_tensor_h2d"] = copy
                    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/worker/patch_mrope_vit_setup.py"
                    spec = importlib.util.spec_from_file_location("_test_vit_mrope_patch", path)
                    assert spec is not None and spec.loader is not None
                    module = importlib.util.module_from_spec(spec)
                    with patch.dict(sys.modules, modules):
                        implementation_name = "vllm_ascend.models.mrope_vit_setup"
                        implementation_path = path.parents[2] / "models/mrope_vit_setup.py"
                        implementation_spec = importlib.util.spec_from_file_location(
                            implementation_name, implementation_path
                        )
                        implementation = importlib.util.module_from_spec(implementation_spec)
                        sys.modules[implementation_name] = implementation
                        implementation_spec.loader.exec_module(implementation)
                        spec.loader.exec_module(module)
                        self.assertIs(upstream.vit_mrope_setup, module.vit_mrope_setup)
                        self.assertIs(upstream._vit_mrope_setup_kernel, kernel)
                        for model in models:
                            if model not in loaded_models:
                                self.assertNotIn(model, sys.modules)
                                modules[model] = types.ModuleType(model)
                                exec(f"from {target} import vit_mrope_setup", vars(modules[model]))
                            self.assertIs(modules[model].vit_mrope_setup, module.vit_mrope_setup)
                        inv = MagicMock()
                        inv.numel.return_value = 21
                        for model in models:
                            modules[model].vit_mrope_setup(inv, inv, inv, [[1, 1, n]], 1)
                    self.assertEqual(kernel.__getitem__.call_count, len(models))
                    kernel.__getitem__.assert_called_with(((n + 15) // 16,))
                    args, kwargs = kernel.__getitem__.return_value.call_args
                    self.assertEqual(args[7:9], (n, 1))
                    self.assertEqual(kwargs["BLOCK_N"], 16)
                    self.assertEqual(kwargs["BLOCK_D"], 64)
                    self.assertEqual(kwargs["num_warps"], 8)
                    self.assertEqual(copy.call_args.args[0], [0, n, 1, 1, n])
                    self.assertEqual(torch.empty.call_args.args, (3, n, 63))


if __name__ == "__main__":
    unittest.main()
