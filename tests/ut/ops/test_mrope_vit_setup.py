# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch


class TestAscendVitMropeSetup(unittest.TestCase):
    def test_registration_and_launch(self):
        class VitMropeSetup:
            register_oot = MagicMock(side_effect=lambda cls: cls)

        kernel = MagicMock()
        copy = MagicMock()
        torch = MagicMock()
        modules = {
            "torch": torch,
            "vllm.model_executor.layers.rotary_embedding.mrope_vit_setup": types.SimpleNamespace(
                VitMropeSetup=VitMropeSetup, _vit_mrope_setup_kernel=kernel
            ),
            "vllm.triton_utils": types.SimpleNamespace(
                triton=types.SimpleNamespace(
                    cdiv=lambda x, y: (x + y - 1) // y,
                    next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
                )
            ),
            "vllm.utils.torch_utils": types.SimpleNamespace(async_tensor_h2d=copy),
        }
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/mrope_vit_setup.py"
        spec = importlib.util.spec_from_file_location("_test_ascend_vit_mrope", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        VitMropeSetup.register_oot.assert_called_once_with(module.AscendVitMropeSetup)

        inv = [MagicMock() for _ in range(3)]
        for freq in inv:
            freq.numel.return_value = 21
        for n, blocks in ((0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (33, 3)):
            with self.subTest(n=n):
                kernel.reset_mock()
                copy.reset_mock()
                torch.empty.reset_mock()
                cos, sin = MagicMock(), MagicMock()
                torch.empty.side_effect = (cos, sin)
                actual = module.AscendVitMropeSetup().forward(*inv, [[1, 1, n]], 1)
                self.assertEqual(actual, (cos, sin))
                kernel.__getitem__.assert_called_once_with((blocks,))
                launch = kernel.__getitem__.return_value
                launch.assert_called_once()
                args, kwargs = launch.call_args
                self.assertEqual(args[:5], (*inv, cos, sin))
                self.assertEqual(args[7:9], (n, 1))
                self.assertEqual(
                    kwargs,
                    dict(half_t=21, half_h=21, half_w=21, spatial_merge=1, BLOCK_N=16, BLOCK_D=64, num_warps=8),
                )
                copy.assert_called_once_with([0, n, 1, 1, n], device=inv[0].device, dtype=torch.int32)
                expected = call(3, n, 63, device=inv[0].device, dtype=torch.float32)
                self.assertEqual(torch.empty.call_args_list, [expected, expected])


if __name__ == "__main__":
    unittest.main()
