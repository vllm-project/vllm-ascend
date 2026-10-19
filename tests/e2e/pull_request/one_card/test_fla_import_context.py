# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
import textwrap


def test_fla_import_does_not_create_npu_context():
    # A fresh interpreter is essential: the pytest parent may already own an
    # NPU context. Real operator execution must still initialize the device.
    script = textwrap.dedent(
        """
        import torch
        import torch_npu
        import vllm_ascend

        assert not torch.npu.is_initialized()
        # Use the plugin's normal bootstrap order before importing adapters.
        # Importing device_op directly can hit an unrelated circular import.
        vllm_ascend._ensure_global_patch()
        from vllm_ascend.device import device_op
        from vllm_ascend.ops import layernorm
        from vllm_ascend.ops import kda
        from vllm_ascend.models.glm5next import kda as model_kda
        from vllm.third_party.flash_linear_attention.ops import utils
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        assert not torch.npu.is_initialized()
        assert utils.check_shared_mem() is False
        torch.manual_seed(42)
        x = torch.randn(4, 128, device='npu', dtype=torch.float32)
        y = l2norm_fwd(x)
        expected = x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-5)
        assert torch.npu.is_initialized()
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
