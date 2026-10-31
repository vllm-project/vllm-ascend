# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
import textwrap


def test_fla_imports_wait_for_worker_device_selection():
    script = textwrap.dedent(
        """
        import sys
        import torch
        import torch_npu
        import vllm_ascend

        assert not torch.npu.is_initialized()
        vllm_ascend._ensure_global_patch()
        from vllm_ascend.device import device_op
        from vllm_ascend.ops import layernorm, kda
        from vllm_ascend.models.glm5next import kda as model_kda

        assert not torch.npu.is_initialized()
        assert 'vllm.third_party.flash_linear_attention.ops' not in sys.modules
        # Follow the worker's ordering: select the NPU, then register FLA.
        torch.npu.set_device(0)
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm_ascend.utils import register_ascend_fla_customop

        cls = register_ascend_fla_customop()
        assert cls is register_ascend_fla_customop()
        assert cls is layernorm.AscendFusedRMSNormGated
        from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        with set_current_vllm_config(VllmConfig()):
            norm = FusedRMSNormGated(128, eps=1e-6, activation='sigmoid').to('npu')
            assert isinstance(norm, cls)
            torch.manual_seed(42)
            x = torch.randn(4, 128, device='npu', dtype=torch.float32)
            gate = torch.randn_like(x)
            actual = norm(x, gate)
            expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + norm.eps)
            expected = expected * norm.weight * torch.sigmoid(gate)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            expected_l2 = x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)
            torch.testing.assert_close(l2norm_fwd(x), expected_l2, rtol=1e-4, atol=1e-5)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
