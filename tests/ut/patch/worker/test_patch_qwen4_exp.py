# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.patch.worker import patch_qwen4_exp


def test_patch_qwen4_exp_module_exports_availability_flag():
    assert isinstance(patch_qwen4_exp.QWEN4_EXP_AVAILABLE, bool)
