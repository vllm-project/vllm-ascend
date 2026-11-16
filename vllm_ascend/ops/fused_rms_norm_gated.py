# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load FLA and register its Ascend subclass only at the point of use.

Import this module after the worker selects its NPU, or when explicitly
requesting the FLA class. Module import locking serializes registration and
the module cache preserves class identity across repeated requests.
"""

# Keep the FLA base class out of layernorm discovery: importing its package
# can initialize an unwanted NPU context through capability queries.
# TODO: Move this class back to layernorm.py and restore eager registration
# once supported vLLM FLA or Triton Ascend versions fix that import side effect.
from vllm.model_executor.custom_op import CustomOp
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

from vllm_ascend.ops.triton.kda.kda import rms_norm_gated


@CustomOp.register_oot(name="FusedRMSNormGated")
class AscendFusedRMSNormGated(FusedRMSNormGated):
    """Use Ascend's fused kernel at the upstream FLA CustomOp boundary."""

    def forward_oot(self, x, g, residual=None, prenorm=False, residual_in_fp32=False, out=None):
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            out=out,
        )
