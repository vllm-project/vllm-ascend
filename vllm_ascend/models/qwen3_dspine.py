# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch_npu
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.qwen3_dspine import DSpineQwen3ForCausalLM

from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.utils import get_rotation_path


class AscendDSpineQwen3ForCausalLM(DSpineQwen3ForCausalLM):
    """Keep weights row-addressable for the inherited DSpine scorer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Before weight postprocessing, keep the three matrices that upstream
        # refinement reads directly in ND. The other projections use the
        # registered linear method and can follow its usual layout policy.
        injection = self.model.injection
        for layer in (injection.predecessor_gate, injection.message, injection.writes[-1]):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                raise ValueError("DSpine requires unquantized exact-refinement projections on Ascend.")
            layer.skip_weight_nz_conversion = True

    def post_process(self, vllm_config: VllmConfig) -> None:
        # AscendDFlashSpeculator calls this after sharing the target LM head.
        # Exact transition scoring indexes its weight directly, so packed
        # quantized weights cannot be interpreted as vocabulary rows.
        if not isinstance(self.lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
            raise ValueError("DSpine requires an unquantized target LM head for exact transition scoring.")
        # Upstream gathers candidate IDs and reduces transition scores over
        # standard TP. Other Ascend vocabulary groups require separate support.
        unsupported_modes = (VocabParallelMode.FINE_GRAINED, VocabParallelMode.PCP_X_TP)
        if getattr(self.lm_head, "parallel_mode", None) in unsupported_modes:
            raise NotImplementedError(
                "DSpine requires standard tensor-parallel LM head sharding on Ascend; "
                "disable fine-grained LM head TP and PCP vocabulary weight sharding."
            )
        # Aligning only the auxiliary projection is insufficient: injection
        # readouts and writes also depend on the target's hidden-state basis.
        if get_rotation_path(vllm_config) is not None:
            raise NotImplementedError("DSpine does not support QuaRot targets on Ascend.")
        injection = self.model.injection
        for layer in (self.lm_head, injection.predecessor_gate, injection.message, injection.writes[-1]):
            if int(torch_npu.get_npu_format(layer.weight)) != int(torch_npu.Format.ND):
                raise NotImplementedError("DSpine exact transition scoring requires ND LM head and refinement weights.")
