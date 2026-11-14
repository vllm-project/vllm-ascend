# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.qwen3_dspine import DSpineQwen3ForCausalLM

from vllm_ascend.ops.dspine import score_dspine_transitions
from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.utils import get_rotation_path

_ACL_FORMAT_ND = 2


class AscendDSpineQwen3ForCausalLM(DSpineQwen3ForCausalLM):
    """Reuse the upstream backbone/loader with NPU exact transition scoring."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # This runs before process_weights_after_loading. Bridge weights are
        # consumed both by Ascend linear methods and by explicit FP32 refinement.
        for layer in self.model.injection.modules():
            if hasattr(layer, "quant_method"):
                if not isinstance(layer.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)):
                    raise ValueError("DSpine requires unquantized causal-injection projections on Ascend.")
                layer.skip_weight_nz_conversion = True
        for name in ("lm_head_nd", "predecessor_weight", "message_weight", "write_weight"):
            self.register_buffer(f"_dspine_{name}", None, persistent=False)

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

        # Materialize stable scoring buffers once, after target weight sharing
        # and before graph capture. ND heads alias their original storage; an
        # NZ target head requires an additional ND copy of this rank's shard.
        self._dspine_lm_head_nd = (
            self.lm_head.weight
            if torch_npu.get_npu_format(self.lm_head.weight) == _ACL_FORMAT_ND
            else torch_npu.npu_format_cast(self.lm_head.weight, _ACL_FORMAT_ND)
        )
        injection = self.model.injection
        self._dspine_predecessor_weight = torch_npu.npu_format_cast(
            injection.predecessor_gate.weight, _ACL_FORMAT_ND
        ).float()
        self._dspine_message_weight = torch_npu.npu_format_cast(injection.message.weight, _ACL_FORMAT_ND).float()
        self._dspine_write_weight = torch_npu.npu_format_cast(injection.writes[-1].weight, _ACL_FORMAT_ND).float()

    def compute_transition_scores(
        self,
        candidate_ids: torch.Tensor,
        final_pre_injection: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._dspine_lm_head_nd is None:
            raise RuntimeError("DSpine scoring buffers must be initialized after loading the draft model.")
        return score_dspine_transitions(
            self.model.injection,
            self.model.norm,
            self.lm_head,
            self.model.transfer_codebook,
            candidate_ids,
            final_pre_injection,
            anchor_token_ids,
            self._dspine_lm_head_nd,
            self._dspine_predecessor_weight,
            self._dspine_message_weight,
            self._dspine_write_weight,
        )
