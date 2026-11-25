import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from vllm_ascend.models.quarot_utils import align_draft_weights


class AscendQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))

    def compute_confidence(self, head_hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token."""
        if not self.enable_confidence_head:
            raise RuntimeError("The DSpark confidence head is disabled.")
        assert self.model.confidence_head is not None
        confidence = torch.sigmoid(self.model.confidence_head(head_hidden, markov_embed))
        # Non-finite estimates must not poison the adaptive budget or its
        # device-side draft ranking. Target verification checks every draft.
        return torch.nan_to_num(confidence, nan=1.0)

    def configure_target_aux_hidden_capture(self, target_model: torch.nn.Module) -> None:
        """Select draft auxiliary inputs, without changing target Eager/Graph mode."""
        set_capture_mode = getattr(target_model, "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is None:
            get_language_model = getattr(target_model, "get_language_model", None)
            if callable(get_language_model):
                set_capture_mode = getattr(get_language_model(), "set_dspark_aux_capture_materialized", None)
        if set_capture_mode is not None:
            set_capture_mode(True)

    def post_process(self, vllm_config: VllmConfig) -> None:
        align_draft_weights(self, self.model.fc, vllm_config)
