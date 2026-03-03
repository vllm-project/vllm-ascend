
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration
)
from vllm.model_executor.models.utils import WeightsMapper


Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "thinker.lm_head.": "language_model.lm_head.",
        "thinker.model.": "language_model.model.",
        "thinker.": "",
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    }
)
