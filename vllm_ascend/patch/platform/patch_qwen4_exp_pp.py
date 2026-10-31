from functools import wraps

from vllm.model_executor.models.config import (
    Qwen4ExpForConditionalGenerationConfig,
)

_verify_config = Qwen4ExpForConditionalGenerationConfig.verify_and_update_config


@wraps(_verify_config)
def _verify_qwen4_exp_config(vllm_config):
    text_config = vllm_config.model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    ple_layer_ids = text_config.ple_layer_ids
    if not ple_layer_ids or parallel_config.pipeline_parallel_size <= 1:
        return _verify_config(vllm_config)

    if parallel_config.enable_dbo or parallel_config.ubatch_size > 1:
        return _verify_config(vllm_config)

    text_config.ple_layer_ids = []
    try:
        return _verify_config(vllm_config)
    finally:
        text_config.ple_layer_ids = ple_layer_ids


Qwen4ExpForConditionalGenerationConfig.verify_and_update_config = staticmethod(_verify_qwen4_exp_config)
