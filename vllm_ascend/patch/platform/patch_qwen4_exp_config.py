# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Allow the Ascend Qwen4Exp DFlash adapter through model-local validation."""

from copy import copy

from vllm.model_executor.models import config as model_configs

_qwen4_exp_config_cls = getattr(model_configs, "Qwen4ExpForConditionalGenerationConfig", None)

# Older supported vLLM releases do not register Qwen4Exp at all.
if _qwen4_exp_config_cls is not None:
    _orig_verify_qwen4_exp_config = _qwen4_exp_config_cls.verify_and_update_config

    def _verify_qwen4_exp_config(vllm_config) -> None:
        speculative_config = vllm_config.speculative_config
        if speculative_config is None or speculative_config.method != "dflash":
            return _orig_verify_qwen4_exp_config(vllm_config)
        if not vllm_config.use_v2_model_runner:
            raise NotImplementedError("Qwen4Exp DFlash requires Model Runner V2; set VLLM_USE_V2_MODEL_RUNNER=1.")

        # This model-local verifier only reads speculative_config for its
        # native-MTP/ngram allowlist. Run all its cache, HC, PLE, parallelism,
        # and MRoPE checks on the shared subconfigs, without that allowlist.
        # The real config retains DFlash for subsequent generic validation,
        # auxiliary-state capture, and draft loading.
        target_config = copy(vllm_config)
        target_config.speculative_config = None
        _orig_verify_qwen4_exp_config(target_config)

    _qwen4_exp_config_cls.verify_and_update_config = staticmethod(_verify_qwen4_exp_config)
