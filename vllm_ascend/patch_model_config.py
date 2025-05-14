#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings
from typing import Any, Dict, Final, List, Literal, Mapping, Optional, Union

import torch
import vllm.envs as envs
from vllm.config import (HfOverrides, ModelConfig, ModelImpl, PoolerConfig,
                         TaskOption, _get_and_verify_dtype,
                         _get_and_verify_max_len, get_min_sliding_window,
                         get_served_model_name, logger)
from vllm.transformers_utils.config import (ConfigFormat, get_config,
                                            get_hf_image_processor_config,
                                            get_hf_text_config)


def new_init(
    self,
    model: str,
    task: Union[TaskOption, Literal["draft"]],
    tokenizer: str,
    tokenizer_mode: str,
    trust_remote_code: bool,
    dtype: Union[str, torch.dtype],
    seed: int,
    allowed_local_media_path: str = "",
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    rope_scaling: Optional[Dict[str, Any]] = None,
    rope_theta: Optional[float] = None,
    tokenizer_revision: Optional[str] = None,
    max_model_len: Optional[int] = None,
    spec_target_max_model_len: Optional[int] = None,
    quantization: Optional[str] = None,
    enforce_eager: Optional[bool] = None,
    max_seq_len_to_capture: Optional[int] = None,
    max_logprobs: int = 20,
    disable_sliding_window: bool = False,
    skip_tokenizer_init: bool = False,
    served_model_name: Optional[Union[str, List[str]]] = None,
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
    use_async_output_proc: bool = True,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    hf_overrides: Optional[HfOverrides] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    disable_mm_preprocessor_cache: bool = False,
    override_neuron_config: Optional[Dict[str, Any]] = None,
    override_pooler_config: Optional["PoolerConfig"] = None,
    logits_processor_pattern: Optional[str] = None,
    generation_config: Optional[str] = None,
    enable_sleep_mode: bool = False,
    override_generation_config: Optional[Dict[str, Any]] = None,
    model_impl: Union[str, ModelImpl] = ModelImpl.AUTO,
) -> None:
    self.model = model
    self.tokenizer = tokenizer
    self.tokenizer_mode = tokenizer_mode
    self.trust_remote_code = trust_remote_code
    self.allowed_local_media_path = allowed_local_media_path
    self.seed = seed
    self.revision = revision
    self.code_revision = code_revision
    self.rope_scaling = rope_scaling
    self.rope_theta = rope_theta
    self.model_impl = model_impl

    if hf_overrides is None:
        hf_overrides = {}

    if callable(hf_overrides):
        hf_overrides_kw: Dict[str, Any] = {}
        hf_overrides_fn = hf_overrides
    else:
        hf_overrides_kw = hf_overrides
        hf_overrides_fn = None

    if rope_scaling is not None:
        hf_override: Dict[str, Any] = {"rope_scaling": rope_scaling}
        hf_overrides_kw.update(hf_override)
        msg = ("`--rope-scaling` will be removed in a future release. "
               f"'Please instead use `--hf-overrides '{hf_override!r}'`")
        warnings.warn(DeprecationWarning(msg), stacklevel=2)
    if rope_theta is not None:
        hf_override = {"rope_theta": rope_theta}
        hf_overrides_kw.update(hf_override)
        msg = ("`--rope-theta` will be removed in a future release. "
               f"'Please instead use `--hf-overrides '{hf_override!r}'`")
        warnings.warn(DeprecationWarning(msg), stacklevel=2)

    self.maybe_pull_model_tokenizer_for_s3(model, tokenizer)

    # The tokenizer version is consistent with the model version by default.
    if tokenizer_revision is None:
        self.tokenizer_revision = revision
    else:
        self.tokenizer_revision = tokenizer_revision
    self.quantization = quantization
    self.enforce_eager = enforce_eager
    self.max_seq_len_to_capture = max_seq_len_to_capture
    self.max_logprobs = max_logprobs
    self.disable_sliding_window = disable_sliding_window
    self.skip_tokenizer_init = skip_tokenizer_init
    self.enable_sleep_mode = enable_sleep_mode

    from vllm.platforms import current_platform

    hf_config = get_config(self.model, trust_remote_code, revision,
                           code_revision, config_format)

    if hf_overrides_kw:
        logger.info("Overriding HF config with %s", hf_overrides_kw)
        hf_config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.info("Overriding HF config with %s", hf_overrides_fn)
        hf_config = hf_overrides_fn(hf_config)

    self.hf_config = hf_config

    self.hf_text_config = get_hf_text_config(self.hf_config)
    self.encoder_config = self._get_encoder_config()
    self.hf_image_processor_config = get_hf_image_processor_config(
        self.model, revision)
    self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
    self.use_async_output_proc = use_async_output_proc
    self.mm_processor_kwargs = mm_processor_kwargs
    self.disable_mm_preprocessor_cache = disable_mm_preprocessor_cache

    # Set enforce_eager to False if the value is unset.
    if self.enforce_eager is None:
        self.enforce_eager = False

    sliding_window = getattr(self.hf_text_config, "sliding_window", None)
    has_interleaved_attention = (sliding_window is not None) and (
        isinstance(sliding_window, list) or
        (self.hf_text_config.model_type in ["gemma2", "cohere2"]))

    if (not self.disable_sliding_window and has_interleaved_attention):
        if (backend :=
                envs.VLLM_ATTENTION_BACKEND) in ("XFORMERS", "FLASHINFER"):
            sliding_window_len_min = get_min_sliding_window(
                self.hf_text_config.sliding_window)

            logger.warning_once(
                f"{self.hf_text_config.model_type} has interleaved "
                "attention, which is currently not supported by the "
                f"{backend} backend. Disabling sliding window and capping "
                "the max length to the sliding window size "
                f"({sliding_window_len_min}).")
            self.disable_sliding_window = True
        else:
            # for a model with interleaved attention,
            # the scheduler and the model treat it as full attention
            # (i.e., not dropping any tokens outside the window).
            # only the attention layer itself is aware of the sliding
            # window, and use the window size to compute the attention.
            self.hf_text_config.interleaved_sliding_window = sliding_window
            delattr(self.hf_text_config, "sliding_window")
            sliding_window = None

    self.max_model_len = _get_and_verify_max_len(
        hf_config=self.hf_text_config,
        max_model_len=max_model_len,
        disable_sliding_window=self.disable_sliding_window,
        sliding_window_len=self.get_hf_config_sliding_window(),
        spec_target_max_model_len=spec_target_max_model_len,
        encoder_config=self.encoder_config)
    self.served_model_name = get_served_model_name(model, served_model_name)
    self.multimodal_config = self._init_multimodal_config(limit_mm_per_prompt)
    if not self.skip_tokenizer_init:
        self._verify_tokenizer_mode()

    self.is_attention_free = self._init_attention_free()
    self.is_hybrid = self._init_is_hybrid()
    self.has_inner_state = self._init_has_inner_state()

    if current_platform.is_neuron():
        self.override_neuron_config = override_neuron_config
    else:
        self.override_neuron_config = None

    supported_tasks, task = self._resolve_task(task, self.hf_config)
    self.supported_tasks = supported_tasks
    self.task: Final = task  # type: ignore
    if self.task in ("draft", "generate"):
        self.truncation_side = "left"
    else:
        self.truncation_side = "right"

    self.pooler_config = self._init_pooler_config(override_pooler_config)
    self.logits_processor_pattern = logits_processor_pattern

    self.generation_config = generation_config
    self.override_generation_config = override_generation_config or {}

    self._verify_quantization()
    self._verify_cuda_graph()
    self._verify_bnb_config()


# The platform assertion is deleted to support the npu platform.
ModelConfig.__init__ = new_init
