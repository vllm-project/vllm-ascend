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

from typing import Any, Dict, Optional

from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.model_executor.layers.rejection_sampler import RejectionSampler
from vllm.model_executor.layers.spec_decode_base_sampler import \
    SpecDecodeBaseSampler
from vllm.model_executor.layers.typical_acceptance_sampler import \
    TypicalAcceptanceSampler
from vllm.spec_decode.medusa_worker import MedusaWorker
from vllm.spec_decode.mlp_speculator_worker import MLPSpeculatorWorker
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker
from vllm.worker.worker_base import WorkerBase

from vllm_ascend.worker.draft_model_runner import TP1DraftModelRunner


def create_worker(
    cls,
    scorer_worker: WorkerBase,
    draft_worker_kwargs: Dict[str, Any],
    disable_mqa_scorer: bool,
    disable_by_batch_size: Optional[int],
    draft_token_acceptance_method: str,
    typical_acceptance_sampler_posterior_threshold: float,
    typical_acceptance_sampler_posterior_alpha: float,
    disable_logprobs: bool,
    disable_log_stats: bool,
    num_speculative_tokens: int,
) -> "SpecDecodeWorker":

    allow_zero_draft_token_step = True
    enable_lm_head_weight_load = False
    num_spec_prefill_steps = 1
    ngram_prompt_lookup_max = (
        draft_worker_kwargs.pop("ngram_prompt_lookup_max"))
    ngram_prompt_lookup_min = (
        draft_worker_kwargs.pop("ngram_prompt_lookup_min"))
    draft_model_config = draft_worker_kwargs["vllm_config"].model_config
    draft_parallel_config: ParallelConfig = draft_worker_kwargs[
        'vllm_config'].parallel_config
    if ngram_prompt_lookup_max > 0:
        draft_worker_kwargs[
            "device_type"] = scorer_worker.device_config.device.type
        proposer_worker = NGramWorker(**draft_worker_kwargs)
        proposer_worker.set_ngram_window_size(ngram_prompt_lookup_min,
                                              ngram_prompt_lookup_max)
    else:
        draft_tp = draft_parallel_config.tensor_parallel_size
        target_tp = scorer_worker.parallel_config.tensor_parallel_size

        if draft_model_config.hf_config.model_type == "mlp_speculator":
            proposer_worker = MLPSpeculatorWorker(**draft_worker_kwargs)
        elif draft_model_config.hf_config.model_type == "medusa":
            proposer_worker = MedusaWorker(**draft_worker_kwargs)
        else:
            # Note: The current version of the MTP module doer not support
            # the use of TP1DraftModelRunner
            if draft_tp == 1 and draft_model_config.hf_config.model_type !=\
                    "deepseek_mtp":
                draft_worker_kwargs["model_runner_cls"] = TP1DraftModelRunner
            else:
                if draft_model_config.hf_config.model_type == "eagle":
                    raise NotImplementedError(
                        f"{draft_model_config.hf_config.model_type} "
                        "does not support TP > 1 yet")

                allow_zero_draft_token_step = False

            # Load lm_head weight for eagle in init_device
            if draft_model_config.hf_config.model_type == "eagle":
                enable_lm_head_weight_load = True

            proposer_worker = MultiStepWorker(**draft_worker_kwargs)
            if draft_model_config.hf_config.model_type == "deepseek_mtp":
                num_spec_prefill_steps = num_speculative_tokens

        proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
            proposer_worker, draft_tp, target_tp)

    logger.info("Configuring SpecDecodeWorker with proposer=%s",
                type(proposer_worker))

    spec_decode_sampler: SpecDecodeBaseSampler = None
    if draft_token_acceptance_method == "rejection_sampler":
        spec_decode_sampler = RejectionSampler()
    elif draft_token_acceptance_method == "typical_acceptance_sampler":
        spec_decode_sampler = TypicalAcceptanceSampler(
            posterior_threshold=\
                typical_acceptance_sampler_posterior_threshold,
            posterior_alpha=typical_acceptance_sampler_posterior_alpha,
        )
    logger.info(
        "[Speculative Decoding] Configuring"
        " SpecDecodeWorker with sampler=%s", type(spec_decode_sampler))

    if not disable_mqa_scorer:
        if scorer_worker.model_runner.attn_backend.get_name() != "FLASH_ATTN":
            disable_mqa_scorer = True
            logger.info("[Speculative Decoding] Disabling MQA scorer as the "
                        "MQA is only available with flash attn backend.")

        if draft_model_config and \
            draft_model_config.max_model_len < \
                scorer_worker.model_config.max_model_len:
            disable_mqa_scorer = True
            logger.info("[Speculative Decoding] Disabling MQA scorer as the "
                        "draft model max_model_len is smaller than the target "
                        "model max_model_len.")

        if not scorer_worker.model_runner.model_config.enforce_eager:
            disable_mqa_scorer = True
            logger.info("[Speculative Decoding] Disabling MQA scorer as the "
                        "target model is not running in eager mode.")

    return SpecDecodeWorker(
        proposer_worker,
        scorer_worker,
        disable_mqa_scorer=disable_mqa_scorer,
        disable_logprobs=disable_logprobs,
        disable_log_stats=disable_log_stats,
        disable_by_batch_size=disable_by_batch_size,
        spec_decode_sampler=spec_decode_sampler,
        allow_zero_draft_token_step=allow_zero_draft_token_step,
        enable_lm_head_weight_load=enable_lm_head_weight_load,
        num_spec_prefill_steps=num_spec_prefill_steps)


SpecDecodeWorker.create_worker = classmethod(create_worker)
