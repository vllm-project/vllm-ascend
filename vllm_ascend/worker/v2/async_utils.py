# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import torch
from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class NonFinalPrefillSamplerOutput(SamplerOutput):
    """Sampler output whose requests are all known to produce zero tokens."""


class AscendAsyncOutput(AsyncOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool = False,
        routed_experts: RoutedExpertsTensors | None = None,
    ):
        # Token counts for a non-final prompt chunk are known on the CPU.
        # Other device outputs still need the upstream D2H completion path.
        self._skip_output_copy = (
            isinstance(sampler_output, NonFinalPrefillSamplerOutput)
            and sampler_output.logprobs_tensors is None
            and sampler_output.num_nans is None
            and sampler_output.sampling_mask_tensors is None
            and routed_experts is None
            and not check_ep_fault
            and all(value is None for value in model_runner_output.prompt_logprobs_dict.values())
        )
        if self._skip_output_copy:
            self.model_runner_output = model_runner_output
            # Keep the same tensor ownership as AsyncOutput. Sampling and
            # postprocessing still run on the main stream; no copy stream uses
            # these tensors on this path.
            self.sampler_output = sampler_output
            self.num_sampled_tokens = num_sampled_tokens
            self.routed_experts = routed_experts
        else:
            super().__init__(
                model_runner_output,
                sampler_output,
                num_sampled_tokens,
                main_stream,
                copy_stream,
                check_ep_fault=check_ep_fault,
                routed_experts=routed_experts,
            )

    def get_output(self) -> ModelRunnerOutput:
        if not self._skip_output_copy:
            return super().get_output()

        # Preserve the request order and connector outputs populated by the
        # runner after construction. This acknowledges an empty token result,
        # not device/KV-transfer completion.
        self.model_runner_output.sampled_token_ids = [[] for _ in self.model_runner_output.req_ids]
        return self.model_runner_output
