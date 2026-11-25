# SPDX-License-Identifier: Apache-2.0
"""Reject stochastic DMTP requests before they reach a running worker."""

from functools import wraps

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams

_original_validate = SamplingParams._validate_spec_decode


@wraps(_original_validate)
def _validate_spec_decode(self, speculative_config):
    if getattr(speculative_config, "custom_method", None) == "dmtp" and self.temperature != 0:
        raise VLLMValidationError("DMTP supports greedy decoding only. Set temperature=0.")
    return _original_validate(self, speculative_config)


SamplingParams._validate_spec_decode = _validate_spec_decode
