# SPDX-License-Identifier: Apache-2.0
"""Process the last known input token through decode in a local PD reference."""

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class InputSuffix:
    """Force only the withheld input token; all answer tokens remain untouched."""

    def __init__(self, token: int):
        self.token = token

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if not output_ids:
            assert 0 <= self.token < logits.shape[-1]
            logits.fill_(float("-inf"))
            logits[self.token] = 0
        return logits


class InputSuffixProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams):
        token = (params.extra_args or {}).get("pd_input_suffix")
        if token is not None and (type(token) is not int or token < 0):
            raise ValueError("pd_input_suffix must be a nonnegative token ID")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        token = (params.extra_args or {}).get("pd_input_suffix")
        return None if token is None else InputSuffix(token)
