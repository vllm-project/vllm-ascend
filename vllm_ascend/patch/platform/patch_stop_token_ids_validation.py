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

from vllm.sampling_params import SamplingParams, VLLMValidationError

# Upstream vLLM (as of 0.23.0) validates logit_bias and allowed_token_ids
# against the vocabulary but not stop_token_ids. Out-of-vocab stop token ids
# flow into all_stop_token_ids and are later used as logits indices on the
# device side (min_tokens / EOS masking does logits.index_put_). On CANN
# 9.1.x the inserted IndexCheck kernel traps on the OOB index and the whole
# engine dies with an unrecoverable vector core exception; on older CANN the
# write silently corrupts logits of neighboring requests in the same batch.
#
# Upstream fix: https://github.com/vllm-project/vllm/pull/54196
# Downstream issue: https://github.com/vllm-project/vllm-ascend/issues/15200


def _validate_stop_token_ids(self, model_config) -> None:
    """Validate stop_token_ids are within vocabulary range."""
    stop_token_ids = self.stop_token_ids
    if not stop_token_ids:
        return

    vocab_size = model_config.get_vocab_size()
    invalid_token_ids = [token_id for token_id in stop_token_ids if token_id < 0 or token_id >= vocab_size]

    if invalid_token_ids:
        raise VLLMValidationError(
            f"token_id(s) {invalid_token_ids} in stop_token_ids contain "
            f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
            parameter="stop_token_ids",
            value=invalid_token_ids,
        )


# Only patch when the bundled vLLM does not already contain the upstream fix;
# once the upstream PR lands in a vLLM release used here, this patch becomes
# a no-op and can be dropped.
if not hasattr(SamplingParams, "_validate_stop_token_ids"):
    SamplingParams._validate_stop_token_ids = _validate_stop_token_ids

    _orig_verify = SamplingParams.verify

    def _verify(self, model_config, *args, **kwargs):
        _validate_stop_token_ids(self, model_config)
        return _orig_verify(self, model_config, *args, **kwargs)

    SamplingParams.verify = _verify
