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

# Upstream vLLM (before #54196) validates logit_bias against the model
# vocabulary but validates stop_token_ids not at all and allowed_token_ids
# against len(tokenizer). Both fields end up as column indices into the
# logits tensor on the device side (min_tokens / EOS masking does
# logits.index_put_ on all_stop_token_ids; the allowed_token_ids mask in
# InputBatch is sized by model_config.get_vocab_size() and the ids are
# written as mask[req_index][allowed_token_ids]). For models whose
# tokenizer knows more ids than the language model (multimodal placeholder
# tokens), gap ids pass the tokenizer-based check yet still index out of
# bounds. On CANN 9.1.x the inserted IndexCheck kernel traps on the OOB
# index and the whole engine dies with an unrecoverable vector core
# exception; on older CANN the write silently corrupts logits of
# neighboring requests in the same batch (vllm-ascend issue #15200).
#
# Upstream fix: https://github.com/vllm-project/vllm/pull/54196
# Downstream issue: https://github.com/vllm-project/vllm-ascend/issues/15200


def _validate_stop_token_ids(self, model_config) -> None:
    """Validate stop_token_ids are within vocabulary range."""
    if not self.stop_token_ids:
        return

    vocab_size = model_config.get_vocab_size()
    invalid_token_ids = [token_id for token_id in self.stop_token_ids if token_id < 0 or token_id >= vocab_size]

    if invalid_token_ids:
        raise VLLMValidationError(
            f"token_id(s) {invalid_token_ids} in stop_token_ids contain "
            f"out-of-vocab token ids. Vocabulary size: {vocab_size}",
            parameter="stop_token_ids",
            value=invalid_token_ids,
        )


def _validate_allowed_token_ids(self, model_config) -> None:
    allowed_token_ids = self.allowed_token_ids
    if allowed_token_ids is None:
        return

    if len(allowed_token_ids) == 0:
        raise VLLMValidationError(
            "allowed_token_ids is not None and empty!",
            parameter="allowed_token_ids",
            value=allowed_token_ids,
        )

    vocab_size = model_config.get_vocab_size()
    invalid_token_ids = [token_id for token_id in allowed_token_ids if token_id < 0 or token_id >= vocab_size]
    if invalid_token_ids:
        raise VLLMValidationError(
            "allowed_token_ids contains out-of-vocab token id!",
            parameter="allowed_token_ids",
            value=invalid_token_ids,
        )


# Only patch when the bundled vLLM does not already contain the upstream
# fix. The guard checks both validators from vllm#54196: if the bundled vLLM
# ships either one, it ships both (they landed in the same commit), and the
# patch becomes a no-op that can be dropped.
if not hasattr(SamplingParams, "_validate_stop_token_ids"):
    SamplingParams._validate_stop_token_ids = _validate_stop_token_ids
    SamplingParams._validate_allowed_token_ids = _validate_allowed_token_ids

    _orig_verify = SamplingParams.verify

    def _verify(self, model_config, *args, **kwargs):
        _validate_stop_token_ids(self, model_config)
        _validate_allowed_token_ids(self, model_config)
        return _orig_verify(self, model_config, *args, **kwargs)

    SamplingParams.verify = _verify
