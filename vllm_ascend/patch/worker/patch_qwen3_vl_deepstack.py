#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Patch DeepStack cross-layer buffer accessors to be thread-safe under
# ubatch overlap.
#
# Without this patch, _get_deepstack_input_embeds and
# _clear_deepstack_input_embeds always operate on [0:num_tokens], which is
# correct for single-stream forward but wrong under ubatch: each ubatch
# worker thread must read/clear the slice [start:stop] corresponding to its
# token_slice. We reuse the thread-local token_slice already bound by
# UBatchRuntimeManager.exec (the same mechanism used by
# get_cos_and_sin_slice in rotary embedding).

from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.ubatch_utils import get_ubatch_runtime_manager


def _ubatch_offsets() -> slice | None:
    # Return the token slice bound to the current ubatch worker thread, or
    # None when ubatch is disabled / caller is not on a worker thread.
    return get_ubatch_runtime_manager().get_current_token_slice()


def _patched_get_deepstack_input_embeds(self, num_tokens: int):
    if not getattr(self, "deepstack_input_embeds", None):
        return None

    token_slice = _ubatch_offsets()
    if token_slice is not None:
        start, stop = token_slice.start, token_slice.stop
    else:
        start, stop = 0, num_tokens

    return IntermediateTensors(
        {
            f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                start:stop
            ]
            for idx in range(self.deepstack_num_level)
        }
    )


def _patched_clear_deepstack_input_embeds(self, num_tokens: int) -> None:
    if not getattr(self, "deepstack_input_embeds", None):
        return

    token_slice = _ubatch_offsets()
    if token_slice is not None:
        start, stop = token_slice.start, token_slice.stop
    else:
        start, stop = 0, num_tokens

    if stop > start:
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][start:stop].zero_()


def _apply_patch(model_cls):
    model_cls._get_deepstack_input_embeds = _patched_get_deepstack_input_embeds
    model_cls._clear_deepstack_input_embeds = _patched_clear_deepstack_input_embeds
    # Marker checked by NPUModelRunner._ubatch_blocked_reason to lift the gate
    model_cls._ubatch_deepstack_patched = True


try:
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    _apply_patch(Qwen3VLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    _apply_patch(Qwen3OmniMoeThinkerForConditionalGeneration)
except ImportError:
    pass
