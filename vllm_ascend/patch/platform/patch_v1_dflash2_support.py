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

from vllm.config.vllm import VllmConfig

DFLASH2_UNSUPPORTED_FEATURE = "dflash2 drafts"

_original_get_v1_unsupported_features = getattr(VllmConfig, "_get_v1_model_runner_unsupported_features", None)


def _patched_get_v1_model_runner_unsupported_features(self) -> list[str]:
    """Keep DFlash2 drafts available on the v1 model runner.

    Upstream rejects DFlash2 checkpoints on the v1 runner because its v1
    proposer never calls the candidate selector, so the draft would silently
    degrade to DFlash1. On Ascend the opposite holds: `get_spec_decode_method`
    routes DFlash2 checkpoints to `AscendDflash2Proposer`, which runs the
    selector, while the NPU v2 speculator has no DFlash2 path at all.
    """
    assert _original_get_v1_unsupported_features is not None
    unsupported = _original_get_v1_unsupported_features(self)
    if DFLASH2_UNSUPPORTED_FEATURE in unsupported:
        unsupported.remove(DFLASH2_UNSUPPORTED_FEATURE)
    return unsupported


# The gate only exists on vLLM versions that ship the v2 model runner split.
if _original_get_v1_unsupported_features is not None:
    VllmConfig._get_v1_model_runner_unsupported_features = _patched_get_v1_model_runner_unsupported_features
