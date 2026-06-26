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

from collections.abc import Iterable

import torch
from vllm.model_executor.models.gpt_oss import GptOssModel


def _remap_gpt_oss_routed_expert_weight_names(
    model: torch.nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    params_dict = dict(model.named_parameters())
    for name, weight in weights:
        # vLLM PR #41184 moved FusedMoE expert parameters under
        # `mlp.experts.routed_experts.*`. Upstream GPT-OSS already remaps
        # this in the MXFP4 loader, but the generic/FP8
        # `_load_weights_other` path still indexes `params_dict` with
        # legacy `mlp.experts.*` names.
        if ".mlp.experts." in name and ".mlp.experts.routed_experts." not in name:
            routed_name = name.replace(".mlp.experts.", ".mlp.experts.routed_experts.", 1)
            if routed_name in params_dict:
                name = routed_name
        yield name, weight


def _patched_load_weights_other(
    self,
    ep_rank_end: int,
    ep_rank_start: int,
    heads_per_rank: int,
    head_start: int,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params_mapping: list[tuple[str, ...]],
) -> set[str]:
    # Call the class-saved original (not a module-level closure) so a second
    # import / importlib.reload does not wrap an already-patched method.
    return GptOssModel._ascend_original_load_weights_other(
        self,
        ep_rank_end,
        ep_rank_start,
        heads_per_rank,
        head_start,
        _remap_gpt_oss_routed_expert_weight_names(self, weights),
        stacked_params_mapping,
    )


# Idempotent guard: only capture the original and install the wrapper once.
if not hasattr(GptOssModel, "_ascend_original_load_weights_other"):
    GptOssModel._ascend_original_load_weights_other = GptOssModel._load_weights_other
    GptOssModel._load_weights_other = _patched_load_weights_other
