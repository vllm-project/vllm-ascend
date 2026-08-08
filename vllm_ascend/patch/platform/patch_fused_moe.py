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

# Patch vllm's FusedMoE factory to use AscendMoERunner by default.
#
# vllm's FusedMoE is a factory function (not a class). deepseek_v2 and other
# models do `from vllm.model_executor.layers.fused_moe import FusedMoE` and
# call it directly, so we must patch the binding in the package __init__ as
# well as the layer module before any model is imported.
#
# Import order in worker.__init__:
#   1. adapt_patch()  ->  this file runs  ->  FusedMoE patched
#   2. from vllm_ascend import ops
#   3. model loading  ->  deepseek_v2 imported  ->  gets patched FusedMoE  ✓

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import is_310p

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoE
_original_fused_moe_make_expert_params_mapping = _fused_moe_layer.fused_moe_make_expert_params_mapping

if is_310p():
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310 as _DefaultAscendMoERunner
else:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner as _DefaultAscendMoERunner


def _resolve_num_redundant_experts(upstream_redundancy: int) -> int:
    """Resolve one redundancy count for allocation and checkpoint loading."""
    eplb_config = get_ascend_config().eplb_config
    if not eplb_config.dynamic_eplb and eplb_config.expert_map_path is None:
        return upstream_redundancy

    configured_redundancy = eplb_config.num_redundant_experts
    if configured_redundancy and upstream_redundancy not in (0, configured_redundancy):
        raise ValueError(
            f"Conflicting EPLB redundant expert counts: vLLM={upstream_redundancy}, Ascend={configured_redundancy}."
        )
    return configured_redundancy or upstream_redundancy


def _ascend_FusedMoE(*args, runner_cls=None, runner_args=None, **kwargs):
    if runner_cls is None:
        runner_cls = _DefaultAscendMoERunner
    # RoutedExperts allocates its parameters before AscendMoERunner is
    # constructed. Propagate Ascend EPLB capacity into the upstream factory.
    eplb_config = get_ascend_config().eplb_config
    if eplb_config.dynamic_eplb or eplb_config.expert_map_path is not None:
        kwargs["enable_eplb"] = True
        kwargs["num_redundant_experts"] = _resolve_num_redundant_experts(kwargs.get("num_redundant_experts", 0))
    # 'hash' is a DeepSeek V4 flag already consumed before FusedMoE is called;
    # 'tid2eid' is Ascend-specific and must reach AscendMoERunner via runner_args.
    kwargs.pop("hash", None)
    tid2eid = kwargs.pop("tid2eid", None)
    if tid2eid is not None:
        runner_args = dict(runner_args) if runner_args is not None else {}
        runner_args["tid2eid"] = tid2eid
    return _original_FusedMoE(*args, runner_cls=runner_cls, runner_args=runner_args, **kwargs)


def _ascend_fused_moe_make_expert_params_mapping(
    model,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int = 0,
    routed_experts_prefix: str = "routed_experts",
):
    # Model implementations read redundancy from vLLM's EPLB config, while
    # Ascend dynamic EPLB is configured through additional_config. Use the
    # same effective count as _ascend_FusedMoE so every allocated redundant
    # physical slot receives the corresponding logical expert checkpoint.
    num_redundant_experts = _resolve_num_redundant_experts(num_redundant_experts)
    return _original_fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name,
        ckpt_down_proj_name,
        ckpt_up_proj_name,
        num_experts,
        num_redundant_experts,
        routed_experts_prefix,
    )


_fused_moe_layer.FusedMoE = _ascend_FusedMoE
_fused_moe_pkg.FusedMoE = _ascend_FusedMoE
_fused_moe_layer.fused_moe_make_expert_params_mapping = _ascend_fused_moe_make_expert_params_mapping
_fused_moe_pkg.fused_moe_make_expert_params_mapping = _ascend_fused_moe_make_expert_params_mapping
