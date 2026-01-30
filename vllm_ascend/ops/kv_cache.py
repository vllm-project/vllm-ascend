# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)


def _unified_kv_cache_update_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    attn_metadata = forward_context.attn_metadata

    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    layer_slot_mapping = slot_mapping.get(layer_name)

    if layer_slot_mapping is not None and key is not None and value is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )

        attn_layer.impl.do_kv_cache_update(
            key,
            value,
            kv_cache,
            attn_metadata,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def _unified_kv_cache_update_impl_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=_unified_kv_cache_update_impl,
    fake_impl=_unified_kv_cache_update_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
