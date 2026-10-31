# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt boundary rules shared by direct and pooled P/D handoffs."""

from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs


def unique_group_specs(group: Any) -> list[Any]:
    if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
        return [group.kv_cache_spec]
    specs = []
    for layer in group.layer_names:
        spec = group.kv_cache_spec.kv_cache_specs[layer]
        if spec not in specs:
            specs.append(spec)
    return specs


def needs_prefill_token_truncation(model_config: Any, group_specs: list[list[Any]]) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    compress_ratios = getattr(hf_config, "compress_ratios", None)
    uses_compressed_cache = isinstance(compress_ratios, (list, tuple, dict))
    return uses_compressed_cache or any(isinstance(spec, MambaSpec) for specs in group_specs for spec in specs)


def truncate_request_for_prefill(request: Any) -> None:
    """Call only on P, once, before its local prefix lookup."""
    params = request.kv_transfer_params
    if params is None or params.get("_p_side_truncated") or request.num_prompt_tokens <= 1:
        return
    if request.prompt_token_ids is not None:
        request.prompt_token_ids.pop()
    elif request.prompt_embeds is not None:
        request.prompt_embeds = request.prompt_embeds[:-1]
    else:
        return
    request._all_token_ids.pop()
    request.num_prompt_tokens -= 1
    request.max_tokens = 1
    params["_p_side_truncated"] = True
