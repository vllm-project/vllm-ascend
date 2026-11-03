# SPDX-License-Identifier: Apache-2.0
"""GLM model semantics for the common registration/transfer layer.

Only this adapter knows KDA projection ordering and the GLM indexer family.
Callers enumerate the loaded model, including its speculative layers, before
matching the resulting mapping against physical cache registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..state_store.cache_schema import (
    ComponentSpecs,
    PagedComponentSpec,
    PrivateComponentSpec,
    StateComponentSpec,
)
from ..state_store.state_keys import StateKeyPolicy

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from vllm_ascend.models.glm5next.config import Glm5NextConfig
    from vllm_ascend.models.glm5next.kda import Glm5NextLinearAttention


def describe_kda(layer: Glm5NextLinearAttention) -> tuple[StateComponentSpec, ...]:
    """Use the loaded layer's native state shape/dtype calculation."""
    conv_shape, recurrent_shape = layer.get_state_shape()
    conv_dtype, recurrent_dtype = layer.get_state_dtype()
    axis = 0 if layer._conv_state_dim_first else 1
    projection = layer.num_heads * layer.head_dim
    return (
        StateComponentSpec(
            "conv",
            (projection, projection, projection),
            axis,
            tuple(conv_shape),
            str(conv_dtype).removeprefix("torch."),
            wire_shard_axis=1,
            history_axis=1 - axis,
            history_rows=layer.conv_size - 1,
        ),
        StateComponentSpec(
            "recurrent",
            (layer.num_heads,),
            0,
            tuple(recurrent_shape),
            str(recurrent_dtype).removeprefix("torch."),
            wire_shard_axis=0,
        ),
    )


def describe_glm_page(spec: KVCacheSpec) -> PagedComponentSpec | PrivateComponentSpec:
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    if getattr(spec, "cache_role", None) == "indexer_tail":
        # The compressor reads old ring rows only before the current pool.
        # At a pool boundary every row of the first new pool comes from the
        # new query; subsequent chunks have already populated their own ring.
        return PrivateComponentSpec("indexer_tail", int(spec.compress_ratio), resume_without_history=True)
    version = getattr(spec, "model_version", None)
    if version != "glm5_next" and not (version is None and isinstance(spec, MLAAttentionSpec)):
        raise ValueError("cache spec is not a supported GLM replicated page")
    ratio = int(spec.tokens_per_state)
    return PagedComponentSpec("main" if ratio == 1 else "indexer", ratio)


def describe_glm_caches(
    native_specs: dict[str, KVCacheSpec],
    state_layers: dict[str, Glm5NextLinearAttention],
) -> dict[str, ComponentSpecs]:
    """Use specs enumerated by the framework from the complete loaded model.

    native_specs expands the framework cache groups, including MTP components
    discovered by the model runner. No layer-index arithmetic or group-position
    assumptions are used here.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    components = {}
    for name, spec in native_specs.items():
        if isinstance(spec, MambaSpec):
            if name not in state_layers:
                raise ValueError(f"missing loaded state layer for {name}")
            components[name] = describe_kda(state_layers[name])
        else:
            components[name] = describe_glm_page(spec)
    return components


def describe_key_policy(
    hf_config: Glm5NextConfig,
    speculative_config: SpeculativeConfig | None,
) -> StateKeyPolicy:
    """GLM's shifted MTP cache consumes token[B] at cache position B-1.

    Draft length does not change this dependency. Multiple predictor layers
    need an independently established cache/input contract before reuse.
    """
    if speculative_config is None or speculative_config.num_speculative_tokens == 0:
        return StateKeyPolicy("engine-prefix")
    if speculative_config.method not in ("mtp", "deepseek_mtp"):
        raise NotImplementedError("GLM state reuse has no key adapter for this draft method")
    if hf_config.num_nextn_predict_layers != 1:
        raise NotImplementedError("multi-layer GLM MTP state dependencies are not yet described")
    return StateKeyPolicy("glm-shifted-mtp-input-v1", extra_input_tokens=1)
