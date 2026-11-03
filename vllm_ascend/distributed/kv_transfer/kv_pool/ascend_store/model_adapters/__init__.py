# SPDX-License-Identifier: Apache-2.0
"""Model dispatch for state semantics; no transport or rank coordination."""

from ..state_store.cache_schema import register_cache
from .glm import describe_glm_caches, describe_key_policy


def model_state_key_policy(vllm_config):
    config = vllm_config.model_config.hf_text_config
    if config.model_type not in ("glm5_next", "glm5_next_text"):
        raise NotImplementedError(f"no accepted-state model adapter for {config.model_type}")
    return describe_key_policy(config, vllm_config.speculative_config)


def describe_model_registration(vllm_config, cache_config, caches, tp_size, tp_rank):
    """Describe framework-allocated members; do not rediscover model ownership."""
    from vllm.v1.kv_cache_interface import MambaSpec

    from ..layerwise.layerwise_cache_layout import get_layerwise_kv_cache_specs
    from ..state_store.cache_schema import PagedComponentSpec

    policy = model_state_key_policy(vllm_config)
    native_specs = get_layerwise_kv_cache_specs(cache_config)
    semantics = describe_glm_caches(native_specs, vllm_config.compilation_config.static_forward_context)
    caches = dict(caches)
    context = vllm_config.compilation_config.static_forward_context
    for name, spec in native_specs.items():
        if isinstance(spec, MambaSpec) and len(caches[name]) == 1:
            # Core may register the packed allocation; the loaded state layer
            # owns its typed views, including convolution lookahead/padding.
            caches[name] = tuple(context[name].kv_cache)
        elif isinstance(semantics[name], PagedComponentSpec):
            (page,) = caches[name]
            if page.ndim == 3:
                page = page.unsqueeze(2)
            elif page.ndim == 4 and page.shape[1] == 1 and page.shape[2] != 1:
                page = page.transpose(1, 2)
            caches[name] = (page,)
    registered = register_cache(cache_config, caches, semantics, tp_size=tp_size, tp_rank=tp_rank)
    return registered, policy
