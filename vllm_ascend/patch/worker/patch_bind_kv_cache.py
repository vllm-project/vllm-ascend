from collections.abc import Sequence

import torch
import vllm.v1.worker.utils as utils
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.worker.utils import defaultdict, extract_layer_index

from vllm_ascend.utils import vllm_version_is

KVCache = torch.Tensor | Sequence[torch.Tensor]


def _bind_layer_kv_cache(
    layer: AttentionLayerBase,
    kv_cache: KVCache,
) -> None:
    """Bind either a canonical cache page or materialized Mamba states.

    Main's ``MambaBase.bind_kv_cache`` unpacks one canonical int8 page whose
    states are interleaved within each block.  Ascend layouts may instead
    materialize each state as a separate cross-block contiguous view.  Those
    views cannot be repacked without either a copy or losing their layout, so
    validate and bind them directly.  Canonical tensors, including ordinary
    packed Mamba pages and QSA cache pages, continue through the layer hook.
    """
    if not isinstance(layer, MambaBase) or isinstance(kv_cache, torch.Tensor):
        layer.bind_kv_cache(kv_cache)  # type: ignore[arg-type]
        return

    expected_shapes = tuple(tuple(shape) for shape in layer.get_state_shape())
    expected_dtypes = tuple(layer.get_state_dtype())
    if len(kv_cache) != len(expected_shapes):
        raise ValueError(
            f"Materialized Mamba cache state count mismatch: got {len(kv_cache)}, expected {len(expected_shapes)}."
        )
    if len(expected_shapes) != len(expected_dtypes):
        raise ValueError(
            "Mamba layer reports different state shape and dtype counts: "
            f"{len(expected_shapes)} shapes, {len(expected_dtypes)} dtypes."
        )

    states: list[torch.Tensor] = []
    num_blocks: int | None = None
    for state_idx, (state, shape, dtype) in enumerate(zip(kv_cache, expected_shapes, expected_dtypes, strict=True)):
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"Materialized Mamba cache states must be tensors; state {state_idx} is {type(state).__name__}."
            )
        if tuple(state.shape[1:]) != shape:
            raise ValueError(
                f"Materialized Mamba cache state {state_idx} shape mismatch: "
                f"got {tuple(state.shape)}, expected (num_blocks, {shape})."
            )
        if state.dtype != dtype:
            raise TypeError(
                f"Materialized Mamba cache state {state_idx} dtype mismatch: got {state.dtype}, expected {dtype}."
            )
        if num_blocks is None:
            num_blocks = state.shape[0]
        elif state.shape[0] != num_blocks:
            raise ValueError(
                "Materialized Mamba cache states have different block counts: "
                f"state 0 has {num_blocks}, state {state_idx} has "
                f"{state.shape[0]}."
            )
        states.append(state)

    layer.kv_cache = tuple(states)


# Ascend keeps a platform-specific runner-cache ordering, but every cache layer
# must still receive its allocation through ``bind_kv_cache``.  The layer hook
# is what turns the canonical allocation into the runtime views required by
# specialized caches such as Mamba and QSA.
def bind_kv_cache(
    kv_caches: dict[str, KVCache],
    forward_context: dict[str, AttentionLayerBase],
    runner_kv_caches: list[KVCache],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention
            layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

    ordered_layer_names: list[str] = []
    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        # remove some codes for the typical case of encoder-decoder model, e.g., bart.
        for layer_name in layer_names:
            runner_kv_caches.append(kv_caches[layer_name])
            ordered_layer_names.append(layer_name)

    # Bind through the layer hook. Main's cache layers use this hook to unpack
    # the canonical allocation into their runtime views (for example QSA
    # creates key_cache/rope_position_cache and Mamba separates its states).
    for layer_name, kv_cache in kv_caches.items():
        _bind_layer_kv_cache(forward_context[layer_name], kv_cache)

    # vLLM #52506 adds ReplaySSM ring trackers on main. v0.29.0 predates
    # that contract and has no tracker helper to invoke.
    if not vllm_version_is("0.29.0"):
        utils.share_replayssm_ring_trackers(ordered_layer_names, forward_context, kv_cache_groups)


utils.bind_kv_cache = bind_kv_cache
