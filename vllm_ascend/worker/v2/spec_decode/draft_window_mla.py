# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own the mutable derived read inputs of an eager MLA draft invocation."""

from collections.abc import Collection, Mapping
from typing import Any


def freeze_mla_read_metadata(metadata: Mapping[str, Any], mla_layer_names: Collection[str]) -> tuple[Any, ...]:
    """Detach only MLA decode RoPE from the builder's process-wide scratch.

    ``AscendMLAMetadataBuilder.build_decode_metadata`` calls
    ``get_cos_and_sin_mla(use_cache=True)``. Its fixed-address tensors are
    overwritten by the next builder call, even when previous work remains
    in flight. Copy them before forward and keep the copies with the proposal
    completion owner. Shared layers keep one metadata object and one copy.

    Read tables/lengths already belong to the read view. Full slot mappings,
    query coordinates, absolute positions and metadata identity are preserved.
    Ordinary prefill RoPE is allocated by the builder. Its context consumer
    allocates per-chunk gather outputs rather than using the workspace field.
    """
    retained = []
    seen = set()
    for name in mla_layer_names:
        layer_metadata = metadata[name]
        if id(layer_metadata) in seen:
            continue
        seen.add(id(layer_metadata))
        if layer_metadata.external_flashmla is not None:
            raise NotImplementedError("Draft Window FlashMLA derived-state provider is not implemented")
        decode = layer_metadata.decode
        if decode is None:
            continue
        # The builder returns slices, so copy the actual logical extent only.
        for field in ("cos", "sin"):
            value = getattr(decode, field)
            if value is not None:
                owned = value.clone()
                setattr(decode, field, owned)
                retained.append(owned)
    return tuple(retained)
