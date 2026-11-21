# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin subclass of DefaultModelLoader for the bigtensorloader format."""

import copy

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def _copy_load_config(load_config: LoadConfig, load_format: str) -> LoadConfig:
    """Return a shallow copy of ``load_config`` with ``load_format`` replaced.

    The shared config object is never mutated: a temporary masquerade on it
    would be visible to concurrently loading draft/target models that share
    one LoadConfig. A shallow copy suffices (only the top-level attribute
    changes) and works for both the dataclass and the pydantic-model shapes
    ``LoadConfig`` takes across upstream releases.
    """
    copied = copy.copy(load_config)
    copied.load_format = load_format
    return copied


class BigTensorDefaultLoader(DefaultModelLoader):
    """DefaultModelLoader that treats 'bigtensorloader' like 'safetensors'.

    The fresh-load path expects a safetensors checkpoint (the format the
    snapshot round-trip is validated with); ``_prepare_weights`` masquerades
    as safetensors so upstream keeps setting its allow_patterns.
    """

    def _prepare_weights(self, *args, **kwargs):
        # Masquerade on a copied config handed to an independent loader --
        # never by mutating the shared LoadConfig, which draft/target models
        # can read concurrently off one config object. Signature-agnostic
        # passthrough (*args, **kwargs): upstream changes this method's
        # positional signature across releases.
        if self.load_config.load_format == "bigtensorloader":
            fallback_loader = DefaultModelLoader(_copy_load_config(self.load_config, "safetensors"))
            return fallback_loader._prepare_weights(*args, **kwargs)
        return super()._prepare_weights(*args, **kwargs)
