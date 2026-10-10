# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin subclass of DefaultModelLoader for the bigtensorloader format."""

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class BigTensorDefaultLoader(DefaultModelLoader):
    """DefaultModelLoader that treats 'bigtensorloader' like 'safetensors'."""

    def _prepare_weights(self, *args, **kwargs):
        # Masquerade as safetensors so upstream sets allow_patterns.
        # Signature-agnostic passthrough (*args, **kwargs): upstream changes
        # this method's positional signature across releases.
        load_config = self.load_config
        original_format = load_config.load_format
        if original_format == "bigtensorloader":
            load_config.load_format = "safetensors"
        try:
            return super()._prepare_weights(*args, **kwargs)
        finally:
            load_config.load_format = original_format
