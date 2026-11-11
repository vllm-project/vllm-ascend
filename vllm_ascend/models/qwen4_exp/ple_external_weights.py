# SPDX-License-Identifier: Apache-2.0
"""Expose attached PLE checkpoint names to vLLM's per-tensor loader filter."""


class ExternalPLEWeightsMixin:
    """Filter only names belonging to PLE tables that have attached successfully."""

    @property
    def filter_unused_checkpoint_weights(self) -> bool:
        names = set()
        for module in self.modules():
            names.update(getattr(module, "ple_external_checkpoint_names", ()))
        self._ple_external_checkpoint_names = frozenset(names)
        if names:
            from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

            if "filter_unused_per_tensor" not in DefaultModelLoader.Source.__dataclass_fields__:
                raise RuntimeError("Shared PLE requires vLLM per-tensor checkpoint filtering")
        return bool(names)

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        names = getattr(self, "_ple_external_checkpoint_names", ())
        if name in names:
            return True
        parent = getattr(super(), "is_unused_checkpoint_weight", None)
        return bool(parent(name)) if parent is not None else False
