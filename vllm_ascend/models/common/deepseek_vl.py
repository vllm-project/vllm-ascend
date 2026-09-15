# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-independent delegation for Ascend DeepSeek multimodal wrappers."""

from collections.abc import Iterable, Iterator

import torch
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def _vision_parameter_name(name: str) -> str | None:
    """Map a checkpoint vision tensor to the wrapper parameter namespace."""
    if name.startswith("model."):
        name = name.removeprefix("model.")
    if name.startswith(("vision.", "aligner.", "image_")):
        return name
    return None


class DeepseekMultiModalMixin:
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded_vision: set[str] = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                vision_name = _vision_parameter_name(name)
                if vision_name is None:
                    yield name, loaded_weight
                    continue
                if vision_name not in params:
                    raise KeyError(f"Vision weight {name!r} has no parameter {vision_name!r}.")
                param = params[vision_name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_vision.add(vision_name)

        loaded_language = self.language_model.load_weights(language_weights())
        return loaded_vision | {f"language_model.{name}" for name in loaded_language}

    def process_weights_after_loading(self) -> None:
        hook = getattr(
            self.language_model,
            "process_weights_after_loading",
            None,
        )
        if hook is not None:
            hook()
