import math

import torch
from vllm.config.lora import LoRAConfig
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping
from vllm.lora.lora_model import LoRAModel
from vllm.lora.model_manager import LoRAModelManager
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.lora.utils import (get_supported_lora_modules, is_moe_model,
                             process_packed_modules_mapping)
from vllm.model_executor.models import SupportsLoRA, supports_multimodal
from vllm.model_executor.models.interfaces import is_pooling_model


class AscendLoRAModelManager:
    """A manager that manages multiple LoRA-fine-tuned models."""

    def __init__(
        self,
        model: SupportsLoRA,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        lora_config: LoRAConfig,
        device: torch.device,
    ):
        """Create a LoRAModelManager and adapter for a given model.

        Args:
            model: the model to be adapted.
            max_num_seqs: the maximum number of sequences model can run in a
                single batch.
            max_num_batched_tokens: the maximum number of tokens model can run
                in a single batch.
            vocab_size: the vocab size of the model.
            lora_config: the LoRA configuration.
        """
        self.model: SupportsLoRA = model
        self._registered_adapters: dict[int, LoRAModel] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._active_adapters: dict[int, None] = {}
        self.adapter_type = "LoRA"
        self.lora_config = lora_config
        self.device = device
        self.max_num_seqs = max_num_seqs
        assert self.capacity >= self.lora_slots  # type:ignore
        self.max_num_batched_tokens = math.ceil(max_num_batched_tokens / 8) * 8
        self.lora_index_to_id: list[int | None] = [
            None
        ] * self.lora_slots  # type:ignore
        self.vocab_size = vocab_size
        self.punica_wrapper = get_punica_wrapper(
            max_num_batched_tokens,
            max_batches=self.max_num_seqs,
            device=self.device,
            lora_config=self.lora_config,
        )

        self.supported_lora_modules = get_supported_lora_modules(self.model)
        assert self.supported_lora_modules, "No supported LoRA modules found in"
        f" {self.model.__class__.__name__}."

        self.packed_modules_mapping = process_packed_modules_mapping(
            self.model)
        # Used to indicate whether the model is a multimodal model
        self.supports_mm: bool = (
            supports_multimodal(self.model)
            # In case the model only supports LoRA for
            # text modules (e.g. ChatGLM)
            and hasattr(self.model, "get_mm_mapping"))
        self.is_pooling_model = is_pooling_model(self.model)
        self.packed_modules: dict[str, list[str]] = {}
        self.modules: dict[str, BaseLayerWithLoRA] = {}
        # Dict instead of a set for compatibility with LRUCache.
        self._last_mapping: LoRAMapping | None = None
        self._is_3d_moe_model = is_moe_model(
            self.model) and self.model.is_3d_moe_weight
        self._create_lora_modules()  # type:ignore

        self.model.lora_manager = self


LoRAModelManager.__init__ = AscendLoRAModelManager.__init__
