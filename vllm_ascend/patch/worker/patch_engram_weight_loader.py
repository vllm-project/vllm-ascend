# SPDX-License-Identifier: Apache-2.0
"""Read an Engram owner's primary checkpoint with a per-tensor lazy filter."""

import time
from collections.abc import Iterator

import torch
from safetensors import safe_open
from torch import nn
from tqdm.auto import tqdm
from vllm.config import ModelConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.weight_utils import (
    _BAR_FORMAT,
    _natural_sort_key,
    enable_tqdm,
    filter_mm_encoder_only_safetensors_files,
    filter_safetensors_files_by_weight_name,
)

_original_get_all_weights = DefaultModelLoader.get_all_weights


def _get_all_weights_with_node_shared_engram(
    self: DefaultModelLoader,
    model_config: ModelConfig,
    model: nn.Module,
) -> Iterator[tuple[str, torch.Tensor]]:
    if not getattr(model, "requires_lazy_checkpoint_loading", False):
        yield from _original_get_all_weights(self, model_config, model)
        return

    is_unused_weight = getattr(model, "is_unused_checkpoint_weight", None)
    if not callable(is_unused_weight):
        raise ValueError("Node-shared Engram requires a checkpoint weight filter")

    load_config = self.load_config
    if (
        load_config.load_format not in ("auto", "safetensors")
        or load_config.safetensors_load_strategy not in (None, "lazy")
        or load_config.model_loader_extra_config.get("enable_multithread_load")
    ):
        raise ValueError("Node-shared Engram requires default lazy safetensors loading")

    folder, shards, use_safetensors, index_file = self._prepare_weights(
        model_config.model,
        subfolder=None,
        revision=model_config.revision,
        fall_back_to_pt=False,
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    if not use_safetensors:
        raise ValueError("Node-shared Engram requires a safetensors checkpoint")

    if self._encoder_only_lm_prefixes is not None and shards:
        shards = filter_mm_encoder_only_safetensors_files(
            shards,
            folder,
            index_file,
            self._encoder_only_lm_prefixes,
            weights_mapper=self._encoder_only_weights_mapper,
        )
        if not shards:
            raise RuntimeError(
                f"mm-encoder-only shard filter removed all weights for {model_config.model!r}; "
                f"check language_model prefixes {self._encoder_only_lm_prefixes}"
            )
    shards = filter_safetensors_files_by_weight_name(shards, is_unused_weight)

    if self.counter_before_loading_weights == 0.0:
        self.counter_before_loading_weights = time.perf_counter()

    # A mixed shard must stay lazy all the way to get_tensor. In particular,
    # auto mode must not prefetch the whole file containing the owner's table.
    for shard in tqdm(
        sorted(shards, key=_natural_sort_key),
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm(load_config.use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(shard, framework="pt") as tensors:
            for name in tensors.keys():  # noqa: SIM118
                if is_unused_weight(name) or should_skip_weight(name, self.local_expert_ids):
                    continue
                yield name, tensors.get_tensor(name)

    for source in getattr(model, "secondary_weights", ()):
        yield from self._get_weights_iterator(source)


DefaultModelLoader.get_all_weights = _get_all_weights_with_node_shared_engram
