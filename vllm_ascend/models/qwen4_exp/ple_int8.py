# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ascend PLE INT8 table support for Qwen4Exp ModelSlim checkpoints."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.qwen4_exp.common import ple as ple_utils
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbeddingMethod

PLE_INT8_GROUP_SIZE = 32
PLE_INT8_QUANT_TYPE = "INT8_SYM"


class AscendQwen4ExpPLEInt8EmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Keep PLE tables INT8 and dequantize only rows selected by n-gram IDs."""

    def create_weights(
        self,
        layer,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        if layer.supports_prefetch:
            raise NotImplementedError(
                "Qwen4Exp PLE INT8 does not yet support pinned-host prefetch; "
                "disable Engram CPU offload for this checkpoint"
            )
        width = input_size_per_partition
        if width % PLE_INT8_GROUP_SIZE:
            raise ValueError(f"PLE INT8 width {width} is not divisible by {PLE_INT8_GROUP_SIZE}")
        rows = sum(output_partition_sizes)
        weight = nn.Parameter(layer.allocate_embedding_weight(rows, width, torch.int8), requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter("weight", weight)
        scale = nn.Parameter(
            layer.allocate_embedding_weight(rows, width // PLE_INT8_GROUP_SIZE, torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(scale, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(scale, extra_weight_attrs)
        layer.register_parameter("weight_scale", scale)
        layer.ple_int8_group_size = PLE_INT8_GROUP_SIZE
        layer.params_dtype = params_dtype

    def process_weights_after_loading(self, layer) -> None:
        expected = getattr(layer, "_ple_expected_scale_shards", None)
        loaded: set[int] = getattr(layer, "_ple_loaded_scale_shards", set())
        if expected is None or loaded != expected:
            missing = sorted((expected or set()) - loaded)
            raise ValueError(f"Missing PLE scale shards after loading: {missing}")

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        weight = F.embedding(input_, layer.weight)
        scale = F.embedding(input_, layer.weight_scale)
        width = weight.shape[-1]
        grouped = weight.float().reshape(*weight.shape[:-1], width // PLE_INT8_GROUP_SIZE, PLE_INT8_GROUP_SIZE)
        restored = grouped * scale.float().unsqueeze(-1)
        return restored.reshape(*weight.shape).to(layer.params_dtype)

    def dequantize(self, layer, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        del layer
        return embeddings.to(output_dtype)


def load_ple_int8_scales(self, weights, original_load_weights):
    """Map checkpoint shard_N.scale rows to the same TP range as shard_N.weight."""
    embedding = self.ngram_embedding
    seen_scale_shards: set[int] = set()
    loaded_scale_shards: set[int] | None = getattr(embedding, "_ple_loaded_scale_shards", None)
    if loaded_scale_shards is None:
        loaded_scale_shards = set()
        embedding._ple_loaded_scale_shards = loaded_scale_shards

    def remaining_weights():
        prefix = "ngram_embedding.shard_"
        for name, value in weights:
            if not (name.startswith(prefix) and name.endswith(".scale")):
                yield name, value
                continue
            if not isinstance(embedding.embedding_method, AscendQwen4ExpPLEInt8EmbeddingMethod):
                raise ValueError(f"Unexpected INT8 PLE scale in non-INT8 embedding: {name}")
            shard_text = name[len(prefix) : -len(".scale")]
            if not shard_text.isdigit():
                raise ValueError(f"Invalid PLE scale shard name: {name}")
            shard_index = int(shard_text)
            if shard_index >= self.split_ngram_parts:
                raise ValueError(f"PLE scale shard index {shard_index} out of range")
            if shard_index in seen_scale_shards:
                raise ValueError(f"Duplicate PLE scale shard {shard_index}")
            shard_size = (embedding.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
            checkpoint_start = shard_index * shard_size
            expected_rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
            expected_shape = (expected_rows, embedding.embedding_dim // PLE_INT8_GROUP_SIZE)
            if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"PLE scale {name} expected FP32 {expected_shape}, got {value.dtype} {tuple(value.shape)}"
                )
            # vLLM main moved the helper onto the embedding in #61454;
            # v0.32.0 still exposes the module-level function.
            copy_shard = getattr(embedding, "_copy_embedding_shard_", None)
            if copy_shard is not None:
                copied = copy_shard(embedding.weight_scale, value, checkpoint_start=checkpoint_start)
            else:
                copied = ple_utils.copy_ple_embedding_shard_(
                    embedding.weight_scale,
                    value,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
            seen_scale_shards.add(shard_index)
            if copied:
                if shard_index in loaded_scale_shards:
                    raise ValueError(f"Duplicate PLE scale shard across batches: {shard_index}")
                loaded_scale_shards.add(shard_index)

    loaded = original_load_weights(self, remaining_weights())
    if isinstance(embedding.embedding_method, AscendQwen4ExpPLEInt8EmbeddingMethod):
        shard_size = (embedding.org_vocab_size + self.split_ngram_parts - 1) // self.split_ngram_parts
        tp_start = embedding.shard_indices.org_vocab_start_index
        tp_end = embedding.shard_indices.org_vocab_end_index
        expected = {
            index
            for index in range(self.split_ngram_parts)
            if index * shard_size < tp_end and (index + 1) * shard_size > tp_start
        }
        embedding._ple_expected_scale_shards = expected
        loaded.add("ngram_embedding.weight_scale")
    return loaded
