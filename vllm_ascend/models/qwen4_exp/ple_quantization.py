# SPDX-License-Identifier: Apache-2.0
"""Load ModelSlim INT8 PLE checkpoints into the BF16 host lookup table.

PLE storage is independent of the attention KV cache: only checkpoint loading
decodes these tables. Native INT8 GBSA continues to consume INT8 K/V at runtime.
"""

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import regex as re
import torch

_PLE_SHARD = re.compile(r"ngram_embedding\.shard_(\d+)\.(weight|scale)")
_DEQUANTIZE_ROWS = 16384


def _load_int8_shard(layer: Any, index: int, weight: torch.Tensor, scale: torch.Tensor) -> None:
    embedding = layer.ngram_embedding
    if not 0 <= index < layer.split_ngram_parts:
        raise ValueError(f"PLE shard index {index} exceeds split_ngram_parts={layer.split_ngram_parts}")
    shard_size = (embedding.org_vocab_size + layer.split_ngram_parts - 1) // layer.split_ngram_parts
    checkpoint_start = index * shard_size
    rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
    width = embedding.embedding_dim
    if weight.dtype != torch.int8 or tuple(weight.shape) != (rows, width):
        raise ValueError(f"PLE INT8 weight must have shape {(rows, width)}, got {weight.dtype} {tuple(weight.shape)}")
    if (
        scale.dtype != torch.float32
        or scale.ndim != 2
        or scale.shape[0] != rows
        or scale.shape[1] <= 0
        or width % scale.shape[1]
    ):
        raise ValueError("PLE INT8 scale must be FP32 [rows, groups] with groups dividing the embedding width")
    if weight.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError("PLE INT8 checkpoint tensors must be loaded on CPU")
    if not bool(torch.all(torch.isfinite(scale) & (scale > 0))):
        raise ValueError("PLE INT8 scales must be finite and positive")
    # Decode only this TP vocabulary intersection, in bounded CPU chunks. The
    # existing weight loader retains shared-host-table ownership/barrier rules.
    start = max(checkpoint_start, embedding.shard_indices.org_vocab_start_index)
    end = min(checkpoint_start + rows, embedding.shard_indices.org_vocab_end_index)
    groups = scale.shape[1]
    for row in range(start, end, _DEQUANTIZE_ROWS):
        source_start = row - checkpoint_start
        source_end = min(end, row + _DEQUANTIZE_ROWS) - checkpoint_start
        codes = weight[source_start:source_end].float().reshape(-1, groups, width // groups)
        decoded = (codes * scale[source_start:source_end, :, None]).reshape(-1, width).to(torch.bfloat16)
        embedding.weight.weight_loader(embedding.weight, decoded, checkpoint_start=row)


def load_modelslim_ple_weights(
    layer: Any,
    weights: Iterable[tuple[str, torch.Tensor]],
    original: Callable[..., set[str]],
) -> set[str]:
    """Pair split INT8 weights/scales, preserving upstream BF16/buffer loading."""
    quantized_loaded: set[str] = set()

    def regular_weights() -> Iterator[tuple[str, torch.Tensor]]:
        pending: dict[int, dict[str, torch.Tensor]] = {}
        for name, tensor in weights:
            match = _PLE_SHARD.fullmatch(name)
            if match is None or (match[2] == "weight" and tensor.dtype != torch.int8):
                yield name, tensor
                continue
            index, kind = int(match[1]), match[2]
            pair = pending.setdefault(index, {})
            if kind in pair:
                raise ValueError(f"Duplicate PLE INT8 checkpoint tensor: {name}")
            pair[kind] = tensor
            if "weight" in pair and "scale" in pair:
                _load_int8_shard(layer, index, pair["weight"], pair["scale"])
                quantized_loaded.add("ngram_embedding.weight")
                del pending[index]
        if pending:
            raise ValueError(f"Unpaired PLE INT8 weight/scale shards: {sorted(pending)}")

    loaded = set(original(layer, regular_weights()))
    return loaded | quantized_loaded
