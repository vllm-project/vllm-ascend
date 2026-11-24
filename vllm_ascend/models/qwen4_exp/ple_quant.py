# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ModelSlim group-32 INT8 and MXFP8 PLE tables.

Keep the table in checkpoint precision and dequantize only selected rows.
INT8 stores FP32 scales; MXFP8 stores unsigned E8M0 scale bytes.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

PLE_GROUP_SIZE = 32
E8M0_BIAS = 127
E8M0_NAN = 255


class AscendPLEQuantizedEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Selected-row lookup for a group-scaled ModelSlim PLE checkpoint."""

    def __init__(self, *, mxfp8: bool) -> None:
        self.mxfp8 = mxfp8
        self.weight_dtype = torch.float8_e4m3fn if mxfp8 else torch.int8
        self.scale_dtype = torch.uint8 if mxfp8 else torch.float32
        self.scale_suffix = ".weight_scale" if mxfp8 else ".scale"

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if layer.supports_prefetch:
            raise NotImplementedError(
                "ModelSlim PLE tables do not support upstream pinned-host prefetch; "
                "use Ascend quantized host lookup or disable Engram CPU offload"
            )
        width = input_size_per_partition
        if width % PLE_GROUP_SIZE:
            raise ValueError(f"PLE embedding width {width} must be divisible by {PLE_GROUP_SIZE}")
        rows = sum(output_partition_sizes)
        for name, columns, dtype in (
            ("weight", width, self.weight_dtype),
            ("weight_scale", width // PLE_GROUP_SIZE, self.scale_dtype),
        ):
            parameter = nn.Parameter(layer.allocate_embedding_weight(rows, columns, dtype), requires_grad=False)
            set_weight_attrs(parameter, {"input_dim": 1, "output_dim": 0})
            set_weight_attrs(parameter, extra_weight_attrs)
            layer.register_parameter(name, parameter)
        layer.params_dtype = params_dtype
        # Per-layer state spans streaming load_weights calls, never requests.
        layer._ple_seen_scale_shards = set()
        layer._ple_loaded_scale_shards = set()

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if self.mxfp8:
            # FP8 embedding lookup is not implemented by every torch backend.
            codes = F.embedding(input_, layer.weight.view(torch.uint8)).view(self.weight_dtype).float()
            exponents = F.embedding(input_, layer.weight_scale).int()
            scales = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents - E8M0_BIAS)
            scales = scales.masked_fill(exponents == E8M0_NAN, float("nan"))
        else:
            codes = F.embedding(input_, layer.weight).float()
            scales = F.embedding(input_, layer.weight_scale)
        grouped = codes.unflatten(-1, (-1, PLE_GROUP_SIZE))
        return (grouped * scales.unsqueeze(-1)).flatten(-2).to(layer.params_dtype)

    def dequantize(self, layer: nn.Module, embeddings: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        # embedding()/host lookup already applied the selected rows' scales.
        return embeddings.to(output_dtype)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        expected = getattr(layer, "_ple_expected_scale_shards", None)
        # Dummy loading does not consume a checkpoint iterator. The streaming
        # loader initializes expected even for an empty iterator, so an actual
        # checkpoint with missing scales still fails the check below.
        if expected is None:
            return
        loaded = layer._ple_loaded_scale_shards
        if loaded != expected:
            raise ValueError(f"Missing PLE scale shards after loading: {sorted(expected - loaded)}")


def modelslim_ple_method(
    descriptions: dict[str, Any], prefix: str, embedding_dtype: str | None
) -> Qwen4ExpPLEEmbeddingMethod:
    """Resolve checkpoint markers without treating MXFP8 as globally scaled FP8."""
    shard_prefix = f"{prefix}.shard_"
    markers = {
        value for key, value in descriptions.items() if key.startswith(shard_prefix) and key.endswith(".quant_type")
    }
    weights = {value for key, value in descriptions.items() if key.startswith(shard_prefix) and key.endswith(".weight")}
    if markers or (weights and weights != {"FLOAT"}):
        if embedding_dtype not in (None, "float8_e4m3fn"):
            raise ValueError(f"Conflicting PLE embedding dtype: {embedding_dtype}")
        if weights == {"W8A8_MXFP8"} and not markers:
            return AscendPLEQuantizedEmbeddingMethod(mxfp8=True)
        if markers == {"INT8_SYM"} and not weights and embedding_dtype is None:
            return AscendPLEQuantizedEmbeddingMethod(mxfp8=False)
        raise ValueError(f"Unsupported or mixed PLE formats for {prefix}: weights={weights}, markers={markers}")
    if embedding_dtype is not None:
        raise ValueError(f"ModelSlim PLE embedding dtype {embedding_dtype} has no matching checkpoint markers")
    return Qwen4ExpPLEUnquantizedEmbeddingMethod()


def load_ple_quantized_weights(
    owner: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    original_load_weights: Callable,
) -> set[str]:
    """Load scales with the same checkpoint-to-ETP row mapping as codes."""
    embedding = owner.ngram_embedding
    method = embedding.embedding_method
    if not isinstance(method, AscendPLEQuantizedEmbeddingMethod):
        return original_load_weights(owner, weights)
    parts = owner.split_ngram_parts
    shard_size = (embedding.org_vocab_size + parts - 1) // parts
    start = embedding.shard_indices.org_vocab_start_index
    end = min(embedding.shard_indices.org_vocab_end_index, embedding.org_vocab_size)
    embedding._ple_expected_scale_shards = {
        index for index in range(parts) if index * shard_size < end and min((index + 1) * shard_size, end) > start
    }

    def remaining_weights():
        prefix = "ngram_embedding.shard_"
        for name, value in weights:
            if not name.startswith(prefix) or not name.endswith((".weight", ".scale", ".weight_scale")):
                yield name, value
                continue
            shard_text, suffix = name[len(prefix) :].rsplit(".", 1)
            if not shard_text.isdigit() or int(shard_text) >= parts:
                raise ValueError(f"Invalid PLE shard name: {name}")
            index = int(shard_text)
            checkpoint_start = index * shard_size
            rows = max(0, min(shard_size, embedding.org_vocab_size - checkpoint_start))
            if suffix == "weight":
                if value.dtype != method.weight_dtype:
                    raise ValueError(f"PLE weight {name} expected {method.weight_dtype}, got {value.dtype}")
                yield name, value
                continue
            if f".{suffix}" != method.scale_suffix:
                raise ValueError(f"Unexpected PLE scale format: {name}; expected {method.scale_suffix}")
            shape = (rows, embedding.embedding_dim // PLE_GROUP_SIZE)
            if value.dtype != method.scale_dtype or tuple(value.shape) != shape:
                raise ValueError(
                    f"PLE scale {name} expected {method.scale_dtype} {shape}, got {value.dtype} {value.shape}"
                )
            if index in embedding._ple_seen_scale_shards:
                raise ValueError(f"Duplicate PLE scale shard: {index}")
            # Load this replica's scale shard with the same ETP row mapping as its codes.
            embedding.weight_scale.weight_loader(embedding.weight_scale, value, checkpoint_start=checkpoint_start)
            embedding._ple_seen_scale_shards.add(index)
            if index in embedding._ple_expected_scale_shards:
                embedding._ple_loaded_scale_shards.add(index)

    loaded = original_load_weights(owner, remaining_weights())
    loaded.add("ngram_embedding.weight_scale")
    return loaded


def patch_modelslim_ple() -> None:
    """Bridge ModelSlim into the upstream AMD PLE construction/load paths.

    Keep these worker-local patches until upstream exposes a PLE quantization
    registry. The registered Ascend host backend handles INT8 and MXFP8 storage.
    """
    from vllm.models.qwen4_exp.amd import ple_layer

    from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

    quant_attr = "_ascend_original_from_quant_config"
    if not hasattr(Qwen4ExpPLEEmbeddingMethod, quant_attr):
        original = Qwen4ExpPLEEmbeddingMethod.from_quant_config
        setattr(Qwen4ExpPLEEmbeddingMethod, quant_attr, original)

        def from_quant_config(quant_config, prefix, embedding_dtype=None):
            if isinstance(quant_config, AscendModelSlimConfig):
                return modelslim_ple_method(quant_config.quant_description, prefix, embedding_dtype)
            return original(quant_config, prefix, embedding_dtype)

        Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)

    ngram_cls = ple_layer.Qwen4ExpNGramEmbedding
    load_attr = "_ascend_original_load_weights"
    if not hasattr(ngram_cls, load_attr):
        original_loader = ngram_cls.load_weights
        setattr(ngram_cls, load_attr, original_loader)

        def load_weights(self, weights):
            return load_ple_quantized_weights(self, weights, original_loader)

        ngram_cls.load_weights = load_weights
