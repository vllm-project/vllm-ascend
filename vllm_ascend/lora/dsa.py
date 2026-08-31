# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA

logger = init_logger(__name__)

DSA_LORA_CONTEXT_ATTR = "_ascend_dsa_lora_context"
_DSA_QKV_PROJECTIONS = frozenset(("wq_a", "wq_b", "wkv"))

DSAParallelMode = Literal["replicated", "column"]


class DSASGMVMetadataLike(Protocol):
    no_lora: bool
    op_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]


@dataclass(frozen=True)
class LoRAIntermediate:
    buffers: tuple[torch.Tensor, ...] | torch.Tensor | None
    sgmv_metadata: DSASGMVMetadataLike


@dataclass(frozen=True)
class DSALoRAContext:
    """LoRA state published on the original Linear cached by the DSA impl."""

    punica_wrapper: object
    lora_a_stacked: tuple[torch.Tensor, ...]
    lora_b_stacked: tuple[torch.Tensor, ...]
    output_slices: tuple[int, ...]
    parallel_mode: DSAParallelMode
    fully_sharded: bool
    tp_size: int
    tp_rank: int


def _is_direct_dsa_projection(source_layer: nn.Module, names: frozenset[str]) -> bool:
    prefix = getattr(source_layer, "prefix", "")
    parts = prefix.split(".")
    return len(parts) >= 2 and parts[-2] == "self_attn" and parts[-1] in names


def get_dsa_lora_context(linear: nn.Module) -> DSALoRAContext | None:
    return getattr(linear, DSA_LORA_CONTEXT_ATTR, None)


def has_dsa_lora(linear: nn.Module) -> bool:
    return get_dsa_lora_context(linear) is not None


class _AscendDSALoRAContextMixin:
    _dsa_parallel_mode: DSAParallelMode

    def set_mapping(self, punica_wrapper) -> None:
        super().set_mapping(punica_wrapper)
        prefix = getattr(self.base_layer, "prefix", "")
        projection_name = prefix.rsplit(".", 1)[-1]
        context = DSALoRAContext(
            punica_wrapper=punica_wrapper,
            lora_a_stacked=self.lora_a_stacked,
            lora_b_stacked=self.lora_b_stacked,
            output_slices=self.output_slices,
            parallel_mode=self._dsa_parallel_mode,
            fully_sharded=(self._dsa_parallel_mode == "column" and bool(self.lora_config.fully_sharded_loras)),
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        if projection_name in _DSA_QKV_PROJECTIONS:
            # Lets the model runner avoid preparing DSA SGMV metadata when a
            # DSA model enables LoRA only for unrelated modules or o-proj.
            punica_wrapper.has_dsa_qkv_lora = True
        # DSAAttentionImpl and CVLinearWrapper keep the original Linear object,
        # while the model manager replaces the module-tree aliases with `self`.
        # Publishing tensor references in a non-Module context makes both views
        # share adapter stacks without registering a module cycle.
        setattr(self.base_layer, DSA_LORA_CONTEXT_ATTR, context)
        logger.debug(
            "Published DeepSeek V4 DSA LoRA context: layer=%s mode=%s "
            "fully_sharded=%s lora_a_shapes=%s lora_b_shapes=%s",
            getattr(self.base_layer, "prefix", type(self.base_layer).__name__),
            context.parallel_mode,
            context.fully_sharded,
            tuple(tuple(weight.shape) for weight in context.lora_a_stacked),
            tuple(tuple(weight.shape) for weight in context.lora_b_stacked),
        )


class AscendDSAReplicatedLinearWithLoRA(_AscendDSALoRAContextMixin, ReplicatedLinearWithLoRA):
    _dsa_parallel_mode: DSAParallelMode = "replicated"

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return _is_direct_dsa_projection(source_layer, _DSA_QKV_PROJECTIONS) and (
            ReplicatedLinearWithLoRA.can_replace_layer(
                source_layer=source_layer,
                lora_config=lora_config,
                packed_modules_list=packed_modules_list,
                model_config=model_config,
            )
        )


class AscendDSAColumnParallelLinearWithLoRA(_AscendDSALoRAContextMixin, ColumnParallelLinearWithLoRA):
    _dsa_parallel_mode: DSAParallelMode = "column"

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return _is_direct_dsa_projection(source_layer, frozenset(("wq_b",))) and (
            ColumnParallelLinearWithLoRA.can_replace_layer(
                source_layer=source_layer,
                lora_config=lora_config,
                packed_modules_list=packed_modules_list,
                model_config=model_config,
            )
        )


class AscendDSAColumnParallelLinearWithShardedLoRA(_AscendDSALoRAContextMixin, ColumnParallelLinearWithShardedLoRA):
    _dsa_parallel_mode: DSAParallelMode = "column"

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        return _is_direct_dsa_projection(source_layer, frozenset(("wq_b",))) and (
            ColumnParallelLinearWithShardedLoRA.can_replace_layer(
                source_layer=source_layer,
                lora_config=lora_config,
                packed_modules_list=packed_modules_list,
                model_config=model_config,
            )
        )


DSA_LORA_CLASSES: tuple[type[BaseLinearLayerWithLoRA], ...] = (
    AscendDSAColumnParallelLinearWithLoRA,
    AscendDSAColumnParallelLinearWithShardedLoRA,
    AscendDSAReplicatedLinearWithLoRA,
)


def _allocate_shrink_buffer(
    context: DSALoRAContext,
    *,
    num_slices: int,
    num_rows: int,
    rank: int,
) -> torch.Tensor:
    """Allocate a fresh zeroed FP32 scratch buffer for one DSA shrink."""

    return torch.zeros(
        (num_slices, num_rows, rank),
        dtype=torch.float32,
        device=context.lora_a_stacked[0].device,
    )


def prepare_dsa_lora(
    linear: nn.Module,
    x: torch.Tensor,
    token_lora_indices: torch.Tensor | None = None,
) -> LoRAIntermediate | None:
    """Run the adapter A projection while the CV base projection is split."""

    context = get_dsa_lora_context(linear)
    if context is None:
        return None
    if context.parallel_mode not in ("replicated", "column"):
        raise ValueError(f"CV LoRA does not support DSA mode {context.parallel_mode!r}.")

    x_2d = x.view(-1, x.shape[-1])
    if token_lora_indices is None:
        token_lora_indices = context.punica_wrapper.get_token_lora_indices(x_2d.shape[0])
    if token_lora_indices.shape[0] != x_2d.shape[0]:
        raise ValueError(
            f"DSA LoRA token mapping length mismatch: expected {x_2d.shape[0]}, got {token_lora_indices.shape[0]}."
        )
    sgmv_metadata = context.punica_wrapper.get_dsa_sgmv_metadata(token_lora_indices)
    if sgmv_metadata.no_lora:
        return LoRAIntermediate(None, sgmv_metadata)

    if context.fully_sharded:
        if context.parallel_mode != "column":
            raise ValueError("Only column-parallel DSA CV projections may shard LoRA A.")
        local_rank = context.lora_a_stacked[0].shape[-2]
        buffers = _allocate_shrink_buffer(
            context,
            num_slices=len(context.lora_a_stacked),
            num_rows=x_2d.shape[0],
            rank=local_rank,
        )
        context.punica_wrapper.add_shrink(
            buffers,
            x_2d,
            context.lora_a_stacked,
            1.0,
            sgmv_metadata=sgmv_metadata,
        )
        buffers = tensor_model_parallel_all_gather(buffers)
        return LoRAIntermediate(buffers, sgmv_metadata)

    shrink_buffer = _allocate_shrink_buffer(
        context,
        num_slices=len(context.lora_a_stacked),
        num_rows=x_2d.shape[0],
        rank=context.lora_a_stacked[0].shape[-2],
    )
    buffers = tuple(shrink_buffer[slice_idx] for slice_idx in range(len(context.lora_a_stacked)))
    context.punica_wrapper.add_shrink(
        buffers,
        x_2d,
        context.lora_a_stacked,
        1.0,
        sgmv_metadata=sgmv_metadata,
    )
    return LoRAIntermediate(buffers, sgmv_metadata)


def apply_prepared_dsa_lora(
    linear: nn.Module,
    output: torch.Tensor,
    intermediate: LoRAIntermediate | None,
) -> torch.Tensor:
    """Add a previously prepared adapter B projection to a CV base result."""

    context = get_dsa_lora_context(linear)
    if context is None:
        return output
    if intermediate is None:
        raise RuntimeError("DSA LoRA context exists but its A projection was not prepared.")
    if intermediate.buffers is None:
        return output

    context.punica_wrapper.add_expand(
        output,
        intermediate.buffers,
        context.lora_b_stacked,
        context.output_slices,
        offset_start=0,
        add_inputs=True,
        sgmv_metadata=intermediate.sgmv_metadata,
    )
    return output
