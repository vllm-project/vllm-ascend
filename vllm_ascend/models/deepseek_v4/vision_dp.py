# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Image-level encoder DP within the language model's TP group.

Weights stay replicated. Each image's ViT, aligner and permutation execute on
one rank; an all-gather restores the complete embedding tuple on every rank.
All TP ranks must enter with the same ordered, uncached image inputs, as in
vLLM's other batch-level encoder DP implementations.
"""

from itertools import accumulate

import torch
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.vision import get_load_balance_assignment


def run_dp_sharded_deepseek_v4_vision(
    vision: nn.Module,
    aligner: nn.Module,
    patches: torch.Tensor,
    vit_grid: torch.Tensor,
    llm_grid: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Balance whole images by patch count; gather final, reordered embeddings.

    Unlike the generic MRoPE helper, output lengths come from ``llm_grid``:
    DeepSeek V4's aligner pads odd grids before spatial merging, so integer
    division of the input patch count would truncate valid output rows.
    """
    if vit_grid.ndim != 2 or vit_grid.shape[1] != 2 or llm_grid.shape != vit_grid.shape:
        raise ValueError("vit_grid and llm_grid must have matching [num_images, 2] shapes.")
    if vit_grid.dtype not in (torch.int32, torch.int64) or llm_grid.dtype not in (torch.int32, torch.int64):
        raise ValueError("Image grids must contain integer dimensions.")
    if patches.ndim != 4 or perm.ndim != 1 or perm.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected patches [num_patches, 3, p, p] and a one-dimensional integer perm.")

    vit_shapes = vit_grid.tolist()
    llm_shapes = llm_grid.tolist()
    merge = aligner.downsample_ratio
    patch_counts: list[int] = []
    output_counts: list[int] = []
    for (n_vit_h, n_vit_w), (n_llm_h, n_llm_w) in zip(vit_shapes, llm_shapes, strict=True):
        if min(n_vit_h, n_vit_w, n_llm_h, n_llm_w) <= 0:
            raise ValueError("Image grid dimensions must be positive.")
        expected = ((n_vit_h + merge - 1) // merge, (n_vit_w + merge - 1) // merge)
        if (n_llm_h, n_llm_w) != expected:
            raise ValueError(f"llm_grid {(n_llm_h, n_llm_w)} does not match merged ViT grid {expected}.")
        patch_counts.append(n_vit_h * n_vit_w)
        output_counts.append(n_llm_h * n_llm_w)
    if patches.shape[0] != sum(patch_counts) or perm.numel() != sum(output_counts):
        raise ValueError("Patch/permutation lengths do not match the image grids.")
    if not patch_counts:
        return ()

    patch_offsets = [0, *accumulate(patch_counts)]
    output_offsets = [0, *accumulate(output_counts)]

    def encode_image(index: int) -> torch.Tensor:
        image_n_vit_h, image_n_vit_w = vit_shapes[index]
        image_patches = patches[patch_offsets[index] : patch_offsets[index + 1]].to(aligner.w1.weight.dtype)
        image_embeds = aligner(vision(image_patches, image_n_vit_h, image_n_vit_w), image_n_vit_h, image_n_vit_w)
        item_perm = perm[output_offsets[index] : output_offsets[index + 1]].to(image_embeds.device)
        return image_embeds[item_perm]

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return tuple(encode_image(index) for index in range(len(patch_counts)))
    tp_rank = get_tensor_model_parallel_rank()
    image_order, samples_per_rank, _ = get_load_balance_assignment(patch_counts, tp_size)
    sample_offsets = [0, *accumulate(samples_per_rank)]
    rank_images = [image_order[sample_offsets[rank] : sample_offsets[rank + 1]] for rank in range(tp_size)]

    rows_per_rank = [sum(output_counts[index] for index in indices) for indices in rank_images]
    max_rows = max(rows_per_rank)

    # Empty ranks participate with the same shape/dtype/device, without running
    # dummy images through the encoder. Zero padding is discarded after gather.
    local_embeds = aligner.w2.weight.new_zeros((max_rows, aligner.w2.out_features))
    cursor = 0
    for index in rank_images[tp_rank]:
        rows = output_counts[index]
        local_embeds[cursor : cursor + rows].copy_(encode_image(index))
        cursor += rows
    gathered = tensor_model_parallel_all_gather(local_embeds, dim=0)

    ordered_embeds: dict[int, torch.Tensor] = {}
    for rank, indices in enumerate(rank_images):
        cursor = rank * max_rows
        for index in indices:
            rows = output_counts[index]
            # Encoder caches retain individual images. Do not let one cached
            # image keep the entire padded, all-rank gather allocation alive.
            ordered_embeds[index] = gathered[cursor : cursor + rows].clone()
            cursor += rows
    return tuple(ordered_embeds[index] for index in range(len(patch_counts)))
