import numpy as np
import torch
import torch.nn as nn
import itertools
from typing import Literal
import math
from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration, Qwen3MoeLLMForCausalLM
from functools import lru_cache
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig
)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageInputs, Qwen2_5_VLForConditionalGeneration
)
from torchvision.transforms.v2 import functional
from vllm.distributed import parallel_state, tensor_model_parallel_all_gather

def get_load_balance_assignment(
    sizes: list[int], num_gpus: int = 2,) -> tuple[list[int], list[int], list[int]]:
    """
    see https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L253 for details.
    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count
    
    # Sort indices by size (largest first for better load balancing)
    large_to_small_indices = sorted(range(n_samples), key=lambda i: sizes[i], reverse=True)

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        shuffle_indices.extend(gpu_assignments[gpu_id])
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)

def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list,
    *,
    rope_type: Literal["rope_3d", "rope_2d"],) -> tuple[torch.Tensor, ...]:
    """
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/vision.py#L322 for details.
    """
    grid_thw_list = grid_thw_list.tolist()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    tp_rank_local = parallel_state.get_tensor_model_parallel_rank()

    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = get_load_balance_assignment(
        patches_per_image, tp_size
    )

    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    image_idxs_local = image_to_tp_rank[cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]]
        
    # Get the pixel values for the local images based on the image_idxs_local
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]] for i in image_idxs_local]
        )
    else:
        # Handle case where this rank has no images
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
    else:
        embed_dim_reduction_factor = vision_model.spatial_merge_size * vision_model.spatial_merge_size

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]
        
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )  
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, torch.tensor(local_grid_thw_list))
        else:
            # Handle empty case
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            
    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work            
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (grouped_pixel_values_len[rank] // embed_dim_reduction_factor)
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [(patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[embed_start : embed_start + img_patches]
                embed_start += img_patches
            current_idx += count
    out_embeddings = tuple(embed for embed in original_order_embeddings if embed is not None)
    if len(out_embeddings) != len(original_order_embeddings):
        raise ValueError("Found unassigned embeddings")

    return torch.concat(out_embeddings)
            

class AscendQwen3VLMoeForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config: Qwen3VLMoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        if not multimodal_config.get_limit_per_prompt(
            "image"
        ) and not multimodal_config.get_limit_per_prompt("video"):
            self.visual = None
        else:
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        self.language_model = Qwen3MoeLLMForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )
        # Whether to include the gate_up_proj mapping is determined by
        # the language model.
        self.packed_modules_mapping = (
            self.packed_modules_mapping | self.language_model.packed_modules_mapping
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        # register buffer for deepstack
        if self.use_deepstack and self.visual is not None:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
        else:
            self.deepstack_input_embeds = None
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        self.image_post_process_config(config.vision_config, vllm_config.model_config)

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def image_post_process_config(self, vision_config, model_config):
        processor = MULTIMODAL_REGISTRY.create_processor(model_config)
        self.channel = vision_config.in_channels
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.do_rescale = True
        self.do_normalize = True
        self.rescale_factor = processor.info.get_hf_processor().image_processor.rescale_factor
        self.image_mean = tuple(processor.info.get_hf_processor().image_processor.image_mean)
        self.image_std = tuple(processor.info.get_hf_processor().image_processor.image_std)
    
    @lru_cache(maxsize=10)
    def _fuse_mean_std_and_rescale_factor(self, do_normalize, image_mean, image_std, do_rescale, rescale_factor, device):
        if do_rescale and do_normalize:
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
            do_rescale = False
        return image_mean, image_std, do_rescale

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        """
        Rescale and normalize images.
        """
        image_mean, image_std, do_rescale = self._fuse_mean_std_and_rescale_factor(
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            device=images.device
        )
        # if/elif as we use fused rescale and normalize if both are set to True
        if do_normalize:
            origin_dtype = images.dtype
            images = functional.normalize(images.to(torch.float32), image_mean, image_std).to(origin_dtype)
        elif do_rescale:
            images = images * rescale_factor
        return images

    def _process_image_input(self, image_input:Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            pixel_values = pixel_values.reshape(-1, self.channel, self.patch_size, self.patch_size)
            pixel_values = self.rescale_and_normalize(pixel_values, self.do_rescale, 
                                                self.rescale_factor, self.do_normalize, self.image_mean, self.image_std)
            pixel_values = pixel_values.reshape(-1, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                        pixel_values,
                                                        grid_thw_list,
                                                        rope_type="rope_3d")
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
        merge_size = self.visual.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1)//
                (merge_size*merge_size)).tolist()
        return image_embeds.split(sizes)

Qwen3VLMoeForConditionalGeneration.__init__ = AscendQwen3VLMoeForConditionalGeneration.__init__
Qwen2_5_VLForConditionalGeneration._process_image_input = AscendQwen3VLMoeForConditionalGeneration._process_image_input
Qwen3VLMoeForConditionalGeneration.image_post_process_config = AscendQwen3VLMoeForConditionalGeneration.image_post_process_config
