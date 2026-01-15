import numpy as np
import torch
import torch.nn as nn
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
