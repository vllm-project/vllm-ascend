import torch
from functools import lru_cache
from typing import Optional

from torchvision.transforms import v2
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
    run_dp_sharded_mrope_vision_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
        try:
            flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        except (AssertionError, AttributeError, KeyError):
            flash_comm_v1_enabled = False
        if flash_comm_v1_enabled:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()
            deepstack_input_embeds.tensors = {
                k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
            }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    qkv, _ = self.qkv_proj(hidden_states)
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


patch_qwen3_vl_moe_pp_layer_range()


# ---------------------------------------------------------------------------
# Move image/video rescale+normalize from HF processor (CPU) onto device (NPU).
# HF processor disable lives in platform/patch_qwen3vl_processor.py so the
# APIServer also skips CPU normalize; otherwise values are normalized twice.
# ---------------------------------------------------------------------------


def _rescale(image: torch.Tensor, scale: float) -> torch.Tensor:
    return image * scale


def _normalize(image: torch.Tensor, mean, std) -> torch.Tensor:
    return v2.functional.normalize(image, mean, std)


@lru_cache(maxsize=10)
def _fuse_mean_std_and_rescale_factor(
    do_normalize: bool | None = None,
    image_mean: float | tuple[float, ...] | None = None,
    image_std: float | tuple[float, ...] | None = None,
    do_rescale: bool | None = None,
    rescale_factor: float | None = None,
    device: Optional[torch.device] = None,
) -> tuple:
    if do_rescale and do_normalize:
        image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
        image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)
        do_rescale = False
    return image_mean, image_std, do_rescale


def rescale_and_normalize(
    images: torch.Tensor,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | tuple[float, ...],
    image_std: float | tuple[float, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Rescale and normalize images on device (fused when both are enabled)."""
    image_mean, image_std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    if do_normalize:
        images = _normalize(images.to(dtype=torch.float32), image_mean, image_std)
    elif do_rescale:
        images = _rescale(images.to(dtype=torch.float32), rescale_factor)
    return images.to(dtype)


def _image_post_process_config(self, vision_config, model_config) -> None:
    image_processor = (
        MULTIMODAL_REGISTRY.create_processor(model_config).info.get_hf_processor().image_processor
    )
    self.channel = vision_config.in_channels
    self.patch_size = vision_config.patch_size
    self.temporal_patch_size = vision_config.temporal_patch_size
    # HF processor side is disabled (platform patch); always apply on device.
    self.do_rescale = True
    self.do_normalize = True
    self.rescale_factor = image_processor.rescale_factor
    self.image_mean = tuple(image_processor.image_mean)
    self.image_std = tuple(image_processor.image_std)


def _apply_rescale_normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
    if not hasattr(self, "channel"):
        _image_post_process_config(self, self.config.vision_config, self.model_config)
    # Keep raw integer/float range until after fp32 normalize; do not cast
    # uint8 -> bf16 before normalize (avoids precision / scale surprises).
    pixel_values = pixel_values.to(dtype=torch.float32).reshape(
        -1, self.channel, self.patch_size, self.patch_size
    )
    pixel_values = rescale_and_normalize(
        pixel_values,
        self.do_rescale,
        self.rescale_factor,
        self.do_normalize,
        self.image_mean,
        self.image_std,
        dtype=self.visual.dtype,
    )
    return pixel_values.reshape(
        -1,
        self.channel * self.temporal_patch_size * self.patch_size * self.patch_size,
    )


_orig_qwen3vl_init = Qwen3VLForConditionalGeneration.__init__


def _patched_qwen3vl_init(self, *, vllm_config, prefix: str = "model"):
    _orig_qwen3vl_init(self, vllm_config=vllm_config, prefix=prefix)
    _image_post_process_config(self, self.config.vision_config, self.model_config)


Qwen3VLForConditionalGeneration.__init__ = _patched_qwen3vl_init

# Qwen3.5 / Qwen3.5-MoE reimplement __init__ without calling Qwen3VL.__init__,
# so they must be patched separately.
try:
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
    )

    _orig_qwen35_init = Qwen3_5ForConditionalGeneration.__init__
    _orig_qwen35_moe_init = Qwen3_5MoeForConditionalGeneration.__init__

    def _patched_qwen35_init(self, *, vllm_config, prefix: str = "model"):
        _orig_qwen35_init(self, vllm_config=vllm_config, prefix=prefix)
        _image_post_process_config(self, self.config.vision_config, self.model_config)

    def _patched_qwen35_moe_init(self, *, vllm_config, prefix: str = "model"):
        _orig_qwen35_moe_init(self, vllm_config=vllm_config, prefix=prefix)
        _image_post_process_config(self, self.config.vision_config, self.model_config)

    Qwen3_5ForConditionalGeneration.__init__ = _patched_qwen35_init
    Qwen3_5MoeForConditionalGeneration.__init__ = _patched_qwen35_moe_init
except Exception:
    pass


def _patched_process_image_input(self, image_input):
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        # Do not cast to visual.dtype before normalize (raw uint8/[0,255]).
        pixel_values = _apply_rescale_normalize(self, image_input["pixel_values"])
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return image_embeds.split(sizes)


def _patched_process_video_input(self, video_input):
    grid_thw = video_input["video_grid_thw"]
    assert grid_thw.ndim == 2

    if video_input["type"] == "video_embeds":
        video_embeds = video_input["video_embeds"].type(self.visual.dtype)
    else:
        pixel_values_videos = _apply_rescale_normalize(
            self, video_input["pixel_values_videos"]
        )
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(
                self.visual,
                pixel_values_videos,
                grid_thw.tolist(),
                rope_type="rope_3d",
            )
        video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

    merge_size = self.visual.spatial_merge_size
    sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
    return video_embeds.split(sizes)


Qwen3VLForConditionalGeneration._process_image_input = _patched_process_image_input
Qwen3VLForConditionalGeneration._process_video_input = _patched_process_video_input
