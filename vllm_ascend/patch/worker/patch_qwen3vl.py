import importlib
from functools import lru_cache, wraps
from typing import NamedTuple

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger
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
from vllm_ascend.patch.platform.patch_qwen3vl_processor import ORIG_PREPROCESS_FLAGS_ATTR


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


@lru_cache(maxsize=10)
def _fuse_mean_std_and_rescale_factor(
    do_normalize: bool,
    image_mean: tuple[float, ...] | None,
    image_std: tuple[float, ...] | None,
    do_rescale: bool,
    rescale_factor: float | None,
    device: torch.device | None = None,
) -> tuple:
    """Build device-side mean/std, folding the rescale factor into them.

    ``(x * s - mean) / std`` equals ``(x - mean / s) / (std / s)``, so when both
    steps are enabled the rescale collapses into the normalize and only one pass
    over the tensor is needed. Cached so the mean/std tensors are built once
    instead of being copied from host on every forward.
    """
    if not do_normalize:
        return None, None, do_rescale
    mean = torch.tensor(image_mean, device=device, dtype=torch.float32)
    std = torch.tensor(image_std, device=device, dtype=torch.float32)
    if do_rescale:
        assert rescale_factor is not None
        mean = mean / rescale_factor
        std = std / rescale_factor
        do_rescale = False
    return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1), do_rescale


def rescale_and_normalize(
    images: torch.Tensor,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: tuple[float, ...],
    image_std: tuple[float, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Rescale and normalize NCHW images on device (fused when both are on)."""
    mean, std, do_rescale = _fuse_mean_std_and_rescale_factor(
        do_normalize=do_normalize,
        image_mean=image_mean,
        image_std=image_std,
        do_rescale=do_rescale,
        rescale_factor=rescale_factor,
        device=images.device,
    )
    if do_normalize:
        images = (images.to(dtype=torch.float32) - mean) / std
    elif do_rescale:
        images = images.to(dtype=torch.float32) * rescale_factor
    return images.to(dtype)


class _VLPreprocessConfig(NamedTuple):
    channel: int
    patch_size: int
    temporal_patch_size: int
    do_rescale: bool
    rescale_factor: float | None
    do_normalize: bool
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]


_PREPROCESS_ATTR = "_ascend_vl_preprocess"


def _image_post_process_config(self, vision_config, model_config) -> None:
    image_processor = MULTIMODAL_REGISTRY.create_processor(model_config).info.get_hf_processor().image_processor
    flags = getattr(image_processor, ORIG_PREPROCESS_FLAGS_ATTR, None)
    if flags is None:
        # The platform patch never ran, so HF is still rescaling/normalizing on
        # the host. Doing nothing here is what keeps values from being
        # normalized twice; the request stays correct, just without the speedup.
        logger.warning(
            "Qwen3-VL HF processor still applies rescale/normalize on host, "
            "skipping device-side preprocess to avoid normalizing twice."
        )
        flags = {"do_rescale": False, "do_normalize": False}
    setattr(
        self,
        _PREPROCESS_ATTR,
        _VLPreprocessConfig(
            channel=vision_config.in_channels,
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            do_rescale=flags["do_rescale"],
            rescale_factor=getattr(image_processor, "rescale_factor", None),
            do_normalize=flags["do_normalize"],
            image_mean=tuple(image_processor.image_mean),
            image_std=tuple(image_processor.image_std),
        ),
    )


def _apply_rescale_normalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
    cfg = getattr(self, _PREPROCESS_ATTR, None)
    if cfg is None:
        _image_post_process_config(self, self.config.vision_config, self.model_config)
        cfg = getattr(self, _PREPROCESS_ATTR)
    if not (cfg.do_rescale or cfg.do_normalize):
        return pixel_values.to(self.visual.dtype)

    # Each row packs one patch as [channel][temporal][ph][pw] (see the flatten
    # step in the HF image/video processors), so folding temporal into the
    # height axis exposes the channel axis to mean/std without moving any data.
    # Reshaping straight to (-1, channel, ph, pw) would instead line temporal
    # slices up under the channel axis and normalize the wrong channels.
    pixel_values = pixel_values.reshape(
        -1,
        cfg.channel,
        cfg.temporal_patch_size * cfg.patch_size,
        cfg.patch_size,
    )
    # Keep the raw integer/float range until after the fp32 normalize; casting
    # uint8 -> bf16 first would lose precision before the scale is applied.
    pixel_values = rescale_and_normalize(
        pixel_values,
        cfg.do_rescale,
        cfg.rescale_factor,
        cfg.do_normalize,
        cfg.image_mean,
        cfg.image_std,
        dtype=self.visual.dtype,
    )
    return pixel_values.reshape(
        -1,
        cfg.channel * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
    )


def _hook_preprocess_config(cls) -> None:
    orig_init = cls.__init__

    # vLLM decides how to construct a model by looking for `vllm_config` and
    # `prefix` in the __init__ signature, so the wrapper has to keep reporting
    # the wrapped signature -- `functools.wraps` is what makes that work.
    @wraps(orig_init)
    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _image_post_process_config(self, self.config.vision_config, self.model_config)

    cls.__init__ = patched_init


# Qwen3-VL-MoE, Qwen3.5 and Qwen3.5-MoE each define their own __init__ that
# bypasses Qwen3VLForConditionalGeneration.__init__, so hooking the base class
# alone would leave them falling back to the lazy path on the first request.
_hook_preprocess_config(Qwen3VLForConditionalGeneration)
for _module_name, _class_names in (
    ("vllm.model_executor.models.qwen3_vl_moe", ("Qwen3VLMoeForConditionalGeneration",)),
    ("vllm.model_executor.models.qwen3_5", ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration")),
):
    try:
        _module = importlib.import_module(_module_name)
    except ImportError:
        continue
    for _class_name in _class_names:
        _cls = getattr(_module, _class_name, None)
        if _cls is not None and "__init__" in _cls.__dict__:
            _hook_preprocess_config(_cls)


def _patched_process_image_input(self, image_input):
    grid_thw = image_input["image_grid_thw"]
    assert grid_thw.ndim == 2

    if image_input["type"] == "image_embeds":
        image_embeds = image_input["image_embeds"].type(self.visual.dtype)
    else:
        # Do not cast to visual.dtype before normalize (raw uint8/[0,255]).
        pixel_values = _apply_rescale_normalize(self, image_input["pixel_values"])
        if self.use_data_parallel:
            return run_dp_sharded_mrope_vision_model(self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d")
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
        pixel_values_videos = _apply_rescale_normalize(self, video_input["pixel_values_videos"])
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
