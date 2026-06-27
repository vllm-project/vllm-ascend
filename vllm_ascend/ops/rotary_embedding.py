#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import math
import os

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    MRotaryEmbedding,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.platform import NPUPlatform

if HAS_TRITON:
    from vllm_ascend.ops.triton.rope import (
        mrope_forward_triton_by_cache,
        rope_forward_triton,
    )


def get_rope_cache(rotary_emb, ref_tensor: torch.Tensor) -> torch.Tensor:
    """Return rotary_emb.cos_sin_cache matched to ref_tensor dtype/device.

    Keep this helper side-effect free. Some upstream cache helpers update the
    module buffer when dtype/device differ, which makes graph capture unstable
    when adjacent calls disagree on the reference tensor dtype.
    """
    cos_sin_cache = rotary_emb.cos_sin_cache
    if cos_sin_cache.device == ref_tensor.device and cos_sin_cache.dtype == ref_tensor.dtype:
        return cos_sin_cache

    return cos_sin_cache.to(ref_tensor.device, dtype=ref_tensor.dtype)


def select_cos_sin_cache(rotary_emb, positions: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """Legacy row materialization helper for callers that still own positions.

    New NPU hot paths should pass positions plus the full cos_sin_cache to a
    by-cache adapter/backend instead of slicing rows in Python.
    """
    return get_rope_cache(rotary_emb, ref_tensor)[positions]


def _expand_rope_dim(cos: torch.Tensor, sin: torch.Tensor, *, is_neox_style: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if is_neox_style:
        return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def select_cos_sin_from_cache(
    rotary_emb,
    positions: torch.Tensor,
    ref_tensor: torch.Tensor,
    *,
    layout: str = "T11D",
    expand_rope_dim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve materialized cos/sin from rotary_emb for legacy NPU ops.

    This is intentionally kept as a boundary helper only. Stage-9 by-cache
    paths should pass positions plus cos_sin_cache through adapter APIs.

    layout values:
    - "TD": [num_tokens, rotary_dim]
    - "T11D": [num_tokens, 1, 1, rotary_dim]
    - "1T1D": [1, num_tokens, 1, rotary_dim]
    """
    cos_sin = select_cos_sin_cache(rotary_emb, positions, ref_tensor)
    cos, sin = cos_sin.chunk(2, dim=-1)
    if expand_rope_dim:
        cos, sin = _expand_rope_dim(cos, sin, is_neox_style=getattr(rotary_emb, "is_neox_style", True))

    if layout == "TD":
        return cos.contiguous().view(positions.shape[-1], -1), sin.contiguous().view(positions.shape[-1], -1)
    if layout == "T11D":
        return cos.contiguous().view(positions.shape[-1], 1, 1, -1), sin.contiguous().view(
            positions.shape[-1], 1, 1, -1
        )
    if layout == "1T1D":
        return cos.contiguous().view(1, positions.shape[-1], 1, -1), sin.contiguous().view(
            1, positions.shape[-1], 1, -1
        )
    raise ValueError(f"Unsupported RoPE cache layout: {layout}")


def rope_forward_oot(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
    offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if offsets is not None:
        raise NotImplementedError("Batched rotary embedding is currently not supported on NPU.")
    if key is None:
        return RotaryEmbedding.forward_static(
            positions,
            query,
            None,
            head_size,
            rotary_dim,
            cos_sin_cache,
            is_neox_style,
        )
    query_shape, key_shape = query.shape, key.shape
    if HAS_TRITON:
        num_tokens = query.shape[0]
        query, key = rope_forward_triton(
            query.view(num_tokens, -1, head_size),
            key.view(num_tokens, -1, head_size),
            cos_sin_cache=cos_sin_cache,
            positions=positions,
            rope_dim=rotary_dim,
            is_neox_style=is_neox_style,
        )
    else:
        if rotary_dim < head_size:
            num_tokens = query.shape[0]
            query = query.view(num_tokens, -1, head_size)
            key = key.view(num_tokens, -1, head_size)
            q_rot = query[..., :rotary_dim]
            q_pass = query[..., rotary_dim:]
            k_rot = key[..., :rotary_dim]
            k_pass = key[..., rotary_dim:]
            q_rot = q_rot.contiguous().view(num_tokens, -1)
            k_rot = k_rot.contiguous().view(num_tokens, -1)
            # only the rotary part is processed here,
            # the dimension should be rotary_dim
            torch_npu._npu_rotary_embedding(
                positions,
                q_rot,
                k_rot,
                rotary_dim,
                cos_sin_cache,
                is_neox_style,
            )
            q_rot = q_rot.view(num_tokens, -1, rotary_dim)
            k_rot = k_rot.view(num_tokens, -1, rotary_dim)
            query = torch.cat((q_rot, q_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_rot, k_pass), dim=-1).reshape(key_shape)
        else:
            # TODO: Remove the contiguous in the future.
            query = query.contiguous().view(query.shape[0], -1)
            key = key.contiguous().view(key.shape[0], -1)
            torch_npu._npu_rotary_embedding(
                positions,
                query,
                key,
                head_size,
                cos_sin_cache,
                is_neox_style,
            )
    return query.view(query_shape), key.view(key_shape)


def rope_forward_oot_op(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    query, key = rope_forward_oot(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
    )
    return query, key


class AscendRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, init_cache)
        vllm_config = get_current_vllm_config()
        self.use_mtp = vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp"

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        is_neox_style = self.is_neox_style
        if is_neox_style_override is not None:
            is_neox_style = is_neox_style_override
        if offsets is not None:
            raise NotImplementedError("Batched rotary embedding with offsets is currently not supported on NPU.")
        is_draft_model = _EXTRA_CTX.is_draft_model
        flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        if is_draft_model and self.use_mtp and flash_comm_v1_enabled:
            positions = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(positions.contiguous(), True)
        cos_sin_cache = get_rope_cache(self, query)
        if key is None:
            return rope_forward_oot(
                positions,
                query,
                None,
                cos_sin_cache,
                self.head_size,
                self.rotary_dim,
                is_neox_style,
            )
        return torch.ops.vllm.npu_rotary_embedding(
            positions, query, key, cos_sin_cache, self.head_size, self.rotary_dim, is_neox_style
        )


class AscendYaRNRotaryEmbedding(YaRNScalingRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_yarn_scaling: bool = True,
        truncate: bool = False,
    ) -> None:
        extra_kwargs = {
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "apply_yarn_scaling": apply_yarn_scaling,
            # TODO: current not support actual truncate，adaptation for extra parameters to be compatible with vllm
            "truncate": truncate,
        }
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, scaling_factor, dtype, **extra_kwargs
        )
        vllm_config = get_current_vllm_config()
        self.use_mtp = vllm_config.speculative_config and vllm_config.speculative_config.method == "mtp"

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        is_neox_style_override: bool | None = None,
    ):
        return AscendRotaryEmbedding.forward_oot(self, positions, query, key, offsets, is_neox_style_override)


class AscendDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        # Note: we adopt the native huggingface deepseek rope initialization code from
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py for
        # its more ascend compute friendly
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            self._yarn_get_mscale(self.scaling_factor, float(mscale))
            / self._yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        super(DeepseekScalingRotaryEmbedding, self).__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        # NOTE: For ascend friendly computing, reorder sin and cos cache
        self.max_seq_len = math.ceil(max_position_embeddings * scaling_factor)
        self._set_cos_sin_cache(self.max_seq_len, device=NPUPlatform.device_type, dtype=dtype)

    def _yarn_get_mscale(self, scale: float = 1, mscale: float = 1) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _yarn_linear_ramp_mask(self, min_value, max_value, dim):
        # Note: The if conditional branch is not used here
        # to solve MTP compilation error.
        max_value += (min_value == max_value).float() * 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Inverse dim formula to find dim based on number of rotations
    def _yarn_find_correction_dim(self, num_rotations, dim, base=10000, max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        return (dim * torch.log(torch.tensor(max_position_embeddings) / (num_rotations * 2 * torch.pi))) / (
            2 * torch.log(torch.tensor(base))
        )

    # Find dim range bounds based on rotations
    def _yarn_find_correction_range(self, low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        # Note: use torch instead of math to solve MTP compilation error.
        low = torch.floor(self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = torch.ceil(self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
        # Note: use torch instead of max/min to solve MTP compilation error.
        return torch.clamp(low, min=0), torch.clamp(high, max=dim - 1)

    # Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.
        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example,
                note that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim].
                Then, if q and k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1
                makes cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly,
                if q and k have the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids]
        sin = sin[position_ids]
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]

        if len(q.shape) == 3:
            q = q[:, :, None, :]
        if len(k.shape) == 2:
            k = k[:, None, None, :]
        elif len(k.shape) == 3:
            k = k[:, :, None, :]

        b, h_q, s, d = q.shape
        q = q.view(b, h_q, s, d // 2, 2).transpose(4, 3).reshape(b, h_q, s, d)

        b, h_k, s, d = k.shape
        k = k.view(b, h_k, s, d // 2, 2).transpose(4, 3).reshape(b, h_k, s, d)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        q_embed = q_embed.view(b, h_q, d)
        k_embed = k_embed.view(b, h_k, d)

        return q_embed, k_embed

    def _set_cos_sin_cache(self, max_seq_len, device, dtype):
        dim = self.rotary_dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)
        cos_cached = torch.cat([freqs, freqs], dim=-1).cos() * self.mscale
        sin_cached = torch.cat([freqs, freqs], dim=-1).sin() * self.mscale
        cos_cached = cos_cached.to(dtype)
        sin_cached = sin_cached.to(dtype)
        cache = torch.cat([freqs.cos() * self.mscale, freqs.sin() * self.mscale], dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: torch.Tensor | None = None
    ):
        if offsets is not None:
            raise NotImplementedError("Batched rotary embedding with offsets is currently not supported on NPU.")
        if len(key.shape) == 2:
            key = key[:, None, :]
        # Note: we implement the non neox_style method with shuffle the last dim and neox style
        # calculation method which is also more compute friendly to the ascend machine
        # https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/modeling_deepseek.py
        is_neox_style = True
        if self.is_neox_style is False:
            b, h_q, d = query.shape
            query = query.view(b, h_q, d // 2, 2).transpose(3, 2).reshape(b, h_q, d)
            b, h_k, d = key.shape
            key = key.view(b, h_k, d // 2, 2).transpose(3, 2).reshape(b, h_k, d)
        cos_sin_cache = get_rope_cache(self, query)
        q_pe, k_pe = torch.ops.vllm.npu_rotary_embedding(
            positions, query, key, cos_sin_cache, self.head_size, self.rotary_dim, is_neox_style
        )
        return q_pe, k_pe


class AscendMRotaryEmbedding(MRotaryEmbedding):
    # Empirical safety threshold for large Triton grids on Ascend NPU
    _ASCEND_TRITON_GRID_LIMIT = 65535

    def forward_triton(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ):
        if offsets is not None:
            raise NotImplementedError("Batched MRoPE with offsets is currently not supported on NPU.")
        assert positions.ndim == 2
        assert key is not None

        query_shape = query.shape
        key_shape = key.shape

        assert self.mrope_section

        # When the grid becomes large, enable TRITON_ALL_BLOCKS_PARALLEL
        # to avoid scheduler/runtime failures.
        if query_shape[0] > self._ASCEND_TRITON_GRID_LIMIT and os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1":
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

        q, k = mrope_forward_triton_by_cache(
            query,
            key,
            get_rope_cache(self, query),
            positions,
            self.mrope_section,
            self.head_size,
            self.rotary_dim,
            self.mrope_interleaved,
        )

        return q.reshape(query_shape), k.reshape(key_shape)

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ):
        if offsets is not None:
            raise NotImplementedError("Batched MRoPE with offsets is currently not supported on NPU.")
        if HAS_TRITON and positions.ndim == 2 and self.mrope_interleaved:
            # todo: need cann update in 8.5.0
            return self.forward_triton(positions, query, key, offsets)

        if self.mrope_section != [16, 24, 24]:
            return super().forward_oot(positions, query, key, offsets)

        import torch_npu

        mrope_section = [0, 0, 0] if positions.ndim == 1 else self.mrope_section

        cos_sin_cache = get_rope_cache(self, query)

        query, key = torch_npu.npu_mrope(
            positions.contiguous(),
            query.contiguous(),
            key.contiguous(),
            cos_sin_cache.contiguous(),
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode="half",
        )

        return query, key


class AscendApplyRotaryEmb(ApplyRotaryEmb):
    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super().__init__(
            enforce_enable=enforce_enable,
            is_neox_style=is_neox_style,
            enable_fp32_compute=enable_fp32_compute,
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from vllm_ascend.ops.rope_cache_ops import rotary_mul_materialized

        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

        head_dim = x.shape[-1]
        # cos, sin: [seq_len, head_dim // 2]
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        # cos, sin: [1, seq_len, 1, head_dim]
        cos = cos.reshape(1, -1, 1, head_dim)
        sin = sin.reshape(1, -1, 1, head_dim)

        output = rotary_mul_materialized(x, cos, sin)

        output = self._post_process(output, origin_shape, origin_dtype)

        return output
