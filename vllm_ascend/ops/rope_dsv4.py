import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from vllm_ascend.ops.rope_cache_ops import inplace_partial_rotary_mul_dsa_by_cache, rotary_mul_materialized
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class RopeGlobalState:
    def __init__(self):
        self.static_cache: dict[str, torch.Tensor] = {}
        self.runtime_buffer: dict[str, dict[str, torch.Tensor]] = {}
        self.layer_info: dict[str, tuple[str, list[str]]] = {}
        self.registry_summary: dict[str, set] = {}


_ROPE_STATE = RopeGlobalState()


def _dsa_rope_layer_key(layername: str, config_key: str, rope_groups: list[str] | None = None) -> str:
    if rope_groups is None:
        return f"{layername}::{config_key}"
    groups_key = ",".join(sorted(rope_groups))
    return f"{layername}::{config_key}::groups={groups_key}"


class DSARopeDataProxy:
    """Lazy per-layer view over DSA RoPE tensors materialized at the op boundary."""

    def __init__(self, data_map, is_cos=True):
        self._data = data_map
        self.idx = 0 if is_cos else 1

    def __getitem__(self, index):
        if not isinstance(index, str):
            new_map: dict = {}
            for config_k, groups_map in self._data.items():
                new_map[config_k] = {}
                for group_name, item in groups_map.items():
                    c_val = item[0][index]
                    s_val = item[1][index]
                    new_map[config_k][group_name] = (c_val, s_val)

            return DSARopeDataProxy(new_map, is_cos=(self.idx == 0))

        else:
            layername = index
            info = _ROPE_STATE.layer_info.get(layername)
            if info is None:
                raise KeyError(f"Layer {layername} not registered.")

            config_key, required_groups = info

            config_data = self._data.get(config_key, {})

            layer_result = {}
            for grp in required_groups:
                if grp in config_data:
                    layer_result[grp] = config_data[grp][self.idx]
                else:
                    pass
            if len(layer_result) == 1:
                return list(layer_result.values())[0]

            return layer_result


def _normalize_dsa_positions(positions: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if isinstance(positions, torch.Tensor):
        return {"default": positions}
    return positions


def _to_dsa_cos_sin_cache(cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Normalize legacy DSA tuple storage into the standard cos_sin_cache tensor."""
    if isinstance(cache, tuple):
        cos_cache, sin_cache = cache
        return torch.cat((cos_cache, sin_cache), dim=-1)
    return cache


def split_dsa_cos_sin_cache(cos_sin_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the standard DSA cos_sin_cache only for legacy materialized users."""
    if isinstance(cos_sin_cache, tuple):
        return cos_sin_cache
    if cos_sin_cache.shape[-1] % 2 != 0:
        raise ValueError(f"DSA cos_sin_cache last dim must be even, got {cos_sin_cache.shape[-1]}")
    half_dim = cos_sin_cache.shape[-1] // 2
    return cos_sin_cache.narrow(-1, 0, half_dim), cos_sin_cache.narrow(-1, half_dim, half_dim)


def _materialize_dsa_group_cos_sin(
    config_key: str,
    group_name: str,
    positions: torch.Tensor,
    *,
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if config_key not in _ROPE_STATE.static_cache:
        return None

    static_cache = _to_dsa_cos_sin_cache(_ROPE_STATE.static_cache[config_key])
    curr_cache = static_cache[positions]

    if use_cache:
        group_buffer = _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name)

        if group_buffer is None:
            return None

        num_tokens = positions.size(0)
        group_buffer[:num_tokens].copy_(curr_cache)
        return split_dsa_cos_sin_cache(group_buffer[:num_tokens])

    return split_dsa_cos_sin_cache(curr_cache)


@dataclass
class DSARopeCacheView:
    """Position-based DSA RoPE view for one layer/group."""

    config_key: str
    group_name: str
    positions: torch.Tensor
    use_cache: bool
    rotary_dim: int
    _legacy_materialized: tuple[torch.Tensor, torch.Tensor] | None = field(default=None, init=False, repr=False)

    @property
    def cos_sin_cache(self) -> torch.Tensor:
        return _to_dsa_cos_sin_cache(_ROPE_STATE.static_cache[self.config_key])

    def materialize(self, *, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return legacy materialized cos/sin for non by-cache DSA callers."""
        if self._legacy_materialized is None:
            materialized = _materialize_dsa_group_cos_sin(
                self.config_key,
                self.group_name,
                self.positions,
                use_cache=self.use_cache,
            )
            if materialized is None:
                raise KeyError(
                    f"DSA RoPE group {self.group_name} is not registered for config {self.config_key}."
                )
            self._legacy_materialized = materialized

        cos, sin = self._legacy_materialized
        if inverse:
            sin = -sin
        return cos, sin

    def backend_cos_sin_cache(self) -> torch.Tensor:
        return self.cos_sin_cache


class DSARopeCacheProxy:
    """Lazy layer lookup for DSA RoPE cache views."""

    def __init__(self, view_map: dict[Any, dict[str, DSARopeCacheView]]):
        self._views = view_map

    def __getitem__(self, layer_name: str):
        info = _ROPE_STATE.layer_info.get(layer_name)
        if info is None:
            raise KeyError(f"Layer {layer_name} not registered.")

        config_key, required_groups = info
        config_views = self._views.get(config_key, {})

        layer_result = {}
        for grp in required_groups:
            if grp in config_views:
                layer_result[grp] = config_views[grp]
        if len(layer_result) == 1:
            return list(layer_result.values())[0]

        return layer_result


def get_dsa_rope_cache_proxy(
    positions: torch.Tensor | dict[str, torch.Tensor],
    *,
    use_cache: bool = False,
) -> DSARopeCacheProxy:
    pos_map = _normalize_dsa_positions(positions)
    view_map: dict[Any, dict[str, DSARopeCacheView]] = {}

    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():
        if config_key not in _ROPE_STATE.static_cache:
            continue
        static_cache = _to_dsa_cos_sin_cache(_ROPE_STATE.static_cache[config_key])
        static_cos, _ = split_dsa_cos_sin_cache(static_cache)

        view_map[config_key] = {}
        for group_name, pos_tensor in pos_map.items():
            if group_name not in registered_groups:
                continue
            view_map[config_key][group_name] = DSARopeCacheView(
                config_key=config_key,
                group_name=group_name,
                positions=pos_tensor,
                use_cache=use_cache,
                rotary_dim=int(static_cos.shape[-1]),
            )

    return DSARopeCacheProxy(view_map)


def materialize_dsa_cos_sin(positions: torch.Tensor | dict[str, torch.Tensor], use_cache: bool = False):
    """Materialize DSA cos/sin only where legacy DSA kernels still require tensors."""
    pos_map = _normalize_dsa_positions(positions)

    batch_result: dict[Any, Any] = {}

    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():
        batch_result[config_key] = {}

        for group_name, pos_tensor in pos_map.items():
            if group_name not in registered_groups:
                continue

            materialized = _materialize_dsa_group_cos_sin(
                config_key,
                group_name,
                pos_tensor,
                use_cache=use_cache,
            )
            if materialized is None:
                continue
            batch_result[config_key][group_name] = materialized

    return DSARopeDataProxy(batch_result, is_cos=True), DSARopeDataProxy(batch_result, is_cos=False)


class ComplexExpRotaryEmbedding(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layername: str,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        rope_groups: list[str] | None = None,
        **extra_kwargs,
    ) -> None:
        super().__init__()
        if rope_groups is None:
            rope_groups = ["default"]
        self.layername = layername
        self.rotary_dim = rotary_dim
        use_fp32_rope = get_ascend_device_type() in {AscendDeviceType.A2, AscendDeviceType.A3}
        dtype = torch.float32 if use_fp32_rope else torch.get_default_dtype()
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        config_key = (
            f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
            f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}"
        )
        self.rope_cache_key = _dsa_rope_layer_key(layername, config_key, rope_groups)
        _ROPE_STATE.layer_info[self.rope_cache_key] = (config_key, rope_groups)

        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)

        if config_key not in _ROPE_STATE.static_cache:
            inv_freq = self.precompute_freqs_cis(
                rotary_dim, max_position_embeddings, max_position_embeddings, base, scaling_factor, beta_fast, beta_slow
            )
            t = torch.arange(
                max_position_embeddings * scaling_factor,
                device=current_platform.device_type,
                dtype=torch.float32,
            )
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
            if not use_fp32_rope:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            cos = cos.to(current_platform.device_type)
            sin = sin.to(current_platform.device_type)

            _ROPE_STATE.static_cache[config_key] = torch.cat(
                (cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1)),
                dim=-1,
            )

        if config_key not in _ROPE_STATE.runtime_buffer:
            _ROPE_STATE.runtime_buffer[config_key] = {}

        target_device = current_platform.device_type
        max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        for grp in rope_groups:
            if grp not in _ROPE_STATE.runtime_buffer[config_key]:
                buf_cos_sin = torch.empty(max_batch_size, 1, 1, rotary_dim * 2, dtype=dtype, device=target_device)
                buf_cos_sin[..., :rotary_dim] = 1
                buf_cos_sin[..., rotary_dim:] = 0
                _ROPE_STATE.runtime_buffer[config_key][grp] = buf_cos_sin

    @staticmethod
    def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
        def yarn_find_correction_dim(
            num_rotations: int,
            dim: int,
            base: float = 10000,
            max_position_embeddings: int = 2048,
        ) -> float:
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        # Find dim range bounds based on rotations
        def yarn_find_correction_range(
            low_rot: int,
            high_rot: int,
            dim: int,
            base: float = 10000,
            max_position_embeddings: int = 2048,
            truncate: bool = True,
        ) -> tuple[float | int, float | int]:
            low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
            high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dim - 1)  # Clamp values just in case

        def yarn_linear_ramp_mask(low: float, high: float, dim: int, dtype: torch.dtype) -> torch.Tensor:
            if low == high:
                high += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        low, high = yarn_find_correction_range(
            beta_fast,
            beta_slow,
            dim,
            base,
            original_seq_len,
        )
        inv_freq_mask = (1 - yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32)) * 1
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy helper for callers that still own materialized DSA cos/sin."""
        ori_shape = x.shape
        y = x

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = rotary_mul_materialized(x, cos, sin, rotary_mode="interleave")

        y.copy_(x.view(ori_shape))
        return y

    def forward_by_cache(
        self,
        x: torch.Tensor,
        rope_cache: DSARopeCacheView,
        *,
        inverse: bool = False,
    ) -> torch.Tensor:
        ori_shape = x.shape
        y = x

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        inplace_partial_rotary_mul_dsa_by_cache(
            x,
            rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, x.shape[-1]],
            inverse=inverse,
        )

        y.copy_(x.view(ori_shape))
        return y

    def extra_repr(self) -> str:
        return f"layername={self.layername}, rotary_dim={self.rotary_dim}"
