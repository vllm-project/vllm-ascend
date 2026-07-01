import math
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.platforms import current_platform


class RopeGlobalState:
    def __init__(self):
        self.full_rope_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.runtime_buffer: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.layer_info: dict[str, tuple[str, list[str]]] = {}
        self.registry_summary: dict[str, set] = {}


_ROPE_STATE = RopeGlobalState()


class RopeDataProxy:
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

            return RopeDataProxy(new_map, is_cos=(self.idx == 0))

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


def get_cos_and_sin_dsa(positions: torch.Tensor | dict[str, torch.Tensor], use_cache: bool = False):
    if isinstance(positions, torch.Tensor):
        pos_map = {"default": positions}
    else:
        pos_map = positions

    batch_result: dict[Any, Any] = {}

    for config_key, registered_groups in _ROPE_STATE.registry_summary.items():
        if config_key not in _ROPE_STATE.full_rope_cache:
            continue
        full_rope_cos, full_rope_sin = _ROPE_STATE.full_rope_cache[config_key]

        batch_result[config_key] = {}

        for group_name, pos_tensor in pos_map.items():
            if group_name not in registered_groups:
                continue

            curr_cos = full_rope_cos[pos_tensor]
            curr_sin = full_rope_sin[pos_tensor]

            if use_cache:
                group_buffers = _ROPE_STATE.runtime_buffer.get(config_key, {}).get(group_name)

                if group_buffers is None:
                    continue

                buf_cos, buf_sin = group_buffers
                num_tokens = pos_tensor.size(0)

                buf_cos[:num_tokens].copy_(curr_cos)
                buf_sin[:num_tokens].copy_(curr_sin)

                batch_result[config_key][group_name] = (buf_cos[:num_tokens], buf_sin[:num_tokens])
            else:
                batch_result[config_key][group_name] = (curr_cos, curr_sin)

    return RopeDataProxy(batch_result, is_cos=True), RopeDataProxy(batch_result, is_cos=False)


def get_full_cos_and_sin_dsa(group_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the full precomputed RoPE cache for a registered DSA RoPE group.

    Unlike get_cos_and_sin_dsa(), this does not index by token positions on
    Python side. The compressor metadata op consumes the full cache and selects
    compressed-row RoPE positions on device.
    """
    config_keys = [
        config_key
        for config_key, registered_groups in _ROPE_STATE.registry_summary.items()
        if group_name in registered_groups
    ]
    if not config_keys:
        raise KeyError(f"RoPE group {group_name} is not registered.")
    if len(config_keys) > 1:
        raise KeyError(f"RoPE group {group_name} is registered with multiple configs: {config_keys}.")

    config_key = config_keys[0]
    if config_key not in _ROPE_STATE.full_rope_cache:
        raise KeyError(f"Rope cache for group {group_name} is not initialized.")

    return _ROPE_STATE.full_rope_cache[config_key]


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
        beta_fast = extra_kwargs.get("beta_fast", 32)
        beta_slow = extra_kwargs.get("beta_slow", 1)
        config_key = (
            f"rotary_dim{rotary_dim}_max_position_embeddings{max_position_embeddings}_"
            f"base{base}_scaling_factor{scaling_factor}_beta_fast{beta_fast}_beta_slow{beta_slow}"
        )
        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)

        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)

        if config_key not in _ROPE_STATE.full_rope_cache:
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
            cos = cos.to(current_platform.device_type)
            sin = sin.to(current_platform.device_type)

            _ROPE_STATE.full_rope_cache[config_key] = (cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1))

        if config_key not in _ROPE_STATE.runtime_buffer:
            _ROPE_STATE.runtime_buffer[config_key] = {}

        target_device = current_platform.device_type
        max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens
        for grp in rope_groups:
            if grp not in _ROPE_STATE.runtime_buffer[config_key]:
                buf_cos = torch.ones(max_batch_size, 1, 1, rotary_dim, dtype=torch.float32, device=target_device)
                buf_sin = torch.zeros(max_batch_size, 1, 1, rotary_dim, dtype=torch.float32, device=target_device)
                _ROPE_STATE.runtime_buffer[config_key][grp] = (buf_cos, buf_sin)

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
        ori_shape = x.shape
        y = x

        if x.dim() == 2:
            x = x.unsqueeze(-2)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

        y.copy_(x.view(ori_shape))
        return y

    def extra_repr(self) -> str:
        return f"layername={self.layername}, rotary_dim={self.rotary_dim}"
