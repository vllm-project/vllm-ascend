import math
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.config import VllmConfig
from vllm.platforms import current_platform


class RopeGlobalState:
    def __init__(self):
        self.static_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
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
        if config_key not in _ROPE_STATE.static_cache:
            continue
        static_cos, static_sin = _ROPE_STATE.static_cache[config_key]

        batch_result[config_key] = {}

        for group_name, pos_tensor in pos_map.items():
            if group_name not in registered_groups:
                continue

            curr_cos = static_cos[pos_tensor]
            curr_sin = static_sin[pos_tensor]

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
        # Persist everything required to rebuild the (device-resident, non
        # nn.Buffer) cos/sin caches after a snapshot restore. These caches live
        # in the module-level ``_ROPE_STATE`` global, so they are NOT part of
        # ``state_dict`` and are never re-materialized by ``restore_model``.
        self._config_key = config_key
        self._max_position_embeddings = max_position_embeddings
        self._base = base
        self._scaling_factor = scaling_factor
        self._beta_fast = beta_fast
        self._beta_slow = beta_slow
        self._rope_groups = rope_groups
        self._max_batch_size = vllm_config.scheduler_config.max_num_batched_tokens

        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)

        if config_key not in _ROPE_STATE.registry_summary:
            _ROPE_STATE.registry_summary[config_key] = set()
        for grp in rope_groups:
            _ROPE_STATE.registry_summary[config_key].add(grp)

        self._build_cos_sin_cache(force=False)

    @staticmethod
    def _static_cache_is_valid(config_key: str, device) -> bool:
        """True iff the cached cos table exists, lives on ``device`` and still
        holds valid data. ``cos`` at position 0 must be all ones (cos(0)=1); a
        snapshot restore invalidates the backing device memory (observed as
        all-zero), so this doubles as a staleness check and lets many layers
        that share a config skip redundant recomputes."""
        entry = _ROPE_STATE.static_cache.get(config_key)
        if entry is None:
            return False
        cos = entry[0]
        if cos.device.type != torch.device(device).type:
            return False
        try:
            return bool(torch.all(cos[0] == 1.0))
        except Exception:  # noqa: BLE001
            return False

    def _build_cos_sin_cache(self, force: bool = False) -> None:
        """(Re)materialize the device-resident cos/sin caches held in the
        global ``_ROPE_STATE``. When ``force`` is True (snapshot restore) the
        static cos/sin table is recomputed if the currently cached one is
        missing or stale (its device memory was invalidated by suspend/resume).
        The runtime scratch buffers only need to *exist* (they are fully
        overwritten per-forward before being read), so they are never rebuilt."""
        config_key = self._config_key
        rotary_dim = self.rotary_dim
        device = current_platform.device_type

        need_static = config_key not in _ROPE_STATE.static_cache
        if force:
            need_static = not self._static_cache_is_valid(config_key, device)

        if need_static:
            # ``precompute_freqs_cis`` builds ``inv_freq`` without an explicit
            # device: at cold start a default device=npu context puts it on the
            # NPU, but the restore path has no such context, so it lands on CPU
            # and the einsum below fails with a CPU/NPU device mismatch. Pin it
            # to ``device`` explicitly to be robust in both paths.
            inv_freq = self.precompute_freqs_cis(
                rotary_dim,
                self._max_position_embeddings,
                self._max_position_embeddings,
                self._base,
                self._scaling_factor,
                self._beta_fast,
                self._beta_slow,
            ).to(device)
            t = torch.arange(
                self._max_position_embeddings * self._scaling_factor,
                device=device,
                dtype=torch.float32,
            )
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos().repeat_interleave(2, dim=-1).to(device)
            sin = freqs.sin().repeat_interleave(2, dim=-1).to(device)
            # Assign only after a successful build so a failure never leaves an
            # empty/half entry (which would surface downstream as ``cos == {}``).
            _ROPE_STATE.static_cache[config_key] = (cos.unsqueeze(1).unsqueeze(1), sin.unsqueeze(1).unsqueeze(1))

        buffers = _ROPE_STATE.runtime_buffer.setdefault(config_key, {})
        max_batch_size = self._max_batch_size
        for grp in self._rope_groups:
            if grp not in buffers:
                buf_cos = torch.ones(max_batch_size, 1, 1, rotary_dim, dtype=torch.float32, device=device)
                buf_sin = torch.zeros(max_batch_size, 1, 1, rotary_dim, dtype=torch.float32, device=device)
                buffers[grp] = (buf_cos, buf_sin)

    def reload_derived_weights_after_restore(self, act_dtype: torch.dtype) -> None:
        """[snapshot] Rebuild the non-persistent rope cos/sin caches after a
        ``state_dict`` restore.

        The cos/sin tables are stored in the module-level ``_ROPE_STATE`` global
        rather than as ``nn.Module`` buffers, so ``dump_model`` never serializes
        them and ``restore_model`` never copies them back. After suspend/resume
        the device memory backing them is stale (observed as all-zero cos/sin),
        which zeroes rotary position encoding and corrupts attention for every
        layer. Recomputing them from the static rope parameters repairs the whole
        decode/prefill path. The ACL graph is re-captured after restore, so
        reallocating is safe."""
        self._build_cos_sin_cache(force=True)

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
