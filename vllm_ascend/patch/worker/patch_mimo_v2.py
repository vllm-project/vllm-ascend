# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# MiMo-V2 on Ascend: fp8 load dequant.
#

from collections.abc import Iterable
from importlib.util import find_spec

import torch
from vllm.platforms import current_platform

_HAS_LEGACY_MIMO_V2 = find_spec("vllm.model_executor.models.mimo_v2") is not None

if _HAS_LEGACY_MIMO_V2:
    import vllm.model_executor.models.mimo_v2 as mimo_v2_mod
    from vllm.config import CacheConfig
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
    from vllm.model_executor.layers.quantization import QuantizationConfig
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.mimo_v2 import (
        MiMoV2Attention,
        MiMoV2FlashForCausalLM,
        MiMoV2ForCausalLM,
    )
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniForCausalLM
    from vllm.v1.attention.backend import AttentionType

FP8_DTYPES = tuple(
    getattr(torch, dtype_name)
    for dtype_name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, dtype_name)
)


def _need_dequantize_fp8_weights(self) -> bool:
    quant_cfg = getattr(self.config, "quantization_config", None)
    return (
        isinstance(quant_cfg, dict)
        and quant_cfg.get("quant_method") == "fp8"
        and current_platform.device_name == "npu"
    )


def _dequantize_fp8_block_weight(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    block_n, block_k = block_size
    n, k = fp8_weight.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    if tuple(weight_scale_inv.shape) != (n_tiles, k_tiles):
        raise ValueError(
            "Unexpected fp8 scale shape: "
            f"weight={tuple(fp8_weight.shape)}, "
            f"scale={tuple(weight_scale_inv.shape)}, "
            f"block_size={block_size}"
        )
    expanded_scale = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )
    expanded_scale = expanded_scale[:n, :k].to(dtype=torch.bfloat16)
    return fp8_weight.to(dtype=torch.bfloat16) * expanded_scale


def _fp8_dequant_weight_iter(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    quant_cfg = getattr(self.config, "quantization_config", {})
    block_cfg = quant_cfg.get("weight_block_size", [128, 128])
    weight_block_size: tuple[int, int] = (128, 128)
    if isinstance(block_cfg, list) and len(block_cfg) == 2:
        weight_block_size = (int(block_cfg[0]), int(block_cfg[1]))

    pending_fp8_weights: dict[str, torch.Tensor] = {}
    pending_fp8_scales: dict[str, torch.Tensor] = {}

    for name, loaded_weight in weights:
        if name.endswith(".weight_scale_inv"):
            paired_weight_name = name[: -len("_scale_inv")]
            pending_weight = pending_fp8_weights.pop(paired_weight_name, None)
            if pending_weight is None:
                pending_fp8_scales[name] = loaded_weight
                continue
            loaded_weight = self._dequantize_fp8_block_weight(
                pending_weight,
                loaded_weight,
                weight_block_size,
            )
            name = paired_weight_name
        elif loaded_weight.dtype in FP8_DTYPES and name.endswith(".weight"):
            scale_name = f"{name}_scale_inv"
            pending_scale = pending_fp8_scales.pop(scale_name, None)
            if pending_scale is None:
                pending_fp8_weights[name] = loaded_weight
                continue
            loaded_weight = self._dequantize_fp8_block_weight(
                loaded_weight,
                pending_scale,
                weight_block_size,
            )
        yield name, loaded_weight

    if pending_fp8_weights or pending_fp8_scales:
        raise ValueError(
            "Unpaired fp8 MiMo-V2 weight/scale tensors detected: "
            f"pending_weights={len(pending_fp8_weights)}, "
            f"pending_scales={len(pending_fp8_scales)}"
        )


if _HAS_LEGACY_MIMO_V2:

    def _patch_load_weights(cls) -> None:
        cls._need_dequantize_fp8_weights = _need_dequantize_fp8_weights
        cls._dequantize_fp8_block_weight = staticmethod(_dequantize_fp8_block_weight)
        cls._fp8_dequant_weight_iter = _fp8_dequant_weight_iter

        original_load_weights = cls.load_weights

        def _patched_load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> set[str]:
            if self._need_dequantize_fp8_weights():
                weights = self._fp8_dequant_weight_iter(weights)
            return original_load_weights(self, weights)

        cls.load_weights = _patched_load_weights

    for _cls in (MiMoV2FlashForCausalLM, MiMoV2ForCausalLM, MiMoV2OmniForCausalLM):
        _patch_load_weights(_cls)

    _original_mimo_v2_attention_init = MiMoV2Attention.__init__

    def _patched_mimo_v2_attention_init(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int | None = None,
        v_scale: float | None = None,
        sliding_window_size: int = -1,
        attention_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        max_position_embeddings: int = 32768,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        partial_rotary_factor: float = 1.0,
        prefix: str = "",
    ) -> None:
        if current_platform.device_name != "npu":
            _original_mimo_v2_attention_init(
                self,
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                v_head_dim,
                v_scale,
                sliding_window_size,
                attention_bias,
                add_swa_attention_sink_bias,
                layer_id,
                rope_theta,
                max_position_embeddings,
                cache_config,
                quant_config,
                partial_rotary_factor,
                prefix,
            )
            return

        super(MiMoV2Attention, self).__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = head_dim
        self.v_head_dim = head_dim if v_head_dim is None else v_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim
        self.scaling = v_scale if v_scale is not None else self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=self.v_head_dim,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config if "mtp.layers" not in prefix else None,
            reduce_results=True,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )

        self.attention_sink_bias = (
            torch.nn.Parameter(torch.empty(self.num_heads), requires_grad=False)
            if add_swa_attention_sink_bias
            else None
        )

        sliding_window = sliding_window_size if sliding_window_size > -1 else None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.attn",
            sinks=self.attention_sink_bias,
            head_size_v=self.v_head_dim,
        )
        if self.v_head_dim != self.head_dim:
            mimo_v2_mod.logger.info_once(
                "Using Ascend attention backend for MiMo-V2 diff-KV attention."
            )

    MiMoV2Attention.__init__ = _patched_mimo_v2_attention_init
