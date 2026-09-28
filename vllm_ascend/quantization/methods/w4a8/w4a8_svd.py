"""Checkpoint-native packed INT4 low-rank MoE parameters and loader."""

from functools import partial

import torch
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoELowRankLinear
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.moe_svd import FORMAT_VERSION, validate_factor_dimensions


def load_factor(
    param, loaded_weight, weight_name=None, *, shard_id, expert_id, return_success=False, local_map, kind, combined
):
    if shard_id not in (("w1", "w3") if combined else ("w2",)):
        raise ValueError(f"Invalid low-rank factor shard: {shard_id}")
    local = local_map.get(expert_id)
    if local is None:
        return False if return_success else None
    target = param.data[(0 if shard_id == "w1" else 1), local] if combined else param.data[local]
    if kind == "weight":
        if loaded_weight.dtype != torch.int8:
            raise ValueError("Low-rank checkpoint weight must be packed INT4 in int8 storage")
        value = loaded_weight.T.contiguous().view(torch.int32)
    elif kind == "scale":
        if (
            loaded_weight.dtype != torch.float32
            or not torch.isfinite(loaded_weight).all()
            or not (loaded_weight > 0).all()
        ):
            raise ValueError("Expected finite positive FP32 low-rank scales")
        value = loaded_weight.T.contiguous().view(torch.int32).to(torch.int64)
    elif kind == "bias":
        if loaded_weight.dtype != torch.float32 or not torch.isfinite(loaded_weight).all():
            raise ValueError("Expected finite FP32 low-rank compensation biases")
        value = loaded_weight
    else:
        raise ValueError(f"Invalid low-rank factor kind: {kind}")
    if target.shape != value.shape:
        raise ValueError(f"Low-rank factor shape mismatch: {target.shape} != {value.shape}")
    target.copy_(value)
    return True if return_success else None


class AscendW4A8SVDMoEScheme(AscendW4A8DynamicFusedMoEMethod):
    supports_eplb = False

    def get_low_rank_weights(self, layer):
        def projection(prefix):
            def get(suffix):
                return getattr(layer, f"{prefix}_{suffix}")

            return MoELowRankLinear(
                get("left_weight"),
                get("right_weight"),
                get("left_scale"),
                get("right_scale"),
                get("left_bias"),
                get("right_bias"),
            )

        return (projection("w1"), projection("w3"), projection("w2"))

    def process_weights_after_loading(self, layer):
        # Give gate/up independent packed allocations, then release their
        # combined loading tensors. Dense expert matrices are never created.
        for side in ("left", "right"):
            for kind in ("weight", "scale", "bias"):
                name = f"w13_{side}_{kind}"
                parameter = getattr(layer, name)
                for branch, prefix in enumerate(("w1", "w3")):
                    layer.register_parameter(
                        f"{prefix}_{side}_{kind}",
                        torch.nn.Parameter(parameter[branch].clone(), requires_grad=False),
                    )
                delattr(layer, name)
        weight_bytes = sum(
            parameter.numel() * parameter.element_size()
            for name, parameter in layer.named_parameters(recurse=False)
            if name.endswith(("left_weight", "right_weight"))
        )
        logger.info_once(
            f"Low-rank MoE packed factors active: local weight bytes per layer={weight_bytes}; "
            "six W4A8 grouped products, no dense expert reconstruction.",
            scope="local",
        )


class AscendW4A8SVDFusedMoEMethod(AscendFusedMoEMethod):
    def __init__(self, moe_config, config, tid2eid=None):
        vllm_config = get_current_vllm_config()
        hf_config = vllm_config.model_config.hf_config
        if hf_config.model_type != "deepseek_v3":
            raise ValueError("Low-rank ModelSlim MoE requires a DeepSeek-V3 checkpoint")
        if (
            config.get("format_version") != FORMAT_VERSION
            or config.get("weight_bits") != 4
            or config.get("group_size") != 0
        ):
            raise ValueError("Unsupported low-rank MoE checkpoint format")
        if config.get("activation_bits") != 8:
            raise ValueError("Low-rank W4A8 requires A8 activations")
        validate_factor_dimensions(config.get("rank"), hf_config.hidden_size, hf_config.moe_intermediate_size)
        activation = getattr(moe_config.activation, "value", moe_config.activation)
        if activation != "silu" or moe_config.has_bias:
            raise ValueError("Low-rank MoE requires bias-free SiLU experts")
        if vllm_config.lora_config is not None:
            raise ValueError("Low-rank MoE does not support LoRA adapters")
        if (
            not vllm_config.parallel_config.enable_expert_parallel
            and vllm_config.parallel_config.tensor_parallel_size > 1
        ):
            raise ValueError("Low-rank MoE with TP > 1 requires expert parallelism")
        scheme = AscendW4A8SVDMoEScheme()
        if scheme.dynamic_eplb or vllm_config.parallel_config.enable_eplb:
            raise ValueError("Low-rank MoE requires static expert placement")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Low-rank MoE currently requires BF16 model activations")
        super().__init__(scheme, moe_config, tid2eid)
        self.rank = config["rank"]

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **extra_weight_attrs
    ):
        validate_factor_dimensions(self.rank, hidden_size, intermediate_size_per_partition)
        mapping = getattr(layer, "expert_map", None)
        if mapping is None:
            mapping = getattr(layer, "_expert_map", None)
        if mapping is None:
            start = getattr(layer, "ep_rank", 0) * num_experts
            local_map = {start + i: i for i in range(num_experts)}
        else:
            local_map = {i: local for i, local in enumerate(mapping.cpu().tolist()) if local >= 0}
        for prefix, out_size, in_size, combined in (
            ("w13", intermediate_size_per_partition, hidden_size, True),
            ("w2", hidden_size, intermediate_size_per_partition, False),
        ):
            for side, out_width, in_width in (("left", out_size, self.rank), ("right", self.rank, in_size)):
                leading = (2, num_experts) if combined else (num_experts,)
                for kind, shape, dtype in (
                    ("weight", (in_width, out_width // 8), torch.int32),
                    ("scale", (1, out_width), torch.int64),
                    ("bias", (out_width,), torch.float32),
                ):
                    param = torch.nn.Parameter(torch.empty((*leading, *shape), dtype=dtype), requires_grad=False)
                    layer.register_parameter(f"{prefix}_{side}_{kind}", param)
                    set_weight_attrs(
                        param,
                        {
                            **extra_weight_attrs,
                            "weight_loader": partial(
                                load_factor,
                                local_map=local_map,
                                kind=kind,
                                combined=combined,
                            ),
                        },
                    )
