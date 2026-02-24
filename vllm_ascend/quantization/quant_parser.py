import torch

from vllm_ascend.quantization.mxfp_compat import (
    FLOAT4_E2M1FN_X2_DTYPE,
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_dtype_available,
    ensure_mxfp8_scale_dtype_available,
)


class QuantTypeMapping:
    quant_configs = {
        "W8A8_MXFP8": {
            "act_quant_type": torch.float8_e4m3fn,
            "weight_quant_type": None,
            "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        },
        "W4A4_MXFP4": {
            "act_quant_type": FLOAT4_E2M1FN_X2_DTYPE,
            "weight_quant_type": FLOAT4_E2M1FN_X2_DTYPE,
            "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        },
        "W4A8_MXFP": {
            "act_quant_type": torch.float8_e4m3fn,
            "weight_quant_type": FLOAT4_E2M1FN_X2_DTYPE,
            "scale_dtype": FLOAT8_E8M0FNU_DTYPE,
            "per_token_scale_dtype": FLOAT8_E8M0FNU_DTYPE,
        },
    }

    @staticmethod
    def get_quant_settings():
        return QuantTypeMapping.quant_configs


def get_rollback_quant_type(rollback_quant_config):
    rollback_quant_type = "W8A8_MXFP8"
    for k, v in rollback_quant_config.items():
        if "down_proj" in k:
            rollback_quant_type = v
    return rollback_quant_type


def parse_mxfp_quant_params(**kwargs):
    act_quant_type = kwargs.get("act_quant_type", torch.float8_e4m3fn)
    weight_quant_type = kwargs.get("weight_quant_type", torch.float8_e4m3fn)
    scale_type = kwargs.get("scale_type")
    per_token_scale_type = kwargs.get("per_token_scale_type")
    round_mode = kwargs.get("round_mode", "rint")
    return act_quant_type, weight_quant_type, scale_type, per_token_scale_type, round_mode


def parse_quant_moe_down_proj_params(rollback_quant_type, parsed_round_mode):
    if rollback_quant_type == "W4A4_MXFP4":
        ensure_mxfp4_dtype_available("W4A4_MXFP4 quantization")
    elif rollback_quant_type in ("W8A8_MXFP8", "W4A8_MXFP"):
        ensure_mxfp8_scale_dtype_available(f"{rollback_quant_type} quantization")

    quant_type_mapping = QuantTypeMapping.get_quant_settings()
    cur_rollback_quant_config = quant_type_mapping[rollback_quant_type]
    if rollback_quant_type in ["W4A4_MXFP4"]:  # w4a4mxfp4 round mode support round、rint
        round_mode = parsed_round_mode
    else:  # mxfp8 only support rint
        round_mode = "rint"
    return (
        cur_rollback_quant_config["act_quant_type"],
        cur_rollback_quant_config["weight_quant_type"],
        cur_rollback_quant_config["scale_dtype"],
        cur_rollback_quant_config["per_token_scale_dtype"],
        round_mode,
    )
