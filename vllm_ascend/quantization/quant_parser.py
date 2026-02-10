import torch
import torch_npu


class QuantTypeMapping:
    class QuantTypeMapping:
    quant_configs = {
        "W8A8_MXFP8": {
            "act_quant_type": torch.float8_e4m3fn,
            "weight_quant_type": None,
            "scale_dtype": torch_npu.float8_e8m0fnu,
            "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
        },
        "W4A4_MXFP4": {
            "act_quant_type": torch_npu.float4_e2m1fn_x2,
            "weight_quant_type": torch_npu.float4_e2m1fn_x2,
            "scale_dtype": torch_npu.float8_e8m0fnu,
            "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
        },
        "W4A8_MXFP": {
            "act_quant_type": torch.float8_e4m3fn,
            "weight_quant_type": torch_npu.float4_e2m1fn_x2,
            "scale_dtype": torch_npu.float8_e8m0fnu,
            "per_token_scale_dtype": torch_npu.float8_e8m0fnu,
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


def parse_a5_quant_params(**kwargs):
    act_quant_type = kwargs.get("act_quant_type", torch.float8_e4m3fn)
    weight_quant_type = kwargs.get("weight_quant_type", torch.float8_e4m3fn)
    scale_type = kwargs.get("scale_type", None)
    per_token_scale_type = kwargs.get("per_token_scale_type", None)
    round_mode = kwargs.get("round_mode", "rint")
    return act_quant_type, weight_quant_type, scale_type, per_token_scale_type, round_mode


def parse_quant_moe_down_proj_params(rollback_quant_type, parsed_round_mode):
    quant_type_mapping = QuantTypeMapping.get_quant_settings()
    cur_rollback_quant_config = quant_type_mapping[rollback_quant_type]
    if rollback_quant_type in ["W4A4_MXFP4"]: # w4a4mxfp4 round mode支持round、rint
        round_mode = parsed_round_mode
    else: # mxfp8只支持rint
        round_mode = "rint"
    return cur_rollback_quant_config["act_quant_type"], cur_rollback_quant_config["weight_quant_type"], \
        cur_rollback_quant_config["scale_dtype"], cur_rollback_quant_config["per_token_scale_dtype"], round_mode
