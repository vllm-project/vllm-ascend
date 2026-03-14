import json
import os
import logging
from typing import Dict, Optional, Any, Union
import torch

from vllm_ascend.device.mxfp_compat import (
    FLOAT4_E2M1FN_X2_DTYPE,
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_dtype_available,
    ensure_mxfp8_scale_dtype_available,
)

logger = logging.getLogger(__name__)


class QuantTypeMapping:
    """量化类型映射配置"""
    
    # 支持的量化类型列表
    SUPPORTED_QUANT_TYPES = ['W8A8', 'W8A8_DYNAMIC', 'C8', 'W8A8_MXFP8', 'W4A4_MXFP4', 'W4A8_MXFP']
    
    # 量化类型配置映射
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
    
    @staticmethod
    def get_supported_types():
        return QuantTypeMapping.SUPPORTED_QUANT_TYPES


def sanitize_quant_type(quant_type: Any, field_name: str = None) -> Optional[str]:
   
    if quant_type is None:
        if field_name:
            logger.debug(f"Quant type for field '{field_name}' is null/empty")
        return None

    # Convert to string, strip whitespace, and check for "null"
    s_quant_type = str(quant_type).strip()
    if not s_quant_type or s_quant_type.lower() == "null":
        if field_name:
            logger.debug(f"Quant type for field '{field_name}' is null/empty")
        return None
    
    return s_quant_type


def load_quant_model_description(model_path: str) -> Dict[str, Any]:
    
    quant_desc_path = os.path.join(model_path, "quant_model_description.json")
    
    if not os.path.exists(quant_desc_path):
        # 尝试其他可能的文件名
        alt_paths = [
            os.path.join(model_path, "quant_config.json"),
            os.path.join(model_path, "quantization_description.json"),
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                quant_desc_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Quantization description not found in: {model_path}")
    
    logger.info(f"Loading quantization description from: {quant_desc_path}")
    
    with open(quant_desc_path, 'r', encoding='utf-8') as f:
        quant_desc = json.load(f)
    
    return quant_desc


def process_quant_description(quant_desc: Dict[str, Any]) -> Dict[str, Any]:
    
    if not isinstance(quant_desc, dict):
        return quant_desc
    
    processed_desc = quant_desc.copy()
    
    # 需要处理的量化类型字段
    quant_type_fields = ['fa_quant_type', 'kv_quant_type', 'weight_quant_type', 'act_quant_type']
    
    for field in quant_type_fields:
        if field in processed_desc:
            original_value = processed_desc[field]
            sanitized_value = sanitize_quant_type(original_value, field)
            
            if sanitized_value is None:
                # 根据字段设置合理的默认值
                if field == 'kv_quant_type':
                    processed_desc[field] = 'C8'
                    logger.info(f"Field '{field}' was null, set to default 'C8'")
                elif field == 'fa_quant_type':
                    processed_desc[field] = 'W8A8'
                    logger.info(f"Field '{field}' was null, set to default 'W8A8'")
                else:
                    # 对于其他字段，如果为null则删除
                    del processed_desc[field]
                    logger.debug(f"Removed null field: {field}")
            else:
                processed_desc[field] = sanitized_value
    
    return processed_desc


def validate_quant_types(quant_desc: Dict[str, Any]) -> bool:
    
    supported_types = QuantTypeMapping.get_supported_types()
    
    # 检查所有量化类型字段
    quant_type_fields = ['fa_quant_type', 'kv_quant_type', 'weight_quant_type', 'act_quant_type']
    
    for field in quant_type_fields:
        if field in quant_desc:
            quant_type = quant_desc[field]
            
            if quant_type is None:
                # 已经在上一步处理中设置为默认值，跳过
                continue
            
            if quant_type not in supported_types:
                raise NotImplementedError(
                    f"Currently, vLLM Ascend only supports following quant types: {supported_types}. "
                    f"Got '{quant_type}' for field '{field}'"
                )
    
    return True


def load_and_validate_quant_config(model_path: str) -> Dict[str, Any]:
    
    # 1. 加载量化描述
    quant_desc = load_quant_model_description(model_path)
    
    # 2. 处理null值
    processed_desc = process_quant_description(quant_desc)
    
    # 3. 验证量化类型
    validate_quant_types(processed_desc)
    
    logger.info(f"Quantization config loaded successfully: {list(processed_desc.keys())}")
    
    return processed_desc


def get_rollback_quant_type(rollback_quant_config: Dict[str, Any]) -> str:
    
    rollback_quant_type = "W8A8_MXFP8"
    
    if not rollback_quant_config:
        return rollback_quant_type
    
    for k, v in rollback_quant_config.items():
        if "down_proj" in k:
            # 清理并验证量化类型
            sanitized_type = sanitize_quant_type(v, f"rollback_{k}")
            if sanitized_type is not None:
                rollback_quant_type = sanitized_type
                break
    
    return rollback_quant_type


def parse_mxfp_quant_params(**kwargs) -> tuple:
   
    act_quant_type = kwargs.get("act_quant_type", torch.float8_e4m3fn)
    weight_quant_type = kwargs.get("weight_quant_type", torch.float8_e4m3fn)
    scale_type = kwargs.get("scale_type")
    per_token_scale_type = kwargs.get("per_token_scale_type")
    round_mode = kwargs.get("round_mode", "rint")
    
    return act_quant_type, weight_quant_type, scale_type, per_token_scale_type, round_mode


def parse_quant_moe_down_proj_params(rollback_quant_type: Optional[str], parsed_round_mode: str) -> tuple:
   
    # 处理可能的null值
    if rollback_quant_type is None or rollback_quant_type == "null":
        rollback_quant_type = "W8A8_MXFP8"
        logger.debug(f"Rollback quant type was null, using default: {rollback_quant_type}")
    
    rollback_quant_type = rollback_quant_type.strip() if isinstance(rollback_quant_type, str) else str(rollback_quant_type)
    
    # 验证量化类型
    if rollback_quant_type not in QuantTypeMapping.get_supported_types():
        logger.warning(f"Unsupported quant type: {rollback_quant_type}, falling back to W8A8_MXFP8")
        rollback_quant_type = "W8A8_MXFP8"
    
    # 确保必要的数据类型可用
    if rollback_quant_type == "W4A4_MXFP4":
        ensure_mxfp4_dtype_available("W4A4_MXFP4 quantization")
    elif rollback_quant_type in ("W8A8_MXFP8", "W4A8_MXFP"):
        ensure_mxfp8_scale_dtype_available(f"{rollback_quant_type} quantization")
    else:
        logger.warning(f"Unhandled quant type: {rollback_quant_type}")
    
    # 获取量化配置
    quant_type_mapping = QuantTypeMapping.get_quant_settings()
    
    if rollback_quant_type not in quant_type_mapping:
        logger.warning(f"Quant type {rollback_quant_type} not in mapping, using W8A8_MXFP8")
        rollback_quant_type = "W8A8_MXFP8"
    
    cur_rollback_quant_config = quant_type_mapping[rollback_quant_type]
    
    # 确定舍入模式
    if rollback_quant_type in ["W4A4_MXFP4"]:
        # w4a4mxfp4 支持 round 和 rint
        round_mode = parsed_round_mode
    else:
        # mxfp8 只支持 rint
        round_mode = "rint"
    
    return (
        cur_rollback_quant_config["act_quant_type"],
        cur_rollback_quant_config["weight_quant_type"],
        cur_rollback_quant_config["scale_dtype"],
        cur_rollback_quant_config["per_token_scale_dtype"],
        round_mode,
    )


def get_quantized_model_config(model_path: str) -> Dict[str, Any]:
   
    try:
        # 加载并验证量化配置
        quant_config = load_and_validate_quant_config(model_path)
        
        # 处理回退量化类型
        if "rollback_quant_config" in quant_config:
            rollback_type = get_rollback_quant_type(quant_config["rollback_quant_config"])
            quant_config["effective_rollback_type"] = rollback_type
        
        logger.info(f"Quantized model config loaded successfully from: {model_path}")
        return quant_config
        
    except Exception as e:
        logger.error(f"Failed to load quantized model config: {e}")
        raise


# 兼容性函数
def ensure_quant_types_valid(quant_desc: Dict[str, Any]) -> Dict[str, Any]:
   
    return process_quant_description(quant_desc)


def main():
    """测试函数"""
    # 示例用法
    test_quant_desc = {
        "fa_quant_type": None,
        "kv_quant_type": "C8",
        "weight_quant_type": "W8A8",
        "act_quant_type": "W8A8",
        "other_field": "value"
    }
    
    print("Original quant description:", test_quant_desc)
    processed = process_quant_description(test_quant_desc)
    print("Processed quant description:", processed)
    
    try:
        validate_quant_types(processed)
        print("Quant types validation passed!")
    except NotImplementedError as e:
        print(f"Validation failed: {e}")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    main()
