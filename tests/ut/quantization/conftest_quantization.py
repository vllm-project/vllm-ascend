from unittest.mock import Mock

import torch
import torch.nn as nn

FAKQUANT_CONFIG = {
    "version": "1.0.0",
    "model_quant_type": "W8A8_DYNAMIC",
    "fa_quant_type": "FAKQuant",
    "model.embed_tokens.weight": "FLOAT",
    "model.layers.3.self_attn.fa_q.scale": "FAQuant",
    "model.layers.3.self_attn.fa_k.scale": "FAQuant",
    "model.layers.3.self_attn.fa_v.scale": "FAQuant",
    "model.layers.3.self_attn.fa_q.offset": "FAQuant",
    "model.layers.3.self_attn.fa_k.offset": "FAQuant",
    "model.layers.3.self_attn.fa_v.offset": "FAQuant",
}

W8A8_CONFIG = {
    "version": "1.0.0",
    "model_quant_type": "W8A8_DYNAMIC",
    "model.embed_tokens.weight": "FLOAT",
    "model.layers.0.self_attn.q_a_proj.weight": "W8A8",
    "model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
    "model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.0.up_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.1.gate_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.1.up_proj.weight": "W8A8_DYNAMIC",
    "model.layers.3.mlp.experts.1.down_proj.weight": "W8A8_DYNAMIC",
}

COMPRESSED_TENSORS_W8A8_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "int-quantized",
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": True,
                "group_size": None,
                "num_bits": 8,
                "observer": None,
                "observer_kwargs": {},
                "strategy": "token",
                "symmetric": True,
                "type": "int",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 8,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "channel",
                "symmetric": True,
                "type": "int",
            },
        }
    },
    "format": "int-quantized",
    "global_compression_ratio": None,
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}


def identity(*args):
    return args[0]


def create_mock_vllm_config(
    quant_description=None,
    model_dtype=torch.bfloat16,
    scheduler_config=None,
    compilation_mode=None,
    enforce_eager=True,
    kv_transfer_config=None,
    parallel_config=None,
):
    if quant_description is None:
        quant_description = {"group_size": 32}

    mock_config = Mock()
    mock_config.quant_config = Mock(quant_description=quant_description)
    mock_config.model_config = Mock(
        dtype=model_dtype,
        hf_config=Mock(model_type=None),
        enforce_eager=enforce_eager,
    )

    if scheduler_config is None:
        mock_config.scheduler_config = Mock(
            max_num_batched_tokens=2048,
            max_model_len=2048,
            enable_chunked_prefill=False,
        )
    else:
        mock_config.scheduler_config = scheduler_config

    if compilation_mode is not None:
        mock_config.compilation_config = Mock(mode=compilation_mode)
    else:
        mock_config.compilation_config = Mock()

    mock_config.kv_transfer_config = kv_transfer_config

    if parallel_config is None:
        mock_config.parallel_config = Mock(enable_expert_parallel=True)
    else:
        mock_config.parallel_config = parallel_config

    return mock_config


def create_mock_ascend_config(
    multistream_overlap_gate=False,
    dynamic_eplb=False,
    flashcomm2_oproj_tensor_parallel_size=1,
):
    mock_config = Mock()
    mock_config.multistream_overlap_gate = multistream_overlap_gate
    mock_config.eplb_config = Mock(dynamic_eplb=dynamic_eplb)
    mock_config.flashcomm2_oproj_tensor_parallel_size = flashcomm2_oproj_tensor_parallel_size
    return mock_config


def create_moe_layer(
    num_experts=8,
    hidden_size=128,
    intermediate_size=128,
    weight_dtype=torch.int8,
    params_dtype=torch.bfloat16,
):
    layer = nn.Module()
    layer.w13_weight = nn.Parameter(
        torch.randint(-8, 8, (num_experts, 2 * intermediate_size, hidden_size), dtype=weight_dtype),
        requires_grad=False,
    )
    layer.w2_weight = nn.Parameter(
        torch.randint(-8, 8, (num_experts, hidden_size, intermediate_size), dtype=weight_dtype),
        requires_grad=False,
    )
    layer.w13_weight_scale = nn.Parameter(
        torch.ones((num_experts, 2 * intermediate_size, 1), dtype=params_dtype), requires_grad=False
    )
    layer.w13_weight_offset = nn.Parameter(
        torch.zeros((num_experts, 2 * intermediate_size, 1), dtype=params_dtype), requires_grad=False
    )
    layer.w2_weight_scale = nn.Parameter(
        torch.ones((num_experts, hidden_size, 1), dtype=params_dtype), requires_grad=False
    )
    layer.w2_weight_offset = nn.Parameter(
        torch.zeros((num_experts, hidden_size, 1), dtype=params_dtype), requires_grad=False
    )
    return layer


def create_mxfp_moe_layer(
    num_experts=8,
    hidden_size=128,
    intermediate_size=128,
    group_size=32,
    weight_dtype=torch.float8_e4m3fn,
    scale_dtype=torch.uint8,
):
    layer = nn.Module()
    layer.w13_weight = nn.Parameter(
        torch.randn(num_experts, 2 * intermediate_size, hidden_size).to(weight_dtype), requires_grad=False
    )
    layer.w2_weight = nn.Parameter(
        torch.randn(num_experts, hidden_size, intermediate_size).to(weight_dtype), requires_grad=False
    )
    layer.w13_weight_scale = nn.Parameter(
        torch.randint(0, 255, (num_experts, 2 * intermediate_size, hidden_size // group_size), dtype=scale_dtype),
        requires_grad=False,
    )
    layer.w2_weight_scale = nn.Parameter(
        torch.randint(0, 255, (num_experts, hidden_size, intermediate_size // group_size), dtype=scale_dtype),
        requires_grad=False,
    )
    return layer
