# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


def _dense_sp_layer(model: Any) -> Any:
    return model.language_model.model.layers[0]


def inspect_dense_sp_model(model: Any) -> dict[str, Any]:
    layer = _dense_sp_layer(model)
    text_config = model.language_model.model.config
    layers = model.language_model.model.layers
    output_projection = layer.linear_attn.out_proj if layer.layer_type == "linear_attention" else layer.self_attn.o_proj
    return {
        "model_class": type(model).__name__,
        "layer_class": type(layer).__name__,
        "dense_sp": layer._use_dense_sequence_parallel,
        "model_sp_contract": layer.use_attn_reduce_scatter_for_moe,
        "attention_reduce_results": output_projection.reduce_results,
        "mlp_reduce_results": layer.mlp.down_proj.reduce_results,
        "hidden_size": text_config.hidden_size,
        "intermediate_size": text_config.intermediate_size,
        "num_layers": len(layers),
        "collectives": getattr(layer, "_dense_sp_collective_events", []),
    }


def instrument_dense_sp_collectives(model: Any) -> dict[str, Any]:
    from vllm_ascend.patch.worker import patch_qwen3_5

    layer = _dense_sp_layer(model)
    events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    layer._dense_sp_collective_events = events
    original_all_gather = patch_qwen3_5.sp_all_gather
    original_reduce_scatter = patch_qwen3_5.sp_reduce_scatter

    def traced_all_gather(value: Any) -> Any:
        output = original_all_gather(value)
        events.append(("all_gather", tuple(value.shape), tuple(output.shape)))
        return output

    def traced_reduce_scatter(value: Any) -> Any:
        output = original_reduce_scatter(value)
        events.append(("reduce_scatter", tuple(value.shape), tuple(output.shape)))
        return output

    patch_qwen3_5.sp_all_gather = traced_all_gather
    patch_qwen3_5.sp_reduce_scatter = traced_reduce_scatter
    return inspect_dense_sp_model(model)
