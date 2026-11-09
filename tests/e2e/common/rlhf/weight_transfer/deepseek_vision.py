# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced DeepSeek-V4 vision configuration shared by IPC and retention tests."""

import json
import os

from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.checkpoint_adapters import (
    deepseek_v4_checkpoint_name,
    deepseek_v4_vision_params,
)


def deepseek_vision_case() -> WeightUpdateModelCase:
    layers = int(os.environ.get("VLLM_RL_DSV4_TEXT_LAYERS", "4"))
    vision_layers = int(os.environ.get("VLLM_RL_DSV4_VISION_LAYERS", "2"))
    experts = int(os.environ.get("VLLM_RL_DSV4_EXPERTS", "8"))
    hash_layers = int(os.environ.get("VLLM_RL_DSV4_HASH_LAYERS", str(min(layers, 3))))
    if layers < 1 or vision_layers < 1 or experts < 6 or not 0 <= hash_layers <= layers:
        raise ValueError(
            "Vision reload needs text/vision layers >= 1, experts >= 6 and 0 <= hash layers <= text layers"
        )
    mode = os.environ.get("VLLM_RL_DSV4_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
    graph = {"cudagraph_mode": mode, "cudagraph_capture_sizes": [1]}
    encoder_graph = os.environ.get("VLLM_RL_DSV4_MM_ENCODER_GRAPH")
    if encoder_graph is not None:
        graph["cudagraph_mm_encoder"] = encoder_graph == "1"
    overrides = {
        "architectures": ["DeepseekV4ForConditionalGeneration"],
        "num_hidden_layers": layers,
        "vision_n_layers": vision_layers,
        "n_routed_experts": experts,
        "num_hash_layers": hash_layers,
        "num_nextn_predict_layers": 0,
        "dspark_target_layer_ids": [],
        "quantization_config": None,
    }
    return WeightUpdateModelCase(
        id=f"dsv4-vision-{layers}L-v{vision_layers}-e{experts}-h{hash_layers}-{mode}-enc{encoder_graph}",
        model=os.environ.get("VLLM_RL_DSV4_MODEL", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"),
        hf_overrides=overrides,
        checkpoint_name_map=deepseek_v4_checkpoint_name,
        expert_intermediate_size=2048,
        extra_param_builder=deepseek_v4_vision_params,
        image_prompt=True,
        compilation_config=json.dumps(graph),
        extra_server_args=(
            "--tokenizer-mode",
            "deepseek_v4",
            "--block-size",
            "32",
            "--limit-mm-per-prompt",
            '{"image": 1}',
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "1024",
            "--kv-cache-memory-bytes",
            str(128 * 2**20),
        ),
    )
