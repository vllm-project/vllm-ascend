# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Validate and rename native BF16/MXFP8 DSpark weights without converting tensors."""

import ast
import copy
import inspect
import json
import textwrap
from pathlib import Path
from types import CodeType

import regex as re

PREFIX = "language_model.model.dspark."
ADAPTER_ARCHITECTURE = "DSparkMiniMaxDraftModel"
MAX_SAFETENSORS_HEADER_BYTES = 10 * 1024 * 1024
MXFP8_GROUP_SIZE = 32
MINIMAX_M31_TARGET_LAYERS = (3, 17, 31, 45, 59)
MINIMAX_M31_DRAFT_BLOCK_SIZE = 7
MINIMAX_M31_DRAFT_WINDOW = 4096
MINIMAX_M31_MARKOV_RANK = 256
MINIMAX_M31_NOISE_TOKEN_ID = 200058
MINIMAX_M31_TARGET_HIDDEN_SIZE = 6144
MINIMAX_M31_TARGET_LAYER_COUNT = 60
MINIMAX_M31_VOCAB_SIZE = 200064


def map_weight_name(name):
    if not name.startswith(PREFIX):
        raise ValueError(f"Unexpected tensor outside standalone DSpark: {name}")
    name = name[len(PREFIX) :]
    name = re.sub(r"^(layers\.\d+)\.decoder_layer\.", r"\1.", name)
    if name.startswith("final_norm."):
        name = "norm." + name[len("final_norm.") :]
    return name


def expected_shapes(config):
    h = config["hidden_size"]
    hd = config["head_dim"]
    q = config["num_attention_heads"] * hd
    kv = config["num_key_value_heads"] * hd
    ffn = config["dspark_ffn_hidden_size"]
    n = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    rank = config["dspark_markov_rank"]
    shapes = {
        "fc.weight": [h, h * len(config["dspark_target_layer_ids"])],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
        "markov_head.markov_w1.weight": [vocab, rank],
        "markov_head.markov_w2.weight": [vocab, rank],
        "confidence_head.proj.weight": [1, h + rank],
        "confidence_head.proj.bias": [1],
    }
    for i in range(n):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[p + norm + ".weight"] = [h]
        for norm in ("q_norm", "k_norm"):
            shapes[p + "self_attn." + norm + ".weight"] = [hd]
        matrices = {
            "self_attn.q_proj": [q, h],
            "self_attn.k_proj": [kv, h],
            "self_attn.v_proj": [kv, h],
            "self_attn.o_proj": [h, q],
            "mlp.gate_proj": [ffn, h],
            "mlp.up_proj": [ffn, h],
            "mlp.down_proj": [h, ffn],
        }
        for name, shape in matrices.items():
            shapes[p + name + ".weight"] = shape
    return shapes


def validate_header(header, config):
    expected = expected_shapes(config)
    mapped = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        mapped_name = map_weight_name(name)
        if mapped_name in mapped:
            raise ValueError(f"Duplicate mapped tensor: {mapped_name}")
        mapped[mapped_name] = info
    uses_mxfp8 = any(info["dtype"] == "F8_E4M3" for info in mapped.values())
    quantized = {name: shape for name, shape in expected.items() if name.startswith("layers.") and len(shape) == 2}
    scales = (
        {
            name.removesuffix("weight") + "weight_scale_inv": [
                shape[0],
                (shape[1] + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE,
            ]
            for name, shape in quantized.items()
        }
        if uses_mxfp8
        else {}
    )
    schema = set(expected) | set(scales)
    if set(mapped) != schema:
        raise ValueError(f"Tensor schema mismatch: missing={schema - set(mapped)}, extra={set(mapped) - schema}")
    for name, shape in {**expected, **scales}.items():
        if list(mapped[name]["shape"]) != shape:
            raise ValueError(f"Tensor shape mismatch: {name}, {mapped[name]['shape']} != {shape}")
        dtype = mapped[name]["dtype"]
        expected_dtype = "U8" if name in scales else "F8_E4M3" if uses_mxfp8 and name in quantized else "BF16"
        if dtype != expected_dtype:
            raise ValueError(f"Expected native {expected_dtype} DSpark tensor, got {name}: {dtype}")
    return mapped


def iter_draft_weights(weights, config):
    # Keep header-only configuration preparation independent of Torch imports.
    import torch

    tensors = {}
    for name, value in weights:
        if name in tensors:
            raise ValueError(f"Duplicate input tensor: {name}")
        tensors[name] = value
    header = {}
    for name, value in tensors.items():
        header[name] = {
            "shape": list(value.shape),
            "dtype": {torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}.get(
                value.dtype, str(value.dtype)
            ),
        }
    validate_header(header, config)
    for name, value in tensors.items():
        yield map_weight_name(name).replace("weight_scale_inv", "weight_scale"), value


def read_draft_header(model_root):
    """Read draft metadata without loading or converting weight tensors."""
    draft = Path(model_root) / "dspark"
    shards = sorted(draft.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"MiniMax DSpark weights are missing from {draft}")
    header = {}
    for shard in shards:
        with shard.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size < MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"Invalid DSpark safetensors header size: {shard}")
            metadata = json.loads(stream.read(size))
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
            if name in header:
                raise ValueError(f"Duplicate DSpark tensor across shards: {name}")
            header[name] = info
    return header


def native_draft_config(source, header, draft_source=None):
    """Normalize checkpoint DSpark settings for the native Ascend implementation.

    Some ModelSlim exports retain the DSpark weights but omit their config.
    Prefer dspark/config.json. The config-less fallback matches MiniMaxAI's
    official M3.1 Flash revision 54e4e69. Restrict it to its target shape and validate
    every draft tensor; an unrelated checkpoint must not inherit this profile.
    """
    text = copy.deepcopy(source.get("text_config", source))
    if (
        text.get("hidden_size") != MINIMAX_M31_TARGET_HIDDEN_SIZE
        or text.get("num_hidden_layers") != MINIMAX_M31_TARGET_LAYER_COUNT
        or text.get("vocab_size") != MINIMAX_M31_VOCAB_SIZE
    ):
        profile = {name: text.get(name) for name in ("hidden_size", "num_hidden_layers", "vocab_size")}
        raise ValueError(
            f"The config-less BF16 DSpark fallback only supports the MiniMax M3.1 Flash profile: {profile}"
        )
    draft = copy.deepcopy(draft_source.get("text_config", draft_source)) if draft_source is not None else {}
    if draft_source is not None:
        architectures = draft_source.get("architectures", draft.get("architectures", []))
        if ADAPTER_ARCHITECTURE not in architectures:
            raise ValueError("Expected a MiniMax DSpark draft config, not a target model config")
    dflash = draft.get("dflash_config", {})
    target_layers = draft.get("dspark_target_layer_ids", dflash.get("target_layer_ids", MINIMAX_M31_TARGET_LAYERS))
    block_size = draft.get("dspark_block_size", dflash.get("block_size", MINIMAX_M31_DRAFT_BLOCK_SIZE))
    draft_layers = draft.get("num_hidden_layers", len(target_layers))
    window = draft.get("sliding_window", dflash.get("sliding_window", MINIMAX_M31_DRAFT_WINDOW))
    rank = draft.get("dspark_markov_rank", draft.get("markov_rank", MINIMAX_M31_MARKOV_RANK))
    noise_token = draft.get("dspark_noise_token_id", dflash.get("mask_token_id", MINIMAX_M31_NOISE_TOKEN_ID))
    layer_types = draft.get("layer_types", dflash.get("layer_types", ["sliding_attention"] * draft_layers))
    config = {**text, **draft}
    config.update(
        {
            "model_type": "minimax_m3_text",
            "architectures": [ADAPTER_ARCHITECTURE],
            "ascend_minimax_dspark": 1,
            "num_hidden_layers": draft_layers,
            "intermediate_size": draft.get("dspark_ffn_hidden_size", text["dense_intermediate_size"]),
            "dspark_ffn_hidden_size": draft.get("dspark_ffn_hidden_size", text["dense_intermediate_size"]),
            "dspark_target_layer_ids": list(target_layers),
            "target_layer_ids": list(target_layers),
            "dspark_block_size": block_size,
            "n_predict": block_size,
            "dspark_markov_rank": rank,
            "markov_rank": rank,
            "dspark_markov_head_type": draft.get("dspark_markov_head_type", "vanilla"),
            "markov_head_type": draft.get("dspark_markov_head_type", "vanilla"),
            "dspark_noise_token_id": noise_token,
            "mask_token_id": noise_token,
            "ptd_token_id": noise_token,
            "dspark_hidden_norm_type": draft.get("dspark_hidden_norm_type", "gemma"),
            "dspark_final_norm_type": draft.get("dspark_final_norm_type", "gemma"),
            "dspark_hybrid_context_fusion": draft.get("dspark_hybrid_context_fusion", False),
            "enable_confidence_head": draft.get("enable_confidence_head", True),
            "confidence_head_with_markov": draft.get("confidence_head_with_markov", True),
            "draft_vocab_size": config["vocab_size"],
            "num_nextn_predict_layers": 0,
            "num_mtp_modules": 0,
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "n_shared_experts": 0,
            "moe_layer_freq": [0] * draft_layers,
            "sparse_attention_config": None,
            "use_aux_hidden_state": True,
            "sample_from_anchor": True,
            "layer_types": layer_types,
            "sliding_window": window,
            "dflash_config": {
                **dflash,
                "num_hidden_layers": draft_layers,
                "num_target_layers": MINIMAX_M31_TARGET_LAYER_COUNT,
                "context_source": "target_layers",
                "block_size": block_size,
                "mask_token_id": noise_token,
                "target_layer_ids": list(target_layers),
                "layer_types": layer_types,
                "sliding_window": window,
            },
            "rope_parameters": draft.get("rope_parameters")
            or {
                "rope_type": "default",
                "rope_theta": text["rope_theta"],
                "partial_rotary_factor": text["rotary_dim"] / text["head_dim"],
            },
            "dtype": "bfloat16",
            "tie_word_embeddings": False,
        }
    )
    config.pop("quantization_config", None)
    mapped = validate_header(header, config)
    if any(info["dtype"] == "F8_E4M3" for info in mapped.values()):
        quant = {"quant_method": "ascend", "group_size": MXFP8_GROUP_SIZE}
        projections = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        )
        # DFlash offsets runtime attention names past the target's layers.
        for layer in range(draft_layers):
            for projection in projections:
                quant[f"model.layers.{MINIMAX_M31_TARGET_LAYER_COUNT + layer}.{projection}.weight"] = "W8A8_MXFP8"
        config["quantization_config"] = quant
    return config


def draft_hf_override(hf_config, *, model_root, target_override):
    # Called only for a MiniMax DSpark draft, through the target's copied
    # hf_overrides. A module-level callable remains picklable for workers.
    if callable(target_override):
        hf_config = target_override(hf_config)
    # HFConfigParser probes callables with an empty sentinel config before
    # loading the real checkpoint. Preserve its source config class selection.
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    config_path = Path(model_root) / "dspark" / "config.json"
    draft_source = json.loads(config_path.read_text()) if config_path.is_file() else None
    config = native_draft_config(hf_config.to_dict(), read_draft_header(model_root), draft_source)
    return MiniMaxM3TextConfig(**config)


def _allow_minimax_draft(post_init):
    # Keep references captured by Ascend's existing __post_init__ wrapper valid.
    # Remove this shim once upstream recognizes this standalone architecture.
    if ADAPTER_ARCHITECTURE in post_init.__code__.co_consts:
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(post_init)))
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.And):
            continue
        for condition in node.values:
            if (
                isinstance(condition, ast.Compare)
                and isinstance(condition.left, ast.Constant)
                and condition.left.value == "K3DSparkModel"
                and len(condition.ops) == 1
                and isinstance(condition.ops[0], ast.NotIn)  # codespell:ignore
                and ast.unparse(condition.comparators[0]) == "self.draft_model_config.architectures"
            ):
                matches.append((node, condition))
    if len(matches) != 1 or post_init.__closure__:
        raise RuntimeError("Core version changed: review the MiniMax DSpark architecture guard")
    node, condition = matches[0]
    addition = ast.Compare(
        left=ast.Constant(value=ADAPTER_ARCHITECTURE),
        ops=[condition.ops[0]],
        comparators=[condition.comparators[0]],
    )
    node.values.append(ast.copy_location(addition, condition))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, post_init.__code__.co_firstlineno - 1)
    module_code = compile(tree, post_init.__code__.co_filename, "exec")
    method_code = next(
        code for code in module_code.co_consts if isinstance(code, CodeType) and code.co_name == post_init.__name__
    )
    post_init.__code__ = method_code
