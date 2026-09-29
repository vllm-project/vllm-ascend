# SPDX-License-Identifier: Apache-2.0
"""Synthetic checkpoint fixtures for glm_reduced tests.

These fixtures are *transformation* fixtures only: tiny byte patterns shaped
like real GLM checkpoints. Weight-name layouts mirror the pinned upstream
index snapshots (see tools/glm_reduced/sources.json): flat ``model.layers.``
for GLM-4.x/5.x, ``model.language_model.layers.`` + ``model.visual.`` for
GLM-4.1V/GLM-5.3-Flash, ``transformer.encoder.layers.`` for ChatGLM. They are
never numerical or performance validation and must not be presented as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.glm_reduced.safetensors_io import write_safetensors_from_bytes

F16 = "F16"
F8 = "F8_E4M3"

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pattern_bytes(nbytes: int, seed: int) -> bytes:
    return bytes(((seed * 31 + i * 7) % 251) for i in range(nbytes))


def f16_tensor(shape: tuple[int, ...], seed: int) -> tuple[str, tuple[int, ...], bytes]:
    nbytes = 2
    for dim in shape:
        nbytes *= dim
    return F16, shape, pattern_bytes(nbytes, seed)


def write_shard_file(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    write_safetensors_from_bytes(str(path), tensors)


def write_aux(root: Path, tokenizer: bool = True) -> None:
    if tokenizer:
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")


def dsa_layer_names(idx: int, indexer_types: list[str]) -> list[str]:
    names = [
        f"model.layers.{idx}.self_attn.q_proj.weight",
        f"model.layers.{idx}.self_attn.kv_a_proj_with_mqa.weight",
        f"model.layers.{idx}.mlp.experts.0.gate_proj.weight",
        f"model.layers.{idx}.mlp.experts.1.down_proj.weight",
        f"model.layers.{idx}.input_layernorm.weight",
    ]
    # Mirrors the real GLM-5.2/5.3 layout: shared-indexer layers ship no
    # indexer weights; full-indexer (producer) layers and MTP layers do.
    if idx < len(indexer_types) and indexer_types[idx] == "full":
        names.append(f"model.layers.{idx}.self_attn.indexer.wq_b.weight")
    if idx >= len(indexer_types):  # MTP layers always carry a full indexer
        names.append(f"model.layers.{idx}.self_attn.indexer.wq_b.weight")
    return names


def default_indexer_types(num_layers: int) -> list[str]:
    types = ["full", "full", "full", "shared", "shared", "shared"]
    types += ["full", "shared", "shared", "shared"] * ((num_layers // 4) + 1)
    return types[:num_layers]


def make_dsa_checkpoint(
    root: Path,
    *,
    num_layers: int = 12,
    mtp_layers: int = 1,
    indexer_types: list[str] | None = None,
    tied: bool = False,
    with_index: bool = True,
    index_name: str = "model.safetensors.index.json",
    with_quant_description: bool = False,
    with_optional_subdir: bool = False,
) -> Path:
    """Flat GlmMoeDsaForCausalLM-style checkpoint (GLM-5.x shape)."""
    root.mkdir(parents=True)
    if indexer_types is None:
        indexer_types = default_indexer_types(num_layers)
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": num_layers,
        "first_k_dense_replace": 3,
        "hidden_size": 16,
        "vocab_size": 64,
        "tie_word_embeddings": tied,
        "num_nextn_predict_layers": mtp_layers,
        "indexer_types": indexer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (num_layers - 3),
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_aux(root)

    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for idx in range(num_layers + mtp_layers):
        for name in dsa_layer_names(idx, indexer_types):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 100 + len(name))
            tensors.append((name, dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=1)
    tensors.append(("model.embed_tokens.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((16,), seed=2)
    tensors.append(("model.norm.weight", dtype, shape, data))
    if not tied:
        dtype, shape, data = f16_tensor((64, 16), seed=3)
        tensors.append(("lm_head.weight", dtype, shape, data))

    if with_index:
        half = len(tensors) // 2
        write_shard_file(root / "model-00001-of-00002.safetensors", tensors[:half])
        write_shard_file(root / "model-00002-of-00002.safetensors", tensors[half:])
        weight_map = {}
        for i, (name, *_rest) in enumerate(tensors):
            weight_map[name] = f"model-0000{1 if i < half else 2}-of-00002.safetensors"
        total = sum(len(data) for *_x, data in tensors)
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (root / index_name).write_text(json.dumps(index), encoding="utf-8")
    else:
        write_shard_file(root / "model.safetensors", tensors)

    if with_quant_description:
        description = {
            "group_size": 0,
            "version": "1.0.0",
            "is_rot_used": True,
            "metadata": {},
            "optional": {},
            "model.embed_tokens.weight": "FLOAT",
        }
        if with_optional_subdir:
            description["optional"] = {"quarot": {"rotation_map": {"global_rotation": "optional/quarot.safetensors"}}}
        for idx in range(num_layers + mtp_layers):
            description[f"model.layers.{idx}.self_attn.q_proj.weight"] = "W8A8_DYNAMIC"
            description[f"model.layers.{idx}.self_attn.q_proj.weight_scale"] = "W8A8_DYNAMIC"
            description[f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"] = "W4A8_DYNAMIC"
        (root / "quant_model_description.json").write_text(json.dumps(description), encoding="utf-8")
    if with_optional_subdir:
        (root / "optional").mkdir()
        dtype, shape, data = f16_tensor((8, 8), seed=42)
        write_shard_file(root / "optional" / "quarot.safetensors", [("global_rotation", dtype, shape, data)])
    return root


def make_flash_checkpoint(root: Path, *, num_layers: int = 12, mtp_layers: int = 1, vision_layers: int = 2) -> Path:
    """Nested Glm5NextForConditionalGeneration-style checkpoint (GLM-5.3-Flash shape).

    Weight prefixes mirror the real index: model.language_model.layers.N.*,
    model.language_model.{embed_tokens,norm}, model.visual.*, lm_head.weight.
    """
    root.mkdir(parents=True)
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"] * 3
    layer_types = layer_types[:num_layers]
    text_config = {
        "num_hidden_layers": num_layers,
        "hidden_size": 16,
        "vocab_size": 64,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": mtp_layers,
        "first_k_dense_replace": 3,
        "layer_types": layer_types,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (num_layers - 3),
        "indexer_types": ["full"] * num_layers,
        "mhc": True,
        "linear_attn_config": {
            "num_heads": 4,
            "kda_layers": [i for i, t in enumerate(layer_types) if t == "linear_attention"],
            "full_attn_layers": [i for i, t in enumerate(layer_types) if t == "deepseek_sparse_attention"],
        },
    }
    not_convert = ["lm_head", "model.language_model.embed_tokens", "model.visual.patch_embed.proj"]
    for idx in range(num_layers + mtp_layers):
        not_convert.append(f"model.language_model.layers.{idx}.hc_attn_fn")
        not_convert.append(f"model.language_model.layers.{idx}.input_layernorm")
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "tie_word_embeddings": False,
        "text_config": text_config,
        "vision_config": {"model_type": "glm5_next_vision", "depth": vision_layers, "hidden_size": 8},
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": not_convert,
        },
        "image_token_id": 66,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_aux(root)
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for idx in range(num_layers + mtp_layers):
        for suffix in (
            "self_attn.q_a_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.wk.weight_scale_inv",
            "mlp.experts.0.gate_proj.weight",
            "hc_attn_fn.weight",
        ):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 97 + len(suffix))
            tensors.append((f"model.language_model.layers.{idx}.{suffix}", dtype, shape, data))
    for idx in range(vision_layers):
        dtype, shape, data = f16_tensor((8, 8), seed=500 + idx)
        tensors.append((f"model.visual.blocks.{idx}.wqkv.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((8, 8), seed=599)
    tensors.append(("model.visual.patch_embed.proj.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=601)
    tensors.append(("model.language_model.embed_tokens.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((16,), seed=602)
    tensors.append(("model.language_model.norm.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=603)
    tensors.append(("lm_head.weight", dtype, shape, data))
    write_shard_file(root / "model.safetensors", tensors)
    return root


def make_chatglm_checkpoint(
    root: Path, *, num_layers: int = 6, both_layer_keys: bool = True, consistent: bool = True
) -> Path:
    """ChatGLM-style checkpoint (transformer.encoder.layers.* layout)."""
    root.mkdir(parents=True)
    config = {
        "architectures": ["ChatGLMModel"],
        "model_type": "chatglm",
        "num_layers": num_layers,
        "hidden_size": 16,
        "padded_vocab_size": 64,
        "tie_word_embeddings": False,
    }
    if both_layer_keys:
        config["num_hidden_layers"] = num_layers if consistent else num_layers + 1
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_aux(root)

    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for idx in range(num_layers):
        for suffix in (
            "self_attention.query_key_value.weight",
            "self_attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
            "mlp.dense_h_to_4h.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 31 + len(suffix))
            tensors.append((f"transformer.encoder.layers.{idx}.{suffix}", dtype, shape, data))
    for name, seed in (
        ("transformer.embedding.word_embeddings.weight", 701),
        ("transformer.encoder.final_layernorm.weight", 702),
        ("transformer.output_layer.weight", 703),
    ):
        dtype, shape, data = f16_tensor((8, 8), seed=seed)
        tensors.append((name, dtype, shape, data))
    write_shard_file(root / "model.safetensors", tensors)
    return root


def make_glm4v_checkpoint(root: Path, *, num_layers: int = 6, vision_layers: int = 2) -> Path:
    """Nested Glm4vForConditionalGeneration-style checkpoint (GLM-4.1V shape)."""
    root.mkdir(parents=True)
    config = {
        "architectures": ["Glm4vForConditionalGeneration"],
        "model_type": "glm4v",
        "text_config": {
            "num_hidden_layers": num_layers,
            "hidden_size": 16,
            "vocab_size": 64,
            "tie_word_embeddings": False,
        },
        "vision_config": {"model_type": "glm4v_vision", "depth": vision_layers},
        "image_token_id": 66,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_aux(root)

    tensors: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for idx in range(num_layers):
        for suffix in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "mlp.gate_proj.weight"):
            dtype, shape, data = f16_tensor((4, 4), seed=idx * 17 + len(suffix))
            tensors.append((f"model.language_model.layers.{idx}.{suffix}", dtype, shape, data))
    for idx in range(vision_layers):
        dtype, shape, data = f16_tensor((8, 8), seed=800 + idx)
        tensors.append((f"model.visual.blocks.{idx}.attn.qkv.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((8, 8), seed=850)
    tensors.append(("model.visual.merger.proj.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=851)
    tensors.append(("model.language_model.embed_tokens.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((16,), seed=852)
    tensors.append(("model.language_model.norm.weight", dtype, shape, data))
    dtype, shape, data = f16_tensor((64, 16), seed=853)
    tensors.append(("lm_head.weight", dtype, shape, data))
    write_shard_file(root / "model.safetensors", tensors)
    return root


@pytest.fixture()
def dsa_checkpoint(tmp_path: Path) -> Path:
    return make_dsa_checkpoint(tmp_path / "src-dsa")


@pytest.fixture()
def flash_checkpoint(tmp_path: Path) -> Path:
    return make_flash_checkpoint(tmp_path / "src-flash")
