# SPDX-License-Identifier: Apache-2.0
"""Export real A3 PD checkpoints without changing tensor widths or quantization.

Run once before CI and retain the output manifest with the model redirect.
This is a functional reduced-model fixture, not full-model accuracy evidence.
"""

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import regex as re
from safetensors import safe_open
from safetensors.torch import save_file


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def prepare(source: Path, output: Path, kind: str) -> None:
    """Select genuine backbone layers; record every output hash."""
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    original_layers = text["num_hidden_layers"]
    layers = {"k3": 4, "dsv4": 4, "glm": 7, "minimax": 5}[kind]
    text["num_hidden_layers"] = layers
    # Reduced backbones do not preserve the full model's MTP acceptance.
    text["num_nextn_predict_layers"] = 0
    # Transformers validates these per-layer lists against num_hidden_layers.
    for key in ("layer_types", "mlp_layer_types"):
        if key in text:
            text[key] = text[key][:layers]
    if kind == "k3":
        config = text
        config.setdefault("model_type", "kimi_linear")
        # Older K3 text implementations read this explicit NoRoPE flag.
        config.setdefault("mla_use_rope", False)
        config["architectures"] = ["KimiK3ForCausalLM"]
        text["num_experts"] = 16
        text["num_expert_group"] = 1
        text["topk_group"] = 1
        text["linear_attn_config"]["kda_layers"] = [1, 2, 3]
        text["linear_attn_config"]["full_attn_layers"] = [4]
    elif kind == "dsv4":
        text["compress_ratios"] = [0, 0, 4, 128]
    elif kind == "glm":
        text["indexer_types"] = text["indexer_types"][:layers]
    else:
        sparse = text["sparse_attention_config"]
        for key in ("sparse_attention_freq", "sparse_disable_index_value"):
            sparse[key] = sparse[key][:layers]

    def selected(name: str) -> str | None:
        # Vision is irrelevant to these text-only cases, but retain its config.
        if name.startswith(("vision_tower.", "vision_model.", "multi_modal_projector.", "mm_projector.", "mtp.")):
            return None
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match:
            layer = int(match[1])
            if layer >= layers:
                return None
        expert = re.search(r"\.experts\.(\d+)\.", name)
        if kind == "k3" and expert and int(expert[1]) >= 16:
            return None
        # Use the registered text backbone across releases, rather than
        # relying on multimodal constructors honoring language-model-only.
        return name.removeprefix("language_model.") if kind == "k3" else name

    index_name = "quant_model_weights.safetensors.index.json"
    index = json.loads((source / index_name).read_text())
    groups = defaultdict(list)
    for name, shard in index["weight_map"].items():
        target = selected(name)
        if target is not None:
            groups[shard].append((name, target))
    weights = {}
    size = 0
    for number, (shard, names) in enumerate(sorted(groups.items())):
        tensors = {}
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            for name, target in names:
                tensor = reader.get_tensor(name)
                if kind == "k3" and name.endswith((".gate.weight", ".gate.e_score_correction_bias")):
                    tensor = tensor[:16].clone()
                tensors[target] = tensor.contiguous()
                size += tensor.numel() * tensor.element_size()
        target_shard = f"model-{number:05d}.safetensors"
        save_file(tensors, output / target_shard)
        weights.update(dict.fromkeys(tensors, target_shard))
        print(f"{kind}: exported {shard}: {len(tensors)} tensors", flush=True)
    # Preserve tokenizer and processor bytes. Do not copy unrelated full shards.
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".py", ".model", ".txt", ".jinja"):
            if path.name not in (
                "config.json",
                "quant_model_description.json",
                "pd_material_manifest.json",
                index_name,
            ):
                if not path.name.endswith(".index.json"):
                    shutil.copyfile(path, output / path.name)
    quant = json.loads((source / "quant_model_description.json").read_text())
    quant = {target: value for name, value in quant.items() if (target := selected(name)) is not None}
    (output / "quant_model_description.json").write_text(json.dumps(quant, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / index_name).write_text(json.dumps({"metadata": {"total_size": size}, "weight_map": weights}, indent=2))
    manifest = {
        "kind": kind,
        "source_config_sha256": digest(source / "config.json"),
        "source_index_sha256": digest(source / index_name),
        "source_quantization_sha256": digest(source / "quant_model_description.json"),
        "source_layers": original_layers,
        "retained_layers": layers,
        "tensor_count": len(weights),
        "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    if (source / "pd_material_manifest.json").is_file():
        manifest["source_material_manifest_sha256"] = digest(source / "pd_material_manifest.json")
    (output / "pd_material_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("k3", "dsv4", "glm", "minimax"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.kind)
