# SPDX-License-Identifier: Apache-2.0
"""Build reduced Flash configs from an actual W8A8 checkpoint, not FP8 defaults."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

LAYER_PATTERN = re.compile(r"(?P<prefix>(?:^|\.)layers\.)(?P<index>\d+)\.")


def cut_config(source, layers=9, mtp=False, multimodal=False):
    config = copy.deepcopy(source)
    text = config["text_config"]
    original_layers = text["num_hidden_layers"]
    if layers not in (5, 9) or layers >= original_layers:
        raise ValueError("Supported initial profiles retain 5 or 9 original layers")
    for name, value in list(text.items()):
        if isinstance(value, list) and len(value) == original_layers:
            if name not in ("layer_types", "mlp_layer_types", "indexer_types"):
                raise ValueError(f"Unreviewed per-layer field: {name}")
            text[name] = value[:layers]
    linear = text["linear_attn_config"]
    for name in ("kda_layers", "full_attn_layers"):
        linear[name] = [i for i in linear[name] if i < layers]
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = 1 if mtp else 0
    for item in (config, text):
        if "quantization_config" in item:
            raise ValueError("Use W8A8 checkpoint config, not an FP8 configuration")
    if not multimodal:
        config = text
        config["architectures"] = ["Glm5NextForCausalLM"]
    return config


def cut_quant(source, original_layers, layers=9, mtp=False, multimodal=False):
    if source.get("model_quant_type") not in ("W8A8", "W8A8_DYNAMIC"):
        raise ValueError("Expected a ModelSlim W8A8 description")
    result = {}
    for name, value in source.items():
        match = LAYER_PATTERN.search(name)
        if match:
            index = int(match["index"])
            if index >= layers:
                if not mtp or index != original_layers:
                    continue
                start, end = match.span("index")
                name = name[:start] + str(layers) + name[end:]
        if not multimodal:
            if name.startswith(("model.visual.", "visual.", "model.vision")):
                continue
            name = name.replace("model.language_model.", "model.", 1)
        if name in result:
            raise ValueError(f"Quant metadata collision: {name}")
        result[name] = value
    return result


def build(checkpoint, output, layers=9, mtp=False, multimodal=False):
    checkpoint, output = Path(checkpoint), Path(output)
    config_path = checkpoint / "config.json"
    quant_path = checkpoint / "quant_model_description.json"
    source = json.loads(config_path.read_text())
    quant = json.loads(quant_path.read_text())
    config = cut_config(source, layers, mtp, multimodal)
    reduced_quant = cut_quant(quant, source["text_config"]["num_hidden_layers"], layers, mtp, multimodal)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("config.json", config), ("quant_model_description.json", reduced_quant)):
        (output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    provenance = {
        "synthetic_weights": True,
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_quant_sha256": hashlib.sha256(quant_path.read_bytes()).hexdigest(),
        "main_layer_mapping": {str(i): i for i in range(layers)},
        "mtp_mapping": {str(source["text_config"]["num_hidden_layers"]): layers} if mtp else {},
        "layers": layers,
        "mtp": mtp,
        "multimodal": multimodal,
        "quant_entries_before": len(quant),
        "quant_entries_after": len(reduced_quant),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=9, choices=(5, 9))
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.checkpoint, args.output, args.layers, args.mtp, args.multimodal), indent=2))
