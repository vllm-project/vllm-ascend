#!/usr/bin/env python3
"""classify_model.py — Classify a model's type from its config.json.

Usage:
    python classify_model.py <MODEL_PATH>

Outputs a structured classification report covering:
  - High-level type: LLM / VLM / Whisper (ASR)
  - LLM attention sub-type: standard / sliding-window / MLA / Mamba / hybrid
  - MoE: yes/no with expert count
  - MTP (multi-token prediction): yes/no
  - Quantization type (if any)
  - Key numeric parameters

Exit codes: 0 = classified successfully, 1 = error reading config
"""

import json
import sys
from pathlib import Path


def load_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def classify(cfg: dict) -> dict:
    result = {}

    architectures = cfg.get("architectures", [])
    model_type = cfg.get("model_type", "")
    result["architectures"] = architectures
    result["model_type"] = model_type

    # ── High-level type ───────────────────────────────────────────────────────
    # Whisper / ASR: encoder-decoder speech model
    is_whisper = (
        any("Whisper" in a for a in architectures)
        or model_type == "whisper"
        or "audio_config" in cfg
        or "encoder_config" in cfg
    )

    # VLM: has a vision sub-config or known VL architecture suffix
    is_vlm = (
        "vision_config" in cfg
        or "thinker_config" in cfg
        or any(
            kw in a
            for a in architectures
            for kw in ("VL", "Vision", "Visual", "Multimodal", "MM", "Image")
        )
    )

    if is_whisper:
        result["high_level_type"] = "Whisper (ASR)"
    elif is_vlm:
        result["high_level_type"] = "VLM (Vision-Language)"
    else:
        result["high_level_type"] = "LLM"

    # ── MoE detection ─────────────────────────────────────────────────────────
    moe_keys = [
        "num_experts", "n_routed_experts", "num_local_experts",
        "moe_intermediate_size", "moe_layer_freq", "num_experts_per_tok",
    ]
    moe_fields = {k: cfg[k] for k in moe_keys if k in cfg}
    # Also recurse one level for nested configs (e.g. text_config)
    for sub_key in ("text_config", "language_config"):
        sub = cfg.get(sub_key, {})
        if isinstance(sub, dict):
            for k in moe_keys:
                if k in sub and k not in moe_fields:
                    moe_fields[k] = sub[k]

    result["is_moe"] = bool(moe_fields)
    if moe_fields:
        result["moe_fields"] = moe_fields

    # ── LLM attention sub-type ────────────────────────────────────────────────
    if result["high_level_type"] == "LLM" or result["high_level_type"] == "VLM":
        sub_types = []

        # MLA: multi-latent attention (DeepSeek-style)
        mla_keys = ["kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim"]
        mla_fields = {k: cfg[k] for k in mla_keys if k in cfg}
        if mla_fields:
            sub_types.append("MLA (multi-latent attention)")
            result["mla_fields"] = mla_fields

        # Mamba / SSM
        mamba_keys = ["state_size", "conv1d_width", "mamba_cache_mode", "ssm_cfg"]
        if any(k in cfg for k in mamba_keys) or "mamba" in model_type.lower():
            sub_types.append("Mamba (SSM)")

        # Sliding window attention
        sw = cfg.get("sliding_window") or cfg.get("sliding_window_size")
        if sw:
            sub_types.append(f"sliding-window (size={sw})")

        if not sub_types:
            sub_types.append("standard full attention")

        result["attention_sub_type"] = sub_types if len(sub_types) > 1 else sub_types[0]
        if len(sub_types) > 1:
            result["attention_note"] = "hybrid"

    # ── MTP (multi-token prediction) ──────────────────────────────────────────
    mtp_keys = ["num_nextn_predict_layers", "num_next_n_predict_layers"]
    mtp_val = next((cfg[k] for k in mtp_keys if k in cfg), None)
    result["mtp_enabled"] = mtp_val is not None and int(mtp_val) > 0
    if mtp_val is not None:
        result["mtp_layers"] = mtp_val

    # ── Quantization ──────────────────────────────────────────────────────────
    quant = cfg.get("quantization_config") or cfg.get("quantization")
    if quant:
        if isinstance(quant, dict):
            result["quantization"] = quant.get("quant_type") or quant.get("type") or str(quant)
        else:
            result["quantization"] = str(quant)
    else:
        result["quantization"] = "none"

    # ── Key numeric parameters ────────────────────────────────────────────────
    numeric_keys = [
        "max_position_embeddings", "num_hidden_layers", "hidden_size",
        "num_attention_heads", "num_key_value_heads", "torch_dtype",
        "vocab_size",
    ]
    result["params"] = {k: cfg[k] for k in numeric_keys if k in cfg}

    return result


def print_report(r: dict) -> None:
    print("=" * 60)
    print("Model Classification Report")
    print("=" * 60)
    print(f"  architectures      : {r.get('architectures', [])}")
    print(f"  model_type         : {r.get('model_type', '')}")
    print()
    print(f"  high_level_type    : {r['high_level_type']}")

    if "attention_sub_type" in r:
        sub = r["attention_sub_type"]
        note = f"  [{r.get('attention_note', '')}]" if "attention_note" in r else ""
        print(f"  attention_sub_type : {sub}{note}")

    print(f"  is_moe             : {r['is_moe']}", end="")
    if r["is_moe"]:
        experts = r.get("moe_fields", {})
        n = experts.get("n_routed_experts") or experts.get("num_experts") or "?"
        print(f"  (routed_experts={n})", end="")
    print()

    print(f"  mtp_enabled        : {r['mtp_enabled']}", end="")
    if r.get("mtp_layers") is not None:
        print(f"  (layers={r['mtp_layers']})", end="")
    print()

    print(f"  quantization       : {r['quantization']}")

    if r.get("params"):
        print()
        print("  Key parameters:")
        for k, v in r["params"].items():
            print(f"    {k:<30} = {v}")

    if r.get("mla_fields"):
        print()
        print("  MLA fields:")
        for k, v in r["mla_fields"].items():
            print(f"    {k:<30} = {v}")

    print("=" * 60)

    # Machine-readable summary line for the LLM to parse
    print()
    print("CLASSIFICATION_SUMMARY:", json.dumps({
        "high_level_type": r["high_level_type"],
        "attention_sub_type": r.get("attention_sub_type", "n/a"),
        "is_moe": r["is_moe"],
        "mtp_enabled": r["mtp_enabled"],
        "quantization": r["quantization"],
    }))


def main():
    if len(sys.argv) < 2:
        print("Usage: classify_model.py <MODEL_PATH>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    cfg = load_config(model_path)
    report = classify(cfg)
    print_report(report)


if __name__ == "__main__":
    main()
