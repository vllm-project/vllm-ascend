#!/usr/bin/env python3
"""inspect_checkpoint.py — Build a structured checkpoint adaptation profile.

Usage:
    python inspect_checkpoint.py <MODEL_PATH> [--vllm-src /path/to/vllm]

Outputs a human-readable report plus a machine-readable INSPECTION_SUMMARY line.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from classify_model import classify, load_config


CODE_SIGNAL_PATTERNS = {
    "torch_ops": re.compile(r"torch\.ops\."),
    "ascend_specific": re.compile(r"torch_npu|torch\.ops\.npu|aclnn[A-Za-z0-9_]*"),
    "triton": re.compile(r"@triton\.jit|triton\.jit"),
    "cuda": re.compile(r"\.cu\b|CUDAExtension|load_inline|cpp_extension"),
}

WEIGHT_HINT_PATTERNS = {
    "split_qkv": ("q_proj", "k_proj", "v_proj"),
    "fused_qkv": ("qkv_proj", "wqkv", "query_key_value", "c_attn", "W_pack"),
    "qk_norm": ("q_norm", "k_norm"),
    "moe": ("experts.", "shared_expert", "router", "gate.", "gate_proj"),
    "mla": ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim"),
    "mtp": ("nextn", "mtp", "spec_module"),
    "vision": ("vision_tower", "vision_model", "visual.", "mm_projector"),
    "audio": ("audio_tower", "speech_encoder", "audio_encoder"),
    "rope": ("rotary_emb", "inv_freq", "rope"),
}


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_asset_files(model_path: Path) -> dict:
    asset_names = [
        "tokenizer_config.json",
        "tokenizer.json",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ]
    return {name: (model_path / name).exists() for name in asset_names}


def summarize_tokenizer_and_processor(model_path: Path) -> dict:
    tokenizer_cfg = load_optional_json(model_path / "tokenizer_config.json") or {}
    processor_cfg = load_optional_json(model_path / "processor_config.json") or {}
    preprocessor_cfg = load_optional_json(model_path / "preprocessor_config.json") or {}

    chat_template_present = bool(
        tokenizer_cfg.get("chat_template")
        or processor_cfg.get("chat_template")
        or (model_path / "chat_template.jinja").exists()
    )

    return {
        "tokenizer_class": tokenizer_cfg.get("tokenizer_class"),
        "processor_class": processor_cfg.get("processor_class"),
        "image_processor_type": preprocessor_cfg.get("image_processor_type"),
        "feature_extractor_type": preprocessor_cfg.get("feature_extractor_type"),
        "chat_template_present": chat_template_present,
    }


def summarize_runtime_signals(cfg: dict, model_path: Path) -> dict:
    auto_map = cfg.get("auto_map") or {}
    rope_scaling = cfg.get("rope_scaling")
    model_py_files = sorted(
        str(path.relative_to(model_path))
        for path in model_path.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    requires_remote_code = bool(auto_map) or bool(model_py_files)

    return {
        "requires_remote_code": requires_remote_code,
        "auto_map_keys": sorted(auto_map.keys()) if isinstance(auto_map, dict) else [],
        "rope_scaling": rope_scaling,
        "has_vision_config": "vision_config" in cfg,
        "has_audio_config": "audio_config" in cfg,
        "custom_code_files": model_py_files,
    }


def find_weight_index(model_path: Path) -> Path | None:
    index_files = sorted(model_path.glob("*index*.json"))
    return index_files[0] if index_files else None


def load_weight_keys(model_path: Path) -> tuple[list[str], dict]:
    index_file = find_weight_index(model_path)
    if not index_file:
        return [], {}

    index_json = load_optional_json(index_file) or {}
    weight_map = index_json.get("weight_map", {})
    return sorted(weight_map.keys()), {
        "index_file": str(index_file.name),
        "total_keys": len(weight_map),
        "shard_count": len(set(weight_map.values())),
    }


def detect_weight_hints(weight_keys: list[str]) -> dict:
    hints = {}
    for hint, patterns in WEIGHT_HINT_PATTERNS.items():
        hints[hint] = any(pattern in key for key in weight_keys for pattern in patterns)
    return hints


def summarize_checkpoint_layout(model_path: Path, weight_keys: list[str]) -> dict:
    safetensors_files = sorted(path.name for path in model_path.glob("*.safetensors"))
    bin_files = sorted(path.name for path in model_path.glob("*.bin"))
    pt_files = sorted(path.name for path in model_path.glob("*.pt"))
    return {
        "has_safetensors": bool(safetensors_files),
        "has_pytorch_bin": bool(bin_files),
        "has_pt_files": bool(pt_files),
        "safetensors_files": safetensors_files[:5],
        "bin_files": bin_files[:5],
        "pt_files": pt_files[:5],
        "sample_weight_keys": weight_keys[:20],
    }


def scan_custom_code(model_path: Path) -> dict:
    findings = {name: [] for name in CODE_SIGNAL_PATTERNS}

    for path in sorted(model_path.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        for name, pattern in CODE_SIGNAL_PATTERNS.items():
            if not pattern.search(text):
                continue
            matches = []
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append({"line": lineno, "text": line.strip()})
                if len(matches) == 3:
                    break
            findings[name].append({
                "file": str(path.relative_to(model_path)),
                "matches": matches,
            })

    return findings


def architecture_support(cfg: dict, vllm_src: Path | None) -> dict:
    architectures = cfg.get("architectures", [])
    if not vllm_src:
        return {"checked": False, "supported_architectures": []}

    registry_path = vllm_src / "vllm" / "model_executor" / "models" / "registry.py"
    if not registry_path.exists():
        return {"checked": False, "supported_architectures": []}

    registry_text = registry_path.read_text(errors="ignore")
    supported = [arch for arch in architectures if arch in registry_text]
    return {"checked": True, "supported_architectures": supported}


def suggest_touch_points(classification: dict, runtime: dict, weight_hints: dict,
                         code_findings: dict, support: dict) -> list[str]:
    suggestions = []

    if not support.get("supported_architectures"):
        suggestions.append("Add or confirm architecture registration in vllm/model_executor/models/registry.py.")
    suggestions.append("Audit explicit weight loading and remap rules against checkpoint key patterns.")

    if runtime["requires_remote_code"]:
        suggestions.append("Review remote-code modeling and processor classes before copying any code into vLLM.")
    if classification["high_level_type"] == "VLM (Vision-Language)":
        suggestions.append("Plan processor and multimodal path validation, not only text generation.")
    if classification["is_moe"]:
        suggestions.append("Validate MoE-specific paths: expert parallel, flashcomm1, and routed expert loading.")
    if classification["mtp_enabled"] or weight_hints["mtp"]:
        suggestions.append("Validate MTP from both config and weight keys; treat config-only signals as insufficient.")
    if "fp8" in str(classification["quantization"]).lower():
        suggestions.append("Prepare fp8-to-bf16 load-time dequantization and strict scale pairing checks.")
    if weight_hints["qk_norm"] or weight_hints["mla"]:
        suggestions.append("Audit head sharding, q/k norm loading, and MLA-specific topology assumptions.")
    if code_findings["cuda"]:
        suggestions.append("Run the operator compatibility gate early for CUDA-only kernels and confirm fallback paths.")
    if code_findings["ascend_specific"]:
        suggestions.append(
            "If blocked on an Ascend-specific operator, search the official HiAscend operator docs before the next fix attempt.")
    if code_findings["triton"]:
        suggestions.append("Treat Triton kernels as correctness-risky on Ascend until validated on NPU.")
    if not suggestions:
        suggestions.append("No special adaptation signals detected; start from registry and loader verification.")

    return suggestions


def print_json_block(title: str, payload: dict) -> None:
    print(title)
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def print_report(model_path: Path, classification: dict, assets: dict, tokenizer_and_processor: dict,
                 runtime: dict, weight_index: dict, checkpoint_layout: dict,
                 weight_hints: dict, code_findings: dict, support: dict,
                 suggestions: list[str]) -> None:
    print("=" * 72)
    print("Checkpoint Inspection Report")
    print("=" * 72)
    print(f"model_path: {model_path}")
    print()
    print_json_block("classification:", classification)
    print()
    print_json_block("asset_files:", assets)
    print()
    print_json_block("tokenizer_and_processor:", tokenizer_and_processor)
    print()
    print_json_block("runtime_signals:", runtime)
    print()
    print_json_block("architecture_support:", support)
    print()
    print_json_block("weight_index:", weight_index)
    print()
    print_json_block("checkpoint_layout:", checkpoint_layout)
    print()
    print_json_block("weight_hints:", weight_hints)
    print()
    print_json_block("operator_signals:", code_findings)
    print()
    print("suggested_touch_points:")
    for item in suggestions:
        print(f"  - {item}")
    print("=" * 72)

    summary = {
        "high_level_type": classification["high_level_type"],
        "attention_sub_type": classification.get("attention_sub_type", "n/a"),
        "is_moe": classification["is_moe"],
        "mtp_enabled": classification["mtp_enabled"],
        "quantization": classification["quantization"],
        "requires_remote_code": runtime["requires_remote_code"],
        "supported_architectures": support.get("supported_architectures", []),
        "operator_signal_counts": {
            name: len(items) for name, items in code_findings.items()
        },
        "weight_hints": weight_hints,
        "suggested_touch_points": suggestions,
    }
    print()
    print("INSPECTION_SUMMARY:", json.dumps(summary, ensure_ascii=True, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--vllm-src", dest="vllm_src")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    cfg = load_config(str(model_path))
    classification = classify(cfg)
    assets = collect_asset_files(model_path)
    tokenizer_and_processor = summarize_tokenizer_and_processor(model_path)
    runtime = summarize_runtime_signals(cfg, model_path)
    weight_keys, weight_index = load_weight_keys(model_path)
    checkpoint_layout = summarize_checkpoint_layout(model_path, weight_keys)
    weight_hints = detect_weight_hints(weight_keys)
    code_findings = scan_custom_code(model_path)
    support = architecture_support(cfg, Path(args.vllm_src) if args.vllm_src else None)
    suggestions = suggest_touch_points(
        classification, runtime, weight_hints, code_findings, support)

    print_report(
        model_path,
        classification,
        assets,
        tokenizer_and_processor,
        runtime,
        weight_index,
        checkpoint_layout,
        weight_hints,
        code_findings,
        support,
        suggestions,
    )


if __name__ == "__main__":
    main()
