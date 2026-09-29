# SPDX-License-Identifier: Apache-2.0
"""Validation against vendored real upstream configs and index key patterns.

Fixtures under ``fixtures/`` are verbatim public config.json snapshots and
key-pattern summaries derived from the pinned upstream safetensors indexes
(see tools/glm_reduced/sources.json for model IDs, revisions and URLs). They
are small metadata files, not weights. Every distinct real key pattern is
driven through the planner at boundary layer indices (first kept, first
dropped, MTP, beyond-MTP) so no real tensor name can fall through the
classifier silently.
"""

import json
from pathlib import Path

import pytest
import regex as re

from tools.glm_reduced.errors import UnknownTensorError
from tools.glm_reduced.inventory import validate_inventory
from tools.glm_reduced.profiles import get_profile, match_profile, read_layer_count, validate_layer_arrays
from tools.glm_reduced.reducer import _classify, _layer_config_of

FIXTURES = Path(__file__).resolve().parent / "fixtures"

CONFIG_CASES = {
    "THUDM--chatglm3-6b.json": ("chatglm", 28, 8),
    "THUDM--glm-4-9b-chat.json": ("chatglm", 40, 8),
    "zai-org--GLM-4.1V-9B-Thinking.json": ("glm4v", 40, 8),
    "zai-org--GLM-4.5.json": ("glm4-moe", 92, 8),
    "zai-org--GLM-4.7-Flash.json": ("glm4-moe-lite", 47, 8),
    "zai-org--GLM-5.2.json": ("glm-moe-dsa", 78, 8),
    "zai-org--GLM-5.3.json": ("glm-moe-dsa", 78, 8),
    "zai-org--GLM-5.3-Flash.json": ("glm5-next", 45, 8),
    "Eco-Tech--GLM-5.2-w4a8--config.json": ("glm-moe-dsa", 78, 8),
    "Eco-Tech--GLM-5.3-w8a8c8--config.json": ("glm-moe-dsa", 78, 8),
    "Eco-Tech--GLM-5.3-Flash-w8a8--config.json": ("glm5-next", 45, 8),
}


@pytest.mark.parametrize("filename", sorted(CONFIG_CASES))
def test_real_config_matches_profile_and_validates(filename):
    profile_name, expected_layers, keep = CONFIG_CASES[filename]
    config = json.loads((FIXTURES / "configs" / filename).read_text(encoding="utf-8"))
    profile = match_profile(config.get("architectures", []))
    assert profile is not None, f"no profile for {filename}"
    assert profile.name == profile_name
    layer_config = _layer_config_of(profile, config)
    assert read_layer_count(profile, layer_config) == expected_layers
    assert validate_layer_arrays(profile, layer_config, keep, expected_layers) == []


def test_real_flash_fp8_not_convert_list_remap():
    config = json.loads((FIXTURES / "configs" / "zai-org--GLM-5.3-Flash.json").read_text(encoding="utf-8"))
    from tools.glm_reduced.reducer import _remap_module_listing

    entries = config["quantization_config"]["modules_to_not_convert"]
    out = _remap_module_listing(entries, source_layers=45, keep_layers=8, mtp_layers=1, what="modules_to_not_convert")
    anchor = re.compile(r"\.layers\.(\d+)\.")
    idxs = sorted({int(m.group(1)) for e in out if (m := anchor.search(e))})
    assert idxs == list(range(9))  # 0..7 kept + MTP 45 remapped to 8
    non_layer_in = [e for e in entries if not anchor.search(e)]
    non_layer_out = [e for e in out if not anchor.search(e)]
    assert non_layer_in == non_layer_out
    mtp_in = [e for e in entries if ".layers.45." in e]
    assert mtp_in and len(mtp_in) == len([e for e in out if ".layers.8." in e])


def test_real_glm53_fp8_not_convert_list_remap():
    config = json.loads((FIXTURES / "configs" / "zai-org--GLM-5.3.json").read_text(encoding="utf-8"))
    from tools.glm_reduced.reducer import _remap_module_listing

    entries = config["quantization_config"]["modules_to_not_convert"]
    out = _remap_module_listing(entries, source_layers=78, keep_layers=8, mtp_layers=1, what="modules_to_not_convert")
    anchor = re.compile(r"\.layers\.(\d+)\.")
    idxs = sorted({int(m.group(1)) for e in out if (m := anchor.search(e))})
    assert max(idxs) <= 8
    # MTP layer 78 module entries (eh_proj/hnorm/enorm/shared_head...) remap to 8.
    assert any(e.startswith("model.layers.8.eh_proj") or ".layers.8.hnorm" in e for e in out)
    assert "model.norm" in out and "lm_head" in out


PATTERN_CASES = {
    "THUDM--chatglm3-6b": ("chatglm", 28, 0),
    "THUDM--glm-4-9b-chat": ("chatglm", 40, 0),
    "zai-org--GLM-4.1V-9B-Thinking": ("glm4v", 40, 0),
    "zai-org--GLM-4.5": ("glm4-moe", 92, 1),
    "zai-org--GLM-4.7-Flash": ("glm4-moe-lite", 47, 1),
    "zai-org--GLM-5": ("glm-moe-dsa", 78, 1),
    "zai-org--GLM-5.2": ("glm-moe-dsa", 78, 1),
    "zai-org--GLM-5.3": ("glm-moe-dsa", 78, 1),
    "zai-org--GLM-5.3-Flash": ("glm5-next", 45, 1),
}


@pytest.mark.parametrize("model", sorted(PATTERN_CASES))
def test_every_real_key_pattern_classifies(model):
    profile_name, source_layers, mtp_layers = PATTERN_CASES[model]
    profile = get_profile(profile_name)
    keep = 8
    data = json.loads((FIXTURES / "key_patterns" / f"{model}.json").read_text(encoding="utf-8"))
    assert data["patterns"], model
    for pattern in data["patterns"]:
        if "layers.{N}." in pattern:
            for idx, expected in ((0, "keep"), (keep - 1, "keep"), (keep, "drop-layer")):
                name = pattern.replace("layers.{N}.", f"layers.{idx}.").replace("experts.{N}.", "experts.0.")
                action = _classify(name, profile.naming, source_layers, keep, mtp_layers, profile.keep_mtp)
                assert action.action == expected, (pattern, idx, action)
            if mtp_layers:
                name = pattern.replace("layers.{N}.", f"layers.{source_layers}.").replace("experts.{N}.", "experts.0.")
                action = _classify(name, profile.naming, source_layers, keep, mtp_layers, profile.keep_mtp)
                assert action.action == "remap", (pattern, action)
                assert f"layers.{keep}." in action.dst_name
            with pytest.raises(UnknownTensorError):
                name = pattern.replace("layers.{N}.", f"layers.{source_layers + mtp_layers}.").replace(
                    "experts.{N}.", "experts.0."
                )
                _classify(name, profile.naming, source_layers, keep, mtp_layers, profile.keep_mtp)
        else:
            name = pattern.replace("blocks.{N}.", "blocks.0.")
            action = _classify(name, profile.naming, source_layers, keep, mtp_layers, profile.keep_mtp)
            assert action.action == "keep", (pattern, action)


def test_real_quant_description_excerpts_remap():
    from tools.glm_reduced.reducer import SourceCheckpoint, _plan_quant_description

    cases = {
        "Eco-Tech--GLM-5.2-w4a8--desc-excerpt.json": (78, 8, 1),
        "Eco-Tech--GLM-5.3-w8a8c8--desc-excerpt.json": (78, 8, 1),
        "Eco-Tech--GLM-5.3-Flash-w8a8--desc-excerpt.json": (45, 8, 1),
    }
    for filename, (source_layers, keep, mtp) in cases.items():
        description = json.loads((FIXTURES / "quant" / filename).read_text(encoding="utf-8"))
        source = SourceCheckpoint("", {}, "", None, {}, {}, [], description, "")
        result = _plan_quant_description(source, source_layers, keep, mtp)
        anchor = re.compile(r"\.layers\.(\d+)\.")
        # Global metadata preserved.
        for key in ("group_size", "metadata", "optional", "version"):
            if key in description:
                assert result[key] == description[key], (filename, key)
        for key, value in result.items():
            match = anchor.search(key)
            if not match:
                continue
            idx = int(match.group(1))
            assert idx < keep or idx == keep  # kept prefix or remapped MTP
            if idx == keep:
                # Value must come from the source MTP layer entry with the
                # same module path, never from a dropped decoder layer.
                src_key = key.replace(f".layers.{keep}.", f".layers.{source_layers}.", 1)
                assert description.get(src_key) == value, (filename, key)
        # Dropped-layer excerpt entries (layer 9) are gone.
        assert not any(".layers.9." in key for key in result)


def test_inventory_consistent_with_profiles():
    assert validate_inventory() == []
