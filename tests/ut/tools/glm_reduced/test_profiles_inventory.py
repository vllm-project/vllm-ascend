# SPDX-License-Identifier: Apache-2.0
"""Profile/inventory consistency and shared-indexer closure rule tests."""

import pytest

from tools.glm_reduced.errors import ProfileError
from tools.glm_reduced.inventory import list_inventory, validate_inventory
from tools.glm_reduced.profiles import (
    get_profile,
    list_profiles,
    match_profile,
    read_layer_count,
    truncate_layer_config,
    validate_layer_arrays,
)


def test_inventory_is_internally_consistent():
    assert validate_inventory() == []


def test_every_inventory_entry_has_evidence():
    for entry in list_inventory():
        assert entry.evidence, entry.model_id
        if entry.status == "unsupported":
            assert entry.reason, entry.model_id


def test_match_profile_by_architecture():
    assert match_profile(["GlmMoeDsaForCausalLM"]).name == "glm-moe-dsa"
    assert match_profile(["Glm4MoeForCausalLM"]).name == "glm4-moe"
    assert match_profile(["Glm5NextForConditionalGeneration"]).name == "glm5-next"
    assert match_profile(["Glm4vForConditionalGeneration"]).name == "glm4v"
    assert match_profile(["ChatGLMModel"]).name == "chatglm"
    assert match_profile(["NotAModel"]) is None
    assert match_profile([]) is None


def test_unknown_profile_raises():
    with pytest.raises(ProfileError, match="unknown profile"):
        get_profile("glm-99")


def _official_52_indexer_types() -> list[str]:
    # zai-org/GLM-5.2: full producers at 0,1,2 then every 4th layer.
    types = ["shared"] * 78
    for idx in (0, 1, 2, *range(6, 78, 4)):
        types[idx] = "full"
    return types


def test_official_glm52_prefix8_satisfies_closure():
    config = {
        "num_hidden_layers": 78,
        "indexer_types": _official_52_indexer_types(),
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
    }
    assert validate_layer_arrays(get_profile("glm-moe-dsa"), config, 8, 78) == []


def test_prefix_starting_with_shared_fails():
    config = {"num_hidden_layers": 12, "indexer_types": ["shared"] + ["full"] * 11}
    problems = validate_layer_arrays(get_profile("glm-moe-dsa"), config, 8, 12)
    assert any("producer" in problem for problem in problems)


def test_truncate_does_not_mutate_source_config():
    config = {
        "num_hidden_layers": 12,
        "indexer_types": ["full"] * 12,
        "layer_types": ["linear_attention"] * 12,
        "linear_attn_config": {"kda_layers": list(range(12)), "full_attn_layers": []},
    }
    snapshot = dict(config)
    subdict_snapshot = dict(config["linear_attn_config"])
    new = truncate_layer_config(get_profile("glm5-next"), config, 8, 12)
    assert new["num_hidden_layers"] == 8
    assert new["linear_attn_config"]["kda_layers"] == list(range(8))
    assert config == snapshot
    assert config["linear_attn_config"] == subdict_snapshot


def test_chatglm_layer_count_keys_sync_and_reject():
    profile = get_profile("chatglm")
    assert read_layer_count(profile, {"num_layers": 28}) == 28
    assert read_layer_count(profile, {"num_layers": 40, "num_hidden_layers": 40}) == 40
    with pytest.raises(ProfileError, match="inconsistent"):
        read_layer_count(profile, {"num_layers": 40, "num_hidden_layers": 39})
    with pytest.raises(ProfileError, match="no layer-count key"):
        read_layer_count(profile, {"hidden_size": 16})


def test_profile_min_layers_cover_family_blocks():
    # The justification invariant: GLM-4.x/5.x MoE/DSA families keep >= 8
    # (past first_k_dense_replace=3 plus a full indexer/hybrid cycle);
    # glm4_moe_lite (first_k_dense_replace=1) and the dense families
    # (chatglm, glm4v) keep >= 4 (attention+MLP+norm stack plus MoE coverage).
    for profile in list_profiles():
        if profile.family in ("glm4_moe", "glm_moe_dsa", "glm5_next"):
            assert profile.min_keep_layers >= 8, profile.name
        else:
            assert profile.min_keep_layers >= 4, profile.name
        assert profile.min_keep_layers <= profile.default_keep_layers
        assert profile.precision_workload.output_tokens > 0
        assert profile.performance_workload.measured_iterations > profile.performance_workload.warmup_iterations


def test_every_profile_naming_covers_required_globals():
    for profile in list_profiles():
        assert profile.naming.layer_roots
        assert profile.naming.embed_names
        assert profile.naming.final_norm_names
        assert profile.naming.lm_head_names
