# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.kv_cache_utils as vllm_kv_cache_utils
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, register_ascend_kv_cache_specs
from vllm_ascend.patch.platform.patch_kv_cache_utils import _ascend_annotate_eagle_groups


def _config(
    *, method="mtp", architecture="Qwen3_5MTP", model_type="qwen3_5", block_drop=True, use_eagle=True, packed=False
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            use_eagle_block_drop=lambda: block_drop,
            use_eagle=lambda: use_eagle,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5_mtp", architectures=[architecture]),
            ),
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(is_block_outermost=packed),
        ),
    )


def _full_spec():
    return FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)


def _hybrid_specs():
    full = _full_spec()
    mamba = MambaSpec(
        block_size=16,
        shapes=((128,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )
    # Both Ascend runners preserve attention order and append Mamba last.
    return {"target.attn": full, "draft.attn": full, "target.mamba.0": mamba, "target.mamba.1": mamba}


def _hybrid_groups(specs):
    return [
        KVCacheGroupSpec(["target.attn", "draft.attn"], specs["target.attn"]),
        KVCacheGroupSpec(["target.mamba.0", "target.mamba.1"], specs["target.mamba.0"]),
    ]


@pytest.mark.parametrize("architecture", ["Qwen3NextMTP", "Qwen3_5MTP", "Qwen3_5MoeMTP"])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"use_deepseek_v4_fallback": True}, {"use_trailing_layer_fallback": True}],
    ids=["general-path", "legacy-packed-keyword", "upstream-pr-keyword"],
)
def test_qwen_hybrid_marks_draft_group_without_reordering(architecture, kwargs):
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)
    original_items = list(specs.items())
    original_layers = [list(group.layer_names) for group in groups]

    _ascend_annotate_eagle_groups(_config(architecture=architecture), specs, groups, **kwargs)

    assert [group.is_eagle_group for group in groups] == [True, False]
    assert list(specs) == [name for name, _ in original_items]
    assert all(specs[name] is spec for name, spec in original_items)
    assert [group.layer_names for group in groups] == original_layers


@pytest.mark.parametrize("packed", [False, True])
def test_patch_is_used_by_real_upstream_hybrid_grouping(packed):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    assert vllm_kv_cache_utils._annotate_eagle_groups is _ascend_annotate_eagle_groups
    specs = _hybrid_specs()
    if packed:
        # The packed entry point requires more than one physical page size.
        for name in ("target.mamba.0", "target.mamba.1"):
            specs[name] = replace(specs[name], shapes=((256,),))
    original_order = list(specs)

    groups = vllm_kv_cache_utils.get_kv_cache_groups(_config(packed=packed), specs)

    assert groups
    assert {name for group in groups if group.is_eagle_group for name in group.layer_names} == {
        "target.attn",
        "draft.attn",
    }
    assert all(not group.is_eagle_group for group in groups if "target.mamba.0" in group.layer_names)
    assert list(specs) == original_order


def test_packed_grouping_annotates_filtered_attention_without_marking_hidden_state():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = _hybrid_specs()
    for name in ("target.mamba.0", "target.mamba.1"):
        specs[name] = replace(specs[name], shapes=((256,),))
    specs["hidden.state"] = HiddenStateCacheSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)

    # Upstream filters hidden-state layers before calling the packed hook and
    # appends their groups afterwards. The exact-partition guard applies to
    # that filtered mapping, not the original registration dictionary.
    groups = vllm_kv_cache_utils.get_kv_cache_groups(_config(packed=True), specs)

    assert {name for group in groups if group.is_eagle_group for name in group.layer_names} == {
        "target.attn",
        "draft.attn",
    }
    assert any(group.layer_names == ["hidden.state"] and not group.is_eagle_group for group in groups)


def test_original_registration_order_and_repeated_annotation():
    specs = _hybrid_specs()
    specs = {name: specs[name] for name in ("target.mamba.0", "target.attn", "target.mamba.1", "draft.attn")}
    groups = _hybrid_groups(specs)

    for _ in range(2):
        _ascend_annotate_eagle_groups(_config(), specs, groups)
        assert [group.is_eagle_group for group in groups] == [True, False]


def test_explicit_false_disables_positional_fallback():
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)

    _ascend_annotate_eagle_groups(
        _config(), specs, groups, use_trailing_layer_fallback=False, use_deepseek_v4_fallback=True
    )

    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("mode", ["no-speculation", "block-drop-disabled", "non-mtp"])
def test_qwen_hybrid_respects_speculative_gates(mode):
    config = _config(method="eagle3" if mode == "non-mtp" else "mtp", block_drop=mode != "block-drop-disabled")
    if mode == "no-speculation":
        config.speculative_config = None
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)

    _ascend_annotate_eagle_groups(config, specs, groups, use_trailing_layer_fallback=True)

    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("architecture", ["InternS2MobiusMTP", "UnknownMTP", None])
def test_same_draft_model_type_does_not_enable_unknown_architectures(architecture):
    # InternS2 also uses qwen3_5_mtp: model_type alone is not a safe guard.
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)
    config = _config(architecture=architecture)
    if architecture is None:
        config.speculative_config.draft_model_config.hf_config.architectures = None

    _ascend_annotate_eagle_groups(config, specs, groups, use_trailing_layer_fallback=True)

    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("extra_spec_type", [HiddenStateCacheSpec, AscendSFAIndexerCacheSpec, MLAAttentionSpec])
def test_hybrid_with_additional_cache_types_does_not_guess(extra_spec_type):
    specs = _hybrid_specs()
    extra_spec = extra_spec_type(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    specs["cache.only"] = extra_spec
    groups = [*_hybrid_groups(specs), KVCacheGroupSpec(["cache.only"], extra_spec)]

    _ascend_annotate_eagle_groups(_config(), specs, groups, use_trailing_layer_fallback=True)

    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("invalid_partition", ["missing", "duplicate", "unknown", "duplicate-in-group"])
def test_incomplete_or_duplicated_groups_disable_positional_fallback(invalid_partition):
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)
    if invalid_partition == "missing":
        groups[1].layer_names.remove("target.mamba.1")
    elif invalid_partition == "duplicate":
        groups.append(KVCacheGroupSpec(["draft.attn"], specs["draft.attn"]))
    elif invalid_partition == "unknown":
        groups[1].layer_names[-1] = "unregistered.layer"
    else:
        groups[0].layer_names.append("draft.attn")

    _ascend_annotate_eagle_groups(_config(), specs, groups, use_trailing_layer_fallback=True)

    assert not any(group.is_eagle_group for group in groups)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"use_trailing_layer_fallback": False}])
def test_spec_marker_remains_effective_without_an_exact_partition(wrapped, kwargs):
    marker_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )
    group_spec = (
        UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"draft.attn": marker_spec}) if wrapped else marker_spec
    )
    groups = [KVCacheGroupSpec(["draft.attn"], group_spec)]
    specs = {"target.attn": _full_spec(), "draft.attn": marker_spec}

    _ascend_annotate_eagle_groups(_config(method="dspark", architecture="UnknownMTP"), specs, groups, **kwargs)

    assert groups[0].is_eagle_group


def test_disabling_block_drop_also_disables_spec_marker_annotation():
    marker_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        non_causal_multi_token_decode=True,
    )
    groups = [KVCacheGroupSpec(["draft.attn"], marker_spec)]

    _ascend_annotate_eagle_groups(_config(method="dspark", block_drop=False), {"draft.attn": marker_spec}, groups)

    assert not groups[0].is_eagle_group


def test_non_hybrid_mtp_keeps_upstream_trailing_layer_rule():
    spec = _full_spec()
    specs = {"target.attn": spec, "draft.attn": spec}
    groups = [KVCacheGroupSpec([name], spec) for name in specs]

    _ascend_annotate_eagle_groups(_config(architecture="UnknownMTP"), specs, groups)

    assert [group.is_eagle_group for group in groups] == [False, True]


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("keyword", ["use_deepseek_v4_fallback", "use_trailing_layer_fallback"])
@pytest.mark.parametrize("use_eagle", [False, True])
def test_deepseek_dspark_keeps_model_scoped_fallback(model_type, keyword, use_eagle):
    spec = _full_spec()
    specs = {"target.attn": spec, "draft.attn": spec}
    groups = [KVCacheGroupSpec([name], spec) for name in specs]

    _ascend_annotate_eagle_groups(
        _config(method="dspark", model_type=model_type, use_eagle=use_eagle), specs, groups, **{keyword: True}
    )

    assert [group.is_eagle_group for group in groups] == [False, use_eagle]


@pytest.mark.parametrize("method", ["mtp", "dspark"])
def test_ascend_deepseek_packed_entry_uses_the_compatible_annotator(method):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    full_specs = {
        f"target.c{ratio}": MLAAttentionSpec(
            block_size=128 * ratio,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            tokens_per_state=ratio,
            model_version="deepseek_v4",
        )
        for ratio in (4, 128)
    }
    swa_spec = SlidingWindowMLASpec(
        block_size=128, num_kv_heads=1, head_size=128, dtype=torch.float16, sliding_window=512
    )
    specs = {**full_specs, "target.swa": swa_spec, "draft.swa": swa_spec}

    groups = vllm_kv_cache_utils._get_packed_kv_cache_groups(_config(method=method, model_type="deepseek_v4"), specs)

    assert groups is not None
    assert {name for group in groups if group.is_eagle_group for name in group.layer_names} == {"draft.swa"}
    assert {name for group in groups for name in group.layer_names} == set(specs)


def test_shared_draft_group_flag_survives_pp_projection_and_scheduler_config():
    specs = _hybrid_specs()
    groups = _hybrid_groups(specs)
    _ascend_annotate_eagle_groups(_config(), specs, groups)
    worker_specs = [
        {name: specs[name] for name in ("target.attn", "target.mamba.0")},
        {name: specs[name] for name in ("draft.attn", "target.mamba.1")},
    ]
    configs = [
        KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=vllm_kv_cache_utils._project_kv_cache_groups_to_worker(groups, local_specs),
        )
        for local_specs in worker_specs
    ]

    # The first PP rank has no draft layer but shares its global group.
    assert configs[0].kv_cache_groups[0].layer_names == ["target.attn"]
    assert all([group.is_eagle_group for group in config.kv_cache_groups] == [True, False] for config in configs)
    scheduler_config = vllm_kv_cache_utils.generate_scheduler_kv_cache_config(configs)
    assert [group.is_eagle_group for group in scheduler_config.kv_cache_groups] == [True, False]


def test_empty_specs_do_not_attempt_a_trailing_layer_lookup():
    _ascend_annotate_eagle_groups(_config(), {}, [], use_trailing_layer_fallback=True)
