# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.attention.indexer import (
    validate_indexer_pp_partition,
    validate_indexer_pp_stage,
)
from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_indexshare_pp_partition_accepts_full_then_shared():
    validate_indexer_pp_stage(
        _config(indexer_types=["full", "shared", "shared", "shared"]),
        start_layer=0,
        end_layer=4,
        pp_rank=0,
        pp_size=2,
    )


def test_indexshare_pp_partition_rejects_shared_stage_start():
    try:
        validate_indexer_pp_stage(
            _config(indexer_types=["full", "shared", "shared", "shared"]),
            start_layer=1,
            end_layer=4,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "crosses a pipeline-parallel stage boundary" in str(exc)
    else:
        raise AssertionError("Expected IndexShare PP boundary validation to fail")


def test_glm52_pp2_equal_partition_is_rejected():
    indexer_types = ["full", "full", "full", "shared", "shared", "shared"]
    indexer_types.extend(["full", "shared", "shared", "shared"] * 18)

    validate_indexer_pp_stage(
        _config(indexer_types=indexer_types),
        start_layer=0,
        end_layer=39,
        pp_rank=0,
        pp_size=2,
    )

    try:
        validate_indexer_pp_stage(
            _config(indexer_types=indexer_types),
            start_layer=39,
            end_layer=78,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "layer 39 uses a shared Indexer" in str(exc)
    else:
        raise AssertionError("Expected GLM-5.2 PP2 equal partition to be rejected")


def test_index_cache_pp_partition_rejects_skip_topk_stage_start():
    config = _config(
        use_index_cache=True,
        index_topk_freq=4,
        index_skip_topk_offset=3,
    )

    validate_indexer_pp_stage(
        config,
        start_layer=0,
        end_layer=3,
        pp_rank=0,
        pp_size=2,
    )

    try:
        validate_indexer_pp_stage(
            config,
            start_layer=3,
            end_layer=6,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "Index cache dependency crosses" in str(exc)
        assert "layer 3 skips Top-K computation" in str(exc)
    else:
        raise AssertionError("Expected IndexCache PP boundary to be rejected")


def test_index_cache_pattern_pp_partition_rejects_skip_topk_stage_start():
    config = _config(
        use_index_cache=True,
        index_topk_pattern="FFSFFS",
    )

    try:
        validate_indexer_pp_stage(
            config,
            start_layer=2,
            end_layer=6,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "layer 2 skips Top-K computation" in str(exc)
    else:
        raise AssertionError("Expected IndexCache pattern PP boundary to be rejected")


def test_indexshare_does_not_validate_disabled_index_cache():
    config = _config(
        use_index_cache=False,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=[
            "full",
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
        ],
    )

    try:
        validate_indexer_pp_stage(
            config,
            start_layer=4,
            end_layer=8,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "IndexShare group crosses" in str(exc)
        assert "Index cache dependency crosses" not in str(exc)
    else:
        raise AssertionError("Expected IndexShare PP boundary validation to fail")


def test_glm52_pp2_suggests_partitions_in_both_directions():
    indexer_types = ["full", "full", "full", "shared", "shared", "shared"]
    indexer_types.extend(["full", "shared", "shared", "shared"] * 18)

    try:
        validate_indexer_pp_partition(
            _config(indexer_types=indexer_types),
            num_hidden_layers=78,
            pp_size=2,
        )
    except ValueError as exc:
        assert "layer 39 uses a shared Indexer" in str(exc)
        assert "The nearest valid layer partitions are" in str(exc)
        assert 'VLLM_PP_LAYER_PARTITION="38,40"' in str(exc)
        assert " or " in str(exc)
        assert 'VLLM_PP_LAYER_PARTITION="42,36"' in str(exc)
    else:
        raise AssertionError("Expected GLM-5.2 PP2 equal partition to be rejected")


def test_pp4_suggests_lower_partition_when_higher_is_unavailable():
    indexer_types = ["shared"] * 16
    for layer_id in (0, 4, 8, 10):
        indexer_types[layer_id] = "full"

    try:
        validate_indexer_pp_partition(
            _config(indexer_types=indexer_types),
            num_hidden_layers=16,
            pp_size=4,
        )
    except ValueError as exc:
        assert "PP rank 3/4" in str(exc)
        assert "The nearest valid layer partition is" in str(exc)
        assert 'VLLM_PP_LAYER_PARTITION="4,4,2,6"' in str(exc)
        assert " or " not in str(exc)
    else:
        raise AssertionError("Expected PP4 IndexShare partition to be rejected")
