# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.attention.indexer import validate_indexshare_pp_partition
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
    validate_indexshare_pp_partition(
        _config(indexer_types=["full", "shared", "shared", "shared"]),
        start_layer=0,
        end_layer=4,
        pp_rank=0,
        pp_size=2,
    )


def test_indexshare_pp_partition_rejects_shared_stage_start():
    try:
        validate_indexshare_pp_partition(
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

    validate_indexshare_pp_partition(
        _config(indexer_types=indexer_types),
        start_layer=0,
        end_layer=39,
        pp_rank=0,
        pp_size=2,
    )

    try:
        validate_indexshare_pp_partition(
            _config(indexer_types=indexer_types),
            start_layer=39,
            end_layer=78,
            pp_rank=1,
            pp_size=2,
        )
    except ValueError as exc:
        assert "layer 39 is shared" in str(exc)
    else:
        raise AssertionError("Expected GLM-5.2 PP2 equal partition to be rejected")
