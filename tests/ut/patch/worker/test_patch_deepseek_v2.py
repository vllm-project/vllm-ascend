# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _is_mtp_layer,
    _resolve_mtp_indexer_permissions,
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


def test_mtp_layer_detection_from_config_and_prefix():
    assert _is_mtp_layer(_config(), "model.layers.79.self_attn") is False
    assert _is_mtp_layer(_config(), "model.layers.80.self_attn") is True


def test_mtp_disables_short_prefill_bypass_and_topk_reuse():
    assert _resolve_mtp_indexer_permissions(True, False) == (True, True)
    assert _resolve_mtp_indexer_permissions(True, True) == (False, False)
