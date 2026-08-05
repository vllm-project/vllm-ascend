# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _is_index_cache_skip_layer,
    _is_skipped_indexer_weight,
    _should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_without_index_cache_keeps_per_layer_indexer():
    """GLM5.1 without use_index_cache: skip_topk=True keeps Indexer."""
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    """GLM5.2 shared indexer_type layers skip Indexer creation."""
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm51_index_cache_skips_indexer_init():
    """GLM5.1 with use_index_cache: skip_topk=True layers skip Indexer."""
    assert _should_skip_indexer_init(
        _config(use_index_cache=True, index_topk_freq=3),
        "model.layers.3.self_attn",
        skip_topk=True,
    )


def test_glm51_index_cache_producer_keeps_indexer():
    """GLM5.1 with use_index_cache: producer layers (skip_topk=False) keep Indexer."""
    assert not _should_skip_indexer_init(
        _config(use_index_cache=True, index_topk_freq=3),
        "model.layers.1.self_attn",
        skip_topk=False,
    )


def test_mtp_layer_keeps_indexer():
    """MTP layers beyond num_hidden_layers never skip Indexer."""
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer_with_index_cache():
    """MTP layers never skip Indexer even with use_index_cache."""
    assert not _should_skip_indexer_init(
        _config(use_index_cache=True, num_hidden_layers=80),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


class TestIsIndexCacheSkipLayer:
    """Tests for _is_index_cache_skip_layer pure function."""

    def test_freq_based_pattern(self):
        """With freq=3, offset=2: layer 1 is producer, 2-3 skip, 4 producer."""
        config = _config(index_topk_freq=3)
        assert not _is_index_cache_skip_layer(config, 1)  # (1-2+1)%3=0 → producer
        assert _is_index_cache_skip_layer(config, 2)  # (2-2+1)%3=1 → skip
        assert _is_index_cache_skip_layer(config, 3)  # (3-2+1)%3=2 → skip
        assert not _is_index_cache_skip_layer(config, 4)  # (4-2+1)%3=0 → producer

    def test_explicit_pattern(self):
        """With pattern "FSSF": layers 0/3 produce, 1/2 skip."""
        config = _config(index_topk_pattern="FSSF")
        assert not _is_index_cache_skip_layer(config, 0)
        assert _is_index_cache_skip_layer(config, 1)
        assert _is_index_cache_skip_layer(config, 2)
        assert not _is_index_cache_skip_layer(config, 3)

    def test_explicit_pattern_out_of_range(self):
        """Layer ID beyond pattern length returns False."""
        config = _config(index_topk_pattern="FS")
        assert not _is_index_cache_skip_layer(config, 10)

    def test_with_custom_offset(self):
        """Custom offset shifts the producer position."""
        config = _config(index_topk_freq=2, index_skip_topk_offset=3)
        # offset=3: layer 3 is first producer
        assert not _is_index_cache_skip_layer(config, 2)  # (2-3+1)%2=0 → producer
        assert _is_index_cache_skip_layer(config, 3)  # (3-3+1)%2=1 → skip


class TestIsSkippedIndexerWeight:
    """Tests for _is_skipped_indexer_weight name filter."""

    skip_ids = frozenset({2, 3, 5})

    def test_skipped_layer_indexer_weight(self):
        assert _is_skipped_indexer_weight(
            "model.layers.2.self_attn.indexer.wk_weights_proj.weight",
            self.skip_ids,
        )

    def test_skipped_layer_indexer_weight_quant_scale(self):
        assert _is_skipped_indexer_weight(
            "model.layers.3.self_attn.indexer.wq_b.weight_scale",
            self.skip_ids,
        )

    def test_non_skipped_layer_indexer_weight(self):
        assert not _is_skipped_indexer_weight(
            "model.layers.1.self_attn.indexer.weights_proj.weight",
            self.skip_ids,
        )

    def test_non_indexer_weight_passes_through(self):
        assert not _is_skipped_indexer_weight(
            "model.layers.2.self_attn.wq_a.weight",
            self.skip_ids,
        )

    def test_compressor_not_filtered(self):
        """compressor weights (not indexer) pass through."""
        assert not _is_skipped_indexer_weight(
            "model.layers.2.self_attn.compressor.wkv.weight",
            self.skip_ids,
        )
