"""Unit tests for ``build_layerwise_reuse_layout`` multi-cache-spec layers.

Covers the layerwise reuse-layout builder of the AscendStore KV pool for
heterogeneous layers that carry more than one main cache spec:

- DeepSeek-V4 DSA layers expose 5 cache specs per physical layer
  (``.attn``, ``.indexer.k_cache``, ``indexer.compressor.state_cache``,
  ``compressor.state_cache``, ``swa_cache``); the original code required
  exactly one main spec + one indexer spec and raised ValueError.
- With the fix, ``.attn`` is selected as the main spec, the remaining
  non-indexer specs are preserved in ``extra_main_specs`` and the indexer
  spec stays optional.
"""

from dataclasses import dataclass

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    NamedKVCacheSpec,
    build_layerwise_reuse_layout,
)


@dataclass(frozen=True)
class _FakeSpec:
    block_size: int = 16
    num_kv_heads: int = 1
    head_size: int = 512


def _spec() -> _FakeSpec:
    return _FakeSpec()


def _dsv4_dsa_layer_specs(layer_idx: int) -> list[NamedKVCacheSpec]:
    """All 5 cache specs of one DSV4 DSA physical layer."""
    names = [
        f"model.layers.{layer_idx}.self_attn.attn",
        f"model.layers.{layer_idx}.self_attn.indexer.k_cache",
        f"model.layers.{layer_idx}.self_attn.indexer.compressor.state_cache",
        f"model.layers.{layer_idx}.self_attn.compressor.state_cache",
        f"model.layers.{layer_idx}.self_attn.swa_cache",
    ]
    return [NamedKVCacheSpec(name, _spec()) for name in names]


def _uniform_layer_specs(layer_idx: int) -> list[NamedKVCacheSpec]:
    """One main + one indexer spec (the previously supported layout)."""
    return [
        NamedKVCacheSpec(f"model.layers.{layer_idx}.self_attn.attn", _spec()),
        NamedKVCacheSpec(f"model.layers.{layer_idx}.self_attn.indexer.k_cache", _spec()),
    ]


def _named_specs_by_layer(num_dsa: int = 4) -> dict[str, "_FakeSpec"]:
    layer_specs: dict[str, _FakeSpec] = {}
    for i in range(num_dsa):
        for named in _dsv4_dsa_layer_specs(i):
            layer_specs[named.layer_name] = named.spec
    return layer_specs


class TestDSV4MultiCacheSpecLayers:
    def test_dsa_layer_with_5_specs_no_longer_raises(self):
        # Original code: ValueError "must have exactly one main spec and one
        # indexer spec". Fixed code must build the layout successfully.
        layout = build_layerwise_reuse_layout(_named_specs_by_layer(4), 4, {})
        assert len(layout.layer_cache_specs) == 4

    def test_attn_selected_as_main(self):
        layout = build_layerwise_reuse_layout(_named_specs_by_layer(4), 4, {})
        for physical_layer, specs in layout.layer_cache_specs.items():
            assert specs.main.layer_name.endswith(".attn"), (
                f"layer {physical_layer}: main spec should be .attn, got {specs.main.layer_name}"
            )

    def test_extra_main_specs_preserved(self):
        layout = build_layerwise_reuse_layout(_named_specs_by_layer(4), 4, {})
        for physical_layer, specs in layout.layer_cache_specs.items():
            extra_names = {s.layer_name.rsplit(".", 1)[-1] for s in specs.extra_main_specs}
            # compressor.state_cache and swa_cache are the extra main specs
            assert "swa_cache" in extra_names
            assert "state_cache" in extra_names
            # the indexer.k_cache must NOT be in extra_main_specs
            for s in specs.extra_main_specs:
                assert not s.layer_name.endswith(".indexer.k_cache")

    def test_indexer_spec_optional(self):
        layout = build_layerwise_reuse_layout(_named_specs_by_layer(4), 4, {})
        for specs in layout.layer_cache_specs.values():
            if specs.indexer is not None:
                assert specs.indexer.layer_name.endswith(".indexer.k_cache")

    def test_layout_without_indexer_only(self):
        # A layer with only main specs (no indexer at all) also works now.
        layer_specs = {
            f"model.layers.0.self_attn.{name}": _spec()
            for name in ("attn", "swa_cache")
        }
        layout = build_layerwise_reuse_layout(layer_specs, 1, {})
        assert layout.layer_cache_specs[0].indexer is None
        assert layout.layer_cache_specs[0].main.layer_name.endswith(".attn")
        assert len(layout.layer_cache_specs[0].extra_main_specs) == 1

    def test_uniform_layout_still_works(self):
        # The previously supported 1-main + 1-indexer layout keeps working.
        layer_specs: dict[str, _FakeSpec] = {}
        for i in range(4):
            for named in _uniform_layer_specs(i):
                layer_specs[named.layer_name] = named.spec
        layout = build_layerwise_reuse_layout(layer_specs, 4, {})
        for specs in layout.layer_cache_specs.values():
            assert specs.main.layer_name.endswith(".attn")
            assert specs.indexer is not None
            assert specs.extra_main_specs == ()

    def test_multi_indexer_only_layer_raises(self):
        # A multi-spec layer where every spec ends with the indexer suffix
        # has no main cache spec: must still raise.
        layer_specs = {
            "model.layers.0.self_attn.attn": _spec(),
            "model.layers.1.self_attn.a.indexer.k_cache": _spec(),
            "model.layers.1.self_attn.b.indexer.k_cache": _spec(),
        }
        with pytest.raises(ValueError, match="no main cache spec"):
            build_layerwise_reuse_layout(layer_specs, 2, {})

    def test_single_spec_layer_takes_fast_path(self):
        # A single-spec layer (len==1) is used as main directly without
        # validation -- pre-existing fast-path behavior preserved.
        layer_specs = {
            "model.layers.0.self_attn.indexer.k_cache": _spec(),
        }
        layout = build_layerwise_reuse_layout(layer_specs, 1, {})
        assert layout.layer_cache_specs[0].main.layer_name.endswith(".indexer.k_cache")
        assert layout.layer_cache_specs[0].extra_main_specs == ()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
