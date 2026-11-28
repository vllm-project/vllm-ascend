"""UT for vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_kv.py."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h import dspark_kv
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_kv import (
    DraftKVCacheMetadata,
    apply_dspark_resident_kv_specs,
    build_draft_kv_metadata,
    build_draft_kv_read_batches,
    get_resident_dspark_layer_names,
)


def _vllm_config(method="dspark", forward_context=None):
    return SimpleNamespace(
        speculative_config=None if method is None else SimpleNamespace(method=method),
        compilation_config=SimpleNamespace(static_forward_context=forward_context or {}),
    )


def _speculator(names):
    if names is None:
        return SimpleNamespace()
    return SimpleNamespace(draft_attn_layer_names=names)


def _metadata(
    group_id=0,
    block_size=4,
    num_blocks=2,
    *,
    base_addrs=(100,),
    block_strides=(64,),
    block_lens=(32,),
    block_scales=(1,),
    shapes=((4, 8),),
    dtypes=("torch.float32",),
):
    return DraftKVCacheMetadata(
        group_id, block_size, num_blocks, base_addrs, block_strides, block_lens, block_scales, shapes, dtypes
    )


class TestGetResidentDSparkLayerNames:
    @pytest.mark.parametrize(
        ("sparse", "last_rank", "method"),
        [
            (False, True, "dspark"),
            (True, False, "dspark"),
            (True, True, None),
            (True, True, "mtp"),
        ],
    )
    def test_inactive_paths_return_empty_set(self, sparse, last_rank, method):
        names = get_resident_dspark_layer_names(
            _vllm_config(method=method),
            _speculator(("a", "b")),
            sparse_offload_enabled=sparse,
            is_last_pp_rank=last_rank,
        )
        assert names == set()

    @pytest.mark.parametrize("names", [None, ()])
    def test_requires_draft_layer_ownership(self, names):
        with pytest.raises(ValueError, match="cache ownership"):
            get_resident_dspark_layer_names(
                _vllm_config(),
                _speculator(names),
                sparse_offload_enabled=True,
                is_last_pp_rank=True,
            )

    def test_valid_names_with_self_sharing(self):
        config = _vllm_config(
            forward_context={
                "a": SimpleNamespace(kv_sharing_target_layer_name="a"),
                "b": SimpleNamespace(kv_sharing_target_layer_name=None),
            }
        )
        names = get_resident_dspark_layer_names(
            config,
            _speculator(("a", "b")),
            sparse_offload_enabled=True,
            is_last_pp_rank=True,
            shared_kv_cache_layers={"b": "b"},
        )
        assert names == {"a", "b"}

    def test_rejects_sharing_target_model_kv(self):
        with pytest.raises(ValueError, match="cannot share target-model KV caches"):
            get_resident_dspark_layer_names(
                _vllm_config(),
                _speculator(("a",)),
                sparse_offload_enabled=True,
                is_last_pp_rank=True,
                shared_kv_cache_layers={"a": "target"},
            )
        with pytest.raises(ValueError, match="cannot share target-model KV caches"):
            get_resident_dspark_layer_names(
                _vllm_config(forward_context={"a": SimpleNamespace(kv_sharing_target_layer_name="ext")}),
                _speculator(("a",)),
                sparse_offload_enabled=True,
                is_last_pp_rank=True,
            )


class TestApplyDSparkResidentKVSpecs:
    def _mla_spec(self, **overrides):
        kwargs = dict(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16, store_on_host=True)
        kwargs.update(overrides)
        return AscendMLAAttentionSpec(**kwargs)

    def test_inactive_path_keeps_specs(self):
        specs = {"a": self._mla_spec()}
        result = apply_dspark_resident_kv_specs(
            specs, _vllm_config(), _speculator(("a",)), sparse_offload_enabled=False, is_last_pp_rank=True
        )
        assert result is specs
        assert specs["a"].store_on_host is True

    def test_active_path_only_flips_placement(self):
        original = self._mla_spec()
        specs = {"a": original, "b": SimpleNamespace()}
        result = apply_dspark_resident_kv_specs(
            specs, _vllm_config(), _speculator(("a",)), sparse_offload_enabled=True, is_last_pp_rank=True
        )
        # The mapping is updated in place; only the flagged spec object is swapped.
        assert result is specs
        updated = result["a"]
        assert updated is not original
        assert original.store_on_host is True
        assert updated.store_on_host is False
        assert updated.block_size == 128
        assert updated.num_kv_heads == 1
        assert updated.head_size == 576
        assert updated.dtype is torch.bfloat16
        assert result["b"] is specs["b"]

    def test_rejects_non_mla_draft_specs(self):
        with pytest.raises(ValueError, match="MLA DSpark draft checkpoint"):
            apply_dspark_resident_kv_specs(
                {"a": SimpleNamespace()},
                _vllm_config(),
                _speculator(("a",)),
                sparse_offload_enabled=True,
                is_last_pp_rank=True,
            )


class TestDraftKVCacheMetadata:
    def test_accepts_valid_component_geometry(self):
        metadata = _metadata()
        assert metadata.group_id == 0
        assert metadata.block_size == 4
        assert metadata.num_blocks == 2
        assert metadata.base_addrs == (100,)
        assert metadata.block_strides == (64,)
        assert metadata.block_lens == (32,)
        assert metadata.block_scales == (1,)
        assert metadata.shapes == ((4, 8),)
        assert metadata.dtypes == ("torch.float32",)

    @pytest.mark.parametrize(
        "overrides",
        [
            pytest.param(
                dict(base_addrs=(), block_strides=(), block_lens=(), block_scales=(), shapes=(), dtypes=()),
                id="empty",
            ),
            pytest.param(dict(base_addrs=(100, 200), block_strides=(64,)), id="mismatched-lengths"),
            pytest.param(dict(base_addrs=(0,)), id="zero-base"),
            pytest.param(dict(base_addrs=(-100,)), id="negative-base"),
            pytest.param(dict(block_lens=(0,)), id="zero-length"),
            pytest.param(dict(block_strides=(10,)), id="stride-below-length"),
            pytest.param(dict(block_scales=(0,)), id="zero-scale"),
        ],
    )
    def test_rejects_invalid_component_geometry(self, overrides):
        kwargs = dict(
            base_addrs=(100,),
            block_strides=(64,),
            block_lens=(32,),
            block_scales=(1,),
            shapes=((4, 8),),
            dtypes=("torch.float32",),
        )
        kwargs.update(overrides)
        with pytest.raises(ValueError):
            DraftKVCacheMetadata(0, 4, 2, **kwargs)

    @pytest.mark.parametrize(
        ("group_id", "block_size", "num_blocks"),
        [(-1, 4, 2), (0, 0, 2), (0, 4, 0)],
    )
    def test_rejects_nonpositive_bounds(self, group_id, block_size, num_blocks):
        with pytest.raises(ValueError, match="bounds must be positive"):
            DraftKVCacheMetadata(
                group_id,
                block_size,
                num_blocks,
                base_addrs=(100,),
                block_strides=(64,),
                block_lens=(32,),
                block_scales=(1,),
                shapes=((4, 8),),
                dtypes=("torch.float32",),
            )


class TestBuildDraftKVMetadata:
    @staticmethod
    def _kv_cache_config(names=("lm_d0",), num_blocks=4, block_size=16):
        groups = [
            SimpleNamespace(layer_names=(name,), kv_cache_spec=SimpleNamespace(block_size=block_size)) for name in names
        ]
        return SimpleNamespace(dspark_draft_layer_names=tuple(names), kv_cache_groups=groups, num_blocks=num_blocks)

    def test_single_component_success(self):
        tensor = torch.empty(16, 4, dtype=torch.float32)
        metadata = build_draft_kv_metadata(self._kv_cache_config(), {"lm_d0": (tensor, None)})
        md = metadata["lm_d0"]
        assert md.group_id == 0
        assert md.block_size == 16
        assert md.num_blocks == 4
        assert md.base_addrs == (tensor.data_ptr(),)
        assert md.block_strides == (tensor.stride(0) * tensor.element_size(),)
        assert md.block_lens == (16,)
        assert md.block_scales == (4,)
        assert md.shapes == ((4,),)
        assert md.dtypes == ("torch.float32",)

    def test_multi_component_and_empty_tensor_filter(self):
        main = torch.empty(16, 4, dtype=torch.float32)
        scale = torch.empty(16, 4, dtype=torch.bfloat16)
        metadata = build_draft_kv_metadata(self._kv_cache_config(), {"lm_d0": (torch.empty(0), main, scale)})
        md = metadata["lm_d0"]
        assert md.base_addrs == (main.data_ptr(), scale.data_ptr())
        assert md.block_lens == (16, 8)
        assert md.block_scales == (4, 4)
        assert md.dtypes == ("torch.float32", "torch.bfloat16")

    @pytest.mark.parametrize(
        "bad_cache",
        [
            pytest.param(torch.empty(16), id="one-dimensional"),
            pytest.param(torch.empty(15, 4), id="rows-not-multiple-of-blocks"),
            pytest.param(torch.empty(16, 8)[:, ::2], id="non-contiguous-inner-page"),
        ],
    )
    def test_rejects_bad_page_layout(self, bad_cache):
        with pytest.raises(ValueError, match="contiguous inner kernel pages"):
            build_draft_kv_metadata(self._kv_cache_config(), {"lm_d0": bad_cache})

    def test_rejects_block_size_mismatch(self):
        with pytest.raises(ValueError, match="cover one logical token block"):
            build_draft_kv_metadata(self._kv_cache_config(), {"lm_d0": torch.empty(16, 8, dtype=torch.float32)})

    def test_rejects_layer_missing_from_kv_caches(self):
        with pytest.raises(ValueError, match="missing from connector registration"):
            build_draft_kv_metadata(self._kv_cache_config(), {})

    def test_rejects_layer_not_owned_by_any_group(self):
        config = self._kv_cache_config()
        config.kv_cache_groups = [
            SimpleNamespace(layer_names=(), kv_cache_spec=SimpleNamespace(block_size=16))
        ]
        with pytest.raises(ValueError, match="missing from connector registration"):
            build_draft_kv_metadata(config, {"lm_d0": torch.empty(16, 4, dtype=torch.float32)})


class TestBuildDraftKVReadBatches:
    @staticmethod
    def _pair(**local_overrides):
        remote = {"a": _metadata(base_addrs=(100,), **local_overrides)}
        local = {"a": _metadata(base_addrs=(200,), **local_overrides)}
        return remote, local

    def test_rejects_coverage_and_layout_mismatches(self):
        remote, local = self._pair()
        with pytest.raises(ValueError, match="cover every loaded draft layer exactly"):
            next(build_draft_kv_read_batches({"b": _metadata()}, local, {"b": [0]}, {0: [0]}, 4))
        with pytest.raises(ValueError, match="cover every loaded draft layer exactly"):
            next(build_draft_kv_read_batches(remote, {}, {}, {0: [0]}, 4))
        with pytest.raises(ValueError, match="cover every loaded draft layer exactly"):
            next(build_draft_kv_read_batches({}, local, {"a": [0]}, {0: [0]}, 4))

        mismatched = {"a": _metadata(base_addrs=(100,), block_size=8)}
        with pytest.raises(ValueError, match="layout mismatch"):
            next(build_draft_kv_read_batches(mismatched, local, {"a": [0]}, {0: [0]}, 4))
        mismatched = {"a": _metadata(base_addrs=(100,), dtypes=("torch.float16",))}
        with pytest.raises(ValueError, match="layout mismatch"):
            next(build_draft_kv_read_batches(mismatched, local, {"a": [0]}, {0: [0]}, 4))

    @pytest.mark.parametrize(
        ("source_blocks", "dest_blocks"),
        [
            pytest.param({"a": []}, {0: [0]}, id="short-source"),
            pytest.param({"a": [0]}, {}, id="short-dest"),
            pytest.param({"a": [0, 0]}, {0: [0, 1]}, id="aliased-source"),
            pytest.param({"a": [0, 1]}, {0: [1, 1]}, id="aliased-dest"),
            pytest.param({"a": [1.0]}, {0: [0]}, id="float-id"),
            pytest.param({"a": [np.int64(0)]}, {0: [0]}, id="numpy-id"),
            pytest.param({"a": [2]}, {0: [0]}, id="source-out-of-range"),
            pytest.param({"a": [0]}, {0: [2]}, id="dest-out-of-range"),
        ],
    )
    def test_rejects_invalid_block_tables(self, source_blocks, dest_blocks):
        remote, local = self._pair()
        prompt = 8 if len(source_blocks["a"]) == 2 else 4
        with pytest.raises(ValueError):
            next(build_draft_kv_read_batches(remote, local, source_blocks, dest_blocks, prompt))

    def test_pointer_arithmetic_single_block(self):
        remote, local = self._pair()
        batches = list(build_draft_kv_read_batches(remote, local, {"a": [0]}, {0: [0]}, 4))
        assert batches == [([100], [200], [32])]

    def test_copies_only_valid_prompt_rows(self):
        remote, local = self._pair()
        batches = list(build_draft_kv_read_batches(remote, local, {"a": [0, 1]}, {0: [0, 1]}, 5))
        assert batches == [([100, 164], [200, 264], [32, 8])]

    def test_multi_kernel_rows_per_block(self):
        remote, local = self._pair(block_lens=(32,), block_scales=(2,), shapes=((2, 4),))
        batches = list(build_draft_kv_read_batches(remote, local, {"a": [0, 1]}, {0: [0, 1]}, 8))
        assert batches == [([100, 164, 228, 292], [200, 264, 328, 392], [32, 32, 32, 32])]

    def test_yields_bounded_batches(self, monkeypatch):
        monkeypatch.setattr(dspark_kv, "MAX_DRAFT_KV_READ_DESCRIPTORS", 3)
        remote, local = self._pair(num_blocks=5)
        blocks = list(range(5))
        batches = list(build_draft_kv_read_batches(remote, local, {"a": blocks}, {0: blocks}, 20))
        assert [len(batch[2]) for batch in batches] == [3, 2]
        assert [len(batch[0]) for batch in batches] == [3, 2]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
