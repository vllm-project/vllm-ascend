import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendCircularBufferSpec
from vllm_ascend.core.six_region_kv_cache_layout import (
    GDN,
    HIDDEN,
    PLE,
    build_six_region_kv_cache_layout,
    make_contiguous_slab_view,
)
from vllm_ascend.models.qwen4_exp.common.qsa_cache import (
    circular_qsa_slot_mapping,
    compressed_qsa_slot_mapping,
)


def _groups():
    sources = [
        "model.layers.11.self_attn",
        "model.layers.47.self_attn",
        "model.mtp.layers.48.self_attn",
    ]
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = AscendCircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    gdn = MambaSpec(
        block_size=65536,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=65536,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
    )
    hidden = HiddenStateCacheSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    raw_sources = [sources[2], sources[0], sources[1]]
    compressed_sources = [sources[1], sources[2], sources[0]]
    return [
        KVCacheGroupSpec([f"{source}.attn" for source in sources], main),
        KVCacheGroupSpec([f"{source}.indexer.raw_key_cache" for source in raw_sources], raw),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in compressed_sources],
            compressed,
        ),
        KVCacheGroupSpec(["model.layers.0.linear_attn", "model.layers.1.linear_attn"], gdn),
        KVCacheGroupSpec(["model.ple"], ple),
        KVCacheGroupSpec(["model.cache_only_layers.0"], hidden),
    ]


def test_six_slab_offsets_and_source_pairing():
    layout = build_six_region_kv_cache_layout(_groups(), num_blocks=3)
    assert layout is not None
    assert layout.slot_count == 3
    assert [(r.name, r.offset, r.page_size_bytes, r.size) for r in layout.regions] == [
        ("r1", 0, 512, 1536),
        ("r2", 1536, 32768, 98304),
        ("r3", 99840, 32768, 98304),
        ("r4", 198144, 512, 1536),
        ("r5", 199680, 4096, 12288),
        ("r6", 211968, 1280, 3840),
    ]
    assert layout.slot_backing_size == 215808
    for left, right in zip(layout.regions, layout.regions[1:]):
        assert right.offset >= left.offset + left.size
    assert layout.owner("model.layers.11.self_attn.attn").spec.block_size == 256
    compressed_spec = layout.owner("model.layers.11.self_attn.indexer.compressed_key_cache").spec
    assert isinstance(compressed_spec, MLAAttentionSpec)
    assert compressed_spec.block_size == 256
    assert compressed_spec.storage_block_size == 64
    assert compressed_spec.real_page_size_bytes == 4096
    assert layout.owner("model.layers.11.self_attn.indexer.raw_key_cache").spec.block_size == 8

    for slot, source in enumerate(
        [
            "model.layers.11.self_attn",
            "model.layers.47.self_attn",
            "model.mtp.layers.48.self_attn",
        ]
    ):
        assert layout.owner(f"{source}.attn").slot == slot
        assert layout.owner(f"{source}.indexer.raw_key_cache").slot == slot
        assert layout.owner(f"{source}.indexer.compressed_key_cache").slot == slot
        shared = layout.slot_shared_by(slot)
        assert f"{source}.attn" in shared
        assert f"{source}.indexer.raw_key_cache" in shared
        assert f"{source}.indexer.compressed_key_cache" in shared
    assert layout.owner("model.layers.0.linear_attn").role == GDN
    assert layout.owner("model.ple").role == PLE
    assert layout.owner("model.cache_only_layers.0").role == HIDDEN


def test_layer_views_are_contiguous_slabs_and_isolated():
    num_blocks = 3
    layout = build_six_region_kv_cache_layout(_groups(), num_blocks=num_blocks)
    assert layout is not None
    backing = torch.zeros(layout.slot_backing_size, dtype=torch.int8)

    def view(region, shape):
        return make_contiguous_slab_view(
            backing,
            dtype=torch.bfloat16,
            num_blocks=num_blocks,
            item_shape=shape,
            storage_offset=layout.region(region).offset,
        )

    views = {
        "k": view("r2", (256, 1, 64)),
        "v": view("r3", (256, 1, 64)),
        "raw": view("r4", (8, 1, 32)),
        "compressed": view("r5", (64, 1, 32)),
        "gdn_conv": view("r1", (64, 4)),
        "gdn_ssm": view("r2", (128, 128)),
        "ple": view("r6", (128, 5)),
    }
    storage_ptr = backing.untyped_storage().data_ptr()
    region_for_view = {
        "k": "r2",
        "v": "r3",
        "raw": "r4",
        "compressed": "r5",
        "gdn_conv": "r1",
        "gdn_ssm": "r2",
        "ple": "r6",
    }
    for name, slab in views.items():
        region = layout.region(region_for_view[name])
        assert slab.untyped_storage().data_ptr() == storage_ptr
        assert slab.is_contiguous()
        assert slab.stride() == torch.empty_like(slab).stride()
        assert slab.stride(0) * slab.element_size() == region.page_size_bytes
        assert slab[1].data_ptr() - slab[0].data_ptr() == region.page_size_bytes

    views["raw"][1, 3, 0, 7] = 17
    assert views["raw"][1, 3, 0, 7].item() == 17
    for name, slab in views.items():
        if name != "raw":
            assert torch.count_nonzero(slab).item() == 0
    assert torch.count_nonzero(views["raw"][0]).item() == 0
    assert torch.count_nonzero(views["raw"][2]).item() == 0

    # R2 is intentionally reusable by QSA K and GDN SSM. Independent groups
    # use independent block tables; different physical block IDs stay isolated.
    views["k"][1, 3, 0, 7] = 23
    assert torch.count_nonzero(views["gdn_ssm"][0]).item() == 0
    assert torch.count_nonzero(views["gdn_ssm"][2]).item() == 0

    hidden_backing = torch.zeros(4096, dtype=torch.int8)
    assert hidden_backing.untyped_storage().data_ptr() != storage_ptr


def test_qsa_source_sets_must_match():
    groups = _groups()
    groups[1].layer_names[-1] = "model.layers.46.self_attn.indexer.raw_key_cache"
    try:
        build_six_region_kv_cache_layout(groups, num_blocks=2)
    except ValueError as error:
        assert "one-to-one source-layer mapping" in str(error)
    else:
        raise AssertionError("mismatched QSA source owners were accepted")


def test_qsa_position_mapping_is_unchanged_by_physical_slabs():
    block_table = torch.tensor([[7, 9], [4, 6]], dtype=torch.int32)
    requests = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    positions = torch.tensor([0, 11, 7, 15], dtype=torch.int64)
    circular = circular_qsa_slot_mapping(
        block_table,
        requests,
        positions,
        compressor_state_size=8,
    )
    assert circular.tolist() == [56, 59, 39, 39]

    compressed = compressed_qsa_slot_mapping(
        block_table,
        requests,
        torch.tensor([3, 7, 11, 15], dtype=torch.int64),
        storage_block_size=64,
        compress_ratio=4,
    )
    assert compressed.tolist() == [448, 449, 258, 259]


def test_real_qsa_768_8_192_geometry_and_boundaries():
    from vllm_ascend.patch.platform.patch_kv_cache_utils import (
        _merge_qsa_composite_group,
        _prepare_qwen4_exp_qsa_groups,
    )

    sources = [
        "model.layers.11.self_attn",
        "model.layers.47.self_attn",
        "model.mtp.layers.48.self_attn",
    ]
    main = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    raw = AscendCircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    gdn = MambaSpec(
        block_size=65536,
        shapes=((64, 4), (384, 512)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=65536,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
    )
    groups = [
        KVCacheGroupSpec([f"{source}.attn" for source in sources], main),
        KVCacheGroupSpec([f"{source}.indexer.raw_key_cache" for source in sources], raw),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in sources],
            compressed,
        ),
        KVCacheGroupSpec(["model.layers.0.linear_attn"], gdn),
        KVCacheGroupSpec(["model.ple"], ple),
    ]
    flat = {name: group.kv_cache_spec for group in groups for name in group.layer_names}
    roles = _prepare_qwen4_exp_qsa_groups(flat)
    assert roles is not None
    assert main.block_size == 768
    assert raw.block_size == 8
    assert compressed.block_size == 768
    assert compressed.storage_block_size == 192
    assert compressed.real_page_size_bytes == 49152

    merged = _merge_qsa_composite_group(groups, flat, roles[0], roles[1], roles[2])
    composite = next(
        group
        for group in merged
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) and group.kv_cache_spec.block_size == 768
    )
    assert composite.kv_cache_spec.block_size == 768
    assert set(composite.layer_names) == roles[0] | roles[1]
    assert not any(name.endswith(".raw_key_cache") for name in composite.layer_names)
    raw_group = next(
        group
        for group in merged
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) and group.kv_cache_spec.block_size == 8
    )
    assert len(raw_group.layer_names) == 3
    assert set(raw_group.layer_names) == roles[2]

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    from vllm_ascend.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        get_manager_class_for_kv_cache_spec,
    )

    assert get_manager_class_for_kv_cache_spec(composite.kv_cache_spec) is FullAttentionManager
    assert get_manager_class_for_kv_cache_spec(raw_group.kv_cache_spec) is CircularBufferManager

    layout = build_six_region_kv_cache_layout(merged, num_blocks=3)
    assert layout is not None
    assert layout.region("r5").page_size_bytes == 49152
    for position in (0, 767, 768, 1535, 1536):
        assert position // main.block_size == position // compressed.block_size

    block_table = torch.tensor([[7, 9]], dtype=torch.int32)
    mapped = compressed_qsa_slot_mapping(
        block_table,
        torch.zeros(3, dtype=torch.int64),
        torch.tensor([767, 768, 771], dtype=torch.int64),
        storage_block_size=compressed.storage_block_size,
        compress_ratio=compressed.compress_ratio,
    )
    assert mapped.tolist() == [7 * 192 + 191, -1, 9 * 192]

    backing = torch.zeros(layout.slot_backing_size, dtype=torch.int8)
    r5 = make_contiguous_slab_view(
        backing,
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(192, 1, 128),
        storage_offset=layout.region("r5").offset,
    )
    assert r5.shape == (3, 192, 1, 128)
    assert r5.is_contiguous()
    assert r5.stride(0) * r5.element_size() == 49152
    assert r5[1].data_ptr() - r5[0].data_ptr() == 49152


def test_planner_emits_one_unstrided_backing_per_slot():
    from types import SimpleNamespace

    from vllm_ascend.patch.platform.patch_kv_cache_utils import (
        _get_qwen4_exp_kv_cache_config,
    )

    config = _get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3)),
        _groups(),
        10**9,
    )
    assert config is not None
    assert config.num_blocks == 3
    slot_tensors = config.kv_cache_tensors[:3]
    hidden_tensor = config.kv_cache_tensors[3]
    assert all(tensor.block_stride == 0 for tensor in slot_tensors)
    assert all(tensor.offset == 0 for tensor in slot_tensors)
    assert all(tensor.size == 215808 for tensor in slot_tensors)
    assert hidden_tensor.shared_by == ["model.cache_only_layers.0"]
    assert hidden_tensor.size == 98304
    assert "model.layers.11.self_attn.attn" in slot_tensors[0].shared_by
    assert "model.layers.0.linear_attn" in slot_tensors[0].shared_by
    assert "model.ple" in slot_tensors[0].shared_by
