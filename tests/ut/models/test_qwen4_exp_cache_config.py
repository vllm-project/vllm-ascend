# mypy: disable-error-code="operator, union-attr"
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.config import KVTransferConfig
from vllm.model_executor.models import ModelRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.models.qwen4_exp.cache_config import (
    GDN,
    HIDDEN,
    PLE,
    _merge_mamba_state_groups,
    _merge_qsa_composite_groups,
    _prepare_qsa_composite_groups,
    build_qwen4_kv_cache_layout,
    get_qwen4_exp_kv_cache_config,
    get_qwen4_exp_kv_cache_groups,
    get_qwen4_exp_pool_bytes_per_block,
    make_slab_view,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_max_memory_usage_bytes_from_groups,
)
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2 import attn_utils as attn_utils_v2


def _groups() -> list[KVCacheGroupSpec]:
    # Match vLLM alignment: 256 tokens cover the full 33KiB GDN page.
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
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    gdn = MambaSpec(
        block_size=128,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=128,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
    )
    hidden = HiddenStateCacheSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
    )
    raw_sources = [sources[2], sources[0], sources[1]]
    compressed_sources = [sources[1], sources[2], sources[0]]
    return [
        KVCacheGroupSpec(
            [f"{source}.attn" for source in sources],
            main,
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.raw_key_cache" for source in raw_sources],
            raw,
        ),
        KVCacheGroupSpec(
            [f"{source}.indexer.compressed_key_cache" for source in compressed_sources],
            compressed,
        ),
        KVCacheGroupSpec(
            [
                "model.layers.0.linear_attn",
                "model.layers.1.linear_attn",
            ],
            gdn,
        ),
        KVCacheGroupSpec(["model.ple"], ple),
        KVCacheGroupSpec(["model.cache_only_layers.0"], hidden),
    ]


def _ungrouped_qwen_specs() -> dict[str, KVCacheSpec]:
    main = FullAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    raw = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=32,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    gdn = MambaSpec(
        block_size=128,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    ple = MambaSpec(
        block_size=128,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
        tp_replicated=True,
    )
    specs: dict[str, KVCacheSpec] = {}
    for index in range(13):
        source = f"model.layers.{index * 4 + 3}.self_attn"
        # Deliberately register the compressed owner first: grouping must
        # still choose the main attention spec as the scheduler representative.
        specs[f"{source}.indexer.compressed_key_cache"] = compressed
        specs[f"{source}.indexer.raw_key_cache"] = raw
        specs[f"{source}.attn"] = main
    for index in range(36):
        specs[f"model.layers.{index}.linear_attn"] = gdn
    specs["model.ple"] = ple
    assert len(specs) == 76
    return specs


def test_block_packed_states_have_disjoint_components_and_aligned_kv_pages() -> None:
    layout = build_qwen4_kv_cache_layout(_groups(), num_blocks=3)
    assert layout is not None
    assert layout.slot_count == 3
    conv, ssm = layout.region("r1"), layout.region("r2")
    assert conv.block_stride == ssm.block_stride == 65536
    assert ssm.offset == conv.page_size_bytes
    occupied = []
    for block in range(3):
        for region in (conv, ssm):
            start = region.offset + block * region.block_stride
            occupied.append((start, start + region.page_size_bytes))
    occupied.sort()
    assert all(left[1] <= right[0] for left, right in zip(occupied, occupied[1:]))
    assert layout.region("r4").offset >= occupied[-1][1]
    assert layout.region("r5").offset >= layout.region("r4").end
    assert layout.slot_backing_size >= layout.region("r5").end

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
    assert layout.owner("model.layers.0.linear_attn").role == GDN
    assert layout.owner("model.ple").role == PLE
    assert layout.owner("model.cache_only_layers.0").role == HIDDEN


def test_qwen4_cache_views_are_contiguous_and_advanced_index_is_bounded() -> None:
    num_blocks = 3
    layout = build_qwen4_kv_cache_layout(
        _groups(),
        num_blocks=num_blocks,
    )
    assert layout is not None
    backing = torch.zeros(layout.slot_backing_size, dtype=torch.int8)

    def view(region_name: str, shape: tuple[int, ...]) -> torch.Tensor:
        return make_slab_view(
            backing,
            dtype=torch.bfloat16,
            num_blocks=num_blocks,
            item_shape=shape,
            storage_offset=layout.region(region_name).offset,
            block_stride=layout.region(region_name).block_stride,
        )

    views = {
        "raw": view("r4", (8, 1, 32)),
        "compressed": view("r5", (64, 1, 32)),
        "gdn_conv": view("r1", (64, 4)),
        "gdn_ssm": view("r2", (128, 128)),
    }
    region_for_view = {
        "raw": "r4",
        "compressed": "r5",
        "gdn_conv": "r1",
        "gdn_ssm": "r2",
    }
    storage_ptr = backing.untyped_storage().data_ptr()
    for name, slab in views.items():
        region = layout.region(region_for_view[name])
        assert slab.untyped_storage().data_ptr() == storage_ptr
        assert slab[0].is_contiguous()
        assert slab.is_contiguous() == (region.block_stride == slab[0].numel() * slab.element_size())
        assert slab.storage_offset() * slab.element_size() == region.offset
        assert slab.stride(0) * slab.element_size() == region.block_stride
        assert slab[0].numel() * slab.element_size() <= region.page_size_bytes

    packed = [
        make_slab_view(
            backing,
            dtype=torch.bfloat16,
            num_blocks=num_blocks,
            item_shape=(128, 1, 64),
            storage_offset=layout.region("r0").offset + offset,
            block_stride=layout.region("r2").block_stride,
        )
        for offset in (0, 16384)
    ]
    keys, values = packed
    assert not keys.is_contiguous() and not values.is_contiguous()
    assert keys.stride(0) * keys.element_size() == layout.region("r2").block_stride
    keys[1].fill_(7)
    values[1].fill_(11)
    ssm = views["gdn_ssm"]
    assert torch.all(views["gdn_conv"][1] == 7)
    assert torch.all(ssm[1].flatten()[:7936] == 7)
    assert torch.all(ssm[1].flatten()[7936:16128] == 11)
    assert torch.count_nonzero(ssm[1].flatten()[16128:]) == 0
    assert torch.count_nonzero(ssm[[0, 2]]) == 0
    ssm.zero_()
    views["gdn_conv"].zero_()

    views["gdn_ssm"][:, 0, 0] = torch.tensor(
        [1, 2, 3],
        dtype=torch.bfloat16,
    )
    selected = views["gdn_ssm"][torch.tensor([0, 2])]
    assert selected.shape == (2, 128, 128)
    assert selected.numel() == 2 * 128 * 128
    assert selected.is_contiguous()
    assert selected.untyped_storage().data_ptr() != storage_ptr
    assert selected.untyped_storage().nbytes() == (selected.numel() * selected.element_size())
    assert selected[:, 0, 0].tolist() == [1, 3]
    views["gdn_ssm"][:, 0, 0] = 0

    views["raw"][1, 3, 0, 7] = 17
    for name, slab in views.items():
        if name != "raw":
            assert torch.count_nonzero(slab).item() == 0


def test_main_planner_uses_one_shared_backing_geometry() -> None:
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=3,
                prefix_cache_retention_interval=None,
            )
        ),
        _groups(),
        10**9,
    )
    assert config is not None
    assert config.num_blocks == 3
    layout = build_qwen4_kv_cache_layout(
        config.kv_cache_groups,
        num_blocks=3,
    )
    assert layout is not None
    role_tensors = config.kv_cache_tensors[:4]
    ple_tensor = config.kv_cache_tensors[4]
    hidden_tensor = config.kv_cache_tensors[5]
    expected_backing_size = layout.slot_count * layout.slot_backing_size
    assert all(tensor.size == expected_backing_size for tensor in role_tensors)
    assert [tensor.offset for tensor in role_tensors] == [
        layout.region(name).offset for name in ("r0", "r4", "r5", "r1")
    ]
    assert all(tensor.layer_stride == layout.slot_backing_size for tensor in role_tensors)
    assert ple_tensor.layers == ["model.ple"]
    assert ple_tensor.size == 3 * 128 * 5 * 2
    assert ple_tensor.block_stride == 128 * 5 * 2
    assert ple_tensor.offset == 0
    assert hidden_tensor.layers == ["model.cache_only_layers.0"]
    assert hidden_tensor.size == 49152


def _allocate_mrv2_kv_cache(
    config,
    monkeypatch,
    *,
    use_main: bool = False,
    kernel_block_sizes: list[int] | None = None,
    kv_transfer_config=None,
):
    """Allocate and materialize Qwen4Exp cache views through MRV2 only."""
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=kv_transfer_config,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils_v2, "enable_sfa", lambda _: False)
    backend = SimpleNamespace(full_cls_name=lambda: "qwen-cache-test")
    layers = {
        name: SimpleNamespace(get_attn_backend=lambda: backend, num_heads=1)
        for group in config.kv_cache_groups
        for name in group.layer_names
    }
    monkeypatch.setattr(
        attn_utils_v2,
        "get_layers_from_vllm_config",
        lambda *_: layers,
    )
    raw = attn_utils_v2._allocate_kv_cache(config, {}, torch.device("cpu"))
    block_sizes = kernel_block_sizes or [128] * len(config.kv_cache_groups)
    if use_main:
        monkeypatch.setattr(attn_utils_v2, "_allocate_kv_cache", lambda *_, **__: raw)
        caches = attn_utils_v2.allocate_kv_cache_main(
            config,
            torch.device("cpu"),
            None,
            block_sizes,
        )
    else:
        groups = [
            SimpleNamespace(
                kv_cache_group_id=i,
                kv_cache_spec=group.kv_cache_spec,
                layer_names=group.layer_names,
                backend=backend,
            )
            for i, group in enumerate(config.kv_cache_groups)
        ]
        caches = attn_utils_v2._reshape_kv_cache_v2(
            groups,
            raw,
            "auto",
            block_sizes,
            {},
            config,
        )
    layout = build_qwen4_kv_cache_layout(
        config.kv_cache_groups,
        config.num_blocks,
    )
    return raw, caches, layout


@pytest.mark.parametrize("runner_kind", ["v2", "v2_main"])
@pytest.mark.parametrize("with_kv_transfer", [False, True])
def test_runner_materializes_packed_kv_and_contiguous_ssm_views(monkeypatch, runner_kind, with_kv_transfer) -> None:
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=3,
                prefix_cache_retention_interval=None,
            )
        ),
        _groups(),
        10**9,
    )
    assert config is not None
    raw, caches, layout = _allocate_mrv2_kv_cache(
        config,
        monkeypatch,
        use_main=runner_kind == "v2_main",
        kv_transfer_config=(
            KVTransferConfig(kv_connector="MooncakeConnector", kv_role="kv_both") if with_kv_transfer else None
        ),
    )
    # Every physical byte is represented once by the planner, including the
    # single PLE and hidden banks outside the shared slot backing.
    allocations = {t.untyped_storage().data_ptr(): t.numel() for t in raw.values()}
    assert sum(allocations.values()) == config.num_blocks * get_qwen4_exp_pool_bytes_per_block(config.kv_cache_groups)
    assert layout is not None

    shared_names = [owner.layer_name for owner in layout.owners if owner.role not in (HIDDEN, PLE)]
    backing = raw[shared_names[0]]
    assert isinstance(backing, torch.Tensor)
    assert all(raw[name] is backing for name in shared_names)
    if with_kv_transfer:
        assert all(t.data_ptr() % (2 * 1024 * 1024) == 0 for t in raw.values())
    assert raw["model.cache_only_layers.0"] is not backing

    expected_regions = {
        "model.layers.0.linear_attn": ("r1", "r2"),
        "model.layers.1.linear_attn": ("r1", "r2"),
    }
    for layer_name, region_names in expected_regions.items():
        owner = layout.owner(layer_name)
        states = caches[layer_name]
        assert isinstance(states, list)
        assert len(states) == len(region_names)
        for state, region_name in zip(
            states,
            region_names,
            strict=True,
        ):
            region = layout.region(region_name)
            assert state[0].is_contiguous()
            assert not state.is_contiguous()
            assert state.storage_offset() * state.element_size() - backing.storage_offset() == (
                owner.slot * layout.slot_backing_size + region.offset
            )
            assert state.stride(0) * state.element_size() == region.block_stride
            assert state.untyped_storage().data_ptr() == (backing.untyped_storage().data_ptr())
            assert state[0].numel() * state.element_size() <= region.page_size_bytes

    ple = caches["model.ple"][0]
    assert ple.is_contiguous()
    assert ple.shape == (3, 128, 5)
    assert ple.stride(0) == 128 * 5
    assert raw["model.ple"] is not backing
    assert raw["model.ple"].numel() == ple.numel() * ple.element_size()
    # Indexed updates address only selected request blocks in the private bank.
    indices = torch.tensor([0, 2])
    selected = ple.index_select(0, indices)
    selected.fill_(7)
    ple.index_copy_(0, indices, selected)
    assert torch.all(ple[indices] == 7)
    assert torch.count_nonzero(ple[1]) == 0
    assert torch.count_nonzero(backing) == 0

    k, v = caches["model.layers.11.self_attn.attn"]
    ssm = caches["model.layers.0.linear_attn"][1]
    assert k.shape == v.shape == (6, 128, 1, 64)
    assert k.stride() == v.stride() == (16384, 64, 64, 1)
    assert v.data_ptr() - k.data_ptr() == 16384
    assert k.data_ptr() == caches["model.layers.0.linear_attn"][0].data_ptr()
    assert ssm.data_ptr() - k.data_ptr() == 512
    k[4].fill_(5)
    v[4].fill_(9)
    torch.testing.assert_close(ssm[2].flatten()[:7936], torch.full((7936,), 5, dtype=ssm.dtype))
    torch.testing.assert_close(ssm[2].flatten()[7936:16128], torch.full((8192,), 9, dtype=ssm.dtype))
    assert torch.count_nonzero(ssm[:2]) == 0

    raw_qsa = caches["model.layers.11.self_attn.indexer.raw_key_cache"]
    assert raw_qsa.shape == (3, 8, 1, 32)
    assert raw_qsa.is_contiguous()

    compressed = caches["model.layers.11.self_attn.indexer.compressed_key_cache"]
    assert compressed.shape == (6, 32, 1, 32)
    assert compressed.is_contiguous()


def test_qsa_source_sets_must_match() -> None:
    groups = _groups()
    groups[1].layer_names[-1] = "model.layers.46.self_attn.indexer.raw_key_cache"
    try:
        build_qwen4_kv_cache_layout(groups, num_blocks=2)
    except ValueError as error:
        assert "one-to-one source-layer mapping" in str(error)
    else:
        raise AssertionError("mismatched QSA source owners were accepted")


def test_qsa_76_input_specs_restore_composite_block_table_lifetimes() -> None:
    specs = _ungrouped_qwen_specs()
    groups = [
        KVCacheGroupSpec(
            [name],
            spec,
            is_eagle_group=name.startswith("model.layers.51.self_attn"),
        )
        for name, spec in specs.items()
    ]
    owners = _prepare_qsa_composite_groups(specs)
    assert owners is not None
    merged = _merge_qsa_composite_groups(
        groups,
        specs,
        *owners,
    )

    assert len(merged) == 39
    composite = [group for group in merged if any(name.endswith(".attn") for name in group.layer_names)]
    raw = [group for group in merged if any(name.endswith(".indexer.raw_key_cache") for name in group.layer_names)]
    assert len(composite) == 1
    assert len(composite[0].layer_names) == 26
    assert sum(name.endswith(".attn") for name in composite[0].layer_names) == 13
    assert sum(name.endswith(".indexer.compressed_key_cache") for name in composite[0].layer_names) == 13
    first_composite_spec = next(iter(composite[0].kv_cache_spec.kv_cache_specs.values()))
    assert isinstance(first_composite_spec, FullAttentionSpec)
    assert not isinstance(first_composite_spec, MLAAttentionSpec)
    assert composite[0].is_eagle_group
    assert len(raw) == 1
    assert len(raw[0].layer_names) == 13
    assert raw[0].is_eagle_group


def test_qwen4_cache_admission_matches_shared_planner_allocation() -> None:
    specs = _ungrouped_qwen_specs()
    owners = _prepare_qsa_composite_groups(specs)
    assert owners is not None
    groups = _merge_qsa_composite_groups(
        [KVCacheGroupSpec([name], spec) for name, spec in specs.items()],
        specs,
        *owners,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=135168),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        max_in_flight_tokens=4096,
    )
    required_bytes = _ascend_max_memory_usage_bytes_from_groups(
        vllm_config,
        groups,
    )
    required_blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(vllm_config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    planned = get_qwen4_exp_kv_cache_config(
        vllm_config,
        groups,
        required_bytes,
    )
    assert planned is not None
    assert planned.num_blocks == required_blocks
    layout = build_qwen4_kv_cache_layout(
        groups,
        num_blocks=required_blocks,
    )
    assert layout is not None
    assert layout.slot_count == 13
    assert (
        layout.slot_count * layout.slot_backing_size
        + sum(owner.spec.real_page_size_bytes * required_blocks for owner in layout.owners if owner.role == PLE)
    ) == required_bytes


def test_mamba_state_groups_merge_shares_one_block_table(monkeypatch) -> None:
    """36 GDN layers use three block tables over 13 shared slots."""
    specs = _ungrouped_qwen_specs()
    owners = _prepare_qsa_composite_groups(specs)
    assert owners is not None
    groups = _merge_qsa_composite_groups(
        [KVCacheGroupSpec([name], spec) for name, spec in specs.items()],
        specs,
        *owners,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=135168),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
        ),
        max_in_flight_tokens=4096,
    )

    def is_gdn_group(group) -> bool:
        spec = group.kv_cache_spec
        members = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else [spec]
        return all(isinstance(member, MambaSpec) and len(member.shapes) == 2 for member in members)

    def required_blocks(group_list) -> int:
        return sum(
            (g.kv_cache_spec.max_memory_usage_bytes(vllm_config) + g.kv_cache_spec.page_size_bytes - 1)
            // g.kv_cache_spec.page_size_bytes
            for g in group_list
        )

    gdn_groups = [g for g in groups if is_gdn_group(g)]
    assert len(gdn_groups) == 36
    assert all(len(g.layer_names) == 1 for g in gdn_groups)
    gdn_blocks_each = required_blocks(gdn_groups[:1])

    blocks_before = required_blocks(groups)
    bytes_before = _ascend_max_memory_usage_bytes_from_groups(vllm_config, groups)

    merged = _merge_mamba_state_groups(groups, 13)

    gdn_merged = [g for g in merged if is_gdn_group(g)]
    assert [len(group.layer_names) for group in gdn_merged] == [13, 13, 10]
    # PLE (single-shape MambaSpec) keeps its own group.
    ple_groups = [g for g in merged if isinstance(g.kv_cache_spec, MambaSpec) and len(g.kv_cache_spec.shapes) == 1]
    assert len(ple_groups) == 1
    assert len(merged) == len(groups) - 33

    blocks_after = required_blocks(merged)
    assert blocks_before - blocks_after == 33 * gdn_blocks_each

    probe = build_qwen4_kv_cache_layout(merged, num_blocks=1)
    assert probe is not None
    assert probe.slot_count == 13
    pool_block_bytes = probe.slot_count * probe.slot_backing_size + sum(
        owner.spec.real_page_size_bytes for owner in probe.owners if owner.role == PLE
    )

    bytes_after = _ascend_max_memory_usage_bytes_from_groups(vllm_config, merged)
    assert bytes_after == blocks_after * pool_block_bytes
    assert bytes_before - bytes_after == 33 * gdn_blocks_each * pool_block_bytes

    # Every group reuses the same 13-slot backing, but its block table owns
    # distinct live IDs. Verify both the alias and the non-overlapping writes.
    layout = build_qwen4_kv_cache_layout(merged, num_blocks=3)
    assert layout is not None
    backing = torch.zeros(layout.slot_count * layout.slot_backing_size, dtype=torch.int8)
    views = []
    for group in gdn_merged:
        slots = [layout.owner(name).slot for name in group.layer_names]
        assert slots == list(range(len(group.layer_names)))
        owner = layout.owner(group.layer_names[0])
        region = layout.region("r2")
        views.append(
            make_slab_view(
                backing,
                storage_offset=owner.slot * layout.slot_backing_size + region.offset,
                num_blocks=3,
                item_shape=owner.spec.shapes[1],
                dtype=owner.spec.dtypes[1],
                block_stride=region.block_stride,
            )
        )
    assert len({view.data_ptr() for view in views}) == 1
    for block_id, view in enumerate(views):
        view[block_id].fill_(block_id + 1)
    for view in views:
        for block_id in range(3):
            torch.testing.assert_close(view[block_id], torch.full_like(view[block_id], block_id + 1))
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        merged,
        10**9,
    )
    assert config is not None
    gdn_names = {name for group in gdn_merged for name in group.layer_names}
    descriptors = [tensor for tensor in config.kv_cache_tensors if set(tensor.layers) & gdn_names]
    assert [len(tensor.layers) for tensor in descriptors] == [13, 13, 10]
    for tensor in descriptors:
        assert tensor.offset + (len(tensor.layers) - 1) * tensor.layer_stride + region.size <= tensor.size

    raw, caches, _ = _allocate_mrv2_kv_cache(config, monkeypatch)
    names = [group.layer_names[0] for group in gdn_merged]
    assert all(raw[name] is raw[names[0]] for name in names)
    assert raw[names[0]].numel() == 13 * layout.slot_backing_size
    states = [caches[name][1] for name in names]
    assert all(state[0].is_contiguous() and not state.is_contiguous() for state in states)
    for block_id, state in enumerate(states):
        state[block_id].fill_(block_id + 1)
    for state in states:
        for block_id in range(3):
            torch.testing.assert_close(state[block_id], torch.full_like(state[block_id], block_id + 1))


def test_state_group_packing_preserves_draft_and_state_lifetime_boundaries():
    spec = next(spec for name, spec in _ungrouped_qwen_specs().items() if name.endswith("linear_attn"))
    groups = [
        KVCacheGroupSpec(["target.0"], spec),
        KVCacheGroupSpec(["draft.0"], spec, is_eagle_group=True),
        KVCacheGroupSpec(["checkpoint.0"], replace(spec, num_prefill_checkpoint_blocks=1)),
        KVCacheGroupSpec(["target.1"], spec),
    ]
    packed = _merge_mamba_state_groups(groups, 13)
    assert [group.layer_names for group in packed] == [
        ["target.0", "target.1"],
        ["draft.0"],
        ["checkpoint.0"],
    ]
    assert [group.is_eagle_group for group in packed] == [False, True, False]
    assert _merge_mamba_state_groups(packed, 13) == packed


@pytest.mark.parametrize("cache_mode", ["align", "all"])
def test_qsa_block_alignment_reuses_vllm_for_whole_gdn_page(monkeypatch, cache_mode):
    # Actual TP4 GDN: 30KiB conv + 768KiB SSM, rounded to 896 token pairs.
    model_class = SimpleNamespace(
        get_mamba_specs_from_config=lambda config: [
            MambaSpec(block_size=128, shapes=((6, 2560), (12, 128, 128)), dtypes=(torch.bfloat16, torch.float32))
        ]
    )
    monkeypatch.setattr(ModelRegistry, "resolve_model_cls", lambda *args, **kwargs: (model_class, "test"))
    cache = SimpleNamespace(
        cache_dtype="auto",
        block_size=128,
        mamba_cache_mode=cache_mode,
        mamba_block_size=None,
        user_specified_mamba_block_size=False,
        mamba_page_size_padded=None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=False,
            hf_text_config=SimpleNamespace(indexer_compress_ratio=4),
            architecture="Qwen4ExpForConditionalGeneration",
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda config: 2,
            get_head_size=lambda: 128,
            get_mamba_chunk_size=lambda: 64,
        ),
        cache_config=cache,
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    from vllm_ascend.models.qwen4_exp.qsa import AscendQSABackend

    monkeypatch.setattr(
        NPUPlatform,
        "_find_non_ssm_backends",
        classmethod(lambda cls, config: [AscendQSABackend]),
    )
    NPUPlatform._align_hybrid_block_size(config, AscendQSABackend)
    assert cache.block_size == 896
    expected_mamba_block_size = 896 if cache_mode == "align" or vllm_version_is("0.31.0") else None
    assert cache.mamba_block_size == expected_mamba_block_size
    assert cache.mamba_page_size_padded == 896 * 1024


def test_large_ple_uses_one_dense_bank_without_expanding_shared_slots(monkeypatch) -> None:
    groups = _groups()
    groups[4] = KVCacheGroupSpec(
        ["model.ple"], replace(groups[4].kv_cache_spec, shapes=((20001,),), page_size_padded=65536)
    )
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        groups,
        10**9,
    )
    assert config is not None
    raw, caches, layout = _allocate_mrv2_kv_cache(config, monkeypatch)
    assert layout.region("r2").page_size_bytes == 32768
    ple = caches["model.ple"][0]
    assert ple.is_contiguous()
    assert ple.shape == (3, 20001)
    assert raw["model.ple"].numel() == 3 * 20001 * 2
    allocations = {tensor.data_ptr(): tensor.numel() for tensor in raw.values()}
    assert sum(allocations.values()) == 3 * get_qwen4_exp_pool_bytes_per_block(groups)
    keys = caches["model.layers.11.self_attn.attn"][0]
    keys.fill_(7)
    ids = torch.tensor([0, 2])
    selected = ple.index_select(0, ids)
    selected.fill_(3)
    ple.index_copy_(0, ids, selected)
    assert torch.all(keys == 7)
    assert torch.all(ple[ids] == 3)
    assert torch.count_nonzero(ple[1]) == 0


def test_public_qwen4_grouping_and_pool_charge_match_physical_backing() -> None:
    specs = _ungrouped_qwen_specs()
    original_groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    calls = []

    def upstream(config, input_specs):
        calls.append((config, input_specs))
        return original_groups

    config = object()
    groups = get_qwen4_exp_kv_cache_groups(config, specs, upstream)
    assert calls == [(config, specs)]
    assert groups is not None
    assert len(groups) == 6
    assert sorted(name for group in groups for name in group.layer_names) == sorted(specs)
    layout = build_qwen4_kv_cache_layout(groups, num_blocks=7)
    assert layout is not None
    assert layout.slot_count == 13
    assert get_qwen4_exp_pool_bytes_per_block(groups) * 7 == (
        layout.slot_count * layout.slot_backing_size
        + 7 * next(owner.spec.real_page_size_bytes for owner in layout.owners if owner.role == PLE)
    )


def test_platform_builder_uses_qwen_pool_accounting() -> None:
    pytest.importorskip("vllm.v1.core.kv_cache_planning")
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

    groups = _groups()
    expected = get_qwen4_exp_pool_bytes_per_block(groups)
    assert expected is not None
    assert AscendKVCacheConfigBuilder()._get_pool_bytes_per_block(SimpleNamespace(), groups) == expected


@pytest.mark.parametrize("manager_block_size", [256, 384, 768])
def test_compressed_qsa_builder_and_cache_use_matching_pages(monkeypatch, manager_block_size):
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.utils import AttentionGroup

    from vllm_ascend.models.qwen4_exp import qsa

    if vllm_version_is("0.32.0"):
        import vllm.models.qwen4_exp.common.qsa_cache as qsa_module  # type: ignore[import-not-found]

        QSARawKeyCache = qsa_module.QSAKeyStateCache
        QSAIndexerBackend = qsa_module.QSAStateBackend
    else:
        import vllm.models.qwen4_exp.common.qsa_indexer as qsa_module  # type: ignore[import-not-found]

        QSARawKeyCache = qsa_module.QSARawKeyCache
        QSAIndexerBackend = qsa_module.QSAIndexerBackend

    groups = [
        KVCacheGroupSpec(
            group.layer_names,
            group.kv_cache_spec
            if isinstance(group.kv_cache_spec, CircularBufferSpec)
            else replace(group.kv_cache_spec, block_size=manager_block_size),
        )
        for group in _groups()
    ]
    config = get_qwen4_exp_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        groups,
        10**9,
    )
    assert config is not None
    runtime_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=manager_block_size),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
            max_model_len=4096,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=2),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        num_speculative_tokens=0,
        use_v2_model_runner=True,
    )
    compressed_group = groups[2]
    spec = compressed_group.kv_cache_spec
    backend = QSAIndexerBackend
    monkeypatch.setattr(attn_utils_v2, "get_current_vllm_config", lambda: runtime_config)
    monkeypatch.setattr(attn_utils_v2, "enable_sfa", lambda _: False)
    raw = attn_utils_v2._allocate_kv_cache(config, {}, torch.device("cpu"))
    attn_groups = [AttentionGroup(backend, g.layer_names, g.kv_cache_spec, i) for i, g in enumerate(groups)]
    caches = attn_utils_v2._reshape_kv_cache_v2(attn_groups, raw, "auto", [128] * len(groups), {}, config)
    # Exercise the actual MRV2 factory, which rewrites the builder spec.
    group = attn_groups[2]
    group.create_metadata_builders(runtime_config, torch.device("cpu"), kernel_block_size=128)
    builder = group.get_metadata_builder(0)
    name = compressed_group.layer_names[0]
    layer = qsa.qsa_indexer.QSACompressedKeyCache(
        head_size=spec.head_size,
        dtype=spec.dtype,
        prefix=name,
        cache_config=runtime_config.cache_config,
        compress_ratio=4,
        vllm_config=runtime_config,
    )
    from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache_to_layers

    bind_kv_cache_to_layers({name: caches[name]}, {name: layer})
    assert layer.kv_cache is caches[name]
    raw_name = groups[1].layer_names[0]
    raw_layer = QSARawKeyCache(
        head_size=20,
        dtype=spec.dtype,
        prefix=raw_name,
        cache_config=runtime_config.cache_config,
        compress_ratio=4,
        cache_rope_positions=True,
        vllm_config=runtime_config,
    )
    bind_kv_cache_to_layers({raw_name: caches[raw_name]}, {raw_name: raw_layer})
    assert raw_layer.key_cache.shape == (3, 8, 1, 20)
    assert raw_layer.rope_position_cache.shape == (3, 8, 1, 3)
    assert raw_layer.key_cache.untyped_storage().data_ptr() == raw[raw_name].untyped_storage().data_ptr()
    positions = torch.tensor([2**40, 2**40 + 1, 2**40 + 2], dtype=torch.int64)
    raw_layer.rope_position_cache[2, 7, 0] = positions
    torch.testing.assert_close(raw_layer.rope_position_cache[2, 7, 0], positions)
    keys_per_page = 128 // 4
    factor = manager_block_size // 128
    assert layer.kv_cache.shape == (3 * factor, keys_per_page, 1, spec.head_size)
    assert builder.storage_block_size == keys_per_page
    assert layer.kv_cache.untyped_storage().data_ptr() == raw[name].untyped_storage().data_ptr()
    manager = torch.tensor([[2, 0]], dtype=torch.int32)
    virtual = (manager.unsqueeze(-1) * factor + torch.arange(factor)).flatten(1).to(torch.int32)
    locs = torch.tensor([0, 8], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=locs,
        query_start_loc_cpu=locs,
        seq_lens=torch.tensor([manager_block_size + 4], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=8,
        max_query_len=8,
        max_seq_len=manager_block_size + 4,
        block_table_tensor=virtual,
        slot_mapping=torch.zeros(8, dtype=torch.int64),
    )
    # One real request owns all eight tokens; avoid the device-only mapping kernel.
    common._token_to_req_indices_cache = torch.zeros(8, dtype=torch.int32)
    monkeypatch.setattr(qsa.qsa_indexer, "build_qsa_metadata", qsa.qsa_indexer._build_qsa_metadata_torch)
    metadata = builder.build(0, common)
    torch.testing.assert_close(metadata.block_table, virtual)
    # Cross the manager-block boundary: flattened storage addresses must agree
    # for metadata writes and PA_BSND indexer reads in both runners.
    for token_row in (3, 7):
        position = manager_block_size - 4 + token_row
        logical_group = position // 4
        page_column, row = divmod(logical_group, keys_per_page)
        block = int(metadata.block_table[0, page_column])
        expected_slot = 2 * (manager_block_size // 4) + logical_group if token_row == 3 else 0
        assert int(metadata.slot_mapping[token_row]) == expected_slot
        selected = layer.kv_cache[block, row]
        expected = layer.kv_cache.view(-1, 1, spec.head_size)[expected_slot]
        assert selected.data_ptr() == expected.data_ptr()
        selected.fill_(token_row + 1)
        torch.testing.assert_close(expected, torch.full_like(expected, token_row + 1))


def test_whole_page_sharing_uses_one_tp4_page_for_qsa_or_complete_gdn():
    groups = _groups()
    groups[0] = KVCacheGroupSpec(
        groups[0].layer_names,
        replace(groups[0].kv_cache_spec, block_size=896, num_kv_heads=2, head_size=128, head_size_v=128),
    )
    groups[2] = KVCacheGroupSpec(groups[2].layer_names, replace(groups[2].kv_cache_spec, block_size=896))
    groups[3] = KVCacheGroupSpec(
        groups[3].layer_names,
        replace(
            groups[3].kv_cache_spec,
            block_size=896,
            shapes=((6, 2560), (12, 128, 128)),
            dtypes=(torch.bfloat16, torch.float32),
        ),
    )
    layout = build_qwen4_kv_cache_layout(groups, 3)
    whole, conv, ssm = (layout.region(name) for name in ("r0", "r1", "r2"))
    assert whole.offset == conv.offset == 0
    assert ssm.offset == 30 * 1024
    assert whole.page_size_bytes == whole.block_stride == conv.block_stride == ssm.block_stride == 896 * 1024
    assert ssm.page_size_bytes == 768 * 1024
    assert layout.region("r4").offset == 3 * whole.block_stride
    backing = torch.zeros(layout.slot_backing_size, dtype=torch.int8)
    qsa = make_slab_view(
        backing,
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(896, 2, 256),
        storage_offset=0,
        block_stride=whole.block_stride,
    )
    gdn_conv = make_slab_view(
        backing,
        dtype=torch.bfloat16,
        num_blocks=3,
        item_shape=(6, 2560),
        storage_offset=conv.offset,
        block_stride=conv.block_stride,
    )
    gdn_ssm = make_slab_view(
        backing,
        dtype=torch.float32,
        num_blocks=3,
        item_shape=(12, 128, 128),
        storage_offset=ssm.offset,
        block_stride=ssm.block_stride,
    )
    # Different groups have disjoint live pool IDs; after recycling a QSA
    # page, GDN initialization must overwrite both components of that page.
    qsa[1].fill_(7)
    assert torch.all(gdn_conv[1] == 7)
    assert torch.count_nonzero(gdn_ssm[1]) == gdn_ssm[1].numel()
    gdn_conv[1].zero_()
    gdn_ssm[1].zero_()
    assert torch.count_nonzero(qsa[1].flatten()[: 798 * 1024 // 2]) == 0
    assert torch.count_nonzero(qsa[[0, 2]]) == 0
