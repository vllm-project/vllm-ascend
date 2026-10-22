from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.six_region_kv_cache_layout import (
    ATTENTION,
    GDN,
    HIDDEN,
    INDEPENDENT_CACHE_ROLES,
    PLE,
    build_six_region_kv_cache_layout,
    make_contiguous_slab_view,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_max_memory_usage_bytes_from_groups,
    _get_ascend_kv_cache_groups,
    _get_qwen4_exp_six_region_kv_cache_config,
    _merge_mamba_state_groups,
    _merge_qsa_composite_groups,
    _prepare_qsa_composite_groups,
)
from vllm_ascend.worker import model_runner_v1
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.v2 import attn_utils


def _groups() -> list[KVCacheGroupSpec]:
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
        block_size=256,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
    )
    ple = MambaSpec(
        block_size=256,
        shapes=((128, 5),),
        dtypes=(torch.bfloat16,),
        tp_replicated=True,
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
        block_size=256,
        shapes=((64, 4), (128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    ple = MambaSpec(
        block_size=256,
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


def _groups_with_dflash(padding: int = 0) -> list[KVCacheGroupSpec]:
    groups = _groups()
    for layer_idx, spec in enumerate(
        (
            FullAttentionSpec(block_size=64, num_kv_heads=2, head_size=32, dtype=torch.bfloat16),
            SlidingWindowSpec(block_size=64, num_kv_heads=2, head_size=32, dtype=torch.bfloat16, sliding_window=128),
        )
    ):
        groups.append(
            KVCacheGroupSpec(
                [f"draft.model.layers.{layer_idx}.self_attn.attn"],
                replace(spec, page_size_padded=spec.page_size_bytes + padding) if padding else spec,
                is_eagle_group=True,
            )
        )
    return groups


@pytest.mark.parametrize("padding", [0, 512])
def test_dflash_attention_keeps_qsa_and_mtp_six_region_geometry(padding: int) -> None:
    baseline = build_six_region_kv_cache_layout(_groups(), num_blocks=3)
    layout = build_six_region_kv_cache_layout(_groups_with_dflash(padding), num_blocks=3)
    assert baseline is not None and layout is not None
    assert layout.regions == baseline.regions
    assert layout.slot_count == baseline.slot_count
    assert layout.slot_backing_size == baseline.slot_backing_size
    assert layout.owner("model.mtp.layers.48.self_attn.attn").role == "qsa_main"
    for layer_idx in range(2):
        name = f"draft.model.layers.{layer_idx}.self_attn.attn"
        assert layout.owner(name).role == ATTENTION
        assert name not in layout.slot_shared_by(layer_idx)


@pytest.mark.parametrize("padding", [0, 512])
def test_dflash_planner_budget_and_admission_include_independent_attention(padding: int) -> None:
    groups = _groups_with_dflash(padding)
    probe = build_six_region_kv_cache_layout(groups, num_blocks=1)
    assert probe is not None
    independent = [owner for owner in probe.owners if owner.role in INDEPENDENT_CACHE_ROLES]
    bytes_per_block = probe.slot_count * probe.slot_backing_size + sum(
        owner.spec.page_size_bytes for owner in independent
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=768),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align", num_gpu_blocks_override=None, prefix_cache_retention_interval=None
        ),
        max_in_flight_tokens=16,
    )
    planned = _get_qwen4_exp_six_region_kv_cache_config(config, groups, bytes_per_block * 3)
    assert planned is not None and planned.num_blocks == 3
    independent_descriptors = {descriptor.layers[0]: descriptor for descriptor in planned.kv_cache_tensors[5:]}
    for owner in independent:
        descriptor = independent_descriptors[owner.layer_name]
        assert descriptor.size == owner.spec.page_size_bytes * planned.num_blocks
        assert descriptor.block_stride == owner.spec.page_size_bytes
        assert descriptor.offset == 0
    required_blocks = sum(
        (group.kv_cache_spec.max_memory_usage_bytes(config) + group.kv_cache_spec.page_size_bytes - 1)
        // group.kv_cache_spec.page_size_bytes
        for group in groups
    )
    assert _ascend_max_memory_usage_bytes_from_groups(config, groups) == bytes_per_block * required_blocks


@pytest.mark.parametrize("block_drop", [False, True])
def test_dflash_groups_only_mark_ordinary_draft_attention(monkeypatch, block_drop: bool) -> None:
    from vllm_ascend.patch.platform import patch_kv_cache_utils

    specs = {
        name: spec
        for group in _groups_with_dflash()
        for name, spec in (
            group.kv_cache_spec.kv_cache_specs.items()
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else ((name, group.kv_cache_spec) for name in group.layer_names)
        )
    }
    original_grouping = Mock(
        side_effect=lambda _config, members: [KVCacheGroupSpec([n], s) for n, s in members.items()]
    )
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_groups", original_grouping)
    groups = _get_ascend_kv_cache_groups(
        SimpleNamespace(speculative_config=SimpleNamespace(method="dflash", use_eagle_block_drop=lambda: block_drop)),
        specs,
    )
    draft_groups = [group for group in groups if any(name.startswith("draft.") for name in group.layer_names)]
    assert len(draft_groups) == 2
    assert all(group.is_eagle_group == block_drop for group in draft_groups)
    assert all(not group.is_eagle_group for group in groups if group not in draft_groups)
    assert len(original_grouping.call_args_list) == 2
    assert all(name.startswith("draft.") for name in original_grouping.call_args_list[1].args[1])


@pytest.mark.parametrize("runner_version", [1, 2])
@pytest.mark.parametrize("padding", [0, 512])
def test_dflash_independent_cache_writes_preserve_target_and_other_requests(
    monkeypatch, runner_version, padding
) -> None:
    config = _get_qwen4_exp_six_region_kv_cache_config(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)),
        _groups_with_dflash(padding),
        10**9,
    )
    assert config is not None
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(model_type="qwen4_exp")
        ),
    )
    backend = SimpleNamespace(get_kv_cache_shape=lambda n, b, h, d, *args: (2, n, b, h, d))
    attn_groups = [
        SimpleNamespace(
            backend=backend,
            kv_cache_group_id=gid,
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
        )
        for gid, group in enumerate(config.kv_cache_groups)
    ]
    if runner_version == 1:
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.ascend_config = SimpleNamespace(
            kvpp_config=SimpleNamespace(size=1), xlite_graph_config=SimpleNamespace(enabled=True)
        )
        runner.vllm_config = vllm_config
        runner.model_config = vllm_config.model_config
        runner.use_dcp = False
        runner.use_sparse = runner.use_compress = runner.use_hybrid_blocks = False
        runner.runner_only_attn_layers = set()
        runner.sparse_kv_offload_enabled = False
        runner._kv_cache_spec_attn_group_iterator = lambda: attn_groups
        raw = runner._allocate_kv_cache_tensors(config)
        caches = runner._reshape_kv_cache_tensors(config, raw, kernel_block_sizes=[16] * len(attn_groups))
    else:
        monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
        monkeypatch.setattr(
            attn_utils, "KVPPConfig", SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1))
        )
        monkeypatch.setattr(attn_utils, "is_c8_mxfp_kv_quant", lambda _: False)
        monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
        raw = attn_utils._allocate_kv_cache(config, {}, torch.device("cpu"))
        caches = attn_utils._reshape_kv_cache_v2(attn_groups, raw, "auto", [16] * len(attn_groups), {}, config)

    qsa_name = "model.layers.11.self_attn.attn"
    draft_names = [f"draft.model.layers.{layer_idx}.self_attn.attn" for layer_idx in range(2)]
    assert raw[draft_names[0]].data_ptr() != raw[qsa_name].data_ptr()
    assert raw[draft_names[1]].data_ptr() != raw[draft_names[0]].data_ptr()
    for name in draft_names:
        assert torch.count_nonzero(raw[name]).item() == 0
        key, value = caches[name]
        assert key.shape == value.shape == (12, 16, 2, 32)
        # Physical scheduler page 1 contains four kernel blocks. Context and
        # query writes must leave another request's pages and the target intact.
        key[4, 0].fill_(3)
        value[7, 15].fill_(5)
        assert torch.all(key[4, 0] == 3)
        assert torch.all(value[4, 0] == 0)
        assert torch.all(value[7, 15] == 5)
        assert torch.all(key[7, 15] == 0)
        assert torch.count_nonzero(key[:4]).item() == 0
        assert torch.count_nonzero(value[:4]).item() == 0
        assert torch.count_nonzero(key[8:]).item() == 0
        assert torch.count_nonzero(value[8:]).item() == 0
        assert torch.count_nonzero(raw[qsa_name]).item() == 0


def test_six_region_offsets_are_contiguous_non_overlapping_slabs() -> None:
    layout = build_six_region_kv_cache_layout(_groups(), num_blocks=3)
    assert layout is not None
    assert layout.slot_count == 3
    assert [(region.name, region.offset, region.page_size_bytes, region.size) for region in layout.regions] == [
        ("r1", 0, 512, 1536),
        ("r2", 1536, 32768, 98304),
        ("r3", 99840, 32768, 98304),
        ("r4", 198144, 512, 1536),
        ("r5", 199680, 4096, 12288),
        ("r6", 211968, 1280, 3840),
    ]
    assert layout.slot_backing_size == 215808
    for left, right in zip(layout.regions, layout.regions[1:]):
        assert right.offset >= left.end
        assert right.offset % layout.alignment == 0

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


def test_six_region_views_are_contiguous_and_advanced_index_is_bounded() -> None:
    num_blocks = 3
    layout = build_six_region_kv_cache_layout(
        _groups(),
        num_blocks=num_blocks,
    )
    assert layout is not None
    backing = torch.zeros(layout.slot_backing_size, dtype=torch.int8)

    def view(region_name: str, shape: tuple[int, ...]) -> torch.Tensor:
        return make_contiguous_slab_view(
            backing,
            dtype=torch.bfloat16,
            num_blocks=num_blocks,
            item_shape=shape,
            storage_offset=layout.region(region_name).offset,
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
    region_for_view = {
        "k": "r2",
        "v": "r3",
        "raw": "r4",
        "compressed": "r5",
        "gdn_conv": "r1",
        "gdn_ssm": "r2",
        "ple": "r6",
    }
    storage_ptr = backing.untyped_storage().data_ptr()
    for name, slab in views.items():
        region = layout.region(region_for_view[name])
        storage_span = (
            sum(
                (size - 1) * stride
                for size, stride in zip(
                    slab.shape,
                    slab.stride(),
                    strict=True,
                )
            )
            + 1
        ) * slab.element_size()
        assert slab.untyped_storage().data_ptr() == storage_ptr
        assert slab.is_contiguous()
        assert slab.stride() == torch.empty_like(slab).stride()
        assert slab.storage_offset() * slab.element_size() == region.offset
        assert slab.stride(0) * slab.element_size() == region.page_size_bytes
        assert storage_span == slab.numel() * slab.element_size()
        assert storage_span == region.size
        assert region.offset + storage_span <= region.end

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
    config = _get_qwen4_exp_six_region_kv_cache_config(
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
    layout = build_six_region_kv_cache_layout(
        config.kv_cache_groups,
        num_blocks=3,
    )
    assert layout is not None
    role_tensors = config.kv_cache_tensors[:5]
    hidden_tensor = config.kv_cache_tensors[5]
    expected_backing_size = layout.slot_count * layout.slot_backing_size
    assert all(tensor.size == expected_backing_size for tensor in role_tensors)
    assert [tensor.offset for tensor in role_tensors] == [
        layout.region(name).offset for name in ("r2", "r4", "r5", "r1", "r6")
    ]
    assert all(tensor.layer_stride == layout.slot_backing_size for tensor in role_tensors)
    assert hidden_tensor.layers == ["model.cache_only_layers.0"]
    assert hidden_tensor.size == 98304


@pytest.mark.parametrize("kv_transfer_config", [None, SimpleNamespace(kv_connector="MooncakeConnectorV1")])
def test_six_region_allocation_requires_no_kv_transfer(monkeypatch, kv_transfer_config) -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(kvpp_config=SimpleNamespace(size=1))
    runner.vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    runner.use_sparse = False
    runner.use_compress = False
    runner._get_layer_kv_cache_specs = Mock(return_value={})
    runner._uses_page_strided_shared_backing = Mock(return_value=False)
    runner._allocate_int8_cache_tensor = Mock()
    layout = SimpleNamespace(slot_count=1, slot_backing_size=16, owners=[])
    monkeypatch.setattr(model_runner_v1, "build_six_region_kv_cache_layout", lambda *args: layout)
    monkeypatch.setattr(model_runner_v1, "vllm_version_is", lambda version: False)
    monkeypatch.setattr(model_runner_v1, "get_layerwise_reuse_config", lambda config: None)
    config = SimpleNamespace(num_blocks=1, kv_cache_groups=[], kv_cache_tensors=[])

    assert runner._allocate_kv_cache_tensors(config) == {}

    if kv_transfer_config is None:
        assert runner._six_region_kv_cache_layout is layout
        runner._allocate_int8_cache_tensor.assert_called_once_with(16, 2 * 1024 * 1024)
    else:
        assert runner._six_region_kv_cache_layout is None
        runner._allocate_int8_cache_tensor.assert_not_called()
    runner._uses_page_strided_shared_backing.assert_not_called()


def test_runner_materializes_contiguous_six_region_views() -> None:
    config = _get_qwen4_exp_six_region_kv_cache_config(
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
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.ascend_config = SimpleNamespace(
        kvpp_config=SimpleNamespace(size=1),
    )
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        cache_config=SimpleNamespace(),
    )
    runner.use_sparse = False
    runner.use_compress = False
    runner.use_hybrid_blocks = False
    runner.runner_only_attn_layers = set()
    runner.sparse_kv_offload_enabled = False
    runner._kv_cache_spec_attn_group_iterator = lambda: [
        SimpleNamespace(
            backend=SimpleNamespace(),
            kv_cache_spec=group.kv_cache_spec,
            layer_names=group.layer_names,
        )
        for group in config.kv_cache_groups
    ]

    raw = runner._allocate_kv_cache_tensors(config)
    caches = runner._reshape_kv_cache_tensors(config, raw)
    layout = runner._six_region_kv_cache_layout
    assert layout is not None

    shared_names = [owner.layer_name for owner in layout.owners if owner.role != HIDDEN]
    backing = raw[shared_names[0]]
    assert isinstance(backing, torch.Tensor)
    assert all(raw[name] is backing for name in shared_names)
    assert raw["model.cache_only_layers.0"] is not backing

    expected_regions = {
        "model.layers.0.linear_attn": ("r1", "r2"),
        "model.layers.1.linear_attn": ("r1", "r2"),
        "model.ple": ("r6",),
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
            assert state.is_contiguous()
            assert state.storage_offset() * state.element_size() == (
                owner.slot * layout.slot_backing_size + region.offset
            )
            assert state.stride() == torch.empty_like(state).stride()
            assert state.untyped_storage().data_ptr() == (backing.untyped_storage().data_ptr())
            assert state.numel() * state.element_size() == region.size

    raw_qsa = caches["model.layers.11.self_attn.indexer.raw_key_cache"]
    assert raw_qsa.shape == (3, 1, 8, 32)
    assert raw_qsa.is_contiguous()

    compressed = caches["model.layers.11.self_attn.indexer.compressed_key_cache"]
    assert compressed.shape == (3, 1, 64, 32)
    assert compressed.is_contiguous()


def test_qsa_source_sets_must_match() -> None:
    groups = _groups()
    groups[1].layer_names[-1] = "model.layers.46.self_attn.indexer.raw_key_cache"
    try:
        build_six_region_kv_cache_layout(groups, num_blocks=2)
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


def test_six_region_admission_matches_shared_planner_allocation() -> None:
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
    planned = _get_qwen4_exp_six_region_kv_cache_config(
        vllm_config,
        groups,
        required_bytes,
    )
    assert planned is not None
    assert planned.num_blocks == required_blocks
    layout = build_six_region_kv_cache_layout(
        groups,
        num_blocks=required_blocks,
    )
    assert layout is not None
    assert layout.slot_count == 36
    assert layout.slot_count * layout.slot_backing_size == required_bytes


def test_mamba_state_groups_merge_shares_one_block_table() -> None:
    """36 per-layer GDN groups must merge into one group sharing a block table.

    Regression for the on-machine 910B2 finding: with per-layer GDN groups the
    admission formula charges each layer's state blocks at the full 36-slot
    pool-block price, inflating the minimum KV memory by tens of times and
    preventing TP4 startup on 64GB cards.
    """
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

    merged = _merge_mamba_state_groups(groups)

    gdn_merged = [g for g in merged if is_gdn_group(g)]
    assert len(gdn_merged) == 1
    assert len(gdn_merged[0].layer_names) == 36
    # PLE (single-shape MambaSpec) keeps its own group.
    ple_groups = [g for g in merged if isinstance(g.kv_cache_spec, MambaSpec) and len(g.kv_cache_spec.shapes) == 1]
    assert len(ple_groups) == 1
    assert len(merged) == len(groups) - 35

    blocks_after = required_blocks(merged)
    assert blocks_before - blocks_after == 35 * gdn_blocks_each

    probe = build_six_region_kv_cache_layout(merged, num_blocks=1)
    assert probe is not None
    assert probe.slot_count == 36
    pool_block_bytes = probe.slot_count * probe.slot_backing_size

    bytes_after = _ascend_max_memory_usage_bytes_from_groups(vllm_config, merged)
    assert bytes_after == blocks_after * pool_block_bytes
    assert bytes_before - bytes_after == 35 * gdn_blocks_each * pool_block_bytes
