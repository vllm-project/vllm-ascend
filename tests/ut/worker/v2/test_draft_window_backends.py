# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for consumer identity and block-table units."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import (
    DraftWindowBackendTypes,
    DraftWindowProviderStatus,
    DraftWindowTopology,
    inventory_draft_window_groups,
    project_draft_window_start,
    require_compatible_direct_prefix_routes,
    require_draft_window_providers,
    select_native_swa_subset_routes,
)


class DenseBackend:
    pass


class DenseBuilder:
    sliding_window: int | None
    sinks: object | None
    use_bnsd_kv_cache: bool


class DenseSubclass(DenseBuilder):
    pass


class CompressedBuilder:
    pass


class FullSpec:
    sliding_window: int | None

    def __init__(self, block_size=256, tokens_per_state=1):
        self.block_size = block_size
        self.tokens_per_state = tokens_per_state


class SlidingSpec(FullSpec):
    pass


class UniformSpec:
    def __init__(self, specs):
        self.kv_cache_specs = specs


@pytest.fixture
def types():
    return DraftWindowBackendTypes(
        dense_backend=DenseBackend,
        dense_builder=DenseBuilder,
        full_attention_spec=FullSpec,
        uniform_spec=UniformSpec,
        builder_families=((CompressedBuilder, "DSA"), (DenseBuilder, "GQA")),
        storage_block_size=lambda spec: spec.block_size // spec.tokens_per_state,
        compression_ratio=lambda spec: spec.tokens_per_state,
    )


@pytest.fixture
def topology():
    return DraftWindowTopology(1, 1, 1, 1, 1, 0, 1, False, False, None, None, False)


def inventory(types, topology, *, specs=None, builders=None, spans=None, full_retention_dflash2=False):
    specs = [FullSpec()] if specs is None else specs
    builders = [DenseBuilder() for _ in specs] if builders is None else builders
    spans = [128 for _ in specs] if spans is None else spans
    groups = [
        [
            SimpleNamespace(
                backend=DenseBackend,
                layer_names=list(specs[gid].kv_cache_specs)
                if isinstance(specs[gid], UniformSpec)
                else [f"draft.layers.{gid}"],
                get_metadata_builder=lambda index, builder=builder: builder,
            )
        ]
        if builder is not None
        else []
        for gid, builder in enumerate(builders)
    ]
    return inventory_draft_window_groups(
        attn_groups=groups,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for spec in specs]),
        block_tables=SimpleNamespace(kernel_block_sizes=spans),
        topology=topology,
        types=types,
        full_retention_dflash2=full_retention_dflash2,
    )


def test_dflash2_reference_full_allocation_keeps_native_read_provider(types, topology):
    spec = FullSpec()
    spec.sliding_window = 2048
    kinds = replace(types, sliding_window_spec=SlidingSpec)
    (original,) = inventory(kinds, topology, specs=[spec])
    assert original.status is DraftWindowProviderStatus.DIRECT_DENSE
    (reference,) = inventory(kinds, topology, specs=[spec], full_retention_dflash2=True)
    assert reference.status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
    assert reference.native_window == 2048


def test_kernel_column_span_is_not_scheduler_block_size(types, topology):
    (route,) = inventory(types, topology)
    assert route.scheduler_block_size == 256
    assert route.table_column_span == route.physical_block_size == 128
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    require_draft_window_providers((route,))


def test_same_named_unregistered_builder_cannot_gain_dense_capability(types, topology):
    # Strings do not certify an implementation, even when names match exactly.
    impostor = type(DenseBuilder.__name__, (), {})()
    (route,) = inventory(types, topology, builders=[impostor])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING
    with pytest.raises(NotImplementedError, match="provider missing"):
        require_draft_window_providers((route,))


def test_subclass_and_native_window_require_their_own_provider(types, topology):
    routes = inventory(types, topology, specs=[FullSpec(), SlidingSpec()], builders=[DenseSubclass(), DenseBuilder()])
    assert all(route.status is DraftWindowProviderStatus.PROVIDER_MISSING for route in routes)


def native_swa_bundle(types, topology):
    spec = SlidingSpec()
    spec.sliding_window = 4096
    routes = inventory(replace(types, sliding_window_spec=SlidingSpec), topology, specs=[spec, FullSpec()])
    impl = DenseBuilder()
    impl.sliding_window = 4096
    impl.sinks = None
    impl.use_bnsd_kv_cache = False
    context = {"draft.layers.0": SimpleNamespace(impl=impl)}
    return routes, impl, context


def test_native_swa_candidate_is_not_a_residency_or_read_provider(types, topology):
    routes, _, _ = native_swa_bundle(types, topology)
    assert routes[0].status is DraftWindowProviderStatus.NATIVE_SWA_CANDIDATE
    with pytest.raises(NotImplementedError, match="provider missing"):
        require_draft_window_providers(routes)


def test_covered_swa_keeps_only_full_group_projection(types, topology):
    routes, _, context = native_swa_bundle(types, topology)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=32768,
        max_query_tokens=2048,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert selected == (routes[1],)
    require_compatible_direct_prefix_routes(selected)
    assert context["draft.layers.0"].impl.sliding_window == 4096


def test_smaller_extra_window_selects_native_intersection(types, topology):
    routes, _, context = native_swa_bundle(types, topology)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 2
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[1] == routes[1]
    require_compatible_direct_prefix_routes(selected)


@pytest.mark.parametrize("window", [1024, 32768])
def test_native_only_dflash2_keeps_a_typed_builder_without_a_full_group(types, topology, window):
    routes, _, context = native_swa_bundle(types, topology)
    selected = select_native_swa_subset_routes(
        routes[:1],
        window_size=window,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 1
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[0].native_window == 4096
    require_compatible_direct_prefix_routes(selected)


@pytest.mark.parametrize("allowed,window", [(True, 1024), (False, 1024), (True, 4096)])
def test_noncausal_native_only_dflash2_needs_explicit_resident_certificate(types, topology, allowed, window):
    routes, _, context = native_swa_bundle(types, topology)
    options = dict(
        window_size=window,
        max_query_tokens=512,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
        group_causal={0: False},
        noncausal_dflash2=allowed,
        query_width=8,
    )
    if allowed and window == 1024:
        (route,) = select_native_swa_subset_routes(routes[:1], **options)
        assert route.status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    else:
        with pytest.raises(NotImplementedError, match="non-causal|resident"):
            select_native_swa_subset_routes(routes[:1], **options)


def test_unmasked_rebased_read_covers_future_queries_without_fabricating_a_causal_band():
    for end in (520, 1151, 1152, 4095, 4096):
        span, window, width, native = 128, 1024, 8, 2048
        assert window + span - 1 + width <= native
        start = max(end - window, 0) // span * span
        assert start >= max(end - native + width, 0)
        # sparse_mode=0 keeps every valid key through E, including query
        # positions to the right of the first query; rebasing changes no mask.
        actual = {start + key for key in range(end - start)}
        assert actual == set(range(start, end))
        assert end - 1 in actual


@pytest.mark.parametrize("change", [{"sliding_window": 1024}, {"sinks": object()}, {"use_bnsd_kv_cache": True}])
def test_native_swa_actual_consumer_must_match_certificate(types, topology, change):
    routes, impl, context = native_swa_bundle(types, topology)
    for name, value in change.items():
        setattr(impl, name, value)
    with pytest.raises(NotImplementedError, match="coverage certificate"):
        select_native_swa_subset_routes(
            routes,
            window_size=32768,
            max_query_tokens=2048,
            future_offset_bound=7,
            context=context,
            dense_impl_type=DenseBuilder,
        )


@pytest.mark.parametrize("future", [0, 7])
def test_native_swa_coverage_against_absolute_read_sets(types, topology, future):
    routes, _, context = native_swa_bundle(types, topology)
    width, native_window = 8, 4096
    window = native_window + width + future
    select_native_swa_subset_routes(
        routes,
        window_size=window,
        max_query_tokens=width,
        future_offset_bound=future,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    for end in (8, 1151, 1152, 4095, 4096, 4223, 33023):
        for block in (64, 128):
            start = max(end + future - window, 0) // block * block
            for query in range(end - width, end):
                native_read = set(range(max(query - native_window, 0), query + 1))
                assert native_read == {key for key in native_read if key >= start}


@pytest.mark.parametrize("ratio,span", [(4, 128), (128, 4096)])
def test_compressed_geometry_records_raw_and_physical_units(types, topology, ratio, span):
    (route,) = inventory(
        types, topology, specs=[FullSpec(span * 2, ratio)], builders=[CompressedBuilder()], spans=[span]
    )
    assert route.compression_ratio == ratio
    assert route.table_column_span == span
    assert route.physical_block_size == span // ratio
    assert route.family == "DSA"
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


@pytest.mark.parametrize(
    "parallel",
    [
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_cp_needs_coordinate_provider(types, topology, parallel):
    (route,) = inventory(types, replace(topology, **parallel))
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


def test_tp_and_effective_sp_are_separate_from_cp(types, topology):
    parallel = replace(topology, tensor_parallel_size=16, sequence_parallel=True, expert_parallel=True)
    (route,) = inventory(types, parallel)
    # Addressing candidate only: this does not certify asynchronous completion.
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    assert route.topology.sequence_parallel
    assert route.topology.flashcomm_requested is None


def test_every_active_gid_is_retained_and_missing_provider_blocks_bundle(types, topology):
    routes = inventory(
        types,
        topology,
        specs=[FullSpec(), FullSpec(), FullSpec()],
        builders=[DenseBuilder(), None, CompressedBuilder()],
    )
    assert [route.gid for route in routes] == [0, 2]
    assert routes[1].status is DraftWindowProviderStatus.PROVIDER_MISSING
    with pytest.raises(NotImplementedError, match="gid=2"):
        require_draft_window_providers(routes)


def test_uniform_wrapper_is_unwrapped_without_inventing_residency(types, topology):
    (route,) = inventory(types, topology, specs=[UniformSpec({"a": FullSpec(), "b": FullSpec()})])
    assert len(route.kv_specs) == 2
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    assert "resident" not in route.as_dict()


@pytest.mark.parametrize(
    "specs,spans",
    [
        ([FullSpec(256)], [96]),
        ([FullSpec(256, 3)], [128]),
        ([UniformSpec({"a": FullSpec(256), "b": FullSpec(512)})], [128]),
    ],
)
def test_invalid_geometry_is_not_zero_padded_or_repaired(types, topology, specs, spans):
    with pytest.raises(ValueError, match="geometry"):
        inventory(types, topology, specs=specs, spans=spans)


def test_empty_inventory_is_not_success():
    with pytest.raises(ValueError, match="at least one"):
        require_draft_window_providers(())


def test_requested_flag_cannot_override_effective_config():
    parallel = SimpleNamespace(
        tensor_parallel_size=16,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        use_sequence_parallel_moe=False,
        enable_expert_parallel=True,
    )
    topology = DraftWindowTopology.from_config(
        SimpleNamespace(parallel_config=parallel),
        SimpleNamespace(cp_rank=0, cp_interleave=1),
        dsa_cp_effective=False,
        flashcomm_requested=True,
        dsa_cp_requested=True,
    )
    assert topology.flashcomm_requested is True
    assert topology.flashcomm_effective is False
    assert topology.dsa_cp_effective is False


def test_unknown_second_gid_is_never_ignored(types, topology):
    routes = inventory(types, topology, specs=[FullSpec(), FullSpec()], builders=[DenseBuilder(), object()])
    assert len(routes) == 2
    assert routes[1].family == "UNKNOWN"
    with pytest.raises(NotImplementedError, match="gid=1"):
        require_draft_window_providers(routes)


@pytest.mark.parametrize("anchor,second", [(256, 128), (128, 64), (128, 128)])
def test_multi_group_projection_preserves_first_group_start(types, topology, anchor, second):
    routes = inventory(types, topology, specs=[FullSpec(256), FullSpec(256)], spans=[anchor, second])
    require_compatible_direct_prefix_routes(routes)
    # The logical read start is decided once using the first group's B.
    end, window, future = 2051, 1024, 8
    start = max(end + future - window, 0) // anchor * anchor
    columns = [project_draft_window_start(start, route) for route in routes]
    assert all(column * route.table_column_span == start for column, route in zip(columns, routes))
    assert [route.gid for route in routes] == [0, 1]


def test_incompatible_multi_group_alignment_rejected_before_request_arrives(types, topology):
    routes = inventory(types, topology, specs=[FullSpec(256), FullSpec(256)], spans=[64, 128])
    # S=128 happens to be representable in both groups, but S=64 is not.
    # Accepting only the first request would create a serving-time failure.
    assert [project_draft_window_start(128, route) for route in routes] == [2, 1]
    with pytest.raises(NotImplementedError, match="first group span 64"):
        require_compatible_direct_prefix_routes(routes)
    with pytest.raises(ValueError, match="not representable"):
        project_draft_window_start(64, routes[1])


def test_multi_group_projection_does_not_round_or_accept_negative_offset(types, topology):
    (route,) = inventory(types, topology)
    for start in [-128, 1, 129]:
        with pytest.raises(ValueError, match="not representable"):
            project_draft_window_start(start, route)


def test_target_dsa_cp_does_not_reclassify_dense_draft(types, topology):
    (route,) = inventory(types, replace(topology, dsa_cp_effective=True, dsa_cp_config_enabled=True))
    assert route.status is DraftWindowProviderStatus.DIRECT_DENSE
    assert route.topology.dsa_cp_config_enabled is True
    assert route.topology.dsa_cp_effective is False


def test_actual_cp_builder_remains_missing_even_when_requested_flag_is_false(types, topology):
    cp_types = replace(types, builder_families=((CompressedBuilder, "DSA_CP"),))
    (route,) = inventory(cp_types, topology, builders=[CompressedBuilder()])
    assert route.topology.dsa_cp_effective is True
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


class MLABuilder:
    flashmla_state: object | None = None


class MLASpec(FullSpec):
    model_version = None
    cache_sparse_sfa_c8 = False
    cache_sparse_sfa_turboquant = False
    store_on_host = False
    indexes_kv_by_block_stride = False
    non_causal_multi_token_decode = False


def test_plain_mla_provider_requires_actual_builder_and_spec(types, topology):
    mla_types = replace(
        types,
        mla_backend=DenseBackend,
        mla_builder=MLABuilder,
        mla_spec=MLASpec,
        builder_families=((MLABuilder, "MLA"),),
    )
    (route,) = inventory(mla_types, topology, specs=[MLASpec()], builders=[MLABuilder()])
    assert route.status is DraftWindowProviderStatus.DIRECT_MLA
    require_compatible_direct_prefix_routes((route,))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_version", "alternate-layout"),
        ("cache_sparse_sfa_c8", True),
        ("cache_sparse_sfa_turboquant", True),
        ("store_on_host", True),
        ("indexes_kv_by_block_stride", True),
    ],
)
def test_mla_special_layouts_do_not_inherit_plain_read_projection(types, topology, field, value):
    mla_types = replace(types, mla_backend=DenseBackend, mla_builder=MLABuilder, mla_spec=MLASpec)
    spec = MLASpec()
    setattr(spec, field, value)
    (route,) = inventory(mla_types, topology, specs=[spec], builders=[MLABuilder()])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


def test_flashmla_has_separate_persistent_state_contract(types, topology):
    mla_types = replace(types, mla_backend=DenseBackend, mla_builder=MLABuilder, mla_spec=MLASpec)
    builder = MLABuilder()
    builder.flashmla_state = object()
    (route,) = inventory(mla_types, topology, specs=[MLASpec()], builders=[builder])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING
    assert "FlashMLA" in route.reason


class SFABuilder:
    nope = False


class IndexerBuilder:
    use_pcp = False
    use_dcp = False
    use_dsa_cp = False


class IndexerSpec(FullSpec):
    cache_sparse_li_c4 = False
    cache_sparse_li_c8 = False
    sfa_dcp_replicated_indexer_size = 1


def sparse_types(types):
    return replace(
        types,
        mla_spec=MLASpec,
        sfa_backend=DenseBackend,
        sfa_builder=SFABuilder,
        indexer_backend=DenseBackend,
        indexer_builder=IndexerBuilder,
        indexer_spec=IndexerSpec,
        builder_families=((SFABuilder, "SFA"), (IndexerBuilder, "SFA_INDEXER")),
    )


def test_mixed_uniform_group_uses_each_consumers_actual_spec(types, topology):
    specs = {"draft.attn": MLASpec(), "draft.indexer": IndexerSpec()}
    groups = [
        SimpleNamespace(
            backend=DenseBackend,
            layer_names=[name],
            get_metadata_builder=lambda index, builder=builder: builder,
        )
        for name, builder in [("draft.attn", SFABuilder()), ("draft.indexer", IndexerBuilder())]
    ]
    routes = inventory_draft_window_groups(
        attn_groups=[groups],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=UniformSpec(specs))]),
        block_tables=SimpleNamespace(kernel_block_sizes=[128]),
        topology=topology,
        types=sparse_types(types),
    )
    assert [route.status for route in routes] == [
        DraftWindowProviderStatus.DIRECT_SFA,
        DraftWindowProviderStatus.DIRECT_SFA_INDEXER,
    ]
    assert all(route.gid == 0 and len(route.kv_specs) == 1 for route in routes)
    require_compatible_direct_prefix_routes(routes)


def test_indexer_c8_quantization_keeps_token_coordinate_span(types, topology):
    spec = IndexerSpec()
    spec.cache_sparse_li_c8 = True
    (route,) = inventory(sparse_types(types), topology, specs=[spec], builders=[IndexerBuilder()])
    assert route.status is DraftWindowProviderStatus.DIRECT_SFA_INDEXER
    assert route.compression_ratio == 1


@pytest.mark.parametrize("field", ["use_pcp", "use_dcp", "use_dsa_cp"])
def test_indexer_actual_parallel_producer_requires_provider(types, topology, field):
    builder = IndexerBuilder()
    setattr(builder, field, True)
    (route,) = inventory(sparse_types(types), topology, specs=[IndexerSpec()], builders=[builder])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING
    assert route.topology.dsa_cp_effective == (field == "use_dsa_cp")


@pytest.mark.parametrize("field", ["non_causal_multi_token_decode", "cache_sparse_sfa_c8", "store_on_host"])
def test_sfa_unsupported_native_semantics_are_explicit(types, topology, field):
    spec = MLASpec()
    setattr(spec, field, True)
    (route,) = inventory(sparse_types(types), topology, specs=[spec], builders=[SFABuilder()])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


def test_sfa_nope_state_is_not_plain_rope_provider(types, topology):
    builder = SFABuilder()
    builder.nope = True
    (route,) = inventory(sparse_types(types), topology, specs=[MLASpec()], builders=[builder])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING
    assert "NoPE" in route.reason


class DSABuilder:
    compressor_ratio = 1
    _device_metadata_enabled = False
    model_config = SimpleNamespace(hf_config=SimpleNamespace(sliding_window=128, vision_n_layers=0))


class DSACPBuilder(DSABuilder):
    pass


class DSASpec(FullSpec):
    model_version = None
    indexes_kv_by_block_stride = False
    bounded_replay = False
    sliding_window = 128


def dsa_types(types):
    return replace(
        types,
        dsa_backend=DenseBackend,
        dsa_builder=DSABuilder,
        dsa_cp_builder=DSACPBuilder,
        dsa_spec=DSASpec,
        builder_families=((DSACPBuilder, "DSA_CP"), (DSABuilder, "DSA")),
    )


@pytest.mark.parametrize("builder,status", [(DSABuilder(), "DIRECT_DSA"), (DSACPBuilder(), "PROVIDER_MISSING")])
def test_uncompressed_dsa_requires_native_window_and_actual_owner(types, topology, builder, status):
    (route,) = inventory(dsa_types(types), topology, specs=[DSASpec()], builders=[builder])
    assert route.status.value == status
    assert route.topology.dsa_cp_effective == isinstance(builder, DSACPBuilder)
    if isinstance(builder, DSACPBuilder):
        assert "MRV2 startup contract" in route.reason
        with pytest.raises(NotImplementedError, match="Legacy DSA-CP"):
            require_draft_window_providers((route,))


@pytest.mark.parametrize(
    "field,value", [("sliding_window", 256), ("bounded_replay", True), ("model_version", "packed")]
)
def test_dsa_native_window_and_replay_contracts_do_not_silently_inherit_provider(types, topology, field, value):
    spec = DSASpec()
    setattr(spec, field, value)
    (route,) = inventory(dsa_types(types), topology, specs=[spec], builders=[DSABuilder()])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING


def test_deferred_dsa_producer_requires_separate_completion_contract(types, topology):
    builder = DSABuilder()
    builder._device_metadata_enabled = True
    (route,) = inventory(dsa_types(types), topology, specs=[DSASpec()], builders=[builder])
    assert route.status is DraftWindowProviderStatus.PROVIDER_MISSING
    assert "executor completion" in route.reason


def test_small_window_native_intersection_gap(types, topology):
    routes, _, context = native_swa_bundle(types, topology)
    selected = select_native_swa_subset_routes(
        routes,
        window_size=1024,
        max_query_tokens=8,
        future_offset_bound=7,
        context=context,
        dense_impl_type=DenseBuilder,
    )
    assert len(selected) == 2
    assert selected[0].status is DraftWindowProviderStatus.NATIVE_SWA_INTERSECTION
    assert selected[1].gid == routes[1].gid
