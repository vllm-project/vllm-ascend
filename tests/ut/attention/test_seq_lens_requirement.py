"""Unit tests for the host seq-lens requirement contract.

See https://github.com/vllm-project/vllm-ascend/issues/17479 for the design.

Covers requirement aggregation, the conservative undeclared-consumer
fallback, subclass inheritance of declarations, and participation of
draft-model speculator backends.
"""

from types import SimpleNamespace

from vllm_ascend.attention.utils import (
    HostSeqLensRequirement,
    resolve_host_seq_lens_requirements,
)


class _NoneBuilder:
    HOST_SEQ_LENS_REQUIREMENT = HostSeqLensRequirement.NONE


class _UpperBoundBuilder:
    HOST_SEQ_LENS_REQUIREMENT = HostSeqLensRequirement.UPPER_BOUND


class _ExactBuilder:
    HOST_SEQ_LENS_REQUIREMENT = HostSeqLensRequirement.EXACT


class _UndeclaredBuilder:
    pass


class _SubclassedExactBuilder(_ExactBuilder):
    """Subclasses inherit the declaration of their parent."""


def _make_backend(builder_cls):
    """A fake backend class exposing the real AttentionBackend API shape."""

    class _FakeBackend:
        @staticmethod
        def get_builder_cls():
            return builder_cls

    return _FakeBackend


def _make_group(builder_cls):
    return SimpleNamespace(backend=_make_backend(builder_cls))


def _make_speculator(builder_clses):
    return SimpleNamespace(attn_backends={f"draft_{i}": _make_backend(cls) for i, cls in enumerate(builder_clses)})


def test_aggregation_takes_strongest_requirement():
    groups = [
        [_make_group(_NoneBuilder)],
        [_make_group(_UpperBoundBuilder), _make_group(_NoneBuilder)],
    ]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.UPPER_BOUND


def test_single_exact_consumer_forces_exact_for_hybrid_models():
    groups = [[_make_group(_NoneBuilder), _make_group(_ExactBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.EXACT


def test_all_none_stays_device_native():
    groups = [[_make_group(_NoneBuilder)], [_make_group(_NoneBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.NONE


def test_no_consumers_defaults_to_none():
    result = resolve_host_seq_lens_requirements([])
    assert result.requirement is HostSeqLensRequirement.NONE


def test_undeclared_consumer_falls_back_to_exact_and_is_reported():
    groups = [[_make_group(_UndeclaredBuilder), _make_group(_NoneBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.EXACT
    assert len(result.undeclared) == 1
    assert "_UndeclaredBuilder" in result.undeclared[0]


def test_declaration_is_inherited_by_subclasses():
    groups = [[_make_group(_SubclassedExactBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.EXACT
    # Inherited declarations are not reported as undeclared.
    assert result.undeclared == []


def test_speculator_draft_backends_participate_in_aggregation():
    groups = [[_make_group(_NoneBuilder)]]
    speculator = _make_speculator([_ExactBuilder])
    result = resolve_host_seq_lens_requirements(groups, speculator)
    assert result.requirement is HostSeqLensRequirement.EXACT


def test_speculator_without_attn_backends_is_ignored():
    groups = [[_make_group(_NoneBuilder)]]
    speculator = SimpleNamespace()
    result = resolve_host_seq_lens_requirements(groups, speculator)
    assert result.requirement is HostSeqLensRequirement.NONE


def test_consumer_names_include_backend_and_builder():
    groups = [[_make_group(_ExactBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    exact_names = result.consumers[HostSeqLensRequirement.EXACT]
    assert any("_ExactBuilder" in name for name in exact_names)


def test_in_tree_builders_are_declared():
    """Every in-tree MRV2 metadata builder must declare its requirement.

    Guards the contract rollout (see
    https://github.com/vllm-project/vllm-ascend/issues/17479): a missing
    declaration silently falls back to EXACT and keeps an unnecessary D2H
    dependency alive.
    """
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
    from vllm_ascend.attention.mla_v1 import AscendMLAMetadataBuilder
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.models.minimax_m3.msa_m3 import (
        AscendMiniMaxM3IndexerMetadataBuilder,
        AscendMiniMaxM3SparseMetadataBuilder,
    )
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    # Exact host materialization is reserved for legacy FIA/MLA host-list
    # consumers; device-native families use bounded or no host view.
    expected = {
        AscendAttentionMetadataBuilder: HostSeqLensRequirement.EXACT,
        AscendMLAMetadataBuilder: HostSeqLensRequirement.EXACT,
        AscendMiniMaxM3IndexerMetadataBuilder: HostSeqLensRequirement.EXACT,
        AscendMiniMaxM3SparseMetadataBuilder: HostSeqLensRequirement.EXACT,
        AscendDSAMetadataBuilder: HostSeqLensRequirement.UPPER_BOUND,
        AscendSFAMetadataBuilder: HostSeqLensRequirement.UPPER_BOUND,
        AscendGDNAttentionMetadataBuilder: HostSeqLensRequirement.UPPER_BOUND,
        AscendSFAIndexerMetadataBuilder: HostSeqLensRequirement.NONE,
    }
    for builder_cls, requirement in expected.items():
        assert getattr(builder_cls, "HOST_SEQ_LENS_REQUIREMENT", None) is requirement, (
            f"{builder_cls.__name__} must declare HOST_SEQ_LENS_REQUIREMENT={requirement.name}"
        )


def test_dsv4_stack_aggregates_to_upper_bound():
    """The DSv4 attention stack (all AscendDSABackend) aggregates through the
    real DSA metadata builder to a non-EXACT requirement, so no exact-host
    snapshot is staged or waited for under speculative decoding."""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    groups = [[_make_group(AscendDSAMetadataBuilder)]]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.UPPER_BOUND
    assert result.undeclared == []


def test_hybrid_fia_gdn_model_stays_exact():
    """A hybrid model with one legacy FIA group (e.g. Qwen3-Next full
    attention) plus a device-native GDN group must keep exact materialization
    available: the single EXACT consumer wins the aggregation."""
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

    groups = [
        [_make_group(AscendAttentionMetadataBuilder)],
        [_make_group(AscendGDNAttentionMetadataBuilder)],
    ]
    result = resolve_host_seq_lens_requirements(groups)
    assert result.requirement is HostSeqLensRequirement.EXACT


def test_sfa_stack_with_draft_stays_bounded():
    """SFA target + SFA draft (speculator backends) aggregates to
    UPPER_BOUND: drafting does not pull a bounded family back to EXACT."""
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder

    groups = [[_make_group(AscendSFAMetadataBuilder)]]
    speculator = _make_speculator([AscendSFAMetadataBuilder])
    result = resolve_host_seq_lens_requirements(groups, speculator)
    assert result.requirement is HostSeqLensRequirement.UPPER_BOUND
