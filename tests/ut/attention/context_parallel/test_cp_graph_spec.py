# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the CP graph-mode contract."""

from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.graph_spec import (
    DSA_FAMILY,
    CPGraphMode,
    CPGraphSpec,
    attention_family,
    cp_piecewise_capability,
    cp_piecewise_capability_for_config,
)


class _CUDAGraphMode:
    """Duck-typed stand-in for ``vllm.config.CUDAGraphMode``."""

    def __init__(self, piecewise: bool, full: bool) -> None:
        self._piecewise = piecewise
        self._full = full

    def has_piecewise_cudagraphs(self) -> bool:
        return self._piecewise

    def has_full_cudagraphs(self) -> bool:
        return self._full


PIECEWISE = _CUDAGraphMode(piecewise=True, full=False)
FULL = _CUDAGraphMode(piecewise=False, full=True)
FULL_AND_PIECEWISE = _CUDAGraphMode(piecewise=True, full=True)


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, CPGraphMode.NONE),
        (FULL, CPGraphMode.FULL),
        (PIECEWISE, CPGraphMode.PIECEWISE),
        # FULL_AND_PIECEWISE is resolved per batch; without a hint the
        # piecewise interpretation wins for mixed and prefill batches.
        (FULL_AND_PIECEWISE, CPGraphMode.PIECEWISE),
        ("FULL_DECODE_ONLY", CPGraphMode.FULL),
        ("NONE", CPGraphMode.NONE),
    ],
)
def test_from_cudagraph_mode(mode, expected):
    assert CPGraphMode.from_cudagraph_mode(mode) is expected


def test_from_runtime_prefers_batch_hint():
    # A decode-only routine on FULL_AND_PIECEWISE replays the full graph.
    decode = CPGraphSpec.from_runtime(FULL_AND_PIECEWISE, full_graph=True, padded_num_tokens=8, padded_num_reqs=8)
    assert decode.is_full
    assert decode.pads_requests

    # The same configuration runs mixed batches through piecewise graphs.
    mixed = CPGraphSpec.from_runtime(FULL_AND_PIECEWISE, full_graph=False, padded_num_tokens=64)
    assert mixed.is_piecewise
    assert mixed.padded_num_reqs is None


def test_full_only_configuration_does_not_upgrade_prefill():
    # FULL_DECODE_ONLY offers full graphs, but prefill on the same config is eager.
    spec = CPGraphSpec.from_runtime(FULL, full_graph=False, padded_num_tokens=64)
    assert spec.is_none
    assert not spec.uses_padded_token_extent


def test_piecewise_spec_never_pads_requests():
    spec = CPGraphSpec.from_runtime(PIECEWISE, padded_num_tokens=64, padded_num_reqs=16)
    assert spec.is_piecewise
    assert spec.padded_num_reqs is None
    assert spec.padded_num_tokens == 64
    assert spec.uses_padded_token_extent
    assert not spec.needs_graph_param_update
    with pytest.raises(ValueError):
        CPGraphSpec(mode=CPGraphMode.PIECEWISE, padded_num_reqs=4)


def test_full_spec_needs_graph_param_update():
    spec = CPGraphSpec.from_runtime(FULL, full_graph=True, padded_num_tokens=8, padded_num_reqs=8)
    assert spec.is_full
    assert spec.needs_graph_param_update
    assert spec.as_capture().capturing


def test_negative_token_extent_is_rejected():
    with pytest.raises(ValueError):
        CPGraphSpec(mode=CPGraphMode.NONE, padded_num_tokens=-1)


@pytest.mark.parametrize("family", ["mla", "sfa", "gqa"])
def test_existing_cp_families_keep_piecewise(family):
    capability = cp_piecewise_capability(
        attention_family=family,
        pcp_size=4,
        dcp_size=2,
        cudagraph_mode=PIECEWISE,
    )
    assert capability.supported


def test_dsa_piecewise_requires_pcp_only():
    ok = cp_piecewise_capability(
        attention_family=DSA_FAMILY,
        pcp_size=4,
        dcp_size=1,
        cudagraph_mode=PIECEWISE,
    )
    assert ok.supported

    rejected = cp_piecewise_capability(
        attention_family=DSA_FAMILY,
        pcp_size=4,
        dcp_size=2,
        cudagraph_mode=PIECEWISE,
    )
    assert not rejected.supported
    assert "PCP-only" in rejected.reason
    with pytest.raises(NotImplementedError):
        rejected.raise_if_unsupported()


def test_unknown_family_is_rejected_but_eager_is_allowed():
    rejected = cp_piecewise_capability(
        attention_family="fa3",
        pcp_size=2,
        dcp_size=1,
        cudagraph_mode=PIECEWISE,
    )
    assert not rejected.supported

    eager = cp_piecewise_capability(
        attention_family="fa3",
        pcp_size=2,
        dcp_size=1,
        cudagraph_mode=None,
    )
    assert eager.supported


@pytest.mark.parametrize(
    "hf_text_config,use_mla,expected",
    [
        (SimpleNamespace(compress_ratios=[1, 2]), True, DSA_FAMILY),
        (SimpleNamespace(index_topk=2048), True, "sfa"),
        (SimpleNamespace(), True, "mla"),
        (SimpleNamespace(), False, "gqa"),
        (None, True, "mla"),
    ],
)
def test_attention_family(hf_text_config, use_mla, expected):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config, use_mla=use_mla),
    )
    assert attention_family(vllm_config) == expected


def test_capability_for_config_reads_parallel_sizes():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(compress_ratios=[1]), use_mla=True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=4, decode_context_parallel_size=2),
        compilation_config=SimpleNamespace(cudagraph_mode=PIECEWISE),
    )
    capability = cp_piecewise_capability_for_config(vllm_config)
    assert not capability.supported
    assert "decode_context_parallel_size=1" in capability.reason
