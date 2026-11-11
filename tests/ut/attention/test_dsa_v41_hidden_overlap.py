# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for overlapping PCP hidden gather with local Q_a projection.

Collectives and device streams are mocked; these tests do not establish real
HCCL overlap or graph replay correctness.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_v41_cp as cp


@pytest.fixture
def overlap_case(monkeypatch):
    impl = cp.AscendDSAV41PCPImpl.__new__(cp.AscendDSAV41PCPImpl)
    impl.multistream_dsv4_dsa_overlap = False
    # Only the fields consumed by the orchestration under test are needed.
    swa = cp.AscendDSAV41PCPMetadata.__new__(cp.AscendDSAV41PCPMetadata)
    swa.num_actual_tokens = 2
    swa.local_num_tokens_after_padding = 3
    swa.hidden_restore_idx = torch.tensor([3, 0, 4, 1])
    swa.global_metadata = SimpleNamespace(num_prefills=1, num_decodes=0)
    metadata = SimpleNamespace(swa=swa)
    projector = SimpleNamespace(
        cv_wq_a=SimpleNamespace(_has_communication=False),
        _parallel_o_proj=SimpleNamespace(all_gather_weights=Mock()),
    )
    attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=projector)))
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(cp, "get_forward_context", lambda: context)
    return SimpleNamespace(impl=impl, metadata=metadata, projector=projector, attn=attn, context=context)


@pytest.mark.parametrize("counts", [(1, 0), (0, 2), (1, 2), (0, 0)])
@pytest.mark.parametrize("local_tokens", [0, 2])
@pytest.mark.parametrize("multistream", [False, True])
def test_overlap_eligibility_uses_global_counts(overlap_case, counts, local_tokens, multistream):
    case = overlap_case
    case.impl.multistream_dsv4_dsa_overlap = multistream
    case.metadata.swa.num_actual_tokens = local_tokens
    case.metadata.swa.global_metadata = SimpleNamespace(num_prefills=counts[0], num_decodes=counts[1])
    assert case.impl._can_overlap_hidden_gather(case.metadata, case.projector) == bool(sum(counts))


@pytest.mark.parametrize("mode", [mode for mode in CUDAGraphMode if mode != CUDAGraphMode.NONE] + [None])
def test_graph_or_unknown_mode_disables_overlap(overlap_case, mode):
    case = overlap_case
    case.context.cudagraph_runtime_mode = mode
    assert not case.impl._can_overlap_hidden_gather(case.metadata, case.projector)


@pytest.mark.parametrize("metadata_kind", ["ordinary", "missing_global"])
def test_non_global_metadata_disables_overlap(overlap_case, metadata_kind):
    case = overlap_case
    if metadata_kind == "ordinary":
        case.metadata.swa = SimpleNamespace()
    else:
        case.metadata.swa.global_metadata = None
    assert not case.impl._can_overlap_hidden_gather(case.metadata, case.projector)


@pytest.mark.parametrize("communication", [True, None])
def test_communicating_or_unknown_cv_projection_disables_overlap(overlap_case, communication):
    case = overlap_case
    case.impl.multistream_dsv4_dsa_overlap = True
    case.projector.cv_wq_a = SimpleNamespace()
    if communication is not None:
        case.projector.cv_wq_a._has_communication = communication
    assert not case.impl._can_overlap_hidden_gather(case.metadata, case.projector)


@pytest.mark.parametrize("live", [0, 1, 2])
@pytest.mark.parametrize("has_work", [False, True])
def test_gather_projects_before_wait_and_restores_padded_layout(monkeypatch, overlap_case, live, has_work):
    case = overlap_case
    case.metadata.swa.num_actual_tokens = live
    # A noncontiguous input checks that the complete padded send is contiguous.
    hidden = torch.arange(12, dtype=torch.float32).view(2, 6).T[::2]
    assert hidden.shape == (3, 2) and not hidden.is_contiguous()
    peer = torch.tensor([[101.0, 102.0], [103.0, 104.0], [-7.0, -7.0]])
    expected_global = torch.stack((peer[0], hidden[0], peer[1], hidden[1]))
    events = []
    group = object()
    monkeypatch.setattr(cp, "get_pcp_group", lambda: group)
    gathered = torch.empty(6, 2)

    def complete():
        gathered.copy_(torch.cat((hidden, peer)))

    def wait():
        events.append("wait")
        complete()

    def gather(send, actual_group):
        assert actual_group is group
        assert send.is_contiguous()
        torch.testing.assert_close(send, hidden)
        events.append("gather")
        if not has_work:
            complete()
        return gathered, SimpleNamespace(wait=wait) if has_work else None

    def project(attn, query):
        events.append("project")
        assert attn is case.attn
        torch.testing.assert_close(query, hidden[:live])
        return query + 10

    def preprocess(attn, query, cache, local_metadata, global_metadata, *, q_a):
        events.append("preprocess")
        assert attn is case.attn and query is hidden
        assert local_metadata is case.metadata and global_metadata is global_layer
        torch.testing.assert_close(cache, expected_global)
        if live:
            torch.testing.assert_close(q_a, hidden[:live] + 10)
        else:
            assert q_a is None
        return "prepared"

    global_layer = object()
    case.impl._global_layer_metadata = lambda mapping: global_layer
    case.projector._parallel_o_proj.all_gather_weights.side_effect = lambda: events.append("weights")
    case.impl._project_q_a = project
    case.impl._preprocess = preprocess
    monkeypatch.setattr(cp, "all_gather_async", gather)
    assert case.impl._prepare_inputs_and_caches(case.attn, hidden, case.metadata, {}) == "prepared"
    expected = ["gather"] + (["project"] if live else []) + (["wait"] if has_work else [])
    assert events == expected + ["weights", "preprocess"]


def test_projection_failure_still_waits_for_hidden_gather(monkeypatch, overlap_case):
    case = overlap_case
    events = []
    work = SimpleNamespace(wait=lambda: events.append("wait"))
    monkeypatch.setattr(cp, "get_pcp_group", lambda: object())
    monkeypatch.setattr(cp, "all_gather_async", lambda *args: (torch.empty(6, 2), work))
    case.impl._project_q_a = Mock(side_effect=RuntimeError("projection failed"))
    case.impl._preprocess = Mock()
    with pytest.raises(RuntimeError, match="projection failed"):
        case.impl._prepare_inputs_and_caches(case.attn, torch.ones(3, 2), case.metadata, {})
    assert events == ["wait"]
    case.projector._parallel_o_proj.all_gather_weights.assert_not_called()
    case.impl._preprocess.assert_not_called()


@pytest.mark.parametrize("reason", ["graph", "dummy", "communication"])
def test_fallback_keeps_synchronous_collective_order(monkeypatch, overlap_case, reason):
    case = overlap_case
    if reason == "graph":
        case.context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    elif reason == "dummy":
        case.metadata.swa.global_metadata = SimpleNamespace(num_prefills=0, num_decodes=0)
    else:
        case.impl.multistream_dsv4_dsa_overlap = True
        case.projector.cv_wq_a._has_communication = True
    hidden, cache, global_layer = torch.ones(3, 2), torch.ones(4, 2), object()
    events = []
    case.impl._gather_and_restore_hidden_states = Mock(side_effect=lambda *args: (events.append("sync"), cache)[1])
    case.projector._parallel_o_proj.all_gather_weights.side_effect = lambda: events.append("weights")
    case.impl._global_layer_metadata = lambda mapping: global_layer
    case.impl._preprocess = Mock(side_effect=lambda *args: events.append("preprocess"))
    async_gather = Mock(side_effect=AssertionError("unexpected async gather"))
    monkeypatch.setattr(cp, "all_gather_async", async_gather)
    case.impl._prepare_inputs_and_caches(case.attn, hidden, case.metadata, {})
    assert events == ["sync", "weights", "preprocess"]
    case.impl._preprocess.assert_called_once_with(case.attn, hidden, cache, case.metadata, global_layer)
    async_gather.assert_not_called()


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("multistream", [False, True])
def test_preprocess_reuses_q_a_and_preserves_outputs(monkeypatch, precomputed, multistream):
    impl = dsa_v41.AscendDSAV41Impl.__new__(dsa_v41.AscendDSAV41Impl)
    impl.multistream_dsv4_dsa_overlap = multistream
    impl.role = SimpleNamespace(is_kv_source=False)
    impl._apply_rotary = Mock()
    impl._prepare_indexer_inputs = Mock(return_value=None)
    impl._quantize_indexer_query = Mock()
    impl._write_swa_cache_on_main_stream = lambda *args: False
    writes = []
    impl._write_swa_cache = lambda attn, metadata, kv: writes.append(kv.clone())
    impl._project_kv = lambda attn, hidden, *args: hidden * 3
    stream = SimpleNamespace(
        record_event=lambda: object(), wait_event=lambda event: None, wait_stream=lambda other: None
    )
    monkeypatch.setattr(dsa_v41.torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(dsa_v41, "dsv4_dsa_overlap_stream", lambda: stream)
    monkeypatch.setattr(dsa_v41, "npu_stream_switch", lambda *args, **kwargs: nullcontext())

    def wrapper(factor):
        return SimpleNamespace(
            _quant_method=None,
            _has_communication=False,
            quantize=Mock(side_effect=lambda value: (value, None)),
            matmul=Mock(side_effect=lambda value, scale, **kwargs: value * factor),
        )

    projector = SimpleNamespace(cv_wq_a=wrapper(2), cv_wkv=wrapper(3), cv_wq_b=wrapper(4))
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=projector)),
        wq_a=Mock(side_effect=lambda value: value * 2),
        wq_b=Mock(side_effect=lambda value: value * 4),
        wkv=SimpleNamespace(bias=None),
        q_norm=lambda value: value + 1,
        kv_norm=lambda value: value,
        head_dim=2,
        rotary_emb=SimpleNamespace(layername="rope"),
    )
    attn.wq_a.bias = attn.wq_b.bias = None
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-7.0, -7.0]])
    cache = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [-9.0, -9.0]])

    def metadata(count):
        return SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=count), rope=lambda *args: (None, None))

    q_a = impl._project_q_a(attn, hidden[:2]) if precomputed else None
    q, qr, prepared = impl._preprocess(attn, hidden, cache, metadata(2), metadata(3), q_a=q_a)
    torch.testing.assert_close(qr, hidden[:2] * 2 + 1)
    torch.testing.assert_close(q, (hidden[:2] * 2 + 1).mul(4).view(2, 1, 2))
    assert prepared is None and len(writes) == 1
    torch.testing.assert_close(writes[0], cache[:3] * 3)
    if multistream:
        assert projector.cv_wq_a.quantize.call_count == projector.cv_wq_a.matmul.call_count == 1
        attn.wq_a.assert_not_called()
        torch.testing.assert_close(projector.cv_wkv.quantize.call_args.args[0], cache[:3])
    else:
        assert attn.wq_a.call_count == 1


def test_ordinary_metadata_delegates_without_new_collectives(monkeypatch, overlap_case):
    case = overlap_case
    metadata = SimpleNamespace(swa=SimpleNamespace())
    hidden, mapping = torch.ones(2, 2), {}
    base_prepare = Mock(return_value="base")
    monkeypatch.setattr(cp._ReplicatedCacheImpl, "_prepare_inputs_and_caches", base_prepare)
    async_gather = Mock(side_effect=AssertionError("unexpected collective"))
    monkeypatch.setattr(cp, "all_gather_async", async_gather)
    assert case.impl._prepare_inputs_and_caches(case.attn, hidden, metadata, mapping) == "base"
    base_prepare.assert_called_once_with(case.attn, hidden, metadata, mapping)
    async_gather.assert_not_called()
    case.projector._parallel_o_proj.all_gather_weights.assert_not_called()


def test_invalid_padding_fails_before_collective(monkeypatch, overlap_case):
    case = overlap_case
    gather = Mock()
    monkeypatch.setattr(cp, "all_gather_async", gather)
    with pytest.raises(AssertionError):
        case.impl._prepare_inputs_and_caches(case.attn, torch.ones(2, 2), case.metadata, {})
    gather.assert_not_called()


def test_wait_failure_stops_before_cache_consumption(monkeypatch, overlap_case):
    case = overlap_case
    work = SimpleNamespace(wait=Mock(side_effect=RuntimeError("gather failed")))
    monkeypatch.setattr(cp, "get_pcp_group", lambda: object())
    monkeypatch.setattr(cp, "all_gather_async", lambda *args: (torch.empty(6, 2), work))
    case.impl._project_q_a = Mock(return_value=torch.ones(2, 2))
    case.impl._preprocess = Mock()
    with pytest.raises(RuntimeError, match="gather failed"):
        case.impl._prepare_inputs_and_caches(case.attn, torch.ones(3, 2), case.metadata, {})
    work.wait.assert_called_once()
    case.projector._parallel_o_proj.all_gather_weights.assert_not_called()
    case.impl._preprocess.assert_not_called()
