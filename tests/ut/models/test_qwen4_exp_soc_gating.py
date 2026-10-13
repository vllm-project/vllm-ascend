# SPDX-License-Identifier: Apache-2.0
"""SoC gating guards for Qwen4Exp QSA performance switches on Ascend.

VLLM_ASCEND_FORCE_QSA_REFERENCE must cover both QSA stages (top-k selection
and sparse attention) so non-validated SoCs keep a fully portable reference
path. The Lightning Indexer and qsa_expand_e3 switches are honored only when
the current hardware profile advertises the matching capability; otherwise
they fall back to the portable implementations instead of crashing.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.qwen4_exp import lightning_indexer
from vllm_ascend.models.qwen4_exp import qsa as ascend_qsa


def _make_indexer() -> ascend_qsa.AscendQSAIndexer:
    indexer = object.__new__(ascend_qsa.AscendQSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.compressed_key_cache = SimpleNamespace(kv_cache=torch.zeros(1))
    indexer.token_topk = 2048
    indexer.compress_ratio = 4
    return indexer


def _select_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        token_to_req=torch.zeros(1, dtype=torch.int64),
        logical_positions=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.arange(2, dtype=torch.int32),
    )


def _recorder(calls: dict, name: str):
    def _impl(*args, **kwargs):
        calls[name] = kwargs
        return None

    return _impl


@pytest.fixture
def select_recorders(monkeypatch) -> dict:
    calls: dict = {}
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_reference", _recorder(calls, "reference"))
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_triton", _recorder(calls, "triton"))
    monkeypatch.setattr(ascend_qsa, "qsa_select_paged_tokens_lightning", _recorder(calls, "lightning"))
    return calls


def _set_switches(monkeypatch, *, force=False, lightning=False, e3v=False) -> None:
    monkeypatch.setattr(ascend_qsa.envs, "VLLM_ASCEND_FORCE_QSA_REFERENCE", force)
    monkeypatch.setattr(ascend_qsa.envs, "VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER", lightning)
    monkeypatch.setattr(ascend_qsa.envs, "VLLM_ASCEND_ENABLE_QSA_E3V", e3v)


def _set_capabilities(monkeypatch, supported: frozenset) -> None:
    profile = SimpleNamespace(supports=lambda capability: capability in supported)
    monkeypatch.setattr(ascend_qsa, "get_current_hardware_profile", lambda: profile)


def test_select_force_reference_wins_over_perf_switches(monkeypatch, select_recorders) -> None:
    _set_switches(monkeypatch, force=True, lightning=True, e3v=True)
    _set_capabilities(monkeypatch, frozenset())
    indexer = _make_indexer()
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["reference"]


@pytest.mark.parametrize("switch", ["e3v", "lightning"])
def test_select_falls_back_on_unsupported_soc(monkeypatch, select_recorders, switch) -> None:
    _set_switches(monkeypatch, **{switch: True})
    _set_capabilities(monkeypatch, frozenset())
    indexer = _make_indexer()
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["triton"]
    assert not select_recorders["triton"].get("use_e3", False)


def test_select_e3v_honored_when_capability_present(monkeypatch, select_recorders) -> None:
    _set_switches(monkeypatch, e3v=True)
    _set_capabilities(monkeypatch, frozenset({ascend_qsa.HardwareCapability.QSA_E3_EXPAND}))
    indexer = _make_indexer()
    monkeypatch.setattr(indexer, "_lightning_indexer_eligible", lambda *args: True)
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["triton"]
    assert select_recorders["triton"].get("use_e3") is True


def test_select_lightning_honored_when_capability_present(monkeypatch, select_recorders) -> None:
    _set_switches(monkeypatch, lightning=True)
    _set_capabilities(monkeypatch, frozenset({ascend_qsa.HardwareCapability.QSA_LIGHTNING_INDEXER}))
    indexer = _make_indexer()
    monkeypatch.setattr(indexer, "_lightning_indexer_eligible", lambda *args: True)
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["lightning"]
    assert select_recorders["lightning"].get("use_e3") is False


def test_select_lightning_full_shapes_bypasses_portable_shape_guard(monkeypatch, select_recorders) -> None:
    _set_switches(monkeypatch, lightning=True)
    _set_capabilities(
        monkeypatch,
        frozenset(
            {
                ascend_qsa.HardwareCapability.QSA_LIGHTNING_INDEXER,
                ascend_qsa.HardwareCapability.QSA_LIGHTNING_INDEXER_FULL_SHAPES,
            }
        ),
    )
    indexer = _make_indexer()
    monkeypatch.setattr(indexer, "_lightning_indexer_eligible", lambda *args: False)
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["lightning"]


@pytest.mark.parametrize("device_type", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_select_defaults_to_lightning_on_a3_and_a5(monkeypatch, select_recorders, device_type) -> None:
    _set_switches(monkeypatch)
    monkeypatch.setattr(
        ascend_qsa,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(device_type),
    )
    indexer = _make_indexer()
    monkeypatch.setattr(indexer, "_lightning_indexer_eligible", lambda *args: False)
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["lightning"]


def test_select_lightning_rejects_incompatible_cache_shape(monkeypatch, select_recorders) -> None:
    _set_switches(monkeypatch, lightning=True)
    _set_capabilities(monkeypatch, frozenset({ascend_qsa.HardwareCapability.QSA_LIGHTNING_INDEXER}))
    indexer = _make_indexer()
    monkeypatch.setattr(indexer, "_lightning_indexer_eligible", lambda *args: False)
    indexer._select(torch.zeros(1, 4, 128), _select_metadata(), None)
    assert list(select_recorders) == ["triton"]


@pytest.mark.parametrize("page_size", [192, 384])
def test_lightning_indexer_accepts_a3_and_a5_page_sizes(page_size: int) -> None:
    output = lightning_indexer.qsa_select_paged_tokens_lightning(
        query=torch.empty(0, 8, 128, dtype=torch.bfloat16),
        compressed_key_cache=torch.empty(1, page_size, 1, 128, dtype=torch.bfloat16),
        block_table=torch.empty(0, 1, dtype=torch.int32),
        token_to_req=torch.empty(0, dtype=torch.int64),
        query_positions=torch.empty(0, dtype=torch.int64),
        sequence_lengths=torch.empty(0, dtype=torch.int32),
        query_start_loc=torch.zeros(1, dtype=torch.int32),
        token_topk=2048,
        compress_ratio=4,
    )
    assert output.shape == (0, 515)


def test_lightning_indexer_rejects_unsupported_page_size() -> None:
    with pytest.raises(ValueError, match=r"\[pages,192\|384,1,128\]"):
        lightning_indexer.qsa_select_paged_tokens_lightning(
            query=torch.empty(0, 8, 128, dtype=torch.bfloat16),
            compressed_key_cache=torch.empty(1, 200, 1, 128, dtype=torch.bfloat16),
            block_table=torch.empty(0, 1, dtype=torch.int32),
            token_to_req=torch.empty(0, dtype=torch.int64),
            query_positions=torch.empty(0, dtype=torch.int64),
            sequence_lengths=torch.empty(0, dtype=torch.int32),
            query_start_loc=torch.zeros(1, dtype=torch.int32),
            token_topk=2048,
            compress_ratio=4,
        )


@pytest.mark.parametrize("page_size, expected", [(192, True), (384, True), (200, False)])
def test_lightning_eligibility_matches_supported_page_sizes(page_size: int, expected: bool) -> None:
    indexer = _make_indexer()
    indexer.compressed_key_cache.kv_cache = torch.empty(1, page_size, 1, 128)
    assert indexer._lightning_indexer_eligible(
        torch.empty(1, 8, 128, dtype=torch.bfloat16),
        _select_metadata(),
    ) is expected


def _make_impl() -> ascend_qsa.AscendQSAImpl:
    impl = object.__new__(ascend_qsa.AscendQSAImpl)
    impl.head_size = 2
    return impl


def _forward_qsa(impl) -> None:
    impl.forward_qsa(
        layer=SimpleNamespace(topk_indices_buffer=torch.zeros(1, 1, dtype=torch.int32)),
        query=torch.zeros(1, 2, 2),
        key=torch.zeros(0),
        value=torch.zeros(0),
        kv_cache=(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2)),
        attn_metadata=SimpleNamespace(
            num_actual_tokens=1,
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
        ),
        output=torch.zeros(1, 2, 2),
        token_to_req=torch.zeros(1, dtype=torch.int64),
    )


@pytest.mark.parametrize("force_reference, backend", [(True, "reference"), (False, "triton")])
def test_forward_qsa_dispatch(monkeypatch, force_reference, backend) -> None:
    calls: dict = {}
    monkeypatch.setattr(ascend_qsa, "qsa_sparse_paged_attention", _recorder(calls, "triton"))
    monkeypatch.setattr(ascend_qsa, "qsa_sparse_paged_attention_reference", _recorder(calls, "reference"))
    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", _recorder(calls, "gbsa"))
    _set_capabilities(monkeypatch, frozenset())
    monkeypatch.setattr(ascend_qsa.envs, "VLLM_ASCEND_FORCE_QSA_REFERENCE", force_reference)
    _forward_qsa(_make_impl())
    assert list(calls) == [backend]


def test_forward_qsa_uses_gbsa_when_capability_present(monkeypatch) -> None:
    calls: dict = {}
    monkeypatch.setattr(ascend_qsa, "qsa_sparse_paged_attention", _recorder(calls, "triton"))
    monkeypatch.setattr(ascend_qsa, "qsa_sparse_paged_attention_reference", _recorder(calls, "reference"))
    monkeypatch.setattr(ascend_qsa, "qsa_gbsa_attention", _recorder(calls, "gbsa"))
    _set_capabilities(monkeypatch, frozenset({ascend_qsa.HardwareCapability.QSA_GBSA_ATTENTION}))
    monkeypatch.setattr(ascend_qsa.envs, "VLLM_ASCEND_FORCE_QSA_REFERENCE", False)
    _forward_qsa(_make_impl())
    assert list(calls) == ["gbsa"]
