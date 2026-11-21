from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import gdn


def _extended(*, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel, A_log, dt_bias):
    pass


@pytest.fixture
def backend(monkeypatch):
    import fla_npu.ops

    gdn._supports_fused_recurrent_gdn.cache_clear()
    gdn._supports_recurrent_qkv_views.cache_clear()
    monkeypatch.setattr(gdn, "version", lambda _: "26.10.0+main.dev0058bb4")
    monkeypatch.setattr(gdn, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    if hasattr(gdn, "_recurrent_gdn_package_is_current"):
        monkeypatch.setattr(gdn, "_recurrent_gdn_package_is_current", lambda: True)
    op = SimpleNamespace(_get_direct_op=Mock(return_value=_extended))
    monkeypatch.setattr(fla_npu.ops, "ascendc", op)
    yield op
    gdn._supports_fused_recurrent_gdn.cache_clear()
    gdn._supports_recurrent_qkv_views.cache_clear()


def test_verified_main_launcher_can_enable_fusion(backend):
    assert gdn._supports_fused_recurrent_gdn()


def test_verified_main_can_consume_strided_qkv(backend):
    assert gdn._supports_recurrent_qkv_views()


@pytest.mark.parametrize("ver", ["26.10.0+main.dev2ef0735", "26.10.1", "999.0"])
def test_unvalidated_recurrent_fusion_stays_disabled(backend, monkeypatch, ver):
    monkeypatch.setattr(gdn, "version", lambda _: ver)
    assert not gdn._supports_fused_recurrent_gdn()


def test_new_version_with_old_launcher_stays_disabled(backend):
    backend._get_direct_op.return_value = lambda: None
    assert not gdn._supports_fused_recurrent_gdn()


def test_version_metadata_cannot_enable_shadowed_python_package(backend, monkeypatch):
    monkeypatch.setattr(gdn, "_recurrent_gdn_package_is_current", lambda: False, raising=False)
    assert not gdn._supports_fused_recurrent_gdn()


def test_fusion_tensor_contract_preserves_noncontiguous_state():
    qkv = torch.empty(8, 8192, dtype=torch.bfloat16)
    a = torch.empty(8, 32, dtype=torch.bfloat16)
    state = torch.empty(10, 32, 128, 160)[..., :128]
    assert gdn._recurrent_fusion_inputs_supported(qkv, a, a, state, 16, 32)


@pytest.mark.parametrize(
    "kind", ["fp16", "bf16_state", "ratio", "state_dtype", "state_dim", "gate_shape", "gate_stride"]
)
def test_fusion_tensor_contract_rejects_unsupported_metadata(kind):
    qkv = torch.empty(8, 8192, dtype=torch.bfloat16)
    a = torch.empty(8, 32, dtype=torch.bfloat16)
    state = torch.empty(10, 32, 128, 128)
    hk, hv = 16, 32
    if kind == "fp16":
        qkv = qkv.to(torch.float16)
    if kind == "bf16_state":
        state = state.to(torch.bfloat16)
    if kind == "ratio":
        hk = 3
    if kind == "state_dtype":
        state = state.to(torch.float16)
    if kind == "state_dim":
        state = state[..., :64]
    if kind == "gate_shape":
        a = a[:, :16]
    if kind == "gate_stride":
        a = torch.empty(8, 64, dtype=torch.bfloat16)[:, :32]
    assert not gdn._recurrent_fusion_inputs_supported(qkv, a, a, state, hk, hv)
