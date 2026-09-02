from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def _fake_npu_tensor(*, shape, device=None):
    tensor = MagicMock()
    tensor.dim.return_value = len(shape)
    tensor.numel.return_value = int(torch.tensor(shape).prod().item())
    tensor.dtype = torch.float16
    tensor.shape = shape
    tensor.device = device or SimpleNamespace(type="npu", index=0)
    tensor.is_contiguous.return_value = True
    return tensor


@pytest.fixture(autouse=True)
def _clear_cached_candidate():
    from vllm_ascend._310p.ops.adn_rms_norm import _get_adn_rms_norm_op

    _get_adn_rms_norm_op.cache_clear()
    yield
    _get_adn_rms_norm_op.cache_clear()


@pytest.mark.parametrize("hidden_size", (256,))
def test_adn_rms_norm_dispatches_supported_fp16_shapes(hidden_size):
    from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback

    device = SimpleNamespace(type="npu", index=0)
    x = _fake_npu_tensor(shape=(3, hidden_size), device=device)
    gamma = _fake_npu_tensor(shape=(hidden_size,), device=device)
    expected = object()

    with (
        patch(
            "vllm_ascend._310p.ops.adn_rms_norm.enable_custom_op",
            return_value=True,
        ),
        patch(
            "torch.ops._C_ascend.adn_rms_norm",
            create=True,
            return_value=expected,
        ) as candidate,
        patch("torch_npu.npu_rms_norm") as baseline,
    ):
        actual = adn_rms_norm_or_fallback(x, gamma, 1e-6)

    candidate.assert_called_once_with(x, gamma, 1e-6)
    baseline.assert_not_called()
    assert actual is expected


def test_adn_rms_norm_falls_back_when_custom_extension_is_unavailable():
    from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback

    device = SimpleNamespace(type="npu", index=0)
    x = _fake_npu_tensor(shape=(3, 256), device=device)
    gamma = _fake_npu_tensor(shape=(256,), device=device)
    expected = object()

    with (
        patch(
            "vllm_ascend._310p.ops.adn_rms_norm.enable_custom_op",
            create=True,
            return_value=False,
        ),
        patch("torch.ops._C_ascend.adn_rms_norm", create=True) as candidate,
        patch("torch_npu.npu_rms_norm", return_value=(expected, None)) as baseline,
    ):
        actual = adn_rms_norm_or_fallback(x, gamma, 1e-6)

    candidate.assert_not_called()
    baseline.assert_called_once_with(x, gamma, 1e-6)
    assert actual is expected


def test_adn_rms_norm_falls_back_for_mismatched_npu_devices():
    from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback

    x = _fake_npu_tensor(shape=(3, 256), device=SimpleNamespace(type="npu", index=0))
    gamma = _fake_npu_tensor(shape=(256,), device=SimpleNamespace(type="npu", index=1))
    expected = object()

    with (
        patch("torch.ops._C_ascend.adn_rms_norm", create=True) as candidate,
        patch("torch_npu.npu_rms_norm", return_value=(expected, None)) as baseline,
    ):
        actual = adn_rms_norm_or_fallback(x, gamma, 1e-6)

    candidate.assert_not_called()
    baseline.assert_called_once_with(x, gamma, 1e-6)
    assert actual is expected


def test_adn_rms_norm_falls_back_for_non_vector_gamma():
    from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback

    device = SimpleNamespace(type="npu", index=0)
    x = _fake_npu_tensor(shape=(3, 256), device=device)
    gamma = _fake_npu_tensor(shape=(1, 256), device=device)
    expected = object()

    with (
        patch("torch.ops._C_ascend.adn_rms_norm", create=True) as candidate,
        patch("torch_npu.npu_rms_norm", return_value=(expected, None)) as baseline,
    ):
        actual = adn_rms_norm_or_fallback(x, gamma, 1e-6)

    candidate.assert_not_called()
    baseline.assert_called_once_with(x, gamma, 1e-6)
    assert actual is expected


@pytest.mark.parametrize(
    ("x", "gamma"),
    (
        (torch.randn(2, 128, dtype=torch.float32), torch.ones(128, dtype=torch.float16)),
        (torch.randn(2, 64, dtype=torch.float16), torch.ones(64, dtype=torch.float16)),
        (torch.randn(2, 128, dtype=torch.float16), torch.ones(128, dtype=torch.float16)),
        (torch.randn(2, 2048, dtype=torch.float16), torch.ones(2048, dtype=torch.float16)),
        (torch.randn(128, 2, dtype=torch.float16).T, torch.ones(128, dtype=torch.float16)),
        (torch.randn(2, 128, dtype=torch.float16), torch.ones(128, dtype=torch.float32)),
        (torch.empty(0, 128, dtype=torch.float16), torch.ones(128, dtype=torch.float16)),
        (torch.randn(2, 256, dtype=torch.float16), torch.ones(256, dtype=torch.float16)),
    ),
    ids=(
        "x-fp32",
        "h64",
        "h128-no-gain",
        "h2048-no-gain",
        "noncontiguous",
        "gamma-fp32",
        "empty",
        "cpu",
    ),
)
def test_adn_rms_norm_falls_back_for_unsupported_inputs(x, gamma):
    from vllm_ascend._310p.ops.adn_rms_norm import adn_rms_norm_or_fallback

    expected = torch.randn_like(x)
    with (
        patch("torch.ops._C_ascend.adn_rms_norm", create=True) as candidate,
        patch("torch_npu.npu_rms_norm", return_value=(expected, None)) as baseline,
    ):
        actual = adn_rms_norm_or_fallback(x, gamma, 1e-5)

    candidate.assert_not_called()
    baseline.assert_called_once_with(x, gamma, 1e-5)
    assert actual is expected
