from unittest.mock import MagicMock

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import set_current_vllm_config

import vllm_ascend._310p.ops.fla.l2norm as l2norm_310
import vllm_ascend._310p.ops.layernorm as layernorm_310
from vllm_ascend._310p.ops.layernorm import AscendRMSNorm310, AscendRMSNormGated310
from vllm_ascend.utils import enable_custom_op

ROWS_BY_H = {
    128: (1, 6, 9, 48, 240, 960, 2048, 9216, 14336),
    256: (1, 6, 17, 48, 240, 960, 2048),
    2048: (1, 6, 48, 240, 241, 960),
}
SHAPES = tuple((rows, hidden) for hidden, rows_values in ROWS_BY_H.items() for rows in rows_values)


@pytest.fixture(scope="module", autouse=True)
def _load_custom_op_extension():
    assert enable_custom_op(), "failed to load vllm_ascend custom-op extension"


def _candidate():
    op = getattr(torch.ops._C_ascend, "adn_rms_norm", None)
    assert op is not None, "adn_rms_norm has not been registered"
    return op


@pytest.mark.parametrize(("rows", "hidden_size"), SHAPES)
@pytest.mark.parametrize("epsilon", (1e-6, 1e-5))
def test_adn_rms_norm_matches_native(rows, hidden_size, epsilon):
    torch.manual_seed(20260831 + rows + hidden_size)
    x = torch.randn(rows, hidden_size, dtype=torch.float16, device="npu")
    gamma = torch.randn(hidden_size, dtype=torch.float16, device="npu")
    x_before = x.clone()
    gamma_before = gamma.clone()

    expected = torch_npu.npu_rms_norm(x, gamma, epsilon)[0]
    actual = _candidate()(x, gamma, epsilon)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device
    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    torch.testing.assert_close(gamma, gamma_before, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("zero", "near-zero", "large", "single-extreme"))
def test_adn_rms_norm_h128_boundary_inputs(kind):
    torch.manual_seed(20260831)
    if kind == "zero":
        x = torch.zeros(60, 4, 128, dtype=torch.float16)
    elif kind == "near-zero":
        x = torch.randn(60, 4, 128, dtype=torch.float16) * 1e-4
    elif kind == "large":
        x = torch.randn(60, 4, 128, dtype=torch.float16) * 100
    else:
        x = torch.zeros(60, 4, 128, dtype=torch.float16)
        x[:, :, 0] = 1000
        x[:, :, 1] = -1000

    x = x.npu()
    gamma = torch.randn(128, dtype=torch.float16, device="npu")
    expected = torch_npu.npu_rms_norm(x, gamma, 1e-6)[0]
    actual = _candidate()(x, gamma, 1e-6)

    assert torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("hidden_size", (128, 256, 2048))
def test_adn_rms_norm_acl_graph_replay_matches_native(hidden_size):
    torch.manual_seed(20260831 + hidden_size)
    rows = 960 if hidden_size != 2048 else 240
    x = torch.randn(rows, hidden_size, dtype=torch.float16, device="npu")
    gamma = torch.randn(hidden_size, dtype=torch.float16, device="npu")
    candidate = _candidate()

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = candidate(x, gamma, 1e-6)

    for _ in range(3):
        replay_x = torch.randn_like(x)
        expected = torch_npu.npu_rms_norm(replay_x, gamma, 1e-6)[0]
        x.copy_(replay_x)
        captured.zero_()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert torch.isfinite(captured).all().item()
        torch.testing.assert_close(captured, expected, rtol=2e-3, atol=2e-3)


def test_adn_rms_norm_binding_rejects_non_vector_gamma():
    x = torch.randn(9, 256, dtype=torch.float16, device="npu")
    gamma = torch.randn(1, 256, dtype=torch.float16, device="npu")

    with pytest.raises(RuntimeError, match="one-dimensional gamma"):
        _candidate()(x, gamma, 1e-6)


def test_adn_rms_norm_binding_rejects_mixed_devices():
    x = torch.randn(9, 256, dtype=torch.float16, device="npu")
    gamma = torch.randn(256, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="same NPU device"):
        _candidate()(x, gamma, 1e-6)


@pytest.mark.parametrize(
    ("x", "gamma", "message"),
    (
        (
            torch.empty(2, 64, dtype=torch.float16, device="meta"),
            torch.empty(64, dtype=torch.float16, device="meta"),
            "shape.*128, 256, 2048",
        ),
        (
            torch.empty(2, 256, dtype=torch.float32, device="meta"),
            torch.empty(256, dtype=torch.float16, device="meta"),
            "FP16 x",
        ),
        (
            torch.empty(2, 256, dtype=torch.float16, device="meta"),
            torch.empty(1, 256, dtype=torch.float16, device="meta"),
            "one-dimensional gamma",
        ),
    ),
)
def test_adn_rms_norm_meta_rejects_invalid_contract(x, gamma, message):
    with pytest.raises(RuntimeError, match=message):
        _candidate()(x, gamma, 1e-6)


def _vllm_config():
    config = MagicMock()
    config.compilation_config.custom_ops = ["all"]
    config.quant_config = None
    return config


def test_adn_rms_norm_real_layernorm_entries_match_native():
    torch.manual_seed(20260831)
    with set_current_vllm_config(_vllm_config()):
        layer = AscendRMSNorm310(hidden_size=256, eps=1e-6, dtype=torch.float16).npu()
        gated = AscendRMSNormGated310(hidden_size=256, eps=1e-6, dtype=torch.float16).npu()
    x = torch.randn(480, 256, dtype=torch.float16, device="npu")

    expected = torch_npu.npu_rms_norm(x, layer.weight, layer.variance_epsilon)[0]
    actual = layer.forward_oot(x)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    expected_gated = torch_npu.npu_rms_norm(x, gated.weight, gated.eps)[0]
    actual_gated = gated.forward_oot(x)
    torch.testing.assert_close(actual_gated, expected_gated, rtol=2e-3, atol=2e-3)


def test_adn_rms_norm_real_l2norm_entry_matches_native():
    torch.manual_seed(20260831)
    x = torch.randn(60, 8, 256, dtype=torch.float16, device="npu")
    weight = torch.full((256,), 1.0 / (256**0.5), dtype=torch.float16, device="npu")
    expected = torch_npu.npu_rms_norm(x.reshape(-1, 256), weight, 1e-6 / 256)[0].reshape_as(x)

    actual = l2norm_310.l2norm_310p(x)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_adn_rms_norm_real_layernorm_entry_acl_graph_replay():
    torch.manual_seed(20260831)
    with set_current_vllm_config(_vllm_config()):
        layer = AscendRMSNorm310(hidden_size=256, eps=1e-6, dtype=torch.float16).npu()
    x = torch.randn(241, 256, dtype=torch.float16, device="npu")

    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = layer.forward_oot(x)

    for _ in range(3):
        replay_x = torch.randn_like(x)
        expected = torch_npu.npu_rms_norm(replay_x, layer.weight, layer.variance_epsilon)[0]
        x.copy_(replay_x)
        captured.zero_()
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, expected, rtol=2e-3, atol=2e-3)


def test_adn_rms_norm_torch_compile_reuses_dynamic_row_graph():
    """Changing only the row count must not specialize a second graph."""
    candidate = _candidate()
    compile_count = 0

    def counting_backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def rms_norm(x, gamma):
        return candidate(x, gamma, 1e-6)

    compiled = torch.compile(
        rms_norm,
        backend=counting_backend,
        dynamic=True,
        fullgraph=True,
    )
    gamma = torch.randn(256, dtype=torch.float16, device="npu")

    for rows in (17, 48):
        torch.manual_seed(20260831 + rows)
        x = torch.randn(rows, 256, dtype=torch.float16, device="npu")
        expected = torch_npu.npu_rms_norm(x, gamma, 1e-6)[0]
        actual = compiled(x, gamma)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    assert compile_count == 1
