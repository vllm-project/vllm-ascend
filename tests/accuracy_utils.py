"""Tensor accuracy assertions shared by CPU unit tests and NPU operator tests."""

import torch

TOLERANCES = {
    torch.float16: (2**-9, 2**-9, 0.1),
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float32: (2**-10, 2**-16, 0.01),
}


def assert_reference_accuracy(actual, expected):
    """Apply both contribution-guide bounds using the output tensor's dtype."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    rtol, atol, max_error = TOLERANCES[actual.dtype]
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    assert torch.isfinite(actual).all(), "Non-finite actual output"
    assert torch.isfinite(expected).all(), "Non-finite reference output"
    error = (actual - expected).abs()
    within_tolerance = error <= atol + rtol * expected.abs()
    assert within_tolerance.sum().item() >= 0.99 * error.numel()
    assert error.max().item() <= max_error
