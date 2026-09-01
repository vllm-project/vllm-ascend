from unittest.mock import patch

import torch

import vllm_ascend._310p.ops.fla.l2norm as l2norm_310


def test_l2norm_310p_uses_adn_dispatch_for_fp16():
    x = torch.randn(2, 3, 256, dtype=torch.float16)
    expected = torch.randn_like(x)
    fallback = torch.randn_like(x)

    with (
        patch.object(
            l2norm_310,
            "adn_rms_norm_or_fallback",
            return_value=expected.reshape(-1, 256),
        ) as experimental_dispatch,
        patch(
            "torch_npu.npu_rms_norm",
            return_value=(fallback.reshape(-1, 256), None),
        ) as baseline,
    ):
        out = l2norm_310.l2norm_310p(x)

    experimental_dispatch.assert_called_once()
    candidate_x, candidate_weight, candidate_eps = experimental_dispatch.call_args.args
    assert experimental_dispatch.call_args.kwargs == {}
    assert torch.equal(candidate_x, x.reshape(-1, 256))
    assert torch.equal(
        candidate_weight,
        torch.full((256,), 1.0 / (256**0.5), dtype=torch.float16),
    )
    assert candidate_eps == 1e-6 / 256
    baseline.assert_not_called()
    assert torch.equal(out, expected)
