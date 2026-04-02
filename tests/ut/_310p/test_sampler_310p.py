from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_ascend._310p.model_runner_310p import (  # noqa: E402
    _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN,
    NPUModelRunner310,
)
from vllm_ascend._310p.sample.sampler import (  # noqa: E402
    AscendSampler310,
    _random_sample_310p,
)
from vllm_ascend.sample.sampler import AscendSampler  # noqa: E402
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ  # noqa: E402


def _clone_cpu_generator(generator: torch.Generator) -> torch.Generator:
    cloned = torch.Generator(device="cpu")
    cloned.set_state(generator.get_state())
    return cloned


def test_random_sample_310p_matches_cpu_reference_and_advances_generators():
    logits = torch.tensor(
        [
            [1.0, 0.5, -0.5, -1.0],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    generators = {
        0: torch.Generator(device="cpu").manual_seed(7),
        1: torch.Generator(device="cpu").manual_seed(11),
    }
    expected_generators = {
        idx: _clone_cpu_generator(generator)
        for idx, generator in generators.items()
    }

    expected_uniforms = torch.empty((logits.shape[0], 1), dtype=torch.float32)
    for idx, generator in expected_generators.items():
        expected_uniforms[idx].uniform_(generator=generator)

    expected_cdf = logits.softmax(dim=-1, dtype=torch.float32).cumsum(dim=-1)
    expected_cdf[:, -1] = 1.0
    expected = torch.searchsorted(
        expected_cdf, expected_uniforms, right=False
    ).to(torch.int64).view(-1)

    sampled = _random_sample_310p(logits, generators)

    assert torch.equal(sampled, expected)
    for idx, generator in generators.items():
        assert torch.equal(
            generator.get_state(),
            expected_generators[idx].get_state(),
        )


def test_apply_temperature_310p_uses_safe_cpu_reciprocal():
    logits = torch.tensor(
        [[2.0, 4.0], [1.5, 3.0]],
        dtype=torch.float16,
    )
    temperature = torch.tensor([0.0, 0.5], dtype=torch.float32)

    out = AscendSampler310.apply_temperature(
        logits.clone(),
        temperature,
        all_random=False,
    )

    expected = torch.tensor(
        [[2.0, 4.0], [3.0, 6.0]],
        dtype=torch.float16,
    )
    assert torch.equal(out, expected)


def test_sample_310p_materializes_logits_before_sampling():
    sampler = AscendSampler310()
    logits = torch.randn(2, 4, dtype=torch.float32)
    copied_logits = torch.randn(2, 4, dtype=torch.float32)
    sampling_metadata = SimpleNamespace(all_greedy=False)
    expected = (torch.tensor([1, 2]), None)

    with (
        patch(
            "vllm_ascend._310p.sample.sampler._copy_to_default_format",
            return_value=copied_logits,
        ) as mock_copy,
        patch.object(
            AscendSampler,
            "sample",
            autospec=True,
            return_value=expected,
        ) as mock_super_sample,
    ):
        out = sampler.sample(logits, sampling_metadata, "processed_logits")

    mock_copy.assert_called_once_with(logits)
    mock_super_sample.assert_called_once_with(
        sampler,
        copied_logits,
        sampling_metadata,
        "processed_logits",
    )
    assert out == expected


def test_sample_310p_skips_materialize_for_all_greedy_requests():
    sampler = AscendSampler310()
    logits = torch.randn(2, 4, dtype=torch.float32)
    sampling_metadata = SimpleNamespace(all_greedy=True)
    expected = (torch.tensor([1, 2]), None)

    with (
        patch(
            "vllm_ascend._310p.sample.sampler._copy_to_default_format",
        ) as mock_copy,
        patch.object(
            AscendSampler,
            "sample",
            autospec=True,
            return_value=expected,
        ) as mock_super_sample,
    ):
        out = sampler.sample(logits, sampling_metadata, "processed_logits")

    mock_copy.assert_not_called()
    mock_super_sample.assert_called_once_with(
        sampler,
        logits,
        sampling_metadata,
        "processed_logits",
    )
    assert out == expected


def test_model_runner_310_installs_sampler_and_rebuilds_rejection_sampler():
    def fake_super_init(self, *args, **kwargs):
        self.rejection_sampler = object()
        self.speculative_config = None
        self.cudagraph_dispatcher = SimpleNamespace(uniform_decode_query_len=8)

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner.__init__",
            new=fake_super_init,
        ),
        patch(
            "vllm_ascend._310p.model_runner_310p.RejectionSampler",
            side_effect=lambda sampler: ("rebuilt", sampler),
        ) as mock_rejection_sampler,
    ):
        runner = NPUModelRunner310(MagicMock(), "npu")

    assert isinstance(runner.sampler, AscendSampler310)
    assert runner._acl_format == ACL_FORMAT_FRACTAL_NZ
    assert runner.rejection_sampler == ("rebuilt", runner.sampler)
    mock_rejection_sampler.assert_called_once_with(runner.sampler)


def test_model_runner_310_aligns_ngram_uniform_decode_query_len():
    def fake_super_init(self, *args, **kwargs):
        self.rejection_sampler = None
        self.speculative_config = SimpleNamespace(method="ngram")
        self.cudagraph_dispatcher = SimpleNamespace(uniform_decode_query_len=8)

    with patch(
        "vllm_ascend._310p.model_runner_310p.NPUModelRunner.__init__",
        new=fake_super_init,
    ):
        runner = NPUModelRunner310(MagicMock(), "npu")

    assert (
        runner.cudagraph_dispatcher.uniform_decode_query_len
        == _NGRAM_GRAPH_UNIFORM_DECODE_QUERY_LEN
    )
