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


def test_random_sample_310p_matches_reference_and_advances_generators():
    probs = torch.tensor(
        [
            [0.50, 0.25, 0.15, 0.10],
            [0.10, 0.20, 0.30, 0.40],
        ],
        dtype=torch.float32,
    )
    original_probs = probs.clone()
    generators = {
        0: torch.Generator(device="cpu").manual_seed(7),
        1: torch.Generator(device="cpu").manual_seed(11),
    }
    expected_generators = {
        idx: _clone_cpu_generator(generator)
        for idx, generator in generators.items()
    }

    expected_uniforms = torch.stack(
        [
            torch.rand(
                (probs.shape[1],),
                dtype=probs.dtype,
                generator=expected_generators[idx],
            )
            for idx in range(probs.shape[0])
        ],
        dim=0,
    )
    min_uniform = torch.tensor(
        torch.finfo(expected_uniforms.dtype).tiny,
        dtype=expected_uniforms.dtype,
    )
    expected_q = -torch.log(torch.maximum(expected_uniforms, min_uniform))
    expected = torch.topk(torch.div(probs, expected_q), k=1, dim=-1).indices.view(-1)

    sampled = _random_sample_310p(probs, generators)

    assert torch.equal(sampled, expected)
    assert torch.equal(probs, original_probs)
    for idx, generator in generators.items():
        assert torch.equal(
            generator.get_state(),
            expected_generators[idx].get_state(),
        )


def test_random_sample_310p_handles_partial_generators_like_upstream():
    probs = torch.tensor(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.10, 0.30, 0.40, 0.20],
        ],
        dtype=torch.float32,
    )
    generators = {
        1: torch.Generator(device="cpu").manual_seed(23),
    }
    expected_generator = _clone_cpu_generator(generators[1])

    expected_uniform_rows = []
    for idx in range(probs.shape[0]):
        if idx == 1:
            expected_uniform_rows.append(
                torch.rand(
                    (probs.shape[1],),
                    dtype=probs.dtype,
                    generator=expected_generator,
                )
            )
        else:
            expected_uniform_rows.append(
                torch.rand(
                    (probs.shape[1],),
                    dtype=probs.dtype,
                )
            )
    expected_uniforms = torch.stack(expected_uniform_rows, dim=0)
    min_uniform = torch.tensor(
        torch.finfo(expected_uniforms.dtype).tiny,
        dtype=expected_uniforms.dtype,
    )
    expected_q = -torch.log(torch.maximum(expected_uniforms, min_uniform))
    expected = torch.topk(torch.div(probs, expected_q), k=1, dim=-1).indices.view(-1)

    sampled = _random_sample_310p(probs, generators)

    assert torch.equal(sampled, expected)
    assert torch.equal(generators[1].get_state(), expected_generator.get_state())


def test_apply_temperature_310p_uses_safe_cpu_reciprocal():
    logits = torch.tensor(
        [[2.0, 4.0], [1.5, 3.0]],
        dtype=torch.float16,
    )
    original_logits = logits.clone()
    temperature = torch.tensor([0.0, 0.5], dtype=torch.float32)

    out = AscendSampler310.apply_temperature(
        logits,
        temperature,
        all_random=False,
    )

    expected = torch.tensor(
        [[2.0, 4.0], [3.0, 6.0]],
        dtype=torch.float16,
    )
    assert torch.equal(out, expected)
    assert torch.equal(logits, original_logits)


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


def test_apply_penalties_310p_is_explicit_noop():
    logits = torch.randn(2, 4, dtype=torch.float32)
    original_logits = logits.clone()
    sampling_metadata = SimpleNamespace(no_penalties=False)

    out = AscendSampler310.apply_penalties(logits, sampling_metadata, [[1], [2]])

    assert out is logits
    assert torch.equal(out, original_logits)


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
