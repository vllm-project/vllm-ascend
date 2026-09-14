# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v2.pp_utils import resolve_spec_pp_support


def _make_config(
    architecture: str,
    *,
    method: str = "dspark",
    pipeline_parallel_size: int = 2,
):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        model_config=SimpleNamespace(architecture=architecture),
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "KimiLinearForCausalLM",
        "KimiK3ForCausalLM",
        "KimiK3ForConditionalGeneration",
    ],
)
def test_kimi_k3_dspark_pp_supports_all_target_aliases(architecture):
    support = resolve_spec_pp_support(_make_config(architecture))

    assert support is not None
    assert support.needs_aux_hidden_states
    assert support.bypass_upstream_pp_guard


@pytest.mark.parametrize(
    "config",
    [
        _make_config("KimiLinearForCausalLM", pipeline_parallel_size=1),
        _make_config("UnsupportedForPP"),
        _make_config("KimiLinearForCausalLM", method="eagle"),
    ],
)
def test_kimi_k3_dspark_pp_support_stays_scoped(config):
    assert resolve_spec_pp_support(config) is None
