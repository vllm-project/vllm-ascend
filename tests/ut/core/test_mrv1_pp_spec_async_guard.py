# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MRV1 + PP + speculation + async-scheduling guard."""

from types import SimpleNamespace

import pytest

from vllm_ascend.platform import _validate_mrv1_pp_spec_async


def _config(*, v2=False, pp=2, spec=True, async_scheduling=False):
    return SimpleNamespace(
        use_v2_model_runner=v2,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        speculative_config=SimpleNamespace(method="mtp") if spec else None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )


def test_rejects_async_with_pp_and_spec_on_mrv1():
    with pytest.raises(ValueError, match="Async scheduling is not supported"):
        _validate_mrv1_pp_spec_async(_config(async_scheduling=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"async_scheduling": False},
        {"pp": 1},
        {"spec": False},
        {"v2": True},
    ],
)
def test_allows_supported_combinations(kwargs):
    _validate_mrv1_pp_spec_async(_config(**kwargs))
