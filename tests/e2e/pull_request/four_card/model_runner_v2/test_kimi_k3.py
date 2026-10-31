# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 Model Runner V2 PR smoke coverage.

The MRV1 Kimi-K3 tests already build the smallest useful local model and cover
the model-specific execution paths.  Reuse the model builder and scenario helpers here
so MRV1 and MRV2 keep the same functional coverage.  These are deliberately
functional smoke tests: the local models use dummy weights and do not replace
the full-checkpoint nightly GPQA or AISBench jobs.
"""

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_kimi_k3 import (
    _engine_args,
    _generate,
    _prompt,
    build_k3_models,
    run_k3_gqa_w4a8_dp2_tp2,
    run_k3_mla_block5_tp4,
    run_k3_mla_pd_tp2,
    run_k3_mtp_image_tp4,
)

pytestmark = pytest.mark.e2e_model("Kimi-K3")


@pytest.fixture(scope="module")
def k3_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    return build_k3_models(tmp_path_factory.mktemp("k3-dummy"))


@pytest.fixture(autouse=True)
def k3_runtime_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the reused Kimi smoke scenarios with Model Runner V2."""

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")


def test_k3_mla_block5_tp4(k3_models: dict[str, str]) -> None:
    """Reuse MLA Block5, prefix-cache, and D-Spark coverage with MRV2."""

    run_k3_mla_block5_tp4(k3_models)


def test_k3_gqa_w4a8_dp2_tp2(k3_models: dict[str, str]) -> None:
    """Reuse quantized GQA and local DP coverage with MRV2."""

    run_k3_gqa_w4a8_dp2_tp2(k3_models)


def test_k3_mtp_image_tp4(k3_models: dict[str, str]) -> None:
    """Reuse MTP and multimodal image coverage with MRV2."""

    run_k3_mtp_image_tp4(k3_models)


def test_k3_mla_pd_tp2(k3_models: dict[str, str]) -> None:
    """Reuse same-host P/D and Mooncake KV-transfer coverage with MRV2."""

    run_k3_mla_pd_tp2(k3_models)


def test_k3_mrv2_basic_generate_tp4(k3_models: dict[str, str]) -> None:
    """Exercise the V2 runner without speculative decoding.

    The reused tests intentionally exercise speculative paths.  This smoke
    keeps one plain generation path so a runner or scheduler regression cannot
    be hidden by the speculative-draft setup.
    """

    args = _engine_args(k3_models, "mla")
    args.pop("speculative_config")
    args["max_model_len"] = 1024
    args["enforce_eager"] = True

    with VllmRunner(k3_models["target"], **args) as runner:
        _generate(
            runner.model,
            [_prompt(1), _prompt(129, salt=137)],
        )
