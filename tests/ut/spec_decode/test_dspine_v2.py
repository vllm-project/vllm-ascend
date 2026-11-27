# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

try:
    from vllm_ascend.worker.v2.spec_decode.dspine.speculator import AscendDSpineSpeculator
except ModuleNotFoundError as exc:
    if exc.name not in (
        "vllm.v1.worker.gpu.spec_decode.dspine",
        "vllm.v1.worker.gpu.spec_decode.dspine.speculator",
    ):
        raise
    pytest.skip("DSpine requires a paired vLLM revision with DSpine support", allow_module_level=True)


@pytest.mark.parametrize(
    "adaptive,draft_tp,target_tp,fp64",
    [
        pytest.param(True, 1, 1, False, id="adaptive-verification"),
        pytest.param(False, 1, 2, False, id="shared-target-tp-runtime"),
        pytest.param(False, 1, 1, True, id="fp64-checked-by-shared-kernel"),
    ],
)
def test_dspine_constructor_defers_to_shared_runtime(monkeypatch, adaptive, draft_tp, target_tp, fp64):
    hf_config = SimpleNamespace(dflash_config={"transition_top_k": 2})
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            enable_adaptive_verification=adaptive,
            draft_tensor_parallel_size=draft_tp,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=target_tp),
    )

    def initialize_runtime(self, vllm_config, device):
        # Device/attention setup is outside this constructor regression.
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.max_num_reqs = 2
        self.num_speculative_steps = 3
        self.device = device
        self.enable_adaptive_verification = vllm_config.speculative_config.enable_adaptive_verification
        self.use_fp64_gumbel = fp64

    monkeypatch.setattr(DFlashSpeculator, "__init__", initialize_runtime)
    monkeypatch.setattr(AscendDFlashSpeculator, "_lmhead_tp_validate_draft_sampling", lambda self: None)

    # Keep the real DSpine and DFlash2 constructors and their sampler allocation.
    speculator = AscendDSpineSpeculator(config, torch.device("cpu"))

    assert speculator.enable_adaptive_verification is adaptive
    assert speculator.use_fp64_gumbel is fp64
    assert type(speculator.candidate_sampler) is CandidateSampler
    assert speculator.candidate_sampler.scores.shape == (2, 3, 2)
