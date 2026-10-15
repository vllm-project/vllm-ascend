#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#


from vllm.logger import logger

from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.domino_proposer import AscendDominoProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


def _is_domino_draft(vllm_config) -> bool:
    """A DFlash draft checkpoint whose projector_type is "domino" is a Domino
    draft. ``hf_config_override`` remaps its architecture to
    ``DFlashDominoDraftModel`` at config-load time (and vllm's EAGLEConfig
    preserves that name), so we detect it via the architecture (falling back to
    ``dflash_config.projector_type`` for robustness)."""
    spec_cfg = getattr(vllm_config, "speculative_config", None)
    if spec_cfg is None:
        return False
    draft_cfg = getattr(spec_cfg, "draft_model_config", None)
    if draft_cfg is None:
        return False
    hf_config = getattr(draft_cfg, "hf_config", None)
    if hf_config is None:
        return False
    architectures = getattr(hf_config, "architectures", None) or []
    if "DFlashDominoDraftModel" in architectures:
        return True
    dflash_config = getattr(hf_config, "dflash_config", {}) or {}
    return dflash_config.get("projector_type") in ("domino", "causal_v5")


def get_spec_decode_method(method, vllm_config, device, runner):
    if method == "ngram":
        return AscendNgramProposer(vllm_config, runner)
    elif method == "ngram_gpu":
        return AscendNgramProposerNPU(vllm_config, device, runner)
    elif method == "suffix":
        return AscendSuffixDecodingProposer(vllm_config, runner)
    elif method == "medusa":
        return AscendMedusaProposer(vllm_config, device)
    elif method == "dspark":
        return AscendDSparkProposer(vllm_config, device, runner)
    elif method in ("eagle", "eagle3", "mtp"):
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.use_step3p5_mtp():
            return AscendStep3p5MTPProposer(vllm_config, device, runner)
        return AscendEagleProposer(vllm_config, device, runner)
    elif method == "dflash":
        # Domino is a DFlash draft with a GRU correction head; it is selected by
        # the checkpoint (projector_type == "domino"), not by a separate method
        # string (vllm's SpeculativeMethod Literal has no "domino").
        if _is_domino_draft(vllm_config):
            logger.info(
                "[spec_decode] Domino draft checkpoint detected "
                "(architectures=DFlashDominoDraftModel); dispatching to "
                "AscendDominoProposer for method='dflash'."
            )
            return AscendDominoProposer(vllm_config, device, runner)
        logger.info("[spec_decode] Dispatching to AscendDflashProposer (DFlash) for method='dflash'.")
        return AscendDflashProposer(vllm_config, device, runner)
    elif method == "draft_model":
        return AscendDraftModelProposer(vllm_config, device, runner)
    elif method == "extract_hidden_states":
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
    else:
        raise ValueError(f"Unknown speculative decoding method: {method}")
