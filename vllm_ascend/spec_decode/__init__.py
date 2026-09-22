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


from vllm_ascend.spec_decode.dflash2_proposer import AscendDflash2Proposer, is_dflash2_draft
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.spec_decode.eagle_proposer import (
    AscendEagleProposer,
    AscendQwen4ExpMTPProposer,
)
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.spec_decode.gemma4_proposer import AscendGemma4Proposer
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.multi_kv_cache_group_proposer import (
    AscendMultiKVCacheGroupMTPProposer,
    is_multi_kv_cache_group_mtp,
)
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


def _use_qwen4_exp_mtp(speculative_config) -> bool:
    checker = getattr(speculative_config, "use_qwen4_exp_mtp", None)
    if callable(checker) and checker():
        return True
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    model_architectures = tuple(getattr(draft_model_config, "architectures", ()) or ())
    if "Qwen4ExpMTP" in model_architectures:
        return True
    text_config = getattr(draft_model_config, "hf_text_config", None)
    if text_config is None:
        return False
    text_architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return getattr(text_config, "model_type", None) == "qwen4_exp_mtp" or ("Qwen4ExpMTP" in text_architectures)


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
    elif method == "mtp" and vllm_config.speculative_config.use_gemma4_mtp():
        return AscendGemma4Proposer(vllm_config, device, runner)
    elif method in ("eagle", "eagle3", "mtp"):
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.use_step3p5_mtp():
            return AscendStep3p5MTPProposer(vllm_config, device, runner)
        if speculative_config is not None and _use_qwen4_exp_mtp(speculative_config):
            return AscendQwen4ExpMTPProposer(vllm_config, device, runner)
        if method == "mtp" and is_multi_kv_cache_group_mtp(vllm_config):
            return AscendMultiKVCacheGroupMTPProposer(vllm_config, device, runner)
        return AscendEagleProposer(vllm_config, device, runner)
    elif method == "dflash":
        if is_dflash2_draft(vllm_config.speculative_config):
            return AscendDflash2Proposer(vllm_config, device, runner)
        return AscendDflashProposer(vllm_config, device, runner)
    elif method == "draft_model":
        return AscendDraftModelProposer(vllm_config, device, runner)
    elif method == "extract_hidden_states":
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
    else:
        raise ValueError(f"Unknown speculative decoding method: {method}")
