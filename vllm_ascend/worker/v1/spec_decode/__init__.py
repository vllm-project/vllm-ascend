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


from vllm_ascend.worker.spec_decode_family import SpecDecodeFamily, resolve_spec_decode_family
from vllm_ascend.worker.v1.spec_decode.dflash2_proposer import AscendDflash2Proposer
from vllm_ascend.worker.v1.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.worker.v1.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.worker.v1.spec_decode.dspark_proposer import AscendDSparkProposer
from vllm_ascend.worker.v1.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.worker.v1.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.worker.v1.spec_decode.gemma4_proposer import AscendGemma4Proposer
from vllm_ascend.worker.v1.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.worker.v1.spec_decode.multi_kv_cache_group_proposer import AscendMultiKVCacheGroupMTPProposer
from vllm_ascend.worker.v1.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.worker.v1.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.worker.v1.spec_decode.step3p5 import AscendStep3p5MTPProposer
from vllm_ascend.worker.v1.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


def get_spec_decode_method(method, vllm_config, device, runner):
    family = resolve_spec_decode_family(vllm_config, method=method)
    if family is SpecDecodeFamily.NGRAM:
        return AscendNgramProposer(vllm_config, runner)
    if family is SpecDecodeFamily.NGRAM_GPU:
        return AscendNgramProposerNPU(vllm_config, device, runner)
    if family is SpecDecodeFamily.SUFFIX:
        return AscendSuffixDecodingProposer(vllm_config, runner)
    if family is SpecDecodeFamily.MEDUSA:
        return AscendMedusaProposer(vllm_config, device)
    if family is SpecDecodeFamily.DSPARK:
        return AscendDSparkProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.GEMMA4_MTP:
        return AscendGemma4Proposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.STEP3P5_MTP:
        return AscendStep3p5MTPProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.MULTI_KV_CACHE_GROUP_MTP:
        return AscendMultiKVCacheGroupMTPProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.EAGLE:
        return AscendEagleProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.DFLASH2:
        return AscendDflash2Proposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.DFLASH:
        return AscendDflashProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.MTP:
        return AscendEagleProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.DRAFT_MODEL:
        return AscendDraftModelProposer(vllm_config, device, runner)
    if family is SpecDecodeFamily.EXTRACT_HIDDEN_STATES:
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
    raise ValueError(f"Unknown speculative decoding method: {method}")
