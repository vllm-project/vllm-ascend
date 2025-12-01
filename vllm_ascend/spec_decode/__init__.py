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
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer
from vllm_ascend.spec_decode.mtp_proposer import MtpProposer
from vllm_ascend.spec_decode.ngram_proposer import NgramProposer
from vllm_ascend.spec_decode.suffix_proposer import SuffixDecodingProposer
from vllm_ascend.torchair.torchair_mtp_proposer import TorchairMtpProposer


def get_spec_decode_method(method,
                           vllm_config,
                           device,
                           runner,
                           is_torchair_graph=False):
    if method == "ngram":
        return NgramProposer(vllm_config, device, runner)
    elif method in ("eagle", "eagle3"):
        return EagleProposer(vllm_config, device, runner)
    elif method in ('deepseek_mtp', 'qwen3_next_mtp'):
        if is_torchair_graph:
            return TorchairMtpProposer(vllm_config, device, runner)
        return MtpProposer(vllm_config, device, runner)
    elif method == 'suffix':
        return SuffixDecodingProposer(vllm_config, device, runner)
    else:
        raise ValueError("Unknown speculative decoding method: "
                         f"{method}")
