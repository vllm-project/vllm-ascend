#!/bin/bash

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
#
set -euo pipefail
trap 'echo "Error on line $LINENO: command \`$BASH_COMMAND\` failed with exit code $?" >&2' ERR

cd /vllm-workspace
# download fused_infer_attention_score related source files
wget https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/fused_infer_attention_score_a3_$(uname -i).tar.gz
tar -zxvf ./fused_infer_attention_score_a3_$(uname -i).tar.gz

# replace fused_infer_attention_score operation files
cd $ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93
rm -rf fused_infer_attention_score
cp -r /vllm-workspace/fused_infer_attention_score_a3_$(uname -i)/fused_infer_attention_score .

# replace related so
cd $ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -i)
rm libopmaster_ct.so libopmaster_rt2.0.so liboptiling.so
cp /vllm-workspace/fused_infer_attention_score_a3_$(uname -i)/*.so .
