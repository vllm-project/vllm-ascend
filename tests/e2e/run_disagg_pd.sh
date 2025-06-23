#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

set -eo errexit

. $(dirname "$0")/common.sh
. $(dirname "$0")/pd_disaggreate/setup_pd.sh

export VLLM_USE_MODELSCOPE="True"

MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
# TODO: add tp case
TP_SIZE=1

# TODO: support multi-card
prefill_ip=$(/usr/local/Ascend/driver/tools/hccn_tool -i 0 -ip -g | grep "ipaddr" | awk -F: '{print $2}' | xargs)
PREFILL_DEVICE_IPS="[\"$prefill_ip\"]"

decode_ip=$(/usr/local/Ascend/driver/tools/hccn_tool -i 1 -ip -g | grep "ipaddr" | awk -F: '{print $2}' | xargs)
DECODE_DEVICE_IPS="[\"$decode_ip\"]"

_info "====> Start pd disaggregated test"
REGISTER_PORT=10101
PREOXY_PORT=10102
run_proxy_server $REGISTER_PORT $PREOXY_PORT
_info "Started pd disaggregated proxy server"

PREFILL_PROC_NAME="Prefill-instance"
PREFILL_PORT=8001
_info "Starting prefill instance"
run_prefill_instance $MODEL_NAME $TP_SIZE $PREFILL_PORT $REGISTER_PORT $PREFILL_DEVICE_IPS $DECODE_DEVICE_IPS &
_info "Waiting for prefill instance ready"
wait_url_ready $PREFILL_PROC_NAME "http://localhost:${PREFILL_PORT}/v1/completions"

DECODE_PROC_NAME="Decode-instance"
DECODE_PORT=8002
_info "Starting decode instance"
run_decode_instance  $MODEL_NAME $TP_SIZE $DECODE_PORT $REGISTER_PORT $PREFILL_DEVICE_IPS $DECODE_DEVICE_IPS &
_info "Waiting for decode instance ready"
wait_url_ready $DECODE_PROC_NAME "http://localhost:${DECODE_PORT}/v1/completions"

_info "pd disaggregated system is ready for handling request"
