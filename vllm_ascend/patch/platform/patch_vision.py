#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

# Historical note: PR #16017 gated an Ascend FusedInputNorm batch_norm/eps
# patch to the release lane. Upstream PR #51734 already rewrote
# FusedInputNorm.forward to a multiply-add on both v0.28.0 and main, so that
# patch must not be installed on either lane. Kept as a no-op module so old
# imports do not break.
