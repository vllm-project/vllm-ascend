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
#

import vllm_ascend.patch.worker.patch_common.patch_metrics  # noqa
# TODO: remove the patch on spec decode when
# https://github.com/vllm-project/vllm/pull/15195 and
# https://github.com/vllm-project/vllm-ascend/pull/395
# is merged
import vllm_ascend.patch.worker.patch_common.patch_multi_step_worker  # noqa
import vllm_ascend.patch.worker.patch_common.patch_spec_decode_worker  # noqa
