#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""V1 model runner zone.

This package contains the model runner v1 core scheduled for removal
(vllm-project/vllm-ascend#18202). Code here may import shared vllm_ascend
modules; nothing outside this package (and its dedicated tests) may import
or reference it. See the V1 code-isolation RFC for the dependency contract.
"""
