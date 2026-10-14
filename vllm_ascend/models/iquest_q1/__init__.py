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
"""IQuest-Q1 (``model_type: iquest_q1``) support for Ascend NPU.

Upstream vLLM does not ship this architecture; it lives in the out-of-tree
plugin ``IQuestLab/vllm-iquest-q1``.  Only the config is imported eagerly so
that ``AutoConfig`` can be registered; the model module is registered lazily
by architecture name.
"""

from .configs import IQuestQ1Config, IQuestQ1MTPConfig

__all__ = ["IQuestQ1Config", "IQuestQ1MTPConfig"]
