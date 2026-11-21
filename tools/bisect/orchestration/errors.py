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
"""Fatal failures that stop the bisect coordinator."""


class BisectFatalError(RuntimeError):
    """Unrecoverable environment problem; abort the whole search.

    Distinct from per-trial failures (which become SKIP for one commit): raised
    when a multi-node barrier times out and no worker node has ever signalled
    ready -- continuing would burn the full barrier timeout on every remaining
    round, so the bisect aborts with exit code 2 instead.
    """
