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
"""Auto-Bisect tooling for nightly E2E test failures.

Given a failing nightly test case, this package binary-searches the
``vllm-ascend`` commit history between the last-known-good commit and the
currently failing commit to pinpoint the first bad commit (and the PR it
belongs to). It deliberately reuses the existing nightly launch entries
(``test_single_node.py`` / ``multi_node/scripts/run.sh``) so that the bisect
reproduces the exact nightly environment.
"""
