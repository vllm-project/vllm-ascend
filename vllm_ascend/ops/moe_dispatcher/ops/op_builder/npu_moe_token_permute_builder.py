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

from vllm_ascend.ops.moe_dispatcher.ops.op_builder.builder import VllmAscendOpBuilder


class MoeTokenPermuteOpBuilder(VllmAscendOpBuilder):
    OP_NAME = "npu_moe_token_permute"

    def __init__(self):
        super(MoeTokenPermuteOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['csrc/kernels/npu_moe_token_permute.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['csrc/kernels/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args
